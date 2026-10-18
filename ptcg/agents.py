"""Agents: random, heuristic, policy (raw network), legacy (old submissions).

Every agent exposes  act(obs: dict) -> list[int]  and  reset().
The MCTS agent lives in ptcg/mcts.py.
"""
import importlib
import os
import random
import sys

import numpy as np
import torch

from .features import encode, stack, MAX_OPTS
from .model import to_torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_deck(path):
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    return [int(x) for x in open(path).read().split()]


def is_deck_request(obs):
    sel = obs.get("select")
    return sel is None or sel.get("maxCount") == 60


class RandomAgent:
    name = "random"

    def __init__(self, deck=None, seed=None):
        self.deck = deck
        self.rng = random.Random(seed)

    def reset(self):
        pass

    def act(self, obs):
        if is_deck_request(obs):
            return list(self.deck)
        s = obs["select"]
        n = len(s["option"])
        k = self.rng.randint(s["minCount"], s["maxCount"]) if s["maxCount"] > 0 else 0
        return sorted(self.rng.sample(range(n), min(k, n)))


class HeuristicAgent:
    """Greedy rule baseline: attack with the biggest attack, otherwise develop the board."""
    name = "heuristic"
    PRIO = {13: 100, 9: 80, 8: 70, 10: 60, 7: 50, 12: 5, 11: 1, 14: -100}

    def __init__(self, deck=None, seed=None):
        self.deck = deck
        self.rng = random.Random(seed)

    def reset(self):
        pass

    def act(self, obs):
        if is_deck_request(obs):
            return list(self.deck)
        from .cards import ATTACK_INFO
        s = obs["select"]
        opts = s["option"]
        scores = []
        for i, o in enumerate(opts):
            t = o.get("type", 0)
            sc = self.PRIO.get(t, 0) + self.rng.random()
            if t == 13:
                a = ATTACK_INFO.get(o.get("attackId") or 0)
                sc += (a.damage if a else 0) / 10.0
            if t == 1:  # YES
                sc += 1
            scores.append(sc)
        order = list(np.argsort(scores)[::-1])
        k = s["maxCount"] if s["type"] in (1,) and s["context"] in (1, 2, 5, 6, 7) else max(s["minCount"], 1 if s["maxCount"] >= 1 else 0)
        return sorted(int(i) for i in order[:k])


@torch.no_grad()
def net_eval(model, obs_list, device="cpu"):
    """Batch network evaluation. Returns list of (probs[n_opt], value, count_ratio, belief_logits|None, enc)."""
    encs = [encode(o) for o in obs_list]
    arr = stack(encs)
    # trim padding: attention/FFN cost scales with sequence length (exact: padded tokens are masked)
    nc = max(1, int(arr["card_mask"].sum(1).max()))
    no = max(1, int(arr["opt_mask"].sum(1).max()))
    for k in ("card_ids", "card_f", "card_mask"):
        arr[k] = arr[k][:, :nc]
    for k in ("opt_f", "opt_attack", "opt_card", "opt_ref", "opt_mask"):
        arr[k] = arr[k][:, :no]
    b = to_torch(arr, device)
    out = model(b)
    logits = out["logits"].float().cpu().numpy()
    values = out["value"].float().cpu().numpy()
    counts = torch.sigmoid(out["count"].float()).cpu().numpy()
    bel = out["belief"].float().cpu().numpy() if "belief" in out else None
    res = []
    for i, e in enumerate(encs):
        n = min(e["n_opt"], MAX_OPTS)
        lg = logits[i, :n]
        p = np.exp(lg - lg.max()); p /= p.sum()
        res.append((p, float(values[i]), float(counts[i]), None if bel is None else bel[i], e))
    return res


def policy_choice(obs, probs, count_ratio, greedy=True, rng=None):
    """Turn network outputs into a legal selection (handles multi-select counts)."""
    s = obs["select"]
    n_all = len(s["option"])
    kmin, kmax = int(s["minCount"]), int(s["maxCount"])
    n = len(probs)
    if kmax == 0 or n == 0:
        return []
    if kmax == kmin:
        k = kmin
    else:
        k = int(round(kmin + count_ratio * (kmax - kmin)))
    k = max(kmin, min(kmax, k, n))
    if greedy or rng is None:
        order = list(np.argsort(-probs))
    else:
        order = list(rng.choice(n, size=n, replace=False, p=probs))
    sel = sorted(int(i) for i in order[:k])
    if len(sel) < kmin:  # options beyond MAX_OPTS (rare): fill deterministically
        extra = [i for i in range(n_all) if i not in sel]
        sel = sorted(sel + extra[:kmin - len(sel)])
    return sel


class PolicyAgent:
    name = "policy"

    def __init__(self, model, deck, greedy=True, seed=None, device="cpu"):
        self.model = model
        self.deck = deck
        self.greedy = greedy
        self.rng = np.random.default_rng(seed)
        self.device = device

    def reset(self):
        pass

    def act(self, obs):
        if is_deck_request(obs):
            return list(self.deck)
        s = obs["select"]
        if len(s["option"]) == 1 and s["minCount"] >= 1:
            return [0]
        p, v, c, _, _ = net_eval(self.model, [obs], self.device)[0]
        return policy_choice(obs, p, c, self.greedy, self.rng)


class LegacyAgent:
    """Wraps an old Kaggle submission folder (main.py with agent(obs)).
    Only one legacy package can be loaded per process (they share module names)."""

    def __init__(self, folder):
        folder = folder if os.path.isabs(folder) else os.path.join(ROOT, folder)
        self.folder = folder
        for m in ["main", "model_dual_cross", "feature_parser", "card_tracker", "ptcg_agents", "search_agent",
                  "card_static_table", "card_text_encoder", "features_spec"]:
            sys.modules.pop(m, None)
        sys.path.insert(0, folder)
        _orig_load = torch.load
        torch.load = lambda *a, **k: _orig_load(*a, **{**k, "map_location": "cpu"})   # old code saved CUDA tensors
        self.mod = importlib.import_module("main")
        self.mod.BASE_DIR = folder
        self.mod.MODEL_PATH = os.path.join(folder, "weights.pt")
        self.mod.DECK_PATH = os.path.join(folder, "deck.csv")
        self.deck = load_deck(os.path.join(folder, "deck.csv"))
        self.name = "legacy:" + os.path.basename(folder)

    def reset(self):
        self.mod.agent({"select": None})   # re-initialise its trackers for a new game

    def act(self, obs):
        if is_deck_request(obs):
            return list(self.deck)
        return [int(x) for x in self.mod.agent(obs)]
