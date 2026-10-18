"""Offline evaluation on the held-out TEST split (never used for model selection).

python -m ptcg.metrics --ckpt experiments/train_full_s0/best.pt --name full_s0
Writes experiments/metrics_<name>.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ptcg.model import load_model  # noqa: E402
from ptcg.train import load_split, batch_at  # noqa: E402

SELTYPE = {0: "MAIN", 1: "CARD", 2: "ATTACHED_CARD", 3: "CARD_OR_ATTACHED", 4: "ENERGY", 5: "SKILL",
           6: "ATTACK", 7: "EVOLVE", 8: "COUNT", 9: "YES_NO", 10: "SPECIAL_COND"}


def auroc(scores, labels):
    scores = np.asarray(scores, np.float64); labels = np.asarray(labels, bool)
    npos, nneg = labels.sum(), (~labels).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def avg_precision(scores, labels):
    o = np.argsort(-np.asarray(scores)); l = np.asarray(labels, bool)[o]
    if l.sum() == 0:
        return float("nan")
    hits = np.cumsum(l); prec = hits / np.arange(1, len(l) + 1)
    return float((prec * l).sum() / l.sum())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--shards", default="data/shards")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(os.path.join(ROOT, args.ckpt), dev)
    data = load_split(args.shards, args.split)
    train_freq = None
    n = len(data["k"])
    print("samples", n, flush=True)
    LOG = defaultdict(list)
    for i in range(0, n, 2048):
        idx = torch.arange(i, min(n, i + 2048))
        b = batch_at(data, idx, dev)
        out = model(b)
        LOG["logits"].append(out["logits"].float().cpu())
        LOG["value"].append(out["value"].float().cpu())
        if "belief" in out:
            LOG["belief"].append(out["belief"].float().cpu())
    logits = torch.cat(LOG["logits"]).numpy()
    value = torch.cat(LOG["value"]).numpy()
    tgt = data["target"].float().numpy()
    om = data["opt_mask"].numpy()
    nopt = om.sum(1)
    single = tgt.sum(1) == 1
    seltype = data["seltype"].numpy()
    drag = data["my_deck_drag"].numpy() == 1
    res = {"ckpt": args.ckpt, "split": args.split, "n": int(n)}
    # ---------------- policy
    lp = logits - logits.max(1, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
    a_true = tgt.argmax(1)
    rank = (logits > logits[np.arange(n), a_true][:, None]).sum(1)
    nll = -lp[np.arange(n), a_true]

    def pol_stats(mask):
        m = mask & single
        if m.sum() == 0:
            return None
        return {"n": int(m.sum()), "top1": float((rank[m] == 0).mean()), "top3": float((rank[m] < 3).mean()),
                "nll": float(nll[m].mean()), "uniform_top1": float((1.0 / nopt[m]).mean()),
                "uniform_nll": float(np.log(nopt[m]).mean()), "mean_options": float(nopt[m].mean())}
    res["policy_all"] = pol_stats(np.ones(n, bool))
    res["policy_dragapult_player"] = pol_stats(drag)
    res["policy_by_seltype"] = {SELTYPE.get(t, str(t)): pol_stats(seltype == t) for t in np.unique(seltype) if pol_stats(seltype == t)}
    res["policy_main_ge5_options"] = pol_stats((seltype == 0) & (nopt >= 5))
    # ---------------- value
    y = data["value"].numpy()
    vm = y != 0
    p = (value[vm] + 1) / 2; yy = (y[vm] + 1) / 2
    turn = data["turn"].numpy()[vm]
    res["value"] = {"brier": float(((p - yy) ** 2).mean()), "brier_const": float(((yy.mean() - yy) ** 2).mean()),
                    "accuracy": float(((p > 0.5) == (yy > 0.5)).mean())}
    by_turn = {}
    for lo, hi in [(0, 4), (4, 8), (8, 12), (12, 18), (18, 100)]:
        mm = (turn >= lo) & (turn < hi)
        if mm.sum() > 50:
            by_turn[f"{lo}-{hi}"] = {"n": int(mm.sum()), "brier": float(((p[mm] - yy[mm]) ** 2).mean()),
                                     "acc": float(((p[mm] > 0.5) == (yy[mm] > 0.5)).mean())}
    res["value"]["by_turn"] = by_turn
    bins = np.linspace(0, 1, 11)
    rel = []
    for j in range(10):
        mm = (p >= bins[j]) & (p < bins[j + 1] + (1e-9 if j == 9 else 0))
        if mm.sum() > 0:
            rel.append({"bin": [float(bins[j]), float(bins[j + 1])], "n": int(mm.sum()), "pred": float(p[mm].mean()), "obs": float(yy[mm].mean())})
    res["value"]["reliability"] = rel
    res["value"]["ece"] = float(sum(r["n"] * abs(r["pred"] - r["obs"]) for r in rel) / max(1, len(p)))
    # ---------------- belief
    if LOG["belief"]:
        bel = torch.cat(LOG["belief"]).numpy()
        ok = data["hand_ok"].numpy() == 1
        hand = data["hand"].numpy(); pool = data["pool"].numpy(); hcount = data["opp_hand_count"].numpy()
        # marginal frequency prior from the same split's pools is "oracle-ish"; use train-free count baseline instead
        S = defaultdict(list)
        aps = defaultdict(list)
        rng = np.random.default_rng(0)
        idxs = np.where(ok & (hcount > 0))[0]
        if len(idxs) > 40000:
            idxs = rng.choice(idxs, 40000, replace=False)
        for i in idxs:
            pl = pool[i][pool[i] > 0]
            if len(pl) == 0:
                continue
            cards, counts = np.unique(pl, return_counts=True)
            inhand = np.isin(cards, hand[i][hand[i] > 0])
            N = len(pl); h = int(hcount[i])
            count_p = 1 - (1 - counts / N) ** h                       # count-based (oracle decklist)
            learned = 1 / (1 + np.exp(-bel[i][cards]))                 # learned belief head (no decklist)
            combined = 1 - (1 - counts / N * (learned / max(learned.mean(), 1e-6))).clip(0, 1) ** h
            randsc = rng.random(len(cards))
            for nm, sc in (("random", randsc), ("count_oracle_decklist", count_p), ("learned", learned), ("learned_x_count", combined)):
                S[nm + "_s"].append(sc); S[nm + "_l"].append(inhand)
                aps[nm].append(avg_precision(sc, inhand))
        res["belief"] = {nm: {"auroc_pooled": auroc(np.concatenate(S[nm + "_s"]), np.concatenate(S[nm + "_l"])),
                              "mAP": float(np.nanmean(aps[nm]))} for nm in ("random", "count_oracle_decklist", "learned", "learned_x_count")}
        res["belief"]["n_samples"] = int(len(idxs))
    out = os.path.join(ROOT, "experiments", f"metrics_{args.name}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "policy_by_seltype"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
