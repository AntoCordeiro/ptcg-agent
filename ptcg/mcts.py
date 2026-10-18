"""Neural-guided determinized MCTS (PIMC-style) on the C++ simulator.

At each of our single-choice decisions with >= 2 options:
  1. sample K "worlds" (determinizations) of the hidden information:
       - our own deck order + face-down prizes (uniform over our unseen cards)
       - opponent decklist  (closest archetype in a library, by visible-card overlap)
       - opponent hand      ("uniform": uniform over unseen copies, i.e. count-based;
                             "learned": weighted by the network's belief head)
       - opponent deck order / prizes (rest of the unseen pool)
  2. run PUCT search in each world with network priors and value leaves;
     forced moves (1 option) and multi-select steps are played by the policy
     without branching;
  3. sum root visit counts over worlds and play the most visited action.
"""
import dataclasses
import itertools
import math
import os
import random
import time
from collections import Counter

import numpy as np

from cg import api
from .agents import net_eval, policy_choice, is_deck_request, load_deck, ROOT
from .cards import is_basic_pokemon, is_pokemon, N_CARD


def obs_to_dict(o):
    return dataclasses.asdict(o) if dataclasses.is_dataclass(o) else o


def _ids(lst):
    return [int(c["id"]) for c in (lst or []) if c]


def visible_of(ps, stadium, pidx):
    out = []
    for p in (ps.get("active") or []) + (ps.get("bench") or []):
        if not p:
            continue
        out.append(int(p["id"]))
        out += _ids(p.get("energyCards")) + _ids(p.get("tools")) + _ids(p.get("preEvolution"))
    out += _ids(ps.get("discard"))
    out += [int(c["id"]) for c in (ps.get("prize") or []) if c]
    out += [int(c["id"]) for c in (stadium or []) if c and c.get("playerIndex") == pidx]
    return out


def multiset_minus(full, seen):
    rem = Counter(full)
    rem.subtract(Counter(seen))
    return [c for c, k in rem.items() for _ in range(max(k, 0))]


class DeckLibrary:
    def __init__(self, folder="decks"):
        folder = folder if os.path.isabs(folder) else os.path.join(ROOT, folder)
        self.decks = {f[:-4]: load_deck(os.path.join(folder, f)) for f in sorted(os.listdir(folder)) if f.endswith(".csv")}

    def guess(self, visible):
        vis = Counter(visible)
        best, best_s = None, -1
        for name, d in self.decks.items():
            s = sum((Counter(d) & vis).values())
            if s > best_s:
                best, best_s = name, s
        return self.decks[best]


class Node:
    __slots__ = ("sid", "obs", "to_move", "terminal", "result", "P", "N", "W", "children", "n", "value", "actions")

    def __init__(self, sid, obs):
        self.sid = sid
        self.obs = obs
        cur = obs.get("current") or {}
        self.to_move = int(cur.get("yourIndex", 0))
        self.result = int(cur.get("result", -1))
        self.terminal = self.result >= 0 or obs.get("select") is None
        self.P = None
        self.N = None
        self.W = None
        self.children = {}
        self.n = 0
        self.value = 0.0
        self.actions = None


class MCTSAgent:
    name = "mcts"

    def __init__(self, model, deck, sims=96, worlds=4, c_puct=1.25, belief="learned", time_budget=None,
                 max_chain=60, seed=None, device="cpu", library=None, use_overage=False, fpu=0.15, max_time=2.0, sample_temp=0.0,
                 reserve=90.0, divisor=45.0, library_dir=None, multi_search=0, multi_top=6, multi_max=24):
        self.model = model
        self.deck = list(deck)
        self.sims = sims
        self.worlds = worlds
        self.c = c_puct
        self.belief = belief
        self.time_budget = time_budget
        self.max_chain = max_chain
        self.rng = random.Random(seed)
        self.nrng = np.random.default_rng(seed)
        self.device = device
        self.lib = library or DeckLibrary(library_dir or "decks")
        self.use_overage = use_overage
        self.fpu = fpu
        self.max_time = max_time
        self.reserve = reserve
        self.multi_search = multi_search
        self.multi_top = multi_top
        self.multi_max = multi_max
        self.divisor = divisor
        self.sample_temp = sample_temp
        self.last_pi = None
        self.stats = Counter()

    def reset(self):
        pass

    # ------------------------------------------------------------------ worlds
    def sample_world(self, obs, belief_logits):
        cur = obs["current"]
        sel = obs["select"]
        me = int(cur["yourIndex"]); op = 1 - me
        P = cur["players"]
        stadium = cur.get("stadium") or []
        # --- our hidden cards
        mine_seen = visible_of(P[me], stadium, me) + _ids(P[me].get("hand"))
        mine_seen += [int(c["id"]) for c in (cur.get("looking") or []) if c and c.get("playerIndex", me) == me]
        my_pool = multiset_minus(self.deck, mine_seen)
        self.rng.shuffle(my_pool)
        prize_slots = P[me].get("prize") or []
        n_fd = sum(1 for c in prize_slots if not c)
        need = int(P[me].get("deckCount", 0)) + n_fd
        while len(my_pool) < need:
            my_pool.append(self.deck[self.rng.randrange(60)])
        your_prize = []
        k = 0
        for c in prize_slots:
            if c:
                your_prize.append(int(c["id"]))
            else:
                your_prize.append(my_pool[k]); k += 1
        your_deck = my_pool[k:k + int(P[me].get("deckCount", 0))]
        if sel.get("deck") is not None:
            your_deck = []
        # --- opponent
        o_vis = visible_of(P[op], stadium, op)
        o_list = self.lib.guess(o_vis)
        pool = multiset_minus(o_list, o_vis)
        self.rng.shuffle(pool)
        h = int(P[op].get("handCount", 0))
        o_prize_slots = P[op].get("prize") or []
        o_fd = sum(1 for c in o_prize_slots if not c)
        o_need = h + int(P[op].get("deckCount", 0)) + o_fd
        while len(pool) < o_need:
            pool.append(o_list[self.rng.randrange(60)])
        # hand
        if self.belief == "learned" and belief_logits is not None and h > 0:
            w = 1.0 / (1.0 + np.exp(-belief_logits[np.clip(pool, 0, N_CARD - 1)]))
            w = np.maximum(w, 1e-4)
            w = w / w.sum()
            idx = self.nrng.choice(len(pool), size=min(h, len(pool)), replace=False, p=w)
            hand = [pool[i] for i in idx]
            rest = [pool[i] for i in range(len(pool)) if i not in set(idx.tolist())]
        else:
            hand, rest = pool[:h], pool[h:]
        self.rng.shuffle(rest)
        opp_prize = []
        k = 0
        for c in o_prize_slots:
            if c:
                opp_prize.append(int(c["id"]))
            else:
                opp_prize.append(rest[k]); k += 1
        opp_deck = rest[k:k + int(P[op].get("deckCount", 0))]
        opp_active = []
        act = P[op].get("active") or []
        if len(act) > 0 and act[0] is None:
            basics = [c for c in o_list if is_basic_pokemon(c)] or [c for c in o_list if is_pokemon(c)]
            opp_active = [self.rng.choice(basics)]
        if not any(is_basic_pokemon(c) for c in opp_deck + hand) and int(cur.get("turn", 0)) == 0:
            basics = [c for c in o_list if is_basic_pokemon(c)]
            if basics and opp_deck:
                opp_deck[0] = basics[0]
        return your_deck, your_prize, opp_deck, opp_prize, hand, opp_active

    # ------------------------------------------------------------------ tree
    def _advance(self, node):
        """From a node, play forced / multi-select steps with the policy until a branching
        node or a terminal is reached. Returns the final node (possibly the same)."""
        chain = 0
        while not node.terminal and chain < self.max_chain:
            sel = node.obs["select"]
            n = len(sel["option"])
            if sel["minCount"] == 1 and sel["maxCount"] == 1 and n >= 2:
                return node
            if n == 0 or sel["maxCount"] == 0:
                a = []
            elif n == 1 and sel["minCount"] >= 1:
                a = [0]
            else:
                p, v, c, _, _ = net_eval(self.model, [node.obs], self.device)[0]
                self.stats["nn"] += 1
                a = policy_choice(node.obs, p, c, greedy=True)
            try:
                st = api.search_step(node.sid, a)
            except Exception:
                node.terminal = True
                node.result = -2
                return node
            node = Node(st.searchId, obs_to_dict(st.observation))
            chain += 1
        return node

    def _expand(self, node):
        p, v, c, _, _ = net_eval(self.model, [node.obs], self.device)[0]
        self.stats["nn"] += 1
        n = len(node.obs["select"]["option"])
        P = np.zeros(n, np.float64)
        P[:len(p)] = p
        node.P = P
        node.N = np.zeros(n)
        node.W = np.zeros(n)
        node.value = v
        return v

    def _leaf_value(self, node, perspective):
        if node.result in (0, 1):
            return 1.0 if node.result == perspective else -1.0
        if node.result == 2 or node.result == -2:
            return 0.0
        return None

    def _select(self, root):
        """Walk down with PUCT. Returns (path, leaf, terminal_value_or_None)."""
        path = []
        node = root
        while True:
            if node.terminal:
                v = self._leaf_value(node, node.to_move)
                return path, node, (0.0 if v is None else v)
            if node.P is None:
                return path, node, None
            sq = math.sqrt(node.n + 1)
            q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), node.value - self.fpu)
            u = q + self.c * node.P * sq / (1 + node.N)
            u[node.P <= 0] = -1e9
            a = int(np.argmax(u))
            child = node.children.get(a)
            if child is None:
                try:
                    sel_list = node.actions[a] if node.actions is not None else [a]
                    st = api.search_step(node.sid, sel_list)
                    child = Node(st.searchId, obs_to_dict(st.observation))
                    child = self._advance(child)
                except Exception:
                    node.P[a] = 0.0
                    self.stats["step_error"] += 1
                    if (node.P > 0).sum() == 0:
                        return path, node, 0.0
                    continue
                node.children[a] = child
            path.append((node, a))
            node = child

    @staticmethod
    def _backup(path, v, persp):
        for nd, a in reversed(path):
            val = v if nd.to_move == persp else -v
            nd.N[a] += 1
            nd.W[a] += val
            nd.n += 1

    def _round(self, roots):
        """One simulation per world; all leaf evaluations in one batched forward pass."""
        pend = []
        for r in roots:
            path, leaf, tv = self._select(r)
            if tv is not None:
                self._backup(path, tv, leaf.to_move)
            else:
                pend.append((path, leaf))
        if pend:
            outs = net_eval(self.model, [lf.obs for _, lf in pend], self.device)
            self.stats["nn"] += len(pend)
            for (path, leaf), (p, v, c, _, _) in zip(pend, outs):
                n = len(leaf.obs["select"]["option"])
                P = np.zeros(n, np.float64)
                P[:len(p)] = p
                leaf.P, leaf.N, leaf.W, leaf.value = P, np.zeros(n), np.zeros(n), v
                self._backup(path, v, leaf.to_move)
        return len(roots)


    def _combos(self, obs, p, count_ratio):
        """Candidate selections for a multi-select / optional decision, with priors."""
        sel = obs["select"]
        kmin, kmax = int(sel["minCount"]), int(sel["maxCount"])
        n = len(p)
        greedy = tuple(policy_choice(obs, p, count_ratio, greedy=True))
        top = [int(i) for i in np.argsort(-p)[:min(n, self.multi_top)]]
        kstar = kmin + count_ratio * (kmax - kmin)
        cands = {}
        for k in range(kmin, min(kmax, 3) + 1):
            for c in itertools.combinations(sorted(top), k):
                lp = float(np.mean(np.log(p[list(c)] + 1e-9))) if k > 0 else float(np.log(1.0 - p.max() + 1e-3))
                cands[c] = lp - 0.7 * abs(k - kstar)
        if greedy not in cands:
            cands[greedy] = max(cands.values()) + 0.1 if cands else 0.0
        items = sorted(cands.items(), key=lambda kv: -kv[1])[:self.multi_max]
        if greedy not in dict(items):
            items[-1] = (greedy, cands[greedy])
        acts = [list(c) for c, _ in items]
        pri = np.array([v for _, v in items], np.float64)
        pri = np.exp(pri - pri.max()); pri /= pri.sum()
        return acts, pri

    def _search_root(self, obs, acts, pri, v, bel):
        """MCTS where the root actions are explicit selection lists."""
        t0 = time.time()
        budget = self._budget(obs)
        m = len(acts)
        total = np.zeros(m)
        try:
            roots = []
            for w in range(self.worlds):
                yd, yp, od, opz, oh, oa = self.sample_world(obs, bel)
                st = api.search_begin(api.to_observation_class(obs), yd, yp, od, opz, oh, oa)
                r = Node(st.searchId, obs_to_dict(st.observation))
                r.P, r.N, r.W, r.value, r.actions = pri.copy(), np.zeros(m), np.zeros(m), v, acts
                roots.append(r)
            i = 0
            while True:
                i += self._round(roots)
                if (budget is not None and (time.time() - t0 > budget or i >= self.sims)) or (budget is None and i >= self.sims):
                    break
            for r in roots:
                total += r.N
        except Exception as e:
            self.stats["search_error"] += 1
        finally:
            try:
                api.search_end()
            except Exception:
                pass
        self.stats["multi_searches"] += 1
        if total.sum() <= 0:
            return None
        return acts[int(np.argmax(total + 1e-6 * pri))]

    # ------------------------------------------------------------------ act
    def _budget(self, obs):
        if self.use_overage:
            rem = obs.get("remainingOverageTime")
            if rem is None:
                return self.max_time
            return max(0.05, min(self.max_time, (float(rem) - self.reserve) / self.divisor))
        return self.time_budget

    def act(self, obs):
        if is_deck_request(obs):
            return list(self.deck)
        sel = obs["select"]
        n = len(sel["option"])
        if n == 1 and sel["minCount"] >= 1:
            return [0]
        self.last_pi = None
        p, v, c, bel, _ = net_eval(self.model, [obs], self.device)[0]
        if not (sel["minCount"] == 1 and sel["maxCount"] == 1 and n >= 2) or obs.get("search_begin_input") is None:
            if self.multi_search and obs.get("search_begin_input") is not None and n >= 2 and sel["maxCount"] >= 1:
                acts, pri = self._combos(obs, p, c)
                if len(acts) >= 2:
                    best = self._search_root(obs, acts, pri, v, bel)
                    if best is not None:
                        return sorted(int(x) for x in best)
            return policy_choice(obs, p, c, greedy=True)
        t0 = time.time()
        budget = self._budget(obs)
        total = np.zeros(n)
        roots = []
        try:
            for w in range(self.worlds):
                yd, yp, od, opz, oh, oa = self.sample_world(obs, bel)
                st = api.search_begin(api.to_observation_class(obs), yd, yp, od, opz, oh, oa)
                r = Node(st.searchId, obs_to_dict(st.observation))
                Pr = np.zeros(n); Pr[:len(p)] = p
                r.P, r.N, r.W, r.value = Pr, np.zeros(n), np.zeros(n), v
                roots.append(r)
            i = 0
            while True:
                i += self._round(roots)
                if budget is not None:
                    if time.time() - t0 > budget or i >= self.sims:
                        break
                elif i >= self.sims:
                    break
            for r in roots:
                total += r.N
        except Exception as e:
            self.stats["search_error"] += 1
            self.stats["last_error"] = repr(e)[:200]
        finally:
            try:
                api.search_end()
            except Exception:
                pass
        self.stats["searches"] += 1
        self.stats["sims"] += int(total.sum())
        if total.sum() <= 0:
            self.last_pi = None
            return policy_choice(obs, p, c, greedy=True)
        self.last_pi = total / total.sum()          # search policy (visit distribution) for expert iteration
        if self.sample_temp > 0 and self.rng.random() < self.sample_temp:
            return [int(self.nrng.choice(n, p=self.last_pi))]
        best = int(np.argmax(total + 1e-6 * np.pad(p, (0, n - len(p)))))
        return [best]
