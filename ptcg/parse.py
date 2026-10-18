"""Replays -> training shards.

Alignment rule (verified, see legacy journal): the observation at steps[s][p]
(status ACTIVE) is answered by steps[s+1][p]["action"].

Per decision we store the encoded observation plus:
  target     multi-hot over options (the chosen option indices)
  k, kmin, kmax   multi-select count info
  value      game result from the acting player's view (+1 / -1 / 0)
  hand       opponent's true hand card ids (from the opponent's own observation
             at the same step) -> belief-head label
  pool       opponent's unseen multiset (their decklist minus visible cards):
             used for the count-based belief baseline
Usage:  python -m ptcg.parse --top 3000 --workers 20 --out data/shards
"""
import argparse
import csv
import json
import os
import sys
import time
import zlib
from collections import Counter
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ptcg.features import encode, ARRAY_KEYS, MAX_OPTS  # noqa: E402
from ptcg.cards import NAME_TO_IDS  # noqa: E402
DRAG = set(NAME_TO_IDS.get("Dragapult ex", []))

MAX_HAND = 32


def _ids(lst):
    return [int(c.get("id", 0)) for c in (lst or []) if c]


def visible_cards(ps):
    """All face-up card ids belonging to a player state that are not in hand/deck/prize."""
    out = []
    for p in (ps.get("active") or []) + (ps.get("bench") or []):
        if not p:
            continue
        out.append(int(p.get("id", 0)))
        out += _ids(p.get("energyCards")) + _ids(p.get("tools")) + _ids(p.get("preEvolution"))
    out += _ids(ps.get("discard"))
    return out


def split_of(ep_id):
    h = zlib.crc32(str(ep_id).encode()) % 100
    return "test" if h < 5 else ("val" if h < 10 else "train")


def game_results(steps):
    """Per-step game index and per-game winner (0/1/2=draw) from RESULT logs."""
    gidx = []
    winners = []
    g = 0
    for s in steps:
        gidx.append(g)
        res = None
        for p in s:
            for lg in ((p.get("observation") or {}).get("logs") or []):
                if lg and int(lg.get("type", -1)) == 23:
                    res = int(lg.get("result", 2))
        if res is not None:
            winners.append(res)
            g += 1
    return gidx, winners


def parse_replay(path, elo):
    d = json.load(open(path, encoding="utf-8"))
    steps = d["steps"]
    ep = os.path.basename(path).split(".")[0]
    decks = [None, None]
    for s in steps[:4]:
        for i, p in enumerate(s):
            a = p.get("action")
            if isinstance(a, list) and len(a) == 60:
                decks[i] = a
    gidx, winners = game_results(steps)
    rewards = d.get("rewards") or [0, 0]
    samples = []
    stats = Counter()
    for si in range(len(steps) - 1):
        for p in (0, 1):
            st = steps[si][p]
            if st.get("status") != "ACTIVE":
                continue
            obs = st.get("observation") or {}
            sel = obs.get("select")
            cur = obs.get("current")
            if not sel or not cur:
                continue
            opts = sel.get("option") or []
            if len(opts) < 2:
                stats["single_option"] += 1
                continue
            act = steps[si + 1][p].get("action")
            if not isinstance(act, list):
                stats["bad_action"] += 1
                continue
            kmin, kmax = int(sel.get("minCount", 0)), int(sel.get("maxCount", 0))
            if not (kmin <= len(act) <= kmax) or any((not isinstance(a, int)) or a < 0 or a >= len(opts) for a in act):
                stats["illegal_label"] += 1
                continue
            if any(a >= MAX_OPTS for a in act):
                stats["label_beyond_max_opts"] += 1
                continue
            me = int(cur.get("yourIndex", p))
            g = gidx[si]
            if g < len(winners):
                w = winners[g]
                value = 0.0 if w == 2 else (1.0 if w == me else -1.0)
            else:
                value = float(np.sign(rewards[me] - rewards[1 - me]))
            enc = encode(obs)
            tgt = np.zeros(MAX_OPTS, np.float32)
            tgt[act] = 1.0
            # belief label: opponent's own hand at this step
            hand = np.zeros(MAX_HAND, np.int64)
            hand_ok = 0
            try:
                oc = steps[si][1 - p]["observation"]["current"]
                oh = oc["players"][1 - me].get("hand")
                if oh is not None and len(oh) == int(cur["players"][1 - me].get("handCount", -1)):
                    ids = _ids(oh)[:MAX_HAND]
                    hand[:len(ids)] = ids
                    hand_ok = 1
            except Exception:
                pass
            pool = np.zeros(60, np.int64)
            if decks[1 - me] is not None:
                rem = Counter(decks[1 - me])
                rem.subtract(Counter(visible_cards(cur["players"][1 - me])))
                ids = [c for c, k in rem.items() for _ in range(max(k, 0))][:60]
                pool[:len(ids)] = ids
            enc.update(target=tgt, k=np.int64(len(act)), kmin=np.int64(kmin), kmax=np.int64(kmax),
                       value=np.float32(value), hand=hand, hand_ok=np.int64(hand_ok), pool=pool,
                       opp_hand_count=np.int64(cur["players"][1 - me].get("handCount", 0)),
                       turn=np.int64(cur.get("turn", 0)), seltype=np.int64(sel.get("type", 0)),
                       elo=np.float32(elo), ep=np.int64(int(ep)), player=np.int64(me),
                       my_deck_drag=np.int64(1 if decks[me] is not None and DRAG.intersection(decks[me]) else 0))
            samples.append(enc)
            stats["ok"] += 1
    return samples, stats


EXTRA_KEYS = ["target", "k", "kmin", "kmax", "value", "hand", "hand_ok", "pool", "opp_hand_count",
              "turn", "seltype", "elo", "ep", "player", "my_deck_drag"]
HALF = {"card_f", "opt_f", "glob", "target"}


def save_shard(samples, path):
    arrs = {}
    for k in ARRAY_KEYS + EXTRA_KEYS:
        a = np.stack([s[k] for s in samples])
        if k in HALF:
            a = a.astype(np.float16)
        arrs[k] = a
    np.savez(path, **arrs)


def worker(args):
    chunk, out_dir, wid = args
    buf = {"train": [], "val": [], "test": []}
    tot = Counter()
    nshard = Counter()
    for path, elo, ep in chunk:
        try:
            samples, st = parse_replay(path, elo)
        except Exception as e:
            tot["replay_error"] += 1
            continue
        tot.update(st)
        sp = split_of(ep)
        buf[sp] += samples
        if len(buf[sp]) >= 20000:
            save_shard(buf[sp], os.path.join(out_dir, f"{sp}_{wid:03d}_{nshard[sp]:02d}.npz"))
            nshard[sp] += 1
            buf[sp] = []
    for sp, b in buf.items():
        if b:
            save_shard(b, os.path.join(out_dir, f"{sp}_{wid:03d}_{nshard[sp]:02d}.npz"))
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", default="data/shards")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    man = list(csv.DictReader(open(os.path.join(ROOT, "data/replays/manifest.csv"))))
    man.sort(key=lambda r: -float(r["avg_score"]))
    items = []
    for r in man:
        fp = os.path.join(ROOT, "data/replays", r["episode_id"] + ".json")
        if os.path.exists(fp):
            items.append((fp, float(r["avg_score"]), r["episode_id"]))
        if len(items) >= args.top:
            break
    print(f"parsing {len(items)} replays (Elo {items[-1][1]:.0f}..{items[0][1]:.0f}) with {args.workers} workers", flush=True)
    chunks = [(items[i::args.workers * 4], args.out, i) for i in range(args.workers * 4)]
    t0 = time.time()
    tot = Counter()
    with Pool(args.workers) as pool:
        for i, st in enumerate(pool.imap_unordered(worker, chunks)):
            tot.update(st)
            print(f"chunk {i+1}/{len(chunks)} done, {time.time()-t0:.0f}s, ok={tot['ok']}", flush=True)
    print("STATS", dict(tot))
    json.dump(dict(tot), open(os.path.join(args.out, "parse_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
