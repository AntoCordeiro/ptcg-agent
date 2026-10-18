"""Gauntlet: one candidate (always piloting --deck) vs a fixed opponent pool.

python -m ptcg.gauntlet --cand "mcts:experiments/train_full_s0_v/best.pt:sims=64,worlds=4" --tag mcts64 \
       --pool standard --games 100 --workers 20
Writes experiments/gauntlet/<tag>.jsonl and <tag>.summary.json
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ptcg.eval import worker, summarize, wilson, elo_diff  # noqa: E402

BC = "policy:experiments/train_full_s0/best.pt"


def pools(name, bc):
    std = [
        ("legacy:baselines/legacy_v1approx", "decks/grimmsnarl.csv"),
        ("legacy:baselines/legacy_latest", "decks/hydrapple.csv"),
        (bc, "decks/dragapult.csv"),
        (bc, "decks/ogerpon.csv"),
        (bc, "decks/clefairy.csv"),
        (bc, "decks/other_top.csv"),
    ]
    if name == "standard":
        return std
    if name == "small":
        return [std[0], std[2], std[3]]
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--deck", default="decks/dragapult.csv")
    ap.add_argument("--pool", default="standard")
    ap.add_argument("--bc", default=BC)
    ap.add_argument("--games", type=int, default=100, help="games per opponent")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    opp = pools(args.pool, args.bc)
    out = os.path.join(ROOT, "experiments", "gauntlet", args.tag + ".jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # split each opponent's games into chunks of 5 so all workers stay busy
    tasks = []
    k = 0
    for spec, deck in opp:
        n_left = args.games
        while n_left > 0:
            n = min(5, n_left)
            tasks.append((args.cand, spec, args.deck, deck, n, args.seed * 100000 + k * 13, k))
            n_left -= n; k += 1
    t0 = time.time()
    recs = []
    with Pool(args.workers) as pool, open(out, "w") as f:
        for res in pool.imap_unordered(worker, tasks):
            for r in res:
                f.write(json.dumps(r) + "\n")
            f.flush()
            recs += res
            sm = summarize(recs)
            print(f"[{time.time()-t0:.0f}s] {len(recs)} games pooled score={sm['score']:.3f} CI[{sm['ci95'][0]:.3f},{sm['ci95'][1]:.3f}] err={sm['errors']}", flush=True)
    per = {}
    for spec, deck in opp:
        rs = [r for r in recs if r["b"] == spec and r["deck_b"] == deck]
        per[f"{spec} | {os.path.basename(deck)}"] = summarize(rs)
    allsm = summarize(recs)
    mean_score = sum(v["score"] for v in per.values()) / len(per)
    summary = {"cand": args.cand, "tag": args.tag, "pool": args.pool, "games_per_opp": args.games,
               "pooled": allsm, "mean_score_over_opponents": mean_score, "per_opponent": per,
               "minutes": (time.time() - t0) / 60}
    json.dump(summary, open(out.replace(".jsonl", ".summary.json"), "w"), indent=1)
    print("SUMMARY", json.dumps({"tag": args.tag, "pooled": allsm["score"], "ci": allsm["ci95"], "mean": mean_score,
                                 "per": {k: round(v["score"], 3) for k, v in per.items()}}), flush=True)


if __name__ == "__main__":
    main()
