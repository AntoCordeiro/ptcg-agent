"""Self-play data for the value head (AlphaGo-style: SL policy -> value net on self-play outcomes).

Both seats are played by the (sampled) policy network across meta-deck matchups.
A random subset of decision points is stored with the final outcome from the
acting player's view.  Output shards have the same keys as replay shards.

python -m ptcg.selfplay --ckpt experiments/train_full_s0/best.pt --games 20000 --workers 20 --out data/selfplay
"""
import argparse
import math
import os
import random
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

KEEP_P = 0.15


def worker(task):
    ckpt, n_games, seed, wid, out_dir, temp = task
    import torch
    torch.set_num_threads(1)
    from cg import game
    from ptcg.model import load_model
    from ptcg.agents import net_eval, policy_choice, load_deck
    from ptcg.features import MAX_OPTS
    from ptcg.parse import save_shard
    model = load_model(os.path.join(ROOT, ckpt), "cpu")
    lib_dir = os.path.join(ROOT, "decks")
    decks = {f[:-4]: load_deck(os.path.join(lib_dir, f)) for f in sorted(os.listdir(lib_dir)) if f.endswith(".csv")}
    names = list(decks)
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    buf, nshard, t0, results = [], 0, time.time(), []
    for g in range(n_games):
        if rng.random() < 0.6:
            d0, d1 = "dragapult", rng.choice(names)
        else:
            d0, d1 = rng.choice(names), rng.choice(names)
        if rng.random() < 0.5:
            d0, d1 = d1, d0
        obs, _ = game.battle_start(decks[d0], decks[d1])
        if obs is None:
            continue
        pending = []
        steps = 0
        try:
            while obs["current"]["result"] < 0 and steps < 3000:
                sel = obs["select"]
                n = len(sel["option"])
                if n == 1 and sel["minCount"] >= 1:
                    a = [0]
                else:
                    p, v, c, _, enc = net_eval(model, [obs])[0]
                    if temp != 1.0:
                        p = p ** (1.0 / temp); p /= p.sum()
                    a = policy_choice(obs, p, c, greedy=False, rng=nrng)
                    if n >= 2 and rng.random() < KEEP_P:
                        enc = dict(enc)
                        tgt = np.zeros(MAX_OPTS, np.float32)
                        tgt[[i for i in a if i < MAX_OPTS]] = 1.0
                        me = obs["current"]["yourIndex"]
                        enc.update(target=tgt, k=np.int64(len(a)), kmin=np.int64(sel["minCount"]), kmax=np.int64(sel["maxCount"]),
                                   hand=np.zeros(32, np.int64), hand_ok=np.int64(0), pool=np.zeros(60, np.int64),
                                   opp_hand_count=np.int64(obs["current"]["players"][1 - me]["handCount"]),
                                   turn=np.int64(obs["current"]["turn"]), seltype=np.int64(sel["type"]),
                                   elo=np.float32(0), ep=np.int64(seed * 100000 + g), player=np.int64(me),
                                   my_deck_drag=np.int64(1 if [d0, d1][me] == "dragapult" else 0))
                        pending.append((enc, me))
                obs = game.battle_select(a)
                steps += 1
            res = obs["current"]["result"]
        finally:
            game.battle_finish()
        for enc, me in pending:
            enc["value"] = np.float32(0.0 if res not in (0, 1) else (1.0 if res == me else -1.0))
            buf.append(enc)
        results.append(res)
        if len(buf) >= 2500:
            save_shard(buf, os.path.join(out_dir, f"sp_{wid:03d}_{nshard:02d}.npz")); nshard += 1; buf = []
        if g % 50 == 0:
            print(f"w{wid} game {g}/{n_games} {time.time()-t0:.0f}s", flush=True)
    if buf:
        save_shard(buf, os.path.join(out_dir, f"sp_{wid:03d}_{nshard:02d}.npz"))
    return len(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--out", default="data/selfplay")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    from multiprocessing import Pool
    per = math.ceil(args.games / args.workers)
    tasks = [(args.ckpt, per, args.seed * 1000 + i, i, out, args.temp) for i in range(args.workers)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        tot = sum(pool.map(worker, tasks))
    print(f"DONE {tot} games in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
