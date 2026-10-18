"""Expert-iteration data: MCTS self-play, storing the search visit distribution as policy target.

python -m ptcg.exit_gen --ckpt experiments/train_full_s0_v/best.pt --games 2400 --workers 20 --out data/exit1
Each searched decision -> (encoded obs, target = visit distribution, value = final outcome).
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


def worker(task):
    ckpt, n_games, seed, wid, out_dir, sims, worlds, main_deck = task
    import torch
    torch.set_num_threads(1)
    from cg import game
    from ptcg.model import load_model
    from ptcg.agents import load_deck
    from ptcg.mcts import MCTSAgent
    from ptcg.features import encode, MAX_OPTS
    from ptcg.parse import save_shard
    model = load_model(os.path.join(ROOT, ckpt), "cpu")
    lib_dir = os.path.join(ROOT, "decks")
    decks = {f[:-4]: load_deck(os.path.join(lib_dir, f)) for f in sorted(os.listdir(lib_dir)) if f.endswith(".csv")}
    names = list(decks)
    rng = random.Random(seed)
    buf, nshard, t0 = [], 0, time.time()
    for g in range(n_games):
        d0 = main_deck if rng.random() < 0.6 else rng.choice(names)
        d1 = rng.choice(names)
        if rng.random() < 0.5:
            d0, d1 = d1, d0
        agents = [MCTSAgent(model, decks[d], sims=sims, worlds=worlds, seed=seed * 7919 + g * 2 + i, sample_temp=0.25)
                  for i, d in enumerate((d0, d1))]
        obs, _ = game.battle_start(decks[d0], decks[d1])
        if obs is None:
            continue
        pending = []
        steps = 0
        try:
            while obs["current"]["result"] < 0 and steps < 3000:
                me = obs["current"]["yourIndex"]
                a = agents[me].act(obs)
                pi = agents[me].last_pi
                if pi is not None:
                    enc = dict(encode(obs))
                    tgt = np.zeros(MAX_OPTS, np.float32)
                    m = min(len(pi), MAX_OPTS)
                    tgt[:m] = pi[:m]
                    tgt /= max(tgt.sum(), 1e-6)
                    sel = obs["select"]
                    enc.update(target=tgt, k=np.int64(1), kmin=np.int64(1), kmax=np.int64(1),
                               hand=np.zeros(32, np.int64), hand_ok=np.int64(0), pool=np.zeros(60, np.int64),
                               opp_hand_count=np.int64(obs["current"]["players"][1 - me]["handCount"]),
                               turn=np.int64(obs["current"]["turn"]), seltype=np.int64(sel["type"]),
                               elo=np.float32(0), ep=np.int64(seed * 100000 + g), player=np.int64(me),
                               my_deck_drag=np.int64(0))
                    pending.append((enc, me))
                obs = game.battle_select(a)
                steps += 1
            res = obs["current"]["result"]
        finally:
            game.battle_finish()
        for enc, me in pending:
            enc["value"] = np.float32(0.0 if res not in (0, 1) else (1.0 if res == me else -1.0))
            buf.append(enc)
        if len(buf) >= 2500:
            save_shard(buf, os.path.join(out_dir, f"sp_{wid:03d}_{nshard:02d}.npz")); nshard += 1; buf = []
        if g % 10 == 0:
            print(f"w{wid} game {g}/{n_games} {time.time()-t0:.0f}s", flush=True)
    if buf:
        save_shard(buf, os.path.join(out_dir, f"sp_{wid:03d}_{nshard:02d}.npz"))
    return n_games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=2400)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--deck", default="other_top")
    ap.add_argument("--out", default="data/exit1")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    from multiprocessing import Pool
    per = math.ceil(args.games / args.workers)
    tasks = [(args.ckpt, per, args.seed * 1000 + i, i, out, args.sims, args.worlds, args.deck) for i in range(args.workers)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        tot = sum(pool.map(worker, tasks))
    print(f"DONE {tot} games in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
