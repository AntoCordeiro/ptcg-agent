"""Parallel match runner + statistics (Wilson CI, Bradley-Terry, SPRT).

Agent spec strings:
  random | heuristic | policy:<ckpt> | mcts:<ckpt>[:key=val,...] | legacy:<folder>
Examples:
  python -m ptcg.eval --a mcts:experiments/train_full_s0/best.pt:sims=96 --b legacy:baselines/legacy_v1approx \
        --deck-a decks/dragapult.csv --games 200 --workers 20 --out experiments/eval/mcts_vs_v1.jsonl
  --deck-b defaults to the legacy agent's own deck; --swap-decks plays both deck assignments.
"""
import argparse
import json
import math
import os
import sys
import time
import traceback
from multiprocessing import Pool

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # eval workers run on CPU (like Kaggle)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_MODELS = {}


def make_agent(spec, deck, seed):
    import torch
    torch.set_num_threads(1)
    from ptcg.agents import RandomAgent, HeuristicAgent, PolicyAgent, LegacyAgent
    parts = spec.split(":")
    kind = parts[0]
    if kind == "random":
        return RandomAgent(deck, seed)
    if kind == "heuristic":
        return HeuristicAgent(deck, seed)
    if kind == "legacy":
        return LegacyAgent(parts[1])
    from ptcg.model import load_model
    ck = parts[1]
    if ck not in _MODELS:
        _MODELS[ck] = load_model(os.path.join(ROOT, ck) if not os.path.isabs(ck) else ck, "cpu")
    model = _MODELS[ck]
    kw = {}
    for p in parts[2:]:
        for kv in p.split(","):
            if "=" in kv:
                k, v = kv.split("=")
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                kw[k] = v
    if kind == "policy":
        return PolicyAgent(model, deck, greedy=kw.get("greedy", 1) == 1, seed=seed)
    if kind == "mcts":
        from ptcg.mcts import MCTSAgent
        return MCTSAgent(model, deck, seed=seed, **kw)
    raise ValueError(spec)


BANK = 600.0   # Kaggle: actTimeout=0, ~600 s overage bank per agent per episode


def play_game(agents, decks, max_steps=4000):
    """agents[i] plays seat i with decks[i]. Returns dict with winner seat (or -1 draw/2 error).
    Simulates Kaggle's time bank: obs["remainingOverageTime"] is provided and running out loses."""
    from cg import game
    times = [0.0, 0.0]
    moves = [0, 0]
    obs, sd = game.battle_start(decks[0], decks[1])
    if obs is None:
        return {"winner": -1, "error": f"start error player {sd.errorPlayer} type {sd.errorType}"}
    for a in agents:
        a.reset()
    steps = 0
    err = None
    try:
        while obs["current"] is None or obs["current"]["result"] < 0:
            who = obs["current"]["yourIndex"] if obs["current"] else 0
            t = time.time()
            obs["remainingOverageTime"] = BANK - times[who]
            try:
                act = agents[who].act(obs)
            except Exception as e:
                err = f"seat{who} act error: {e!r}"
                return {"winner": 1 - who, "error": err, "steps": steps, "trace": traceback.format_exc()[-600:]}
            times[who] += time.time() - t
            moves[who] += 1
            if times[who] > BANK:
                return {"winner": 1 - who, "error": f"seat{who} timeout", "steps": steps}
            try:
                obs = game.battle_select(act)
            except Exception as e:
                return {"winner": 1 - who, "error": f"seat{who} illegal {act}: {e!r}", "steps": steps}
            steps += 1
            if steps > max_steps:
                return {"winner": -1, "error": "max_steps", "steps": steps}
        res = obs["current"]["result"]
        return {"winner": res if res in (0, 1) else -1, "steps": steps,
                "sec_per_move": [times[i] / max(1, moves[i]) for i in (0, 1)], "moves": moves,
                "total_time": times}
    finally:
        game.battle_finish()


def worker(task):
    spec_a, spec_b, deck_a, deck_b, n, seed, wid = task
    from ptcg.agents import load_deck
    da = load_deck(deck_a)
    a = make_agent(spec_a, da, seed)
    db = load_deck(deck_b) if deck_b else getattr(None, "deck", None)
    b = make_agent(spec_b, db, seed + 1)
    if db is None:
        db = b.deck
    out = []
    for g in range(n):
        seat_a = (g + wid) % 2
        agents = [a, b] if seat_a == 0 else [b, a]
        decks = [da, db] if seat_a == 0 else [db, da]
        t0 = time.time()
        r = play_game(agents, decks)
        w = r.get("winner", -1)
        score_a = 0.5 if w == -1 else (1.0 if w == seat_a else 0.0)
        rec = {"a": spec_a, "b": spec_b, "deck_a": deck_a, "deck_b": deck_b or "own", "seat_a": seat_a,
               "score_a": score_a, "time": time.time() - t0, **{k: v for k, v in r.items() if k != "winner"}}
        if hasattr(a, "stats"):
            rec["a_stats"] = {k: v for k, v in a.stats.items() if k != "last_error"}
        out.append(rec)
    return out


def wilson(s, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 1.0
    p = s / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, (c - h) / d, (c + h) / d


def elo_diff(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return -400 * math.log10(1 / p - 1)


def sprt_llr(wins, losses, draws, elo0=0.0, elo1=30.0):
    """GSPRT log-likelihood ratio (normal approximation, as in chess-engine testing)."""
    n = wins + losses + draws
    if n == 0:
        return 0.0
    w, d = wins / n, draws / n
    s = w + d / 2
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + (1 - w - d) * s ** 2)
    if var <= 0:
        return 0.0
    s0 = 1 / (1 + 10 ** (-elo0 / 400))
    s1 = 1 / (1 + 10 ** (-elo1 / 400))
    return (s1 - s0) * (2 * s - s0 - s1) / (2 * var / n)


def summarize(recs):
    n = len(recs)
    s = sum(r["score_a"] for r in recs)
    wins = sum(1 for r in recs if r["score_a"] == 1)
    losses = sum(1 for r in recs if r["score_a"] == 0)
    draws = n - wins - losses
    p, lo, hi = wilson(s, n)
    llr = sprt_llr(wins, losses, draws)
    a, b = math.log(0.05 / 0.95), math.log(0.95 / 0.05)
    errs = sum(1 for r in recs if r.get("error"))
    return {"games": n, "score": p, "ci95": [lo, hi], "elo": elo_diff(p), "elo_ci": [elo_diff(lo), elo_diff(hi)],
            "W": wins, "L": losses, "D": draws, "errors": errs, "sprt_llr": llr,
            "sprt": "H1 accepted (+30)" if llr > b else ("H0 accepted (<=0)" if llr < a else "continue"),
            "sec_per_move_a": sum(r.get("sec_per_move", [0, 0])[r["seat_a"]] for r in recs) / max(1, n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--deck-a", default="decks/dragapult.csv")
    ap.add_argument("--deck-b", default=None)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    per = math.ceil(args.games / args.workers)
    tasks = [(args.a, args.b, args.deck_a, args.deck_b, per, args.seed * 1000 + i * 7, i) for i in range(args.workers)]
    t0 = time.time()
    recs = []
    with Pool(args.workers) as pool, open(out, "a") as f:
        for res in pool.imap_unordered(worker, tasks):
            for r in res:
                f.write(json.dumps(r) + "\n")
            recs += res
            sm = summarize(recs)
            print(f"[{time.time()-t0:.0f}s] {len(recs)} games  score_a={sm['score']:.3f} "
                  f"CI[{sm['ci95'][0]:.3f},{sm['ci95'][1]:.3f}] elo={sm['elo']:+.0f} err={sm['errors']} sprt={sm['sprt']}", flush=True)
    sm = summarize(recs)
    sm.update({"a": args.a, "b": args.b, "deck_a": args.deck_a, "deck_b": args.deck_b})
    print("SUMMARY", json.dumps(sm), flush=True)
    with open(out.replace(".jsonl", ".summary.json"), "w") as f:
        json.dump(sm, f, indent=1)


if __name__ == "__main__":
    main()
