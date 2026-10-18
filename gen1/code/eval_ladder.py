# eval_ladder.py
"""
Rigorous multi-agent evaluation ladder.

- Round-robin between agents (Random, Heuristic, BC, PPO, anchored PPO, capstone league).
- MIRROR decks: both agents play the SAME deck in a game, so the result isolates policy skill
  from deck strength. Decks rotate over a pool; agent seats alternate.
- Parallel (multiprocessing, spawn), raw games saved to JSONL (resumable analysis).
- Bradley-Terry / Elo with bootstrap 95% CIs (draws/timeouts count as 0.5 and are reported).
- Output: results/ladder_games.jsonl, results/ladder_summary.json,
          results_plots/figure10_ladder.png

Usage:
    python eval_ladder.py --agents random heuristic bc ppo ppo_anchored capstone --games_per_pair 300
"""

import os
import sys
import json
import time
import argparse
import itertools
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

SUBMISSION_DIR = os.path.abspath(
    "the-pokemon-company-ptcg-ai-battle-challenge-playground/sample_submission/sample_submission/sample_submission"
)
if SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, SUBMISSION_DIR)

DECK_FILES = {
    "gardevoir": "deck_gardevoir.csv",
    "dragapult": "deck_dragapult.csv",
    "dragapult_evolved": "deck_dragapult_evolved.csv",
    "water": "deck_water.csv",
}
MAX_STEPS = 1500

_CACHE = {}


def load_deck(path):
    with open(path, "r", encoding="utf-8") as f:
        return [int(l.strip()) for l in f if l.strip()]


def _get_agent(spec, device):
    if spec not in _CACHE:
        from agents import make_agent
        _CACHE[spec] = make_agent(spec, device)
    return _CACHE[spec]


def play_game(task):
    """task: dict(a, b, deck, a_seat, game_id). Returns result dict."""
    import torch
    from cg import game
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_spec, b_spec, deck_name, a_seat = task["a"], task["b"], task["deck"], task["a_seat"]
    random.seed(task["game_id"])
    torch.manual_seed(task["game_id"])
    agent_a, agent_b = _get_agent(a_spec, device), _get_agent(b_spec, device)
    deck = load_deck(DECK_FILES[deck_name])
    seats = {a_seat: agent_a, 1 - a_seat: agent_b}
    # NN agents may be shared between a and b if same spec: use distinct trackers by resetting per seat
    if a_spec == b_spec:
        from agents import make_agent
        seats[1 - a_seat] = make_agent(b_spec, device, seed=task["game_id"] + 1)

    obs, _ = game.battle_start(deck, deck)
    if obs is None:
        return {**task, "result": "error"}
    for ag in seats.values():
        ag.reset(deck)

    steps, first_player = 0, -1
    try:
        while steps < MAX_STEPS:
            cur = obs.get("current") or {}
            if cur.get("result", -1) != -1:
                break
            first_player = cur.get("firstPlayer", first_player)
            seat = cur.get("yourIndex", 0)
            try:
                action = seats[seat].act(obs)
            except Exception as e:  # agent crash -> forfeit
                return {**task, "result": "crash", "winner_seat": 1 - seat, "steps": steps,
                        "err": repr(e)[:120], "a_won": float(seat != a_seat)}
            obs = game.battle_select(action)
            steps += 1
        cur = obs.get("current") or {}
        res = cur.get("result", -1)
        players = cur.get("players", [])
        left = [len(p.get("prize") or []) if p.get("prizeCount") is None else p.get("prizeCount") for p in players]
    finally:
        game.battle_finish()

    if res in (0, 1):
        a_won = 1.0 if res == a_seat else 0.0
        status = "ok"
    else:
        a_won, status = 0.5, "timeout"
    return {**task, "result": status, "winner_seat": res, "a_won": a_won, "steps": steps,
            "first_player": first_player, "a_first": int(first_player == a_seat),
            "prizes_left": left}


# ---------------- Bradley-Terry -----------------
def fit_bt(names, games, reg=1.0, iters=200):
    """games: list of (i, j, score_i in [0,1]). MM algorithm with virtual half-win regularization."""
    n = len(names)
    W = np.full((n, n), 0.0)  # wins of i over j (fractional)
    for i, j, s in games:
        W[i, j] += s
        W[j, i] += 1 - s
    for i in range(n):
        for j in range(n):
            if i != j:
                W[i, j] += reg * 0.5
    N = W + W.T
    p = np.ones(n)
    for _ in range(iters):
        new = np.zeros(n)
        for i in range(n):
            denom = sum(N[i, j] / (p[i] + p[j]) for j in range(n) if j != i)
            new[i] = W[i].sum() / denom
        new /= np.exp(np.mean(np.log(new)))
        p = new
    return 400.0 * np.log10(p)  # Elo scale, mean 0


def bootstrap_elo(names, games, B=300, seed=0):
    rng = np.random.default_rng(seed)
    g = np.array(games, dtype=float)
    out = []
    for _ in range(B):
        idx = rng.integers(0, len(g), len(g))
        samp = [(int(a), int(b), float(s)) for a, b, s in g[idx]]
        out.append(fit_bt(names, samp, iters=60))
    out = np.array(out)
    return np.percentile(out, 2.5, axis=0), np.percentile(out, 97.5, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", default=["random", "heuristic", "bc", "ppo", "ppo_anchored", "capstone"])
    ap.add_argument("--games_per_pair", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--decks", nargs="+", default=list(DECK_FILES))
    ap.add_argument("--tag", default="ladder")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("results_plots", exist_ok=True)
    names = args.agents
    tasks, gid = [], 0
    for a, b in itertools.combinations(names, 2):
        for g in range(args.games_per_pair):
            tasks.append({"a": a, "b": b, "deck": args.decks[g % len(args.decks)],
                          "a_seat": (g // len(args.decks)) % 2, "game_id": gid})
            gid += 1
    random.Random(0).shuffle(tasks)
    print(f"{len(tasks)} games over {len(names)} agents, {args.workers} workers")

    out_path = f"results/{args.tag}_games.jsonl"
    done = []
    t0 = time.time()
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex, \
            open(out_path, "w", encoding="utf-8") as f:
        futs = [ex.submit(play_game, t) for t in tasks]
        for k, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            done.append(r)
            f.write(json.dumps(r) + "\n")
            if k % 100 == 0:
                print(f"  {k}/{len(tasks)} games | {time.time()-t0:.0f}s", flush=True)

    summarize(names, done, args.tag)


def summarize(names, results, tag="ladder"):
    from metrics_eval import compute_wilson_score_interval
    idx = {n: i for i, n in enumerate(names)}
    valid = [r for r in results if r["result"] in ("ok", "timeout")]
    crashes = [r for r in results if r["result"] in ("crash", "error")]
    games = [(idx[r["a"]], idx[r["b"]], r["a_won"]) for r in valid]
    elo = fit_bt(names, games)
    lo, hi = bootstrap_elo(names, games)

    n = len(names)
    wr = np.full((n, n), np.nan)
    cnt = np.zeros((n, n), dtype=int)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                continue
            gs = [r["a_won"] if r["a"] == a and r["b"] == b else 1 - r["a_won"]
                  for r in valid if {r["a"], r["b"]} == {a, b}]
            if gs:
                wr[i, j] = 100 * np.mean(gs)
                cnt[i, j] = len(gs)

    per_deck = {}
    for r in valid:
        per_deck.setdefault(r["deck"], []).append(r)
    first_adv = np.mean([r["a_first"] == (r["a_won"] == 1.0) for r in valid if r["a_won"] != 0.5]) * 100

    summary = {
        "agents": names, "elo": dict(zip(names, map(float, elo))),
        "elo_ci95": {n: [float(l), float(h)] for n, l, h in zip(names, lo, hi)},
        "winrate_matrix_pct": np.nan_to_num(wr, nan=-1).tolist(), "games_matrix": cnt.tolist(),
        "n_games": len(valid), "n_timeouts": sum(r["result"] == "timeout" for r in valid),
        "n_crashes": len(crashes), "crash_examples": [r.get("err") for r in crashes[:5]],
        "first_player_win_pct": float(first_adv),
        "avg_steps": float(np.mean([r["steps"] for r in valid])),
    }
    # vs-random and vs-heuristic Wilson CIs
    anchors = {}
    for anchor in ("random", "heuristic"):
        if anchor not in idx:
            continue
        for a in names:
            if a == anchor:
                continue
            gs = [r["a_won"] if r["a"] == a else 1 - r["a_won"] for r in valid if {r["a"], r["b"]} == {a, anchor}]
            wins = int(round(sum(gs)))
            p, l, h = compute_wilson_score_interval(wins, len(gs))
            anchors[f"{a}_vs_{anchor}"] = {"win_pct": p, "ci95": [l, h], "n": len(gs)}
    summary["anchor_results"] = anchors
    with open(f"results/{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=========== LADDER ===========")
    for n_ in sorted(names, key=lambda x: -summary["elo"][x]):
        print(f"{n_:>18s}  Elo {summary['elo'][n_]:+7.1f}  CI [{summary['elo_ci95'][n_][0]:+.0f}, {summary['elo_ci95'][n_][1]:+.0f}]")
    for k, v in anchors.items():
        print(f"{k:>34s}: {v['win_pct']:.1f}%  CI [{v['ci95'][0]:.1f}, {v['ci95'][1]:.1f}]  n={v['n']}")
    print(f"timeouts={summary['n_timeouts']} crashes={summary['n_crashes']} avg_steps={summary['avg_steps']:.0f}")
    plot(names, summary, wr, tag)


def plot(names, summary, wr, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = sorted(range(len(names)), key=lambda i: -summary["elo"][names[i]])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    e = [summary["elo"][names[i]] for i in order]
    err = [[summary["elo"][names[i]] - summary["elo_ci95"][names[i]][0] for i in order],
           [summary["elo_ci95"][names[i]][1] - summary["elo"][names[i]] for i in order]]
    ax1.barh([names[i] for i in order][::-1], e[::-1], xerr=[err[0][::-1], err[1][::-1]], color="#2980b9", capsize=4)
    ax1.set_xlabel("Bradley-Terry Elo (mean = 0), 95% bootstrap CI")
    ax1.set_title("Agent ladder (mirror decks, seat-balanced)")
    m = wr[np.ix_(order, order)]
    im = ax2.imshow(m, cmap="RdYlGn", vmin=0, vmax=100)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels([names[i] for i in order], rotation=45, ha="right")
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([names[i] for i in order])
    for i in range(len(order)):
        for j in range(len(order)):
            if not np.isnan(m[i, j]):
                ax2.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=9)
    ax2.set_title("Row win-rate (%) vs column")
    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    plt.savefig(f"results_plots/figure10_{tag}.png", dpi=250)
    print(f"saved results_plots/figure10_{tag}.png")


if __name__ == "__main__":
    main()
