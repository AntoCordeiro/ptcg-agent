# benchmark_approach_ac.py
"""
Evaluates the new Approach A (Relational Pairwise-Bias + Option Self-Attn + Conditional Pointer)
+ Approach C (Offline Implicit Q-Learning) model (`ptcg_iql_relational.pt` and `search_iql`)
against our previous strongest checkpoints (`wise`, `search_v3`, `bc`, `heuristic`, `old_submission`).
"""

import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))
SUBMISSION_DIR = os.path.abspath(
    "the-pokemon-company-ptcg-ai-battle-challenge-playground/sample_submission/sample_submission/sample_submission"
)
if SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, SUBMISSION_DIR)

from cg import game
from ptcg_agents import make_agent


def load_deck(path):
    with open(path, "r", encoding="utf-8") as f:
        return [int(l.strip()) for l in f if l.strip()]


def fit_bradley_terry_elo(win_matrix, names, reg=1.0, iters=200):
    n = len(names)
    W = win_matrix.astype(np.float64).copy()
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
    elos = 400.0 * np.log10(p)
    return {names[i]: float(elos[i]) for i in range(n)}


def play_matchup(ag_a, ag_b, deck_a, deck_b, n_games=30):
    wins_a, wins_b, draws = 0, 0, 0
    for g in range(n_games):
        seat_a = g % 2
        if seat_a == 0:
            obs, _ = game.battle_start(deck_a, deck_b)
            ag_a.reset(deck_a)
            ag_b.reset(deck_b)
            seats = {0: ag_a, 1: ag_b}
        else:
            obs, _ = game.battle_start(deck_b, deck_a)
            ag_b.reset(deck_b)
            ag_a.reset(deck_a)
            seats = {0: ag_b, 1: ag_a}

        steps = 0
        while steps < 800:
            cur = obs.get("current") or {}
            if cur.get("result", -1) != -1:
                break
            s = cur.get("yourIndex", 0)
            act = seats[s].act(obs)
            obs = game.battle_select(act)
            steps += 1

        res = (obs.get("current") or {}).get("result", -1)
        game.battle_finish()
        if res == seat_a:
            wins_a += 1
        elif res == 1 - seat_a:
            wins_b += 1
        else:
            draws += 1
    wr_a = (wins_a + 0.5 * draws) / n_games
    return wr_a, wins_a, wins_b, draws


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_deck = load_deck("deck_dragapult_evolved.csv")
    champ_deck = (
        load_deck("championship_decks/best_championship_deck.csv")
        if os.path.exists("championship_decks/best_championship_deck.csv")
        else base_deck
    )

    agent_names = [
        "search_v4",
        "wise_v3",
        "selfplay_v3",
        "iql_v3",
        "iql_rel",
        "wise",
        "bc",
        "heuristic",
        "old_submission",
    ]
    agents = {name: make_agent(name, device=device) for name in agent_names}

    n = len(agent_names)
    win_matrix = np.zeros((n, n), dtype=np.float64)
    games_per_pair = 20

    print(f"=== Grand Model Benchmark ({n} Agents, {games_per_pair} Games/Pair, Mirror Deck) ===")
    t0 = time.time()
    pair_results = {}
    for i in range(n):
        for j in range(i + 1, n):
            na, nb = agent_names[i], agent_names[j]
            wr, wa, wb, dr = play_matchup(agents[na], agents[nb], champ_deck, champ_deck, n_games=games_per_pair)
            win_matrix[i, j] = wa + 0.5 * dr
            win_matrix[j, i] = wb + 0.5 * dr
            pair_results[f"{na}_vs_{nb}"] = {
                "wr_pct": round(wr * 100.0, 2),
                "wins_a": wa,
                "wins_b": wb,
                "draws": dr,
            }
            print(f"  {na:<15} vs {nb:<15} : {wr*100:5.1f}% ({wa}W - {wb}L - {dr}D)")

    elos = fit_bradley_terry_elo(win_matrix, agent_names)
    ranked = sorted(elos.items(), key=lambda x: x[1], reverse=True)

    print("\n=== Grand Bradley-Terry Elo Leaderboard (All Approaches A + B + C + D + Self-Play) ===")
    for rank, (name, rating) in enumerate(ranked, 1):
        idx = agent_names.index(name)
        total_w = np.sum(win_matrix[idx])
        total_g = (n - 1) * games_per_pair
        avg_wr = (total_w / total_g) * 100.0
        print(f"  #{rank} {name:<16} | Elo: {rating:+7.1f} | Overall Win-Rate: {avg_wr:5.1f}%")

    # Direct Full-System Comparison: New Submission (search_v4 + best_championship_deck) vs Rebuild v1 (wise + Grimmsnarl)
    print("\n=== Direct Full-System Matchup: New Submission (Search v4 + Champ Deck) vs Rebuild v1 (WiSE v1 + Grimmsnarl) ===")
    wr_full, wa_f, wb_f, dr_f = play_matchup(agents["search_v4"], agents["wise"], champ_deck, base_deck, n_games=40)
    print(f"  New Submission vs Rebuild v1 (~600 Elo): {wr_full*100:5.1f}% ({wa_f}W - {wb_f}L - {dr_f}D)")

    os.makedirs("results", exist_ok=True)
    summary = {
        "agent_names": agent_names,
        "games_per_pair": games_per_pair,
        "elapsed_sec": round(time.time() - t0, 1),
        "elo_ratings": {k: round(v, 2) for k, v in ranked},
        "new_submission_vs_rebuild_v1": {
            "wr_pct": round(wr_full * 100.0, 2),
            "wins": wa_f,
            "losses": wb_f,
            "draws": dr_f,
        },
        "pair_results": pair_results,
        "win_matrix": (win_matrix / games_per_pair).round(4).tolist(),
    }
    with open("results/approach_ac_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved results to 'results/approach_ac_benchmark.json'.")


if __name__ == "__main__":
    main()
