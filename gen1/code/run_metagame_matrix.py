# run_metagame_matrix.py
"""
Cross-Deck Metagame Payoff Matrix & Game-Theoretic Nash Equilibrium.

Evaluates Dual-Cross Anchored PPO across 3 competitive archetypes:
1. Dragapult (Spread Damage)
2. Gardevoir (Discard Energy Acceleration)
3. Water (Chien-Pao / Baxcalibur Deluge)

Generates:
- 3x3 Empirical Win-Rate Matrix M
- Wilson 95% Confidence Intervals for each matchup
- Balanced Win Rates (first vs. second seat disaggregation)
- Game-theoretic Nash Equilibrium meta distribution
"""

import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm

SUBMISSION_DIR = os.path.abspath(
    "the-pokemon-company-ptcg-ai-battle-challenge-playground/sample_submission/sample_submission/sample_submission"
)
if SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, SUBMISSION_DIR)

from cg import game
from model_dual_cross import PTCGDualCrossNet
from match_sim import select_action
from metrics_eval import compute_wilson_score_interval, compute_binomial_pvalue


def load_deck(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def evaluate_deck_matchup(
    model,
    name_d0: str,
    deck0: list[int],
    name_d1: str,
    deck1: list[int],
    num_games: int = 24,
    device: torch.device = None
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d0_wins = 0
    d1_wins = 0
    d0_wins_as_p0 = 0
    d0_games_as_p0 = 0
    d0_wins_as_p1 = 0
    d0_games_as_p1 = 0

    p0_prizes_diff = []

    for g_idx in range(num_games):
        # Alternate who is Player 0 to eliminate seat order bias
        d0_is_p0 = (g_idx % 2 == 0)
        p0_deck = deck0 if d0_is_p0 else deck1
        p1_deck = deck1 if d0_is_p0 else deck0

        obs, start_data = game.battle_start(p0_deck, p1_deck)
        if obs is None:
            continue

        steps = 0
        try:
            while steps < 600:
                current = obs.get("current", {})
                res = current.get("result", -1)
                if res != -1:
                    break

                action, _, _, _, _ = select_action(model, obs, device, stochastic=False)
                obs = game.battle_select(action)
                steps += 1

            final_res = obs.get("current", {}).get("result", -1)
            players = obs.get("current", {}).get("players", [])
            p0_left = players[0].get("prizeCount", 0) if len(players) > 0 else 0
            p1_left = players[1].get("prizeCount", 0) if len(players) > 1 else 0

            if final_res in (0, 1):
                # Did Deck 0 win?
                winner_is_d0 = (final_res == 0 and d0_is_p0) or (final_res == 1 and not d0_is_p0)
                if winner_is_d0:
                    d0_wins += 1
                else:
                    d1_wins += 1

                # Track seat dynamics for Deck 0
                if d0_is_p0:
                    d0_games_as_p0 += 1
                    if winner_is_d0:
                        d0_wins_as_p0 += 1
                    p0_prizes_diff.append(p1_left - p0_left)
                else:
                    d0_games_as_p1 += 1
                    if winner_is_d0:
                        d0_wins_as_p1 += 1
                    p0_prizes_diff.append(p0_left - p1_left)
        finally:
            game.battle_finish()

    total_valid = d0_wins + d1_wins
    raw_win_rate = (d0_wins / max(total_valid, 1)) * 100.0
    _, ci_low, ci_high = compute_wilson_score_interval(d0_wins, total_valid)

    wr_as_p0 = (d0_wins_as_p0 / max(d0_games_as_p0, 1)) * 100.0
    wr_as_p1 = (d0_wins_as_p1 / max(d0_games_as_p1, 1)) * 100.0
    balanced_wr = 0.5 * (wr_as_p0 + wr_as_p1)
    p_val = compute_binomial_pvalue(d0_wins, total_valid)

    return {
        "deck_a": name_d0,
        "deck_b": name_d1,
        "total_games": total_valid,
        "d0_wins": d0_wins,
        "d1_wins": d1_wins,
        "raw_win_rate": round(raw_win_rate, 2),
        "ci_95": [round(ci_low, 2), round(ci_high, 2)],
        "p_value": round(p_val, 4),
        "win_rate_p0": round(wr_as_p0, 2),
        "win_rate_p1": round(wr_as_p1, 2),
        "balanced_win_rate": round(balanced_wr, 2),
        "avg_prize_margin": round(float(np.mean(p0_prizes_diff)), 2) if p0_prizes_diff else 0.0
    }


def compute_nash_equilibrium(payoff_matrix: np.ndarray) -> np.ndarray:
    """
    Computes symmetric two-player zero-sum Nash Equilibrium meta distribution
    via linear programming / fictitious play.
    """
    try:
        from scipy.optimize import linprog
        # Zero-sum game: M_ij in [-1, 1]
        # Maximize v subject to sum_i x_i M_ij >= v, sum x_i = 1, x_i >= 0
        n = payoff_matrix.shape[0]
        c = np.zeros(n + 1)
        c[-1] = -1.0  # Maximize v <=> minimize -v

        # Constraints: -sum_i x_i M_ij + v <= 0 => [ -M^T, 1 ] [x, v]^T <= 0
        A_ub = np.hstack([-payoff_matrix.T, np.ones((n, 1))])
        b_ub = np.zeros(n)

        # Equality: sum x_i = 1
        A_eq = np.zeros((1, n + 1))
        A_eq[0, :n] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0, 1) for _ in range(n)] + [(None, None)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if res.success:
            return res.x[:n]
    except Exception:
        pass
    # Fallback uniform
    return np.ones(payoff_matrix.shape[0]) / payoff_matrix.shape[0]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Cross-Deck Metagame Payoff Matrix Evaluation on {device} ---")

    decks = {
        "Dragapult": load_deck("deck_dragapult.csv"),
        "Gardevoir": load_deck("deck_gardevoir.csv"),
        "Water": load_deck("deck_water.csv"),
    }
    deck_names = list(decks.keys())

    # Load Dual-Cross Anchored PPO model
    model = PTCGDualCrossNet().to(device)
    model.load_state_dict(torch.load("ptcg_dual_cross_ppo_anchored.pt", map_location=device, weights_only=True))
    model.eval()

    n_decks = len(deck_names)
    win_matrix = np.zeros((n_decks, n_decks))
    matchup_details = []

    print(f"\nEvaluating {n_decks}x{n_decks} archetype round-robin (24 balanced games per pair)...")

    for i, name_a in enumerate(deck_names):
        for j, name_b in enumerate(deck_names):
            if i == j:
                # Mirror match theoretical parity = 50.0%
                win_matrix[i, j] = 50.0
                matchup_details.append({
                    "deck_a": name_a,
                    "deck_b": name_b,
                    "raw_win_rate": 50.0,
                    "ci_95": [50.0, 50.0],
                    "balanced_win_rate": 50.0,
                    "note": "Mirror match parity"
                })
                print(f"Matchup [{name_a} vs {name_b}]: Mirror Match -> 50.0%")
            elif i < j:
                # Run balanced games
                res = evaluate_deck_matchup(
                    model, name_a, decks[name_a], name_b, decks[name_b], num_games=24, device=device
                )
                bwr = res["balanced_win_rate"]
                win_matrix[i, j] = bwr
                win_matrix[j, i] = 100.0 - bwr

                matchup_details.append(res)
                print(
                    f"Matchup [{name_a} vs {name_b}]: "
                    f"{name_a} BWR: {bwr:.1f}% | 95% CI: [{res['ci_95'][0]}%, {res['ci_95'][1]}%] | "
                    f"P0: {res['win_rate_p0']:.1f}% vs P1: {res['win_rate_p1']:.1f}%"
                )

    # Compute Game-Theoretic Metagame Nash Equilibrium
    # Transform win matrix into zero-sum payoff [-1, 1]
    payoff_zero_sum = (win_matrix - 50.0) / 50.0
    nash_meta = compute_nash_equilibrium(payoff_zero_sum)

    meta_results = {
        "deck_names": deck_names,
        "balanced_win_rate_matrix": win_matrix.tolist(),
        "nash_equilibrium_distribution": [round(float(w) * 100.0, 2) for w in nash_meta],
        "matchup_details": matchup_details
    }

    print("\n================ METAGAME PAYOFF MATRIX (BWR %) ================")
    print(f"{'Archetype':<12} | " + " | ".join(f"{name:<10}" for name in deck_names) + " | Avg Meta Win%")
    for i, name_a in enumerate(deck_names):
        row_str = " | ".join(f"{win_matrix[i, j]:>9.1f}%" for j in range(n_decks))
        avg_meta = np.mean(win_matrix[i, :])
        print(f"{name_a:<12} | {row_str} | {avg_meta:>11.1f}%")

    print("\n================ NASH EQUILIBRIUM META DISTRIBUTION ================")
    for name, weight in zip(deck_names, nash_meta):
        print(f"Optimal Meta Share for {name:<10}: {weight * 100.0:>5.1f}%")

    with open("metagame_matrix_results.json", "w", encoding="utf-8") as f:
        json.dump(meta_results, f, indent=2)

    print("\nSaved metagame analysis to 'metagame_matrix_results.json'.")


if __name__ == "__main__":
    main()
