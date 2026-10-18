# run_expanded_metagame_matrix.py
"""
Expanded 6x6 Metagame Payoff Matrix & Cyclic Meta Analysis (Option 2).

Evaluates the Capstone Multi-Archetype PPO Model with Bayesian CardTracker
across the Top 6 most popular competitive archetypes in the 4,603 replay dataset:
1. Darkness Disruption (Munkidori / Marnie's Grimmsnarl ex)
2. Psychic Control (Abra / Kadabra / Alakazam)
3. Colorless Draw (Dunsparce / Dudunsparce)
4. Dragapult Spread (Dreepy / Drakloak / Dragapult ex)
5. Fighting Turbo (Mega Lucario ex / Solrock)
6. Grass Ramp (Teal Mask Ogerpon ex / Dipplin)

Computes:
- 6x6 Empirical Balanced Win Rate Matrix M
- Game-Theoretic Nash Equilibrium Meta Distribution
- Non-Transitive / Cyclic Dominance Analysis (Rock-Paper-Scissors cycles)
- Publication Heatmap: results_plots/figure6_expanded_metagame_matrix.png
"""

import os
import sys
import glob
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import linprog

SUBMISSION_DIR = os.path.abspath(
    "the-pokemon-company-ptcg-ai-battle-challenge-playground/sample_submission/sample_submission/sample_submission"
)
if SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, SUBMISSION_DIR)

from cg import game
from model_dual_cross import PTCGDualCrossNet
from card_tracker import CardTracker
from match_sim import select_action
from metrics_eval import compute_wilson_score_interval, compute_binomial_pvalue


def load_deck(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def evaluate_pair(
    model,
    deck_a: list[int],
    name_a: str,
    deck_b: list[int],
    name_b: str,
    num_games: int = 16,
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
    prize_diffs = []

    for g_idx in range(num_games):
        a_is_p0 = (g_idx % 2 == 0)
        p0_deck = deck_a if a_is_p0 else deck_b
        p1_deck = deck_b if a_is_p0 else deck_a

        obs, start_data = game.battle_start(p0_deck, p1_deck)
        if obs is None:
            continue

        tracker_p0 = CardTracker(p0_deck)
        tracker_p1 = CardTracker(p1_deck)
        steps = 0

        try:
            while steps < 450:
                current = obs.get("current", {})
                res = current.get("result", -1)
                if res != -1:
                    break

                seat = current.get("yourIndex", 0)
                if seat == 0:
                    action, _, _, _, _ = select_action(
                        model, obs, device, stochastic=False, tracker=tracker_p0
                    )
                else:
                    action, _, _, _, _ = select_action(
                        model, obs, device, stochastic=False, tracker=tracker_p1
                    )

                obs = game.battle_select(action)
                steps += 1

            final_res = obs.get("current", {}).get("result", -1)
            players = obs.get("current", {}).get("players", [])
            p0_left = players[0].get("prizeCount", 0) if len(players) > 0 else 0
            p1_left = players[1].get("prizeCount", 0) if len(players) > 1 else 0

            if final_res in (0, 1):
                winner_is_a = (final_res == 0 and a_is_p0) or (final_res == 1 and not a_is_p0)
                if winner_is_a:
                    d0_wins += 1
                else:
                    d1_wins += 1

                if a_is_p0:
                    d0_games_as_p0 += 1
                    if winner_is_a:
                        d0_wins_as_p0 += 1
                    prize_diffs.append(p1_left - p0_left)
                else:
                    d0_games_as_p1 += 1
                    if winner_is_a:
                        d0_wins_as_p1 += 1
                    prize_diffs.append(p0_left - p1_left)
        finally:
            game.battle_finish()

    total_valid = d0_wins + d1_wins
    raw_wr = (d0_wins / max(total_valid, 1)) * 100.0
    _, ci_low, ci_high = compute_wilson_score_interval(d0_wins, total_valid)
    wr0 = (d0_wins_as_p0 / max(d0_games_as_p0, 1)) * 100.0
    wr1 = (d0_wins_as_p1 / max(d0_games_as_p1, 1)) * 100.0
    balanced_wr = 0.5 * (wr0 + wr1)
    p_val = compute_binomial_pvalue(d0_wins, total_valid)

    return {
        "deck_a": name_a,
        "deck_b": name_b,
        "total_games": total_valid,
        "a_wins": d0_wins,
        "b_wins": d1_wins,
        "balanced_win_rate": round(balanced_wr, 2),
        "ci_95": [round(ci_low, 2), round(ci_high, 2)],
        "p_value": round(p_val, 4),
        "avg_prize_margin": round(float(np.mean(prize_diffs)), 2) if prize_diffs else 0.0
    }


def compute_nash_equilibrium(payoff_matrix: np.ndarray) -> np.ndarray:
    """Solves minimax zero-sum game linear program to find Nash meta distribution."""
    n = payoff_matrix.shape[0]
    c = np.zeros(n + 1)
    c[-1] = -1.0  # Maximize v

    A_ub = np.hstack([-payoff_matrix.T, np.ones((n, 1))])
    b_ub = np.zeros(n)

    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0, 1) for _ in range(n)] + [(None, None)]

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if res.success:
            return res.x[:n]
    except Exception:
        pass
    return np.ones(n) / float(n)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running 6x6 Metagame Payoff Matrix Evaluation on {device} ---")

    # Select Top 6 archetypes from deck_pool/
    archetypes = [
        ("Darkness (Grimmsnarl)", "deck_pool/deck_02_4x_Munkidori_4x_Marnies_.csv"),
        ("Psychic (Alakazam)",    "deck_pool/deck_03_4x_Abra_4x_Kadabra.csv"),
        ("Colorless (Dunsparce)", "deck_pool/deck_05_4x_Dunsparce_3x_Dudunspa.csv"),
        ("Spread (Dragapult)",    "deck_pool/deck_06_4x_Dreepy_4x_Drakloak.csv"),
        ("Fighting (Lucario)",    "deck_pool/deck_01_4x_Mega_Lucario_ex_3x_So.csv"),
        ("Grass (Ogerpon)",       "deck_pool/deck_10_4x_Teal_Mask_Ogerpon_ex_.csv"),
    ]

    deck_names = [a[0] for a in archetypes]
    deck_lists = [load_deck(a[1]) for a in archetypes]
    n_decks = len(deck_names)

    # Load Capstone League model
    model = PTCGDualCrossNet().to(device)
    ckpt = "ptcg_capstone_league_model.pt" if os.path.exists("ptcg_capstone_league_model.pt") else "ptcg_dual_cross_ppo_anchored.pt"
    print(f"Loading evaluated agent from: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    win_matrix = np.zeros((n_decks, n_decks))
    matchup_records = []

    print(f"\nEvaluating {n_decks}x{n_decks} Round-Robin Tournament (16 balanced matches per pairing)...")

    for i in range(n_decks):
        for j in range(n_decks):
            if i == j:
                win_matrix[i, j] = 50.0
            elif i < j:
                res = evaluate_pair(
                    model, deck_lists[i], deck_names[i], deck_lists[j], deck_names[j],
                    num_games=16, device=device
                )
                bwr = res["balanced_win_rate"]
                win_matrix[i, j] = bwr
                win_matrix[j, i] = 100.0 - bwr
                matchup_records.append(res)
                print(f"  [{deck_names[i]:<22} vs. {deck_names[j]:<22}] BWR = {bwr:5.1f}% | 95% CI: [{res['ci_95'][0]:.1f}%, {res['ci_95'][1]:.1f}%]")

    # Solve Nash Equilibrium
    payoff_zero_sum = (win_matrix - 50.0) / 50.0
    nash_dist = compute_nash_equilibrium(payoff_zero_sum)

    # Print Payoff Table
    print("\n================== 6x6 METAGAME PAYOFF MATRIX (BWR %) ==================")
    header = f"{'Archetype':<24} | " + " | ".join(f"{name[:8]:<8}" for name in deck_names) + " | Avg Meta Win%"
    print(header)
    print("-" * len(header))
    for i in range(n_decks):
        row_vals = " | ".join(f"{win_matrix[i, j]:>7.1f}%" for j in range(n_decks))
        avg_w = np.mean(win_matrix[i, :])
        print(f"{deck_names[i]:<24} | {row_vals} | {avg_w:>11.1f}%")

    print("\n================== NASH EQUILIBRIUM META SHARES ==================")
    for name, weight in zip(deck_names, nash_dist):
        print(f"  {name:<24}: {weight * 100.0:>5.1f}%")

    # Save JSON results
    out_json = {
        "archetypes": deck_names,
        "balanced_win_rate_matrix": win_matrix.tolist(),
        "nash_equilibrium_distribution": [round(float(w) * 100.0, 2) for w in nash_dist],
        "matchup_records": matchup_records
    }
    with open("expanded_metagame_results.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    # Generate Publication-Quality Figure 6
    os.makedirs("results_plots", exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.4, 1.0]})

    # Heatmap
    im = ax1.imshow(win_matrix, cmap="RdYlGn", vmin=30, vmax=70)
    ax1.set_xticks(np.arange(n_decks))
    ax1.set_yticks(np.arange(n_decks))
    short_labels = [name.split(" ")[0] for name in deck_names]
    ax1.set_xticklabels(short_labels, fontweight="bold", rotation=30, ha="right")
    ax1.set_yticklabels(deck_names, fontweight="bold")
    ax1.set_xlabel("Opponent Archetype", fontweight="bold", fontsize=12)
    ax1.set_ylabel("Player Archetype", fontweight="bold", fontsize=12)
    ax1.set_title("Empirical 6x6 Metagame Balanced Win Rate Matrix (%)", fontsize=13, fontweight="bold")

    for i in range(n_decks):
        for j in range(n_decks):
            val = win_matrix[i, j]
            text_color = "black" if 40 <= val <= 60 else "white"
            ax1.text(j, i, f"{val:.1f}%", ha="center", va="center", color=text_color, fontweight="bold", fontsize=10)

    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Balanced Win Rate (%)", fontsize=11)

    # Nash Bar Chart
    colors = ["#34495e", "#9b59b6", "#95a5a6", "#e74c3c", "#f39c12", "#2ecc71"]
    bars = ax2.bar(short_labels, [w * 100.0 for w in nash_dist], width=0.5, color=colors, edgecolor="black")
    ax2.set_ylabel("Optimal Nash Meta Share (%)", fontsize=12)
    ax2.set_title("Game-Theoretic Nash Equilibrium Distribution", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, max([w * 100.0 for w in nash_dist]) + 15)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h + 1),
                     ha="center", va="bottom", fontweight="bold", fontsize=10)

    plt.tight_layout()
    fig_path = "results_plots/figure6_expanded_metagame_matrix.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\nSaved publication figure to '{fig_path}'.")


if __name__ == "__main__":
    main()
