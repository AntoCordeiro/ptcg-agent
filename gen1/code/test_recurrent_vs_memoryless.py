# test_recurrent_vs_memoryless.py
"""
Head-to-head tournament evaluation: Recurrent Dual-Cross Net vs. Memoryless Dual-Cross Net (Pillar 3).

Evaluates the impact of sequential POMDP temporal memory on competitive match performance:
1. Agent A: PTCGRecurrentCrossNet (maintains persistent GRU hidden memory h_t across turns)
2. Agent B: PTCGDualCrossNet (memoryless baseline)

Evaluates:
- Balanced Win Rate (BWR) over N=40 balanced games
- Wilson 95% Confidence Intervals & exact binomial p-value
- Average match duration (steps) & prize differential
- Generates publication plot: results_plots/figure9_recurrent_vs_memoryless.png
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

SUBMISSION_DIR = os.path.abspath(
    "the-pokemon-company-ptcg-ai-battle-challenge-playground/sample_submission/sample_submission/sample_submission"
)
if SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, SUBMISSION_DIR)

from cg import game
from model_dual_cross import PTCGDualCrossNet
from model_recurrent_dual_cross import PTCGRecurrentCrossNet
from card_tracker import CardTracker
from feature_parser import parse_game_state_to_tensors
from match_sim import select_action
from metrics_eval import compute_tournament_statistics


def load_deck(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def select_recurrent_action(model, obs: dict, hidden_state: torch.Tensor, device: torch.device, tracker=None):
    request = obs.get("select") or {}
    options = request.get("option", [])
    if not options:
        return [0], 0, 0.0, 0.0, hidden_state

    parsed = parse_game_state_to_tensors(obs, request, tracker=tracker)

    g = torch.from_numpy(parsed["global_vec"]).unsqueeze(0).to(device)
    o = torch.from_numpy(parsed["obj_features"]).unsqueeze(0).to(device)
    c = torch.from_numpy(parsed["card_ids"]).unsqueeze(0).to(device)
    e = torch.from_numpy(parsed["evo_ids"]).unsqueeze(0).to(device)
    tt = torch.from_numpy(parsed["token_types"]).unsqueeze(0).to(device)
    opt_mask = torch.from_numpy(parsed["option_mask"]).unsqueeze(0).to(device)
    val_mask = torch.from_numpy(parsed["valid_action_mask"]).unsqueeze(0).to(device)
    pad_mask = torch.from_numpy(parsed["padding_mask"]).unsqueeze(0).to(device)

    with torch.no_grad():
        probs, scores, val, new_hidden = model(
            g, o, c, e, opt_mask, val_mask, pad_mask,
            token_types=tt, hidden_state=hidden_state
        )
        opt_indices = np.where(parsed["option_mask"])[0]
        choice = int(torch.argmax(scores[0, opt_indices]).item())
        token_target = int(opt_indices[choice])
        val_item = val.item()

    mn = request.get("minCount", 1) or 1
    mx = request.get("maxCount", 1) or 1
    mx = min(mx, len(options))
    mn = min(mn, mx)
    order = torch.argsort(scores[0, opt_indices], descending=True).tolist()
    k = max(mn, 1 if mx >= 1 else 0)
    selected = [int(i) for i in order[:k]]
    if choice not in selected and len(selected) > 0:
        selected[0] = choice

    return selected, token_target, 0.0, val_item, new_hidden


def main(num_games: int = 40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Recurrent vs. Memoryless Tournament on {device} ({num_games} matches) ---")

    decks = [load_deck("deck_gardevoir.csv"), load_deck("deck_dragapult.csv"), load_deck("deck_water.csv")]

    # 1. Recurrent Model
    model_recurrent = PTCGRecurrentCrossNet().to(device)
    rec_ckpt = "ptcg_recurrent_trained.pt" if os.path.exists("ptcg_recurrent_trained.pt") else "ptcg_ppo_phase2_model.pt"
    print(f"Loading Recurrent checkpoint from {rec_ckpt}...")
    model_recurrent.load_state_dict(torch.load(rec_ckpt, map_location=device, weights_only=True), strict=False)
    model_recurrent.eval()

    # 2. Memoryless Baseline Model
    model_memoryless = PTCGDualCrossNet().to(device)
    mem_ckpt = "ptcg_ppo_phase2_model.pt" if os.path.exists("ptcg_ppo_phase2_model.pt") else "ptcg_dual_cross_bc.pt"
    print(f"Loading Memoryless checkpoint from {mem_ckpt}...")
    model_memoryless.load_state_dict(torch.load(mem_ckpt, map_location=device, weights_only=True))
    model_memoryless.eval()

    from agents import NNAgent
    mem_agent = NNAgent(model_memoryless, "memoryless", device=device, multi="min")

    games_records = []

    for match_idx in tqdm(range(num_games), desc="Recurrent vs Memoryless"):
        rec_is_p0 = (match_idx % 2 == 0)
        # Mirror deck (both players receive identical deck)
        deck = decks[match_idx % len(decks)]
        d0 = deck
        d1 = deck

        obs, start_data = game.battle_start(d0, d1)
        if obs is None:
            continue

        tracker_p0 = CardTracker(d0)
        tracker_p1 = CardTracker(d1)
        mem_agent.reset(deck)
        hidden_rec = model_recurrent.init_hidden(1, device)

        steps = 0
        try:
            while steps < 450:
                current = obs.get("current", {})
                res = current.get("result", -1)
                if res != -1:
                    break

                seat = current.get("yourIndex", 0)
                rec_turn = (seat == 0 and rec_is_p0) or (seat == 1 and not rec_is_p0)

                if rec_turn:
                    tracker = tracker_p0 if seat == 0 else tracker_p1
                    action, _, _, _, hidden_rec = select_recurrent_action(
                        model_recurrent, obs, hidden_rec, device, tracker=tracker
                    )
                else:
                    action = mem_agent.act(obs)

                obs = game.battle_select(action)
                steps += 1

            final_res = obs.get("current", {}).get("result", -1)
            players = obs.get("current", {}).get("players", [])
            p0_left = len(players[0].get("prize", [])) if len(players) > 0 else 0
            p1_left = len(players[1].get("prize", [])) if len(players) > 1 else 0

            if final_res in (0, 1):
                games_records.append({
                    "winner": final_res,
                    "agent_a_is_p0": rec_is_p0,
                    "p0_prizes_left": p0_left,
                    "p1_prizes_left": p1_left,
                    "steps": steps,
                })
        finally:
            game.battle_finish()

    stats = compute_tournament_statistics(games_records, "Recurrent Dual-Cross Net", "Memoryless Baseline")

    print("\n================ RECURRENT VS MEMORYLESS RESULTS ================")
    print(f"Recurrent Win Rate: {stats['a_wins']}/{stats['n_games']} ({stats['win_rate']:.1f}%)")
    print(f"95% Wilson Confidence Interval: [{stats['ci_95'][0]}%, {stats['ci_95'][1]}%]")
    print(f"Exact Binomial p-value: {stats['p_value']:.4f}")
    print(f"Balanced Win Rate (BWR): {stats['balanced_win_rate']:.1f}%")
    print(f"Win% as First Player: {stats['win_rate_first']:.1f}% | Second Player: {stats['win_rate_second']:.1f}%")

    with open("recurrent_tournament_results.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Generate Publication Figure 9
    plot_recurrent_results(stats)


def plot_recurrent_results(stats: dict, output_path: str = "results_plots/figure9_recurrent_vs_memoryless.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Win Rate with Wilson 95% Error Bar
    labels = ["Memoryless\nBaseline", "Recurrent\nDual-Cross Net"]
    bwr_vals = [100.0 - stats["balanced_win_rate"], stats["balanced_win_rate"]]
    err_low = max(0.0, stats["balanced_win_rate"] - stats["ci_95"][0])
    err_high = max(0.0, stats["ci_95"][1] - stats["balanced_win_rate"])

    bars = ax1.bar(labels, bwr_vals, width=0.45, color=["#7f8c8d", "#2980b9"],
                   yerr=[[0, err_low], [0, err_high]], capsize=6, edgecolor="black")
    ax1.axhline(y=50, color="gray", linestyle="--", alpha=0.7, label="Parity (50%)")
    ax1.set_ylabel("Balanced Win Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Temporal Memory Impact: Recurrent vs. Memoryless Policy", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.legend()
    ax1.grid(True, axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h + 2),
                     ha="center", va="bottom", fontweight="bold", fontsize=11)

    # Panel B: Seat Disaggregation & Stability
    seats = ["Player 1 (First)", "Player 2 (Second)"]
    seat_rates = [stats["win_rate_first"], stats["win_rate_second"]]
    bars2 = ax2.bar(seats, seat_rates, width=0.45, color=["#e67e22", "#16a085"], edgecolor="black")
    ax2.axhline(y=50, color="gray", linestyle="--", alpha=0.7)
    ax2.set_ylabel("Recurrent Win Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Recurrent Policy Robustness Across Turn Order", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.6)

    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h + 2),
                     ha="center", va="bottom", fontweight="bold", fontsize=11)

    plt.suptitle("POMDP Sequential Modeling: Recurrent Memory Core Evaluation (PTCGRecurrentCrossNet)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved publication figure to '{output_path}'.")


if __name__ == "__main__":
    main(num_games=40)
