# test_evolved_deck_head_to_head.py
"""
Head-to-head tournament comparing Evolved Gardevoir Deck vs. Original Gardevoir Deck.
Uses Dual-Cross Anchored PPO agent with CardTracker.
Evaluates Balanced Win Rate (BWR), Wilson 95% Confidence Interval, and exact binomial p-value.
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
from card_tracker import CardTracker
from match_sim import select_action
from metrics_eval import compute_tournament_statistics


def load_deck(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def main(num_games: int = 40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Evolved Deck vs Original Deck Tournament on {device} ({num_games} matches) ---")

    deck_evolved = load_deck("deck_gardevoir_evolved.csv")
    deck_original = load_deck("deck_gardevoir.csv")

    model = PTCGDualCrossNet().to(device)
    model.load_state_dict(torch.load("ptcg_dual_cross_ppo_anchored.pt", map_location=device, weights_only=True))
    model.eval()

    games_records = []

    for match_idx in tqdm(range(num_games), desc="Evolved vs Original"):
        evolved_is_p0 = (match_idx % 2 == 0)
        d0 = deck_evolved if evolved_is_p0 else deck_original
        d1 = deck_original if evolved_is_p0 else deck_evolved

        obs, start_data = game.battle_start(d0, d1)
        if obs is None:
            continue

        tracker_p0 = CardTracker(d0)
        tracker_p1 = CardTracker(d1)
        steps = 0

        try:
            while steps < 500:
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
                games_records.append({
                    "winner": final_res,
                    "agent_a_is_p0": evolved_is_p0,
                    "p0_prizes_left": p0_left,
                    "p1_prizes_left": p1_left,
                    "steps": steps,
                })
        finally:
            game.battle_finish()

    stats = compute_tournament_statistics(games_records, "Evolved Gardevoir Deck", "Original Gardevoir Deck")

    print("\n================ EVOLVED DECK TOURNAMENT RESULTS ================")
    print(f"Outcome: Evolved Deck won {stats['a_wins']}/{stats['n_games']} ({stats['win_rate']:.1f}%)")
    print(f"95% Wilson CI: [{stats['ci_95'][0]}%, {stats['ci_95'][1]}%]")
    print(f"Exact Binomial p-value: {stats['p_value']:.4f}")
    print(f"Balanced Win Rate (BWR): {stats['balanced_win_rate']:.1f}%")
    print(f"Win% as First: {stats['win_rate_first']:.1f}% | Win% as Second: {stats['win_rate_second']:.1f}%")
    print(f"Average Steps per Game: {stats['avg_steps']:.1f}")

    with open("evolved_deck_h2h_results.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main(num_games=40)
