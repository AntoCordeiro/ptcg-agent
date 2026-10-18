# eval_arena.py
import sys
import os
import json
import random
import torch
import numpy as np
from tqdm import tqdm

SUBMISSION_DIR = os.path.abspath(
    "the-pokemon-company-ptcg-ai-battle-challenge-playground/sample_submission/sample_submission/sample_submission"
)
if SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, SUBMISSION_DIR)

from cg import game
from model import PTCGNet
from match_sim import select_action

def load_deck(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]

def select_random_action(obs: dict) -> list[int]:
    request = obs.get("select") or {}
    options = request.get("option", [])
    if not options:
        return [0]
    max_count = request.get("maxCount", 1) or 1
    min_count = request.get("minCount", 1) or 1
    choices = list(range(len(options)))
    k = min(max_count, len(choices))
    picked = random.sample(choices, k)
    return picked if len(picked) >= min_count else [0]

def evaluate_head_to_head(
    agent_a_name: str,
    agent_a_model,
    agent_b_name: str,
    agent_b_model,
    deck_a: list[int],
    deck_b: list[int],
    num_games: int = 50,
    device: torch.device = None
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    games_records = []

    print(f"\n[Tournament Match] {agent_a_name} vs. {agent_b_name} ({num_games} games)...")

    for match_idx in tqdm(range(num_games), desc=f"{agent_a_name} vs {agent_b_name}"):
        # Strictly alternate seat assignment
        a_is_p0 = (match_idx % 2 == 0)
        d0 = deck_a if a_is_p0 else deck_b
        d1 = deck_b if a_is_p0 else deck_a

        obs, start_data = game.battle_start(d0, d1)
        if obs is None:
            continue

        steps = 0
        try:
            while steps < 600:
                current = obs.get("current", {})
                res = current.get("result", -1)
                if res != -1:
                    break

                seat = current.get("yourIndex", 0)
                a_turn = (seat == 0 and a_is_p0) or (seat == 1 and not a_is_p0)

                if a_turn:
                    if agent_a_model is not None:
                        action, _, _, _, _ = select_action(agent_a_model, obs, device, stochastic=False)
                    else:
                        action = select_random_action(obs)
                else:
                    if agent_b_model is not None:
                        action, _, _, _, _ = select_action(agent_b_model, obs, device, stochastic=False)
                    else:
                        action = select_random_action(obs)

                obs = game.battle_select(action)
                steps += 1

            final_res = obs.get("current", {}).get("result", -1)
            players = obs.get("current", {}).get("players", [])
            p0_left = players[0].get("prizeCount", 0) if len(players) > 0 else 0
            p1_left = players[1].get("prizeCount", 0) if len(players) > 1 else 0

            if final_res in (0, 1):
                games_records.append({
                    "winner": final_res,
                    "agent_a_is_p0": a_is_p0,
                    "p0_prizes_left": p0_left,
                    "p1_prizes_left": p1_left,
                    "steps": steps,
                })
        finally:
            game.battle_finish()

    from metrics_eval import compute_tournament_statistics
    stats = compute_tournament_statistics(games_records, agent_a_name, agent_b_name)

    print(f"Results: {agent_a_name} ({stats['a_wins']}/{stats['n_games']}, {stats['win_rate']:.1f}%) | 95% Wilson CI: [{stats['ci_95'][0]}%, {stats['ci_95'][1]}%] | p-val: {stats['p_value']:.4f}")
    print(f"Seat Breakdown: Win% as First: {stats['win_rate_first']:.1f}% | Win% as Second: {stats['win_rate_second']:.1f}% | Balanced Win Rate: {stats['balanced_win_rate']:.1f}% | Avg Steps: {stats['avg_steps']:.1f}")
    print(f"Prize Dynamics: Avg Margin: {stats['avg_prize_margin']:+.2f} prizes | Sweep Rate: {stats['sweep_rate']:.1f}% | Close Games: {stats['close_game_rate']:.1f}%\n")

    return stats

def run_full_arena(num_games_per_pair: int = 40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Arena Evaluation Suite on {device} ({torch.cuda.get_device_name(0)}) ---")

    # Load decks
    deck_drag = load_deck("deck_dragapult.csv")
    deck_gard = load_deck("deck_gardevoir.csv") if os.path.exists("deck_gardevoir.csv") else deck_drag

    # Load models
    ppo_model = PTCGNet().to(device)
    ppo_model.load_state_dict(torch.load("ptcg_ppo_latest.pt", map_location=device, weights_only=True))
    ppo_model.eval()

    bc_model = PTCGNet().to(device)
    bc_model.load_state_dict(torch.load("ptcg_bc_checkpoint.pt", map_location=device, weights_only=True))
    bc_model.eval()

    all_results = []

    # 1. PPO Agent vs BC Agent (Dragapult Mirror)
    res_ppo_bc = evaluate_head_to_head(
        "PPO Agent", ppo_model, "BC Agent", bc_model,
        deck_drag, deck_drag, num_games=num_games_per_pair, device=device
    )
    all_results.append(res_ppo_bc)

    # 2. PPO Agent vs Random Agent (Dragapult Mirror)
    res_ppo_rnd = evaluate_head_to_head(
        "PPO Agent", ppo_model, "Random Baseline", None,
        deck_drag, deck_drag, num_games=num_games_per_pair, device=device
    )
    all_results.append(res_ppo_rnd)

    # 3. BC Agent vs Random Agent (Dragapult Mirror)
    res_bc_rnd = evaluate_head_to_head(
        "BC Agent", bc_model, "Random Baseline", None,
        deck_drag, deck_drag, num_games=num_games_per_pair, device=device
    )
    all_results.append(res_bc_rnd)

    # 4. Cross-Deck Archetype: PPO (Dragapult) vs BC (Gardevoir)
    res_cross = evaluate_head_to_head(
        "PPO (Dragapult)", ppo_model, "BC (Gardevoir)", bc_model,
        deck_drag, deck_gard, num_games=num_games_per_pair, device=device
    )
    all_results.append(res_cross)

    # Save summary
    with open("arena_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n================ FINAL ARENA SUMMARY (WITH CLASSICAL METRICS) ================")
    for r in all_results:
        w = r.get("win_rate", r.get("a_win_rate", 0))
        ci = r.get("ci_95", (0.0, 0.0))
        p = r.get("p_value", 1.0)
        bwr = r.get("balanced_win_rate", w)
        margin = r.get("avg_prize_margin", 0.0)
        n = r.get("n_games", r.get("valid_games", 0))
        print(f"{r['agent_a']} vs {r['agent_b']}: Win Rate {w:.1f}% [95% CI: {ci[0]:.1f}%, {ci[1]:.1f}%] | BWR: {bwr:.1f}% | Avg Margin: {margin:+.2f} | p-val: {p:.4f} (Games: {n})")
    print("Results saved to 'arena_results.json'.")

if __name__ == "__main__":
    run_full_arena(num_games_per_pair=40)