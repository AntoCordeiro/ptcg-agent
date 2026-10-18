# run_large_scale_tournament.py
"""
High-Statistical-Power Tournament Evaluation Suite for Pokémon TCG AI.

Executes large-scale pairwise competitions (N=60 per pair) to narrow Wilson 95% CIs
and establish formal p < 0.05 statistical significance across all key model comparisons:
1. Dual-Cross PPO vs. Flat PPO (Architecture Showdown after RL)
2. Dual-Cross PPO vs. Dual-Cross BC (Reinforcement Learning Gain)
3. Dual-Cross PPO vs. Random Baseline (Sanity Benchmark)
4. Value-Guided Deliberative Search vs. Greedy Policy (Test-Time Search Impact)
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
from model import PTCGNet
from model_dual_cross import PTCGDualCrossNet
from match_sim import select_action
from search_agent import ValueGuidedSearchAgent, select_search_action
from metrics_eval import compute_tournament_statistics


def load_deck(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def select_random_action(obs: dict) -> list[int]:
    import random
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


def run_tournament_matchup(
    name_a: str,
    model_or_agent_a,
    name_b: str,
    model_or_agent_b,
    deck_a: list[int],
    deck_b: list[int],
    num_games: int = 60,
    device: torch.device = None,
    is_search_a: bool = False,
    is_search_b: bool = False,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    games_records = []
    print(f"\n[High-Power Tournament] {name_a} vs. {name_b} ({num_games} matches)...")

    for match_idx in tqdm(range(num_games), desc=f"{name_a} vs {name_b}"):
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
                    if is_search_a:
                        action, _, _, _ = select_search_action(model_or_agent_a, obs, stochastic=False)
                    elif model_or_agent_a is not None:
                        action, _, _, _, _ = select_action(model_or_agent_a, obs, device, stochastic=False)
                    else:
                        action = select_random_action(obs)
                else:
                    if is_search_b:
                        action, _, _, _ = select_search_action(model_or_agent_b, obs, stochastic=False)
                    elif model_or_agent_b is not None:
                        action, _, _, _, _ = select_action(model_or_agent_b, obs, device, stochastic=False)
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

    stats = compute_tournament_statistics(games_records, name_a, name_b)

    sig_flag = "STATISTICALLY SIGNIFICANT (p < 0.05)" if stats["statistically_significant"] else f"p = {stats['p_value']:.4f}"
    print(f"Results: {name_a} ({stats['a_wins']}/{stats['n_games']}, {stats['win_rate']:.1f}%) | 95% Wilson CI: [{stats['ci_95'][0]}%, {stats['ci_95'][1]}%] | {sig_flag}")
    print(f"Seat Breakdown: Win% as First: {stats['win_rate_first']:.1f}% | Win% as Second: {stats['win_rate_second']:.1f}% | Balanced Win Rate (BWR): {stats['balanced_win_rate']:.1f}%")
    print(f"Prize Dynamics: Avg Margin: {stats['avg_prize_margin']:+.2f} prizes | Sweep Rate: {stats['sweep_rate']:.1f}% | Close Games: {stats['close_game_rate']:.1f}%\n")

    return stats


def run_full_large_scale_tournament(games_per_matchup: int = 60):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================================================")
    print(f"   LAUNCHING HIGH-STATISTICAL-POWER TOURNAMENT (N = {games_per_matchup} per pair) on {device}")
    print(f"==================================================================================")

    deck_drag = load_deck("deck_dragapult.csv")
    deck_gard = load_deck("deck_gardevoir.csv") if os.path.exists("deck_gardevoir.csv") else deck_drag

    # 1. Load Dual-Cross PPO Model
    dual_ppo = PTCGDualCrossNet().to(device)
    if os.path.exists("ptcg_dual_cross_ppo.pt"):
        dual_ppo.load_state_dict(torch.load("ptcg_dual_cross_ppo.pt", map_location=device, weights_only=True))
    dual_ppo.eval()

    # 2. Load Flat PPO Model
    flat_ppo = PTCGNet().to(device)
    if os.path.exists("ptcg_ppo_latest.pt"):
        flat_ppo.load_state_dict(torch.load("ptcg_ppo_latest.pt", map_location=device, weights_only=True))
    flat_ppo.eval()

    # 3. Load Dual-Cross BC Model
    dual_bc = PTCGDualCrossNet().to(device)
    if os.path.exists("ptcg_dual_cross_bc.pt"):
        dual_bc.load_state_dict(torch.load("ptcg_dual_cross_bc.pt", map_location=device, weights_only=True))
    dual_bc.eval()

    # 4. Initialize Search Agent
    search_agent = ValueGuidedSearchAgent(dual_ppo, device=device, top_k=3, beta=0.65)

    tournament_results = []

    # Matchup 1: Dual-Cross PPO vs Flat PPO (Architecture Showdown under RL)
    m1 = run_tournament_matchup(
        "Dual-Cross PPO", dual_ppo,
        "Flat PPO", flat_ppo,
        deck_drag, deck_drag,
        num_games=games_per_matchup,
        device=device
    )
    tournament_results.append(m1)

    # Matchup 2: Dual-Cross PPO vs Dual-Cross BC (Reinforcement Learning Gain)
    m2 = run_tournament_matchup(
        "Dual-Cross PPO", dual_ppo,
        "Dual-Cross BC", dual_bc,
        deck_drag, deck_drag,
        num_games=games_per_matchup,
        device=device
    )
    tournament_results.append(m2)

    # Matchup 3: Dual-Cross PPO vs Random Baseline
    m3 = run_tournament_matchup(
        "Dual-Cross PPO", dual_ppo,
        "Random Baseline", None,
        deck_drag, deck_drag,
        num_games=games_per_matchup,
        device=device
    )
    tournament_results.append(m3)

    # Matchup 4: Value-Guided Deliberative Search vs Greedy Policy (Track 4 Test)
    m4 = run_tournament_matchup(
        "Value Search (K=3)", search_agent,
        "Greedy Dual-Cross", dual_ppo,
        deck_drag, deck_drag,
        num_games=games_per_matchup,
        device=device,
        is_search_a=True,
        is_search_b=False
    )
    tournament_results.append(m4)

    # Save to JSON
    with open("large_scale_tournament_results.json", "w", encoding="utf-8") as f:
        json.dump(tournament_results, f, indent=2)

    print("\n" + "="*85)
    print("                    FINAL HIGH-POWER TOURNAMENT SUMMARY")
    print("="*85)
    print(f"{'Matchup':<36} {'Win Rate':<10} {'95% Wilson CI':<18} {'BWR':<8} {'Avg Margin':<12} {'p-value'}")
    print("-" * 95)
    for r in tournament_results:
        w_str = f"{r['win_rate']:.1f}%"
        ci_str = f"[{r['ci_95'][0]}%, {r['ci_95'][1]}%]"
        bwr_str = f"{r['balanced_win_rate']:.1f}%"
        m_str = f"{r['avg_prize_margin']:+.2f}"
        p_str = f"{r['p_value']:.4f}*" if r['statistically_significant'] else f"{r['p_value']:.4f}"
        print(f"{r['agent_a'] + ' vs ' + r['agent_b']:<36} {w_str:<10} {ci_str:<18} {bwr_str:<8} {m_str:<12} {p_str}")
    print("="*85 + "\n")
    print("Saved complete statistical results to 'large_scale_tournament_results.json'.")


if __name__ == "__main__":
    run_full_large_scale_tournament(games_per_matchup=60)
