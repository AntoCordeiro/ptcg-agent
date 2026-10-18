# run_anchored_tournament.py
"""
High-Statistical-Power Tournament Evaluation for Demonstration-Anchored Dual-Cross PPO.
Compares:
1. Dual-Cross Anchored PPO vs. Dual-Cross BC (Did Anchored PPO preserve and improve over BC?)
2. Dual-Cross Anchored PPO vs. Dual-Cross Unconstrained PPO (Anchored vs. Unconstrained RL Ablation)
3. Dual-Cross Ensemble (Anchored + BC) vs. Dual-Cross BC (Logit-Ensemble Test-Time Deliberation)
4. Dual-Cross Anchored PPO vs. Random Baseline (Sanity Benchmark)

All metrics: Wilson 95% Confidence Intervals, exact two-sided binomial p-values,
Balanced Win Rate (BWR), Prize Margins, and Sweep Rates.
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
from feature_parser import parse_game_state_to_tensors
from metrics_eval import compute_tournament_statistics


def load_deck(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


class DualCrossEnsembleAgent:
    def __init__(self, model_anchored, model_bc, device=None, weight_anchored=0.6):
        self.m1 = model_anchored
        self.m2 = model_bc
        self.w1 = weight_anchored
        self.w2 = 1.0 - weight_anchored
        self.device = device or next(model_anchored.parameters()).device
        self.m1.eval()
        self.m2.eval()

    def select_action(self, obs: dict):
        request = obs.get("select") or {}
        options = request.get("option", [])
        if not options:
            return [0]

        parsed = parse_game_state_to_tensors(obs, request)
        g = torch.from_numpy(parsed["global_vec"]).unsqueeze(0).to(self.device)
        o = torch.from_numpy(parsed["obj_features"]).unsqueeze(0).to(self.device)
        c = torch.from_numpy(parsed["card_ids"]).unsqueeze(0).to(self.device)
        e = torch.from_numpy(parsed["evo_ids"]).unsqueeze(0).to(self.device)
        tt = torch.from_numpy(parsed["token_types"]).unsqueeze(0).to(self.device)
        opt_mask = torch.from_numpy(parsed["option_mask"]).unsqueeze(0).to(self.device)
        val_mask = torch.from_numpy(parsed["valid_action_mask"]).unsqueeze(0).to(self.device)
        pad_mask = torch.from_numpy(parsed["padding_mask"]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, scores1, _ = self.m1(g, o, c, e, opt_mask, val_mask, pad_mask, token_types=tt)
            _, scores2, _ = self.m2(g, o, c, e, opt_mask, val_mask, pad_mask, token_types=tt)

            opt_indices = np.where(parsed["option_mask"])[0]
            s1 = scores1[0, opt_indices]
            s2 = scores2[0, opt_indices]

            blended_scores = self.w1 * s1 + self.w2 * s2
            choice = int(torch.argmax(blended_scores).item())

        max_count = request.get("maxCount", 1) or 1
        selected = [choice]
        if max_count > 1 and len(options) > 1:
            extra = [i for i in range(len(options)) if i != choice][:max_count - 1]
            selected.extend(extra)
        return selected


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


def run_matchup(
    name_a: str,
    agent_a,
    name_b: str,
    agent_b,
    deck_a: list[int],
    deck_b: list[int],
    num_games: int = 60,
    device: torch.device = None,
    mode_a: str = "model",  # 'model', 'ensemble', 'random'
    mode_b: str = "model",
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    games_records = []
    print(f"\n=======================================================")
    print(f"Tournament Matchup: {name_a} vs. {name_b} ({num_games} matches)")
    print(f"=======================================================")

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
                    if mode_a == "ensemble":
                        action = agent_a.select_action(obs)
                    elif mode_a == "model":
                        action, _, _, _, _ = select_action(agent_a, obs, device, stochastic=False)
                    else:
                        action = select_random_action(obs)
                else:
                    if mode_b == "ensemble":
                        action = agent_b.select_action(obs)
                    elif mode_b == "model":
                        action, _, _, _, _ = select_action(agent_b, obs, device, stochastic=False)
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
    print(f"Outcome: {name_a} wins {stats['a_wins']}/{stats['n_games']} ({stats['win_rate']:.1f}%) | 95% CI: [{stats['ci_95'][0]}%, {stats['ci_95'][1]}%] | {sig_flag}")
    print(f"Seat Breakdown: First: {stats['win_rate_first']:.1f}% | Second: {stats['win_rate_second']:.1f}% | Balanced Win Rate (BWR): {stats['balanced_win_rate']:.1f}%")
    print(f"Prize Dynamics: Avg Margin: {stats['avg_prize_margin']:+.2f} prizes | Sweep: {stats['sweep_rate']:.1f}% | Close: {stats['close_game_rate']:.1f}%\n")

    return stats


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Anchored PPO Tournament Evaluation on {device} ({torch.cuda.get_device_name(0)}) ---")

    # Load decks
    deck_dragapult = load_deck("deck_dragapult.csv")
    deck_gardevoir = load_deck("deck_gardevoir.csv")

    # 1. Load Dual-Cross Anchored PPO model
    anchored_ppo = PTCGDualCrossNet().to(device)
    anchored_ppo.load_state_dict(torch.load("ptcg_dual_cross_ppo_anchored.pt", map_location=device, weights_only=True))
    anchored_ppo.eval()
    print("Loaded Dual-Cross Anchored PPO checkpoint.")

    # 2. Load Dual-Cross BC model
    dual_cross_bc = PTCGDualCrossNet().to(device)
    dual_cross_bc.load_state_dict(torch.load("ptcg_dual_cross_bc.pt", map_location=device, weights_only=True))
    dual_cross_bc.eval()
    print("Loaded Dual-Cross BC checkpoint.")

    # 3. Load Dual-Cross Unconstrained PPO model
    dual_cross_ppo = PTCGDualCrossNet().to(device)
    dual_cross_ppo.load_state_dict(torch.load("ptcg_dual_cross_ppo.pt", map_location=device, weights_only=True))
    dual_cross_ppo.eval()
    print("Loaded Dual-Cross Unconstrained PPO checkpoint.")

    # 4. Create Fused Ensemble Agent (Anchored PPO + BC)
    ensemble_agent = DualCrossEnsembleAgent(anchored_ppo, dual_cross_bc, device=device, weight_anchored=0.6)
    print("Initialized DualCrossEnsembleAgent (60% Anchored PPO + 40% BC).")

    all_results = {}

    # Matchup 1: Dual-Cross Anchored PPO vs. Dual-Cross BC (N=60)
    all_results["anchored_ppo_vs_bc"] = run_matchup(
        name_a="Dual-Cross Anchored PPO",
        agent_a=anchored_ppo,
        name_b="Dual-Cross BC",
        agent_b=dual_cross_bc,
        deck_a=deck_dragapult,
        deck_b=deck_gardevoir,
        num_games=60,
        device=device,
        mode_a="model",
        mode_b="model",
    )

    # Matchup 2: Dual-Cross Anchored PPO vs. Dual-Cross Unconstrained PPO (N=60)
    all_results["anchored_ppo_vs_unconstrained_ppo"] = run_matchup(
        name_a="Dual-Cross Anchored PPO",
        agent_a=anchored_ppo,
        name_b="Dual-Cross Unconstrained PPO",
        agent_b=dual_cross_ppo,
        deck_a=deck_dragapult,
        deck_b=deck_gardevoir,
        num_games=60,
        device=device,
        mode_a="model",
        mode_b="model",
    )

    # Matchup 3: Dual-Cross Ensemble vs. Dual-Cross BC (N=60)
    all_results["ensemble_vs_bc"] = run_matchup(
        name_a="Dual-Cross Ensemble",
        agent_a=ensemble_agent,
        name_b="Dual-Cross BC",
        agent_b=dual_cross_bc,
        deck_a=deck_dragapult,
        deck_b=deck_gardevoir,
        num_games=60,
        device=device,
        mode_a="ensemble",
        mode_b="model",
    )

    # Matchup 4: Dual-Cross Anchored PPO vs. Random Baseline (N=60)
    all_results["anchored_ppo_vs_random"] = run_matchup(
        name_a="Dual-Cross Anchored PPO",
        agent_a=anchored_ppo,
        name_b="Random Baseline",
        agent_b=None,
        deck_a=deck_dragapult,
        deck_b=deck_dragapult,
        num_games=60,
        device=device,
        mode_a="model",
        mode_b="random",
    )

    out_json = "anchored_tournament_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll tournament results saved to '{out_json}'.")


if __name__ == "__main__":
    main()
