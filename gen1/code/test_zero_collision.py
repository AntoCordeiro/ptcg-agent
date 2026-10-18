# test_zero_collision.py
import os
import sys
import numpy as np

SUBMISSION_DIR = os.path.abspath(
    "the-pokemon-company-ptcg-ai-battle-challenge-playground/sample_submission/sample_submission/sample_submission"
)
if SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, SUBMISSION_DIR)

from cg import game, api
from feature_parser import parse_game_state_to_tensors

def run_collision_audit(num_games: int = 15):
    deck_path = "deck_gardevoir.csv"
    with open(deck_path, "r", encoding="utf-8") as f:
        deck = [int(line.strip()) for line in f if line.strip()]


    total_decision_points = 0
    total_multi_options = 0
    total_attach_options = 0
    total_evolve_options = 0
    total_card_options = 0
    total_number_options = 0
    collisions = 0
    unresolved_cids = 0

    print(f"Running zero-collision verification over {num_games} complete matches...")

    for g in range(num_games):
        obs, start_data = game.battle_start(deck, deck)
        try:
            while True:
                current = obs.get("current", {})
                if current.get("result", -1) != -1:
                    break
                req = obs.get("select", {})
                if not req or "option" not in req:
                    break

                options = req.get("option") or []
                if len(options) > 1:
                    total_decision_points += 1
                    total_multi_options += len(options)

                    parsed = parse_game_state_to_tensors(obs, req)
                    opt_mask = parsed["option_mask"]
                    opt_feats = parsed["obj_features"][opt_mask]
                    c_ids = parsed["card_ids"][opt_mask]
                    evo_ids = parsed["evo_ids"][opt_mask]

                    N = opt_feats.shape[0]
                    for i in range(N):
                        opt_i = options[i]
                        t_i = opt_i.get("type", 0)
                        if t_i == 8: total_attach_options += 1
                        elif t_i == 9: total_evolve_options += 1
                        elif t_i == 3: total_card_options += 1
                        elif t_i == 0: total_number_options += 1

                        for j in range(i + 1, N):
                            opt_j = options[j]
                            diff_feats = np.max(np.abs(opt_feats[i] - opt_feats[j]))
                            diff_cid = abs(c_ids[i] - c_ids[j])
                            diff_evo = abs(evo_ids[i] - evo_ids[j])
                            total_diff = diff_feats + diff_cid + diff_evo

                            if total_diff == 0.0:
                                collisions += 1
                                print(f"[COLLISION DETECTED] Game {g}, Context {req.get('context')}:")
                                print(f"  Opt {i}: {opt_i}")
                                print(f"  Opt {j}: {opt_j}")

                # Choose legal action
                n_opts = len(options)
                action = [0] if n_opts > 0 else []
                min_c = req.get("minCount", 1) or 1
                if min_c > 1 and n_opts >= min_c:
                    action = list(range(min_c))
                obs = game.battle_select(action)
        finally:
            game.battle_finish()


    print("=" * 60)
    print("ZERO-COLLISION VERIFICATION SUMMARY:")
    print(f"Total Matches Simulated: {num_games}")
    print(f"Total Decision Points with >1 Option: {total_decision_points}")
    print(f"Total Multi-Option Candidates Tested: {total_multi_options}")
    print(f"  - ATTACH Options Tested: {total_attach_options}")
    print(f"  - EVOLVE Options Tested: {total_evolve_options}")
    print(f"  - CARD Target Options Tested: {total_card_options}")
    print(f"  - NUMBER Options Tested: {total_number_options}")
    print(f"Total Collisions Found: {collisions}")
    print("=" * 60)

    assert collisions == 0, f"Found {collisions} option collisions!"
    print("VERIFICATION PASSED: ZERO COLLISIONS DETECTED ACROSS ALL MATCHES.")

if __name__ == "__main__":
    run_collision_audit(15)
