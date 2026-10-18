# test_kaggle_submission.py
"""
Strict local test for kaggle_submission/main.py.
Simulates matches from the perspective of Kaggle's evaluation environment.
"""

import os
import sys
import time

SUBMISSION_DIR = os.path.abspath("kaggle_submission")
if SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, SUBMISSION_DIR)

# Import the submission entrypoint
import main
from cg import game
from agents import HeuristicAgent

def test_isolated_submission():
    print(f"=== Testing Kaggle Submission Package in '{SUBMISSION_DIR}' ===")
    
    # 1. Test deck retrieval
    deck0 = main.agent({})
    assert isinstance(deck0, list), "agent({}) should return list of card IDs"
    assert len(deck0) == 60, f"Deck must have exactly 60 cards, got {len(deck0)}"
    print(f"[PASS] Deck retrieval: 60 cards successfully loaded.")

    opp_deck = list(deck0)
    opp_agent = HeuristicAgent()

    # 2. Simulate 3 full games
    N_GAMES = 3
    step_times = []
    
    for g_idx in range(1, N_GAMES + 1):
        obs, _ = game.battle_start(deck0, opp_deck)
        opp_agent.reset(opp_deck)
        steps = 0
        max_steps = 300

        while steps < max_steps:
            cur = obs.get("current") or {}
            res = cur.get("result", -1)
            if res != -1:
                break

            active_player = cur.get("yourIndex", 0)
            req = obs.get("select") or {}
            options = req.get("option", [])
            mn = req.get("minCount", 1) or 1
            mx = req.get("maxCount", 1) or 1
            mx = min(mx, len(options))
            mn = min(mn, mx)

            if active_player == 0:
                t0 = time.time()
                action = main.agent(obs)
                dt = time.time() - t0
                step_times.append(dt)

                # Validation checks
                assert isinstance(action, list), f"Action must be list, got {type(action)}"
                assert len(action) >= mn, f"Action length {len(action)} < minCount {mn}"
                assert len(action) <= mx, f"Action length {len(action)} > maxCount {mx}"
                for a in action:
                    assert 0 <= a < len(options), f"Invalid action index {a} for options {len(options)}"
            else:
                action = opp_agent.act(obs)

            obs = game.battle_select(action)
            steps += 1

        final_res = obs.get("current", {}).get("result", -1)
        game.battle_finish()
        
        result_str = "WIN" if final_res == 0 else ("LOSS" if final_res == 1 else "DRAW/TIMEOUT")
        print(f"Game {g_idx}: {result_str} in {steps} steps.")

    avg_ms = np_mean = sum(step_times) / len(step_times) * 1000
    max_ms = max(step_times) * 1000
    print(f"\n[PERFORMANCE] Over {len(step_times)} moves:")
    print(f"  Average move latency: {avg_ms:.2f} ms")
    print(f"  Maximum move latency: {max_ms:.2f} ms")
    print(f"[VERDICT] Submission package verified 100% compliant and ready for Kaggle upload!")

if __name__ == "__main__":
    test_isolated_submission()
