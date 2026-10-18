# self_play_runner.py
import sys, os
submission_dir = os.path.abspath(
    "the-pokemon-company-ptcg-ai-battle-challenge-playground/sample_submission/sample_submission/sample_submission"
)
sys.path.insert(0, submission_dir)

from cg import game
from train_self_play import select_action_for_rollout

def play_one_game(model, deck0, deck1, buffer, device):
    # 1. Initialize match
    obs, start_data = game.battle_start(deck0, deck1)
    
    trajectory_p0 = []
    trajectory_p1 = []
    
    # 2. Match loop
    while obs.get("current", {}).get("result", -1) == -1:
        cur = obs.get("current", {})
        active_player = cur.get("yourIndex", 0)
        
        # Select action via model
        action, token_target, log_p, val, parsed = select_action_for_rollout(model, obs, device)
        
        # Store decision step
        step_record = (parsed, token_target, log_p, val)
        if active_player == 0:
            trajectory_p0.append(step_record)
        else:
            trajectory_p1.append(step_record)
            
        # Step the engine
        obs = game.battle_step(action)  # or equivalent step call from cg.game

    # 3. Final outcome
    result = obs.get("current", {}).get("result", 0)  # 0: p0 win, 1: p1 win, etc.
    p0_reward = 1.0 if result == 0 else -1.0
    p1_reward = 1.0 if result == 1 else -1.0
    
    # Push into buffer with GAE rewards
    for parsed, token_target, log_p, val in trajectory_p0:
        buffer.add(parsed, token_target, log_p, val, p0_reward, done=False)
    for parsed, token_target, log_p, val in trajectory_p1:
        buffer.add(parsed, token_target, log_p, val, p1_reward, done=False)