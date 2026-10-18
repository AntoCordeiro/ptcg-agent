# match_sim.py
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

# Path to the directory holding cg/
SUBMISSION_DIR = os.path.abspath(
    "the-pokemon-company-ptcg-ai-battle-challenge-playground/sample_submission/sample_submission/sample_submission"
)
if SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, SUBMISSION_DIR)

from cg import game
from feature_parser import parse_game_state_to_tensors
from card_tracker import CardTracker

def select_action(model, obs: dict, device: torch.device, stochastic: bool = True, tracker=None):
    request = obs.get("select")
    if not request or "option" not in request:
        return [0], 0, 0.0, 0.0, None

    options = request.get("option", [])
    if len(options) == 0:
        return [0], 0, 0.0, 0.0, None

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
        _, scores, val = model(
            g, o, c, e, opt_mask, val_mask, pad_mask, token_types=tt
        )
        
        opt_indices = np.where(parsed["option_mask"])[0]
        # Restrict logits to valid options
        opt_scores = scores[0, opt_indices]
        opt_probs = F.softmax(opt_scores, dim=-1)

        if stochastic:
            dist = torch.distributions.Categorical(opt_probs)
            choice = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(choice, device=device)).item()
        else:
            choice = int(torch.argmax(opt_scores).item())
            log_prob = 0.0

        token_target = int(opt_indices[choice])
        value = val.item()

    # The engine expects a list of chosen indices, constrained by maxCount
    max_count = request.get("maxCount", 1) or 1
    selected_indices = [choice]
    
    if max_count > 1 and len(options) > 1:
        extra = [i for i in range(len(options)) if i != choice][:max_count - 1]
        selected_indices.extend(extra)

    return selected_indices, token_target, log_prob, value, parsed

def simulate_one_match(
    model_p0,
    model_p1=None,
    deck0: list[int] = None,
    deck1: list[int] = None,
    device: torch.device = None,
    stochastic_p0: bool = True,
    stochastic_p1: bool = True
):
    """
    Runs a full match between two agent instances.
    Returns: trajectories for (P0, P1), winner index (0 or 1), and turn count.
    Each trajectory item is (parsed, token_target, log_p, val, intermediate_prize_reward).
    """
    if model_p1 is None:
        model_p1 = model_p0
    if device is None:
        device = next(model_p0.parameters()).device

    obs, start_data = game.battle_start(deck0, deck1)
    if obs is None:
        return [], [], -1, 0

    tracker_p0 = CardTracker(deck0) if deck0 is not None else None
    tracker_p1 = CardTracker(deck1) if deck1 is not None else None

    traj_p0 = []
    traj_p1 = []
    steps_taken = 0
    max_steps = 600

    prev_p0_prizes = None
    prev_p1_prizes = None
    pending_p0_rew = 0.0
    pending_p1_rew = 0.0

    try:
        while steps_taken < max_steps:
            current = obs.get("current", {})
            result = current.get("result", -1)
            if result != -1:
                break

            players = current.get("players", [])
            p0_prizes_raw = players[0].get("prize", []) if len(players) > 0 else []
            p1_prizes_raw = players[1].get("prize", []) if len(players) > 1 else []
            curr_p0_prizes = len(p0_prizes_raw) if isinstance(p0_prizes_raw, list) else 6
            curr_p1_prizes = len(p1_prizes_raw) if isinstance(p1_prizes_raw, list) else 6
            turn = current.get("turn", 0)

            # In PTCG, prize pile decreases when taking prizes:
            # p0 takes prizes when p0's prize pile shrinks!
            # p1 takes prizes when p1's prize pile shrinks!
            if prev_p0_prizes is not None and turn >= 1:
                p0_prizes_taken = max(0, prev_p0_prizes - curr_p0_prizes)
                p0_prizes_lost = max(0, prev_p1_prizes - curr_p1_prizes)
                delta_p0 = float(p0_prizes_taken - p0_prizes_lost)
                if delta_p0 != 0.0:
                    r0 = 0.2 * delta_p0
                    r1 = -r0
                    if len(traj_p0) > 0:
                        p, t, lp, v, r = traj_p0[-1]
                        traj_p0[-1] = (p, t, lp, v, r + r0)
                    else:
                        pending_p0_rew += r0

                    if len(traj_p1) > 0:
                        p, t, lp, v, r = traj_p1[-1]
                        traj_p1[-1] = (p, t, lp, v, r + r1)
                    else:
                        pending_p1_rew += r1

            prev_p0_prizes = curr_p0_prizes
            prev_p1_prizes = curr_p1_prizes

            active_player = current.get("yourIndex", 0)
            if active_player == 0:
                action, token_target, log_p, val, parsed = select_action(
                    model_p0, obs, device, stochastic=stochastic_p0, tracker=tracker_p0
                )
                if parsed is not None:
                    step_rew = pending_p0_rew
                    pending_p0_rew = 0.0
                    traj_p0.append((parsed, token_target, log_p, val, step_rew))
            else:
                action, token_target, log_p, val, parsed = select_action(
                    model_p1, obs, device, stochastic=stochastic_p1, tracker=tracker_p1
                )
                if parsed is not None:
                    step_rew = pending_p1_rew
                    pending_p1_rew = 0.0
                    traj_p1.append((parsed, token_target, log_p, val, step_rew))

            obs = game.battle_select(action)
            steps_taken += 1

        final_result = obs.get("current", {}).get("result", -1)
    finally:
        game.battle_finish()

    return traj_p0, traj_p1, final_result, steps_taken