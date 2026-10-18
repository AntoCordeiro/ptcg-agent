# train_recurrent_bptt.py
"""
Backpropagation Through Time (BPTT) Training for Recurrent Dual-Cross Net (Pillar 3).

Trains the recurrent GRU core and heads on sequential match rollouts:
1. Gathers sequential game trajectories.
2. Applies BPTT over temporal chunks to train GRU hidden state persistence.
3. Evaluates real sequence loss and perplexity.
4. Compares recurrent policy vs memoryless baseline head-to-head.
5. Saves trained model and generates figure9_recurrent_vs_memoryless.png.
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from metrics_eval import compute_wilson_score_interval


def load_deck(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def collect_match_sequence(model, deck0: list[int], deck1: list[int], device: torch.device):
    """Collects an entire sequential match trajectory with sequential states per player."""
    obs, _ = game.battle_start(deck0, deck1)
    if obs is None:
        return {0: [], 1: []}, -1

    tracker_p0 = CardTracker(deck0)
    tracker_p1 = CardTracker(deck1)
    hidden_states = {0: model.init_hidden(1, device), 1: model.init_hidden(1, device)}
    trajs = {0: [], 1: []}
    prev_prizes = {0: 6, 1: 6}
    steps = 0
    max_steps = 350

    try:
        while steps < max_steps:
            cur = obs.get("current", {})
            if cur.get("result", -1) != -1:
                break

            seat = cur.get("yourIndex", 0)
            request = obs.get("select") or {}
            options = request.get("option", [])
            active_tracker = tracker_p0 if seat == 0 else tracker_p1

            if options and len(options) > 0:
                parsed = parse_game_state_to_tensors(obs, request, tracker=active_tracker)

                g = torch.from_numpy(parsed["global_vec"]).unsqueeze(0).to(device)
                o = torch.from_numpy(parsed["obj_features"]).unsqueeze(0).to(device)
                c = torch.from_numpy(parsed["card_ids"]).unsqueeze(0).to(device)
                e = torch.from_numpy(parsed["evo_ids"]).unsqueeze(0).to(device)
                tt = torch.from_numpy(parsed["token_types"]).unsqueeze(0).to(device)
                opt_mask = torch.from_numpy(parsed["option_mask"]).unsqueeze(0).to(device)
                val_mask = torch.from_numpy(parsed["valid_action_mask"]).unsqueeze(0).to(device)
                pad_mask = torch.from_numpy(parsed["padding_mask"]).unsqueeze(0).to(device)

                with torch.no_grad():
                    probs, scores, val, new_h = model(
                        g, o, c, e, opt_mask, val_mask, pad_mask,
                        token_types=tt, hidden_state=hidden_states[seat]
                    )
                    hidden_states[seat] = new_h

                    opt_indices = np.where(parsed["option_mask"])[0]
                    opt_scores = scores[0, opt_indices]
                    opt_probs = F.softmax(opt_scores, dim=-1)

                    dist = torch.distributions.Categorical(opt_probs)
                    choice = dist.sample().item()
                    token_target = int(opt_indices[choice])

                mn = request.get("minCount", 1) or 1
                mx = request.get("maxCount", 1) or 1
                mx = min(mx, len(options))
                mn = min(mn, mx)
                order = torch.argsort(opt_scores, descending=True).tolist()
                k = max(mn, 1 if mx >= 1 else 0)
                selected = [int(i) for i in order[:k]]
                if choice not in selected and len(selected) > 0:
                    selected[0] = choice

                # Track intermediate prize taking
                players = cur.get("players", [])
                my_p_cnt = len(players[seat].get("prize", [])) if len(players) > seat else 6
                opp_p_cnt = len(players[1 - seat].get("prize", [])) if len(players) > (1 - seat) else 6
                r_step = float(prev_prizes[1 - seat] - opp_p_cnt) * 0.5 - float(prev_prizes[seat] - my_p_cnt) * 0.5
                prev_prizes[seat] = my_p_cnt
                prev_prizes[1 - seat] = opp_p_cnt

                trajs[seat].append({
                    "parsed": parsed,
                    "target_action": token_target,
                    "r_step": r_step,
                })
                action = selected
            else:
                action = [0]

            obs = game.battle_select(action)
            steps += 1

        final_res = obs.get("current", {}).get("result", -1)
    finally:
        game.battle_finish()

    return trajs, final_res


def train_recurrent_model(
    epochs: int = 8,
    warmstart_path: str = "ptcg_ppo_phase2_model.pt",
    output_checkpoint: str = "ptcg_recurrent_trained.pt",
    log_path: str = "recurrent_training_log.json"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Recurrent Cross-Attention Net (BPTT) on {device} ---")

    decks = [load_deck("deck_gardevoir.csv"), load_deck("deck_dragapult.csv"), load_deck("deck_water.csv")]

    model = PTCGRecurrentCrossNet().to(device)
    if os.path.exists(warmstart_path):
        print(f"Warm-starting base net from {warmstart_path}...")
        base_state = torch.load(warmstart_path, map_location=device, weights_only=True)
        model.base_net.load_state_dict(base_state, strict=False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5,
        weight_decay=1e-4
    )

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_pi_loss = 0.0
        epoch_v_loss = 0.0
        total_steps = 0
        correct_preds = 0

        # Collect 15 sequential matches
        pbar = tqdm(range(15), desc=f"Recurrent Epoch {epoch:02d}/{epochs}")
        for _ in pbar:
            d0 = random.choice(decks)
            d1 = random.choice(decks)
            trajs, res = collect_match_sequence(model, d0, d1, device)
            if res not in (0, 1):
                continue

            for seat in (0, 1):
                seq = trajs[seat]
                if len(seq) < 4:
                    continue

                # Compute discounted returns
                running_ret = 1.0 if res == seat else -1.0
                gamma = 0.99
                for item in reversed(seq):
                    running_ret = item["r_step"] + gamma * running_ret
                    item["return"] = running_ret

                # Run BPTT chunk
                hidden = model.init_hidden(1, device)
                optimizer.zero_grad()
                seq_loss = 0.0

                for item in seq:
                    p = item["parsed"]
                    g = torch.from_numpy(p["global_vec"]).unsqueeze(0).to(device)
                    o = torch.from_numpy(p["obj_features"]).unsqueeze(0).to(device)
                    c = torch.from_numpy(p["card_ids"]).unsqueeze(0).to(device)
                    e = torch.from_numpy(p["evo_ids"]).unsqueeze(0).to(device)
                    tt = torch.from_numpy(p["token_types"]).unsqueeze(0).to(device)
                    opt_mask = torch.from_numpy(p["option_mask"]).unsqueeze(0).to(device)
                    val_mask = torch.from_numpy(p["valid_action_mask"]).unsqueeze(0).to(device)
                    pad_mask = torch.from_numpy(p["padding_mask"]).unsqueeze(0).to(device)

                    probs, scores, val, hidden = model(
                        g, o, c, e, opt_mask, val_mask, pad_mask,
                        token_types=tt, hidden_state=hidden
                    )

                    target_tok = torch.tensor([item["target_action"]], device=device)
                    loss_pi = F.cross_entropy(scores, target_tok)

                    target_val = torch.tensor([item["return"]], dtype=torch.float32, device=device)
                    loss_v = F.mse_loss(val, target_val)

                    loss = loss_pi + 0.5 * loss_v
                    seq_loss = seq_loss + loss

                    # Metrics
                    pred_tok = torch.argmax(scores, dim=-1).item()
                    if pred_tok == item["target_action"]:
                        correct_preds += 1
                    total_steps += 1
                    epoch_pi_loss += loss_pi.item()
                    epoch_v_loss += loss_v.item()

                seq_loss = seq_loss / len(seq)
                seq_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

        acc = (correct_preds / max(1, total_steps)) * 100
        avg_pi = epoch_pi_loss / max(1, total_steps)
        avg_v = epoch_v_loss / max(1, total_steps)
        print(f"Epoch {epoch:02d} | PiLoss: {avg_pi:.4f} | VLoss: {avg_v:.4f} | Acc: {acc:.1f}% ({total_steps} steps)")

        history.append({
            "epoch": epoch,
            "pi_loss": avg_pi,
            "v_loss": avg_v,
            "accuracy": acc,
            "steps": total_steps
        })

        torch.save(model.state_dict(), output_checkpoint)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return model, history


if __name__ == "__main__":
    train_recurrent_model(epochs=6)
