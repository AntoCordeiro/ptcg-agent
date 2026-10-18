# train_ppo_dual_cross.py
"""
Multi-Deck PPO League Self-Play for PTCGDualCrossNet (Decoupled Cross-Attention).
Logs all classical RL metrics: Explained Variance R^2, Approx KL, Clip Fraction, Entropy, Loss.
"""

import os
import json
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from model_dual_cross import PTCGDualCrossNet
from rollout_buffer import RolloutBuffer
from gae import compute_gae
from match_sim import simulate_one_match


def load_deck(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def train_ppo_dual_cross(
    total_iterations: int = 20,
    target_steps_per_iter: int = 1500,
    lr: float = 1e-4,
    bc_checkpoint: str = "ptcg_dual_cross_bc.pt",
    output_checkpoint: str = "ptcg_dual_cross_ppo.pt",
    log_file: str = "ppo_dual_cross_training_log.json"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Launching Dual-Cross PPO Multi-Deck League Self-Play on {device} ({torch.cuda.get_device_name(0)}) ---")

    # Load deck pool
    deck_paths = ["deck_dragapult.csv", "deck_gardevoir.csv", "deck_water.csv"]
    decks = []
    for dp in deck_paths:
        if os.path.exists(dp):
            decks.append(load_deck(dp))
            print(f"Loaded deck pool archetype from {dp}")
    if not decks:
        fb = "the-pokemon-company-ptcg-ai-battle-challenge-playground/sample_submission/sample_submission/sample_submission/deck.csv"
        decks = [load_deck(fb)]

    # Initialize current training model
    model = PTCGDualCrossNet().to(device)
    if os.path.exists(bc_checkpoint):
        print(f"Initializing from Dual-Cross BC warm-start: {bc_checkpoint}")
        model.load_state_dict(torch.load(bc_checkpoint, map_location=device, weights_only=True))

    # Opponent pool starts with frozen BC policy
    opponent_pool = [copy.deepcopy(model).eval()]

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4
    )
    buffer = RolloutBuffer(capacity=target_steps_per_iter * 2, device=device)

    history = []

    for iteration in range(1, total_iterations + 1):
        model.eval()
        buffer.clear()

        total_games = 0
        p0_wins = 0

        pbar = tqdm(total=target_steps_per_iter, desc=f"Dual-Cross PPO Iter {iteration:02d}/{total_iterations}")

        while buffer.ptr < target_steps_per_iter:
            d0 = random.choice(decks)
            d1 = random.choice(decks)

            # League play: 75% self-play, 25% past checkpoints
            if random.random() < 0.25 and len(opponent_pool) > 0:
                opp_model = random.choice(opponent_pool)
                traj_p0, traj_p1, result, steps = simulate_one_match(
                    model, opp_model, d0, d1, device, stochastic_p0=True, stochastic_p1=True
                )
                include_p1_in_buffer = False
            else:
                traj_p0, traj_p1, result, steps = simulate_one_match(
                    model, model, d0, d1, device, stochastic_p0=True, stochastic_p1=True
                )
                include_p1_in_buffer = True

            if result == -1:
                continue

            total_games += 1
            if result == 0:
                p0_wins += 1

            r0_terminal = 1.0 if result == 0 else -1.0
            r1_terminal = 1.0 if result == 1 else -1.0

            # Store P0 steps
            for i, (parsed, token_target, log_p, val, step_rew) in enumerate(traj_p0):
                is_done = (i == len(traj_p0) - 1)
                r = r0_terminal if is_done else step_rew
                buffer.add(parsed, token_target, log_p, val, r, is_done)

            # Store P1 steps if pure self-play
            if include_p1_in_buffer:
                for i, (parsed, token_target, log_p, val, step_rew) in enumerate(traj_p1):
                    is_done = (i == len(traj_p1) - 1)
                    r = r1_terminal if is_done else step_rew
                    buffer.add(parsed, token_target, log_p, val, r, is_done)

            pbar.n = min(buffer.ptr, target_steps_per_iter)
            pbar.refresh()

        pbar.close()

        # Compute GAE across rollout trajectories
        valid_len = buffer.ptr
        adv, ret = compute_gae(
            buffer.rewards[:valid_len],
            buffer.values[:valid_len],
            buffer.dones[:valid_len],
            gamma=0.99,
            lam=0.95
        )

        # Global advantage normalization
        adv_std = adv.std()
        if adv_std > 1e-6:
            adv = (adv - adv.mean()) / (adv_std + 1e-8)
        else:
            adv = adv - adv.mean()

        buffer.advantages[:valid_len] = adv
        buffer.returns[:valid_len] = ret

        # PPO Update
        model.train()
        pi_loss_acc, val_loss_acc, ent_acc = 0.0, 0.0, 0.0
        kl_acc, clip_acc, exp_var_acc = 0.0, 0.0, 0.0
        n_updates = 0

        for _ in range(3):  # 3 PPO epochs per rollout
            for b in buffer.get_batches(batch_size=256):
                g = b["global_vec"]
                o = b["obj_features"]
                c = b["card_ids"]
                e = b["evo_ids"]
                tt = b["token_types"]
                opt_mask = b["option_mask"]
                val_mask = b["valid_action_mask"]
                pad_mask = b["padding_mask"]
                actions = b["actions"]
                old_log_probs = b["old_log_probs"]
                advantages = b["advantages"]
                returns = b["returns"]

                probs, scores, values = model(
                    g, o, c, e, opt_mask, val_mask, pad_mask, token_types=tt
                )
                
                dist = torch.distributions.Categorical(logits=scores)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
                pi_loss = -torch.min(surr1, surr2).mean()

                v_loss = F.mse_loss(values, returns)
                total_loss = pi_loss + 0.5 * v_loss - 0.01 * entropy

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print("Warning: NaN/Inf detected in loss, skipping batch")
                    continue

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                # Classical Diagnostics
                with torch.no_grad():
                    logratio = new_log_probs - old_log_probs
                    batch_kl = ((torch.exp(logratio) - 1.0) - logratio).mean().item()
                    batch_clip = ((torch.exp(logratio) - 1.0).abs() > 0.2).float().mean().item()
                    
                    y_pred = values.detach().cpu().numpy()
                    y_true = returns.cpu().numpy()
                    var_y = np.var(y_true)
                    batch_exp_var = float(1.0 - np.var(y_true - y_pred) / var_y) if var_y > 1e-6 else 0.0

                pi_loss_acc += pi_loss.item()
                val_loss_acc += v_loss.item()
                ent_acc += entropy.item()
                kl_acc += batch_kl
                clip_acc += batch_clip
                exp_var_acc += batch_exp_var
                n_updates += 1

        avg_pi_loss = pi_loss_acc / max(n_updates, 1)
        avg_val_loss = val_loss_acc / max(n_updates, 1)
        avg_ent = ent_acc / max(n_updates, 1)
        avg_kl = kl_acc / max(n_updates, 1)
        avg_clip = clip_acc / max(n_updates, 1)
        avg_exp_var = exp_var_acc / max(n_updates, 1)
        win_rate = (p0_wins / max(total_games, 1)) * 100.0

        print(
            f"Iter {iteration:02d}/{total_iterations} | "
            f"Games: {total_games} | Steps: {valid_len} | "
            f"P0 Win: {win_rate:.1f}% | "
            f"Pi Loss: {avg_pi_loss:.4f} | Val MSE: {avg_val_loss:.4f} | Ent: {avg_ent:.3f} | "
            f"KL: {avg_kl:.4f} | Clip: {avg_clip*100:.1f}% | ExpVar R2: {avg_exp_var:.3f}"
        )

        history.append({
            "iteration": iteration,
            "games": total_games,
            "steps": valid_len,
            "p0_win_rate": win_rate,
            "pi_loss": avg_pi_loss,
            "val_loss": avg_val_loss,
            "entropy": avg_ent,
            "approx_kl": avg_kl,
            "clip_fraction": avg_clip,
            "explained_variance": avg_exp_var,
        })

        # Save checkpoint
        torch.save(model.state_dict(), output_checkpoint)

        # Snapshot league opponent
        if iteration % 6 == 0:
            frozen_ckpt = copy.deepcopy(model).eval()
            opponent_pool.append(frozen_ckpt)
            print(f"Added iteration {iteration} checkpoint to League Opponent Pool (Pool size: {len(opponent_pool)})")

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nDual-Cross PPO League training completed. Final model saved to '{output_checkpoint}'.")
    return model


if __name__ == "__main__":
    train_ppo_dual_cross(total_iterations=20, target_steps_per_iter=1500)
