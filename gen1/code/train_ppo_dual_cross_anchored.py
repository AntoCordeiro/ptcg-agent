# train_ppo_dual_cross_anchored.py
"""
Demonstration-Anchored Multi-Deck PPO League Self-Play for PTCGDualCrossNet.

Incorporates:
1. PPO-Clipped Policy Gradient on live game rollouts against League opponents.
2. Critic Generalized Advantage Estimation (GAE) with Explained Variance R^2 tracking.
3. Demonstration Auxiliary Anchor (AlphaStar / DQfD style) on 44,812 human expert replay states:
   L_total = L_PPO + 0.5 * L_Value - 0.01 * Entropy + lambda_demo * L_demo
4. Classical RL Diagnostics: Explained Variance R^2, Approx KL, Clip Fraction, Entropy, Demo Accuracy.
"""

import os
import sys
import copy
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model_dual_cross import PTCGDualCrossNet
from rollout_buffer import RolloutBuffer
from gae import compute_gae
from match_sim import simulate_one_match
from train_bc_dual_cross import PTCGPartitionDataset, compute_topk_correct


def load_deck(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def train_ppo_dual_cross_anchored(
    total_iterations: int = 20,
    target_steps_per_iter: int = 1500,
    lr: float = 4e-5,
    lambda_demo: float = 0.25,
    replays_path: str = "parsed_replays_compact.pt",
    bc_checkpoint: str = "ptcg_dual_cross_bc.pt",
    output_checkpoint: str = "ptcg_dual_cross_ppo_anchored.pt",
    log_file: str = "ppo_dual_cross_anchored_training_log.json"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Launching Demonstration-Anchored Dual-Cross PPO on {device} ({torch.cuda.get_device_name(0)}) ---")

    # Load deck pool
    deck_paths = ["deck_dragapult.csv", "deck_gardevoir.csv", "deck_water.csv"]
    decks = []
    for dp in deck_paths:
        if os.path.exists(dp):
            decks.append(load_deck(dp))
            print(f"Loaded deck archetype: {dp}")
    if not decks:
        fb = "the-pokemon-company-ptcg-ai-battle-challenge-playground/sample_submission/sample_submission/sample_submission/deck.csv"
        decks = [load_deck(fb)]

    # Load human demonstration dataset
    print(f"Loading expert human demonstration dataset from {replays_path}...")
    dataset_dict = torch.load(replays_path, map_location="cpu", weights_only=False)
    train_dataset = PTCGPartitionDataset(dataset_dict["train"])
    demo_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True
    )
    demo_iter = iter(demo_loader)
    print(f"Human demonstration pool ready: {len(train_dataset)} expert decisions.")

    # Initialize current training model
    model = PTCGDualCrossNet().to(device)
    if os.path.exists(bc_checkpoint):
        print(f"Initializing from Dual-Cross BC warm-start: {bc_checkpoint}")
        model.load_state_dict(torch.load(bc_checkpoint, map_location=device, weights_only=True))

    # Opponent pool begins with frozen BC teacher
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

        pbar = tqdm(total=target_steps_per_iter, desc=f"Anchored PPO Iter {iteration:02d}/{total_iterations}")

        while buffer.ptr < target_steps_per_iter:
            d0 = random.choice(decks)
            d1 = random.choice(decks)

            # League sampling: 70% self-play, 30% against historical pool
            if random.random() < 0.30 and len(opponent_pool) > 0:
                opp_model = random.choice(opponent_pool)
                traj_p0, traj_p1, result, steps = simulate_one_match(
                    model, opp_model, d0, d1, device, stochastic_p0=True, stochastic_p1=True
                )
                include_p1 = False
            else:
                traj_p0, traj_p1, result, steps = simulate_one_match(
                    model, model, d0, d1, device, stochastic_p0=True, stochastic_p1=True
                )
                include_p1 = True

            if result == -1:
                continue

            total_games += 1
            if result == 0:
                p0_wins += 1

            r0_terminal = 1.0 if result == 0 else -1.0
            r1_terminal = 1.0 if result == 1 else -1.0

            for i, (parsed, token_target, log_p, val, step_rew) in enumerate(traj_p0):
                is_done = (i == len(traj_p0) - 1)
                r = r0_terminal if is_done else step_rew
                buffer.add(parsed, token_target, log_p, val, r, is_done)

            if include_p1:
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

        adv_std = adv.std()
        if adv_std > 1e-6:
            adv = (adv - adv.mean()) / (adv_std + 1e-8)
        else:
            adv = adv - adv.mean()

        buffer.advantages[:valid_len] = adv
        buffer.returns[:valid_len] = ret

        # Optimization Phase with Demonstration Replay Anchor
        model.train()
        pi_loss_acc, val_loss_acc, demo_loss_acc, ent_acc = 0.0, 0.0, 0.0, 0.0
        kl_acc, clip_acc, exp_var_acc, demo_acc_sum = 0.0, 0.0, 0.0, 0.0
        n_updates = 0

        for _ in range(3):  # 3 PPO epochs
            for b in buffer.get_batches(batch_size=256):
                # 1. PPO Rollout Loss
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

                # 2. Demonstration Auxiliary Anchor Loss (Human Expert Replay)
                try:
                    d_batch = next(demo_iter)
                except StopIteration:
                    demo_iter = iter(demo_loader)
                    d_batch = next(demo_iter)

                dg = d_batch["global_vec"].to(device)
                do = d_batch["obj_features"].to(device)
                dc = d_batch["card_ids"].to(device)
                de = d_batch["evo_ids"].to(device)
                dtt = d_batch["token_types"].to(device)
                dopt = d_batch["option_mask"].to(device)
                dval = d_batch["valid_action_mask"].to(device)
                dpad = d_batch["padding_mask"].to(device)
                dtgt = d_batch["target"].to(device)

                _, d_scores, _ = model(
                    dg, do, dc, de, dopt, dval, dpad, token_types=dtt
                )
                demo_loss = F.cross_entropy(d_scores, dtgt)
                demo_pred = torch.argmax(d_scores, dim=-1)
                batch_demo_acc = (demo_pred == dtgt).float().mean().item()

                # Joint Objective
                total_loss = pi_loss + 0.5 * v_loss - 0.01 * entropy + lambda_demo * demo_loss

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print("Warning: NaN/Inf detected in loss, skipping batch")
                    continue

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                # Diagnostics
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
                demo_loss_acc += demo_loss.item()
                demo_acc_sum += batch_demo_acc
                ent_acc += entropy.item()
                kl_acc += batch_kl
                clip_acc += batch_clip
                exp_var_acc += batch_exp_var
                n_updates += 1

        avg_pi = pi_loss_acc / max(n_updates, 1)
        avg_val = val_loss_acc / max(n_updates, 1)
        avg_demo = demo_loss_acc / max(n_updates, 1)
        avg_demo_acc = (demo_acc_sum / max(n_updates, 1)) * 100.0
        avg_ent = ent_acc / max(n_updates, 1)
        avg_kl = kl_acc / max(n_updates, 1)
        avg_clip = clip_acc / max(n_updates, 1)
        avg_exp_var = exp_var_acc / max(n_updates, 1)
        win_rate = (p0_wins / max(total_games, 1)) * 100.0

        print(
            f"Iter {iteration:02d}/{total_iterations} | "
            f"Games: {total_games} | Steps: {valid_len} | "
            f"Rollout Win: {win_rate:.1f}% | "
            f"PiLoss: {avg_pi:.4f} | ValLoss: {avg_val:.4f} | DemoAcc: {avg_demo_acc:.2f}% | "
            f"KL: {avg_kl:.4f} | Clip: {avg_clip*100:.1f}% | ExpVar R2: {avg_exp_var:.3f}"
        )

        history.append({
            "iteration": iteration,
            "games": total_games,
            "steps": valid_len,
            "p0_win_rate": win_rate,
            "pi_loss": avg_pi,
            "val_loss": avg_val,
            "demo_loss": avg_demo,
            "demo_accuracy": avg_demo_acc,
            "entropy": avg_ent,
            "approx_kl": avg_kl,
            "clip_fraction": avg_clip,
            "explained_variance": avg_exp_var,
        })

        torch.save(model.state_dict(), output_checkpoint)

        # Snapshot league opponent every 5 iterations
        if iteration % 5 == 0:
            frozen = copy.deepcopy(model).eval()
            opponent_pool.append(frozen)
            print(f"Added iteration {iteration} checkpoint to League Pool (Pool size: {len(opponent_pool)})")

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nAnchored Dual-Cross PPO training completed! Checkpoint saved to '{output_checkpoint}'.")
    return model


if __name__ == "__main__":
    train_ppo_dual_cross_anchored(total_iterations=20, target_steps_per_iter=1500)
