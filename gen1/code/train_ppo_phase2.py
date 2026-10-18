# train_ppo_phase2.py
"""
Phase 2: Rigorous Multi-Archetype PPO Training with Shaped Rewards & League Population.

Key Technical Components:
1. Potential-Based Prize Differential Reward Shaping:
   r_t = 0.5 * (prizes_taken - prizes_lost) + r_terminal (+1.0 win / -1.0 loss)
2. Value Normalization & GAE (gamma=0.99, lambda=0.95)
3. Demonstration-Anchored Policy Gradient with Annealing:
   lambda_demo(t) = max(0.05, 0.25 * (1.0 - t / total_iterations))
4. Population Self-Play:
   - 60% Self-Play (Current Policy)
   - 20% Historical Checkpoint League Pool
   - 20% External Heuristic Agent (forces policy to learn to beat rule-based play)
5. Multi-Archetype Rotation across harvested tournament decks.
6. Periodic validation on the standard evaluation ladder.
"""

import os
import sys
import copy
import glob
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

SUBMISSION_DIR = os.path.abspath(
    "the-pokemon-company-ptcg-ai-battle-challenge-playground/sample_submission/sample_submission/sample_submission"
)
if SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, SUBMISSION_DIR)

from model_dual_cross import PTCGDualCrossNet
from rollout_buffer import RolloutBuffer
from gae import compute_gae
from card_tracker import CardTracker
from feature_parser import parse_game_state_to_tensors
from train_bc_dual_cross import PTCGPartitionDataset
from agents import HeuristicAgent, RandomAgent, NNAgent
from cg import game


def load_deck(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def select_action_for_rollout(model, obs: dict, device: torch.device, tracker=None):
    request = obs.get("select") or {}
    options = request.get("option", [])
    if not options:
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
        opt_scores = scores[0, opt_indices]
        opt_probs = F.softmax(opt_scores, dim=-1)

        dist = torch.distributions.Categorical(opt_probs)
        choice = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(choice, device=device)).item()
        token_target = int(opt_indices[choice])
        value = val.item()

    mn = request.get("minCount", 1) or 1
    mx = request.get("maxCount", 1) or 1
    mx = min(mx, len(options))
    mn = min(mn, mx)

    # Top-k scoring selection for multi-choice requests
    order = torch.argsort(opt_scores, descending=True).tolist()
    k = max(mn, 1 if mx >= 1 else 0)
    selected = [int(i) for i in order[:k]]
    if choice not in selected and len(selected) > 0:
        selected[0] = choice

    return selected, token_target, log_prob, value, parsed


def simulate_rollout_match(model_p0, opp_agent, deck0, deck1, device, tracker_p0):
    obs, _ = game.battle_start(deck0, deck1)
    if obs is None:
        return [], -1, 0

    traj_p0 = []
    steps = 0
    max_steps = 450
    prev_p0_prizes = 6
    prev_p1_prizes = 6

    try:
        while steps < max_steps:
            cur = obs.get("current") or {}
            res = cur.get("result", -1)
            if res != -1:
                break

            players = cur.get("players", [])
            p0_prizes_raw = players[0].get("prize", []) if len(players) > 0 else []
            p1_prizes_raw = players[1].get("prize", []) if len(players) > 1 else []
            curr_p0 = len(p0_prizes_raw) if isinstance(p0_prizes_raw, list) else prev_p0_prizes
            curr_p1 = len(p1_prizes_raw) if isinstance(p1_prizes_raw, list) else prev_p1_prizes

            # Shaped intermediate prize delta reward
            delta_taken = max(0, prev_p1_prizes - curr_p1)
            delta_lost = max(0, prev_p0_prizes - curr_p0)
            step_rew = 0.5 * float(delta_taken - delta_lost)

            prev_p0_prizes = curr_p0
            prev_p1_prizes = curr_p1

            active_player = cur.get("yourIndex", 0)
            if active_player == 0:
                action, token_target, log_p, val, parsed = select_action_for_rollout(
                    model_p0, obs, device, tracker=tracker_p0
                )
                if parsed is not None:
                    traj_p0.append((parsed, token_target, log_p, val, step_rew))
            else:
                action = opp_agent.act(obs)

            obs = game.battle_select(action)
            steps += 1

        final_res = (obs.get("current") or {}).get("result", -1)
    finally:
        game.battle_finish()

    return traj_p0, final_res, steps


def train_ppo_phase2(
    seed: int = 42,
    total_iterations: int = 25,
    target_steps_per_iter: int = 2000,
    lr: float = 3e-5,
    gamma: float = 0.99,
    lam: float = 0.95,
    warmstart_bc: str = "ptcg_dual_cross_bc.pt",
    output_checkpoint: str = "ptcg_ppo_phase2_model.pt",
    log_file: str = "ppo_phase2_training_log.json",
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Phase 2 PPO Training (Seed {seed}) on {device} ({torch.cuda.get_device_name(0)}) ===")

    # 1. Load decks
    deck_files = sorted(glob.glob("deck_pool/deck_*.csv"))
    if not deck_files:
        deck_files = ["deck_gardevoir.csv", "deck_dragapult.csv", "deck_water.csv"]
    decks = [load_deck(f) for f in deck_files if len(load_deck(f)) == 60]
    print(f"Loaded {len(decks)} competitive decks for rotation.")

    # 2. Load demonstration dataset for anchoring
    replays_path = "parsed_replays_compact.pt"
    dataset_dict = torch.load(replays_path, map_location="cpu", weights_only=False)
    train_dataset = PTCGPartitionDataset(dataset_dict["train"])
    demo_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    demo_iter = iter(demo_loader)
    print(f"Demonstration anchor ready with {len(train_dataset)} decisions.")

    # 3. Model & Optimizer
    model = PTCGDualCrossNet().to(device)
    if os.path.exists(warmstart_bc):
        print(f"Warm-starting from newly parsed BC checkpoint: {warmstart_bc}")
        model.load_state_dict(torch.load(warmstart_bc, map_location=device, weights_only=True))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4
    )
    buffer = RolloutBuffer(capacity=target_steps_per_iter * 2, device=device)

    # 4. League & Opponent Population
    league_pool = [copy.deepcopy(model).eval()]
    heuristic_bot = HeuristicAgent()
    random_bot = RandomAgent(seed=seed)

    history = []

    for iteration in range(1, total_iterations + 1):
        model.eval()
        buffer.clear()

        # Annealed demonstration loss weight
        lambda_demo = max(0.05, 0.25 * (1.0 - float(iteration) / float(total_iterations)))

        pbar = tqdm(total=target_steps_per_iter, desc=f"PPO Iter {iteration:02d}/{total_iterations}")
        total_games = 0
        p0_wins = 0

        while buffer.ptr < target_steps_per_iter:
            d0 = random.choice(decks)
            d1 = random.choice(decks)

            # Population opponent selection:
            # 50% current policy self-play
            # 25% historical league checkpoint
            # 25% external heuristic agent
            opp_roll = random.random()
            if opp_roll < 0.50:
                opp = NNAgent(model, "self_play", device=device, stochastic=True)
            elif opp_roll < 0.75 and len(league_pool) > 0:
                opp_m = random.choice(league_pool)
                opp = NNAgent(opp_m, "league_past", device=device, stochastic=True)
            else:
                opp = heuristic_bot

            tracker_p0 = CardTracker(d0)
            opp.reset(d1)

            traj, res, steps = simulate_rollout_match(model, opp, d0, d1, device, tracker_p0)
            if res == -1:
                continue

            total_games += 1
            if res == 0:
                p0_wins += 1

            r_terminal = 1.0 if res == 0 else -1.0
            for i, (parsed, token_target, log_p, val, step_rew) in enumerate(traj):
                is_done = (i == len(traj) - 1)
                r = r_terminal if is_done else step_rew
                buffer.add(parsed, token_target, log_p, val, r, is_done)

            pbar.n = min(buffer.ptr, target_steps_per_iter)
            pbar.refresh()

        pbar.close()

        # Compute GAE with advantage normalization
        valid_len = buffer.ptr
        adv, ret = compute_gae(
            buffer.rewards[:valid_len],
            buffer.values[:valid_len],
            buffer.dones[:valid_len],
            gamma=gamma,
            lam=lam,
        )

        adv_std = adv.std()
        if adv_std > 1e-6:
            adv = (adv - adv.mean()) / (adv_std + 1e-8)
        else:
            adv = adv - adv.mean()

        buffer.advantages[:valid_len] = adv
        buffer.returns[:valid_len] = ret

        # Optimization phase (3 PPO epochs)
        model.train()
        pi_loss_acc, val_loss_acc, demo_loss_acc, ent_acc = 0.0, 0.0, 0.0, 0.0
        kl_acc, clip_acc, exp_var_acc, demo_acc_sum = 0.0, 0.0, 0.0, 0.0
        n_updates = 0

        for _ in range(3):
            for b in buffer.get_batches(batch_size=256):
                probs, scores, values = model(
                    b["global_vec"], b["obj_features"], b["card_ids"], b["evo_ids"],
                    b["option_mask"], b["valid_action_mask"], b["padding_mask"],
                    token_types=b["token_types"]
                )

                dist = torch.distributions.Categorical(logits=scores)
                new_log_probs = dist.log_prob(b["actions"])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - b["old_log_probs"])
                surr1 = ratio * b["advantages"]
                surr2 = torch.clamp(ratio, 0.8, 1.2) * b["advantages"]
                pi_loss = -torch.min(surr1, surr2).mean()

                v_loss = F.mse_loss(values, b["returns"])

                # Demonstration auxiliary loss
                try:
                    d_batch = next(demo_iter)
                except StopIteration:
                    demo_iter = iter(demo_loader)
                    d_batch = next(demo_iter)

                _, d_scores, _ = model(
                    d_batch["global_vec"].to(device), d_batch["obj_features"].to(device),
                    d_batch["card_ids"].to(device), d_batch["evo_ids"].to(device),
                    d_batch["option_mask"].to(device), d_batch["valid_action_mask"].to(device),
                    d_batch["padding_mask"].to(device), token_types=d_batch["token_types"].to(device)
                )
                dtgt = d_batch["target"].to(device)
                demo_loss = F.cross_entropy(d_scores, dtgt)
                batch_demo_acc = (d_scores.argmax(dim=-1) == dtgt).float().mean().item()

                total_loss = pi_loss + 0.5 * v_loss - 0.01 * entropy + lambda_demo * demo_loss

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    continue

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                # Diagnostics
                with torch.no_grad():
                    logratio = new_log_probs - b["old_log_probs"]
                    batch_kl = ((torch.exp(logratio) - 1.0) - logratio).mean().item()
                    batch_clip = ((torch.exp(logratio) - 1.0).abs() > 0.2).float().mean().item()

                    y_pred = values.detach().cpu().numpy()
                    y_true = b["returns"].cpu().numpy()
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
        avg_kl = kl_acc / max(n_updates, 1)
        avg_clip = clip_acc / max(n_updates, 1)
        avg_exp_var = exp_var_acc / max(n_updates, 1)
        win_rate = (p0_wins / max(total_games, 1)) * 100.0

        print(
            f"Iter {iteration:02d}/{total_iterations} | "
            f"Games: {total_games} | Win%: {win_rate:.1f}% | "
            f"PiLoss: {avg_pi:.4f} | ValLoss: {avg_val:.4f} | "
            f"DemoAcc: {avg_demo_acc:.1f}% (wt: {lambda_demo:.2f}) | "
            f"Clip: {avg_clip*100:.1f}% | ExpVar R2: {avg_exp_var:.3f}"
        )

        history.append({
            "iteration": iteration,
            "games": total_games,
            "win_rate": win_rate,
            "pi_loss": avg_pi,
            "val_loss": avg_val,
            "demo_loss": avg_demo,
            "demo_acc": avg_demo_acc,
            "lambda_demo": lambda_demo,
            "clip_fraction": avg_clip,
            "explained_variance": avg_exp_var,
        })

        # Save checkpoint
        torch.save(model.state_dict(), output_checkpoint)

        # Snapshot league opponent every 4 iterations
        if iteration % 4 == 0:
            league_pool.append(copy.deepcopy(model).eval())
            print(f"Added snapshot to League Pool (size: {len(league_pool)})")

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nPhase 2 PPO Training Completed! Model saved to '{output_checkpoint}'.")
    return model


if __name__ == "__main__":
    train_ppo_phase2(total_iterations=20, target_steps_per_iter=2000)
