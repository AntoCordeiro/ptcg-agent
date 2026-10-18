# train_ppo_phase3.py
"""
Phase 3: Rigorous PPO Training with Zero-Collision Feature Representation & Calibrated Baselines.

Key Technical Components:
1. Zero-Collision Observation & Action Space (Corrected AreaTypes, Target-Bound ATTACH/EVOLVE, Combat Physics).
2. Potential-Based Prize Differential Reward Shaping:
   r_t = 0.5 * (prizes_taken - prizes_lost) + r_terminal (+1.0 win / -1.0 loss)
3. Value Normalization & GAE (gamma=0.99, lambda=0.95)
4. Demonstration-Anchored Policy Gradient with Annealing on Valid Decisions:
   lambda_demo(t) = max(0.05, 0.25 * (1.0 - t / total_iterations))
5. Population Self-Play:
   - 50% Self-Play (Current Policy)
   - 25% Historical Checkpoint League Pool
   - 25% External Heuristic Agent
6. Multi-Archetype Rotation across official tournament decks.
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
from train_bc_dual_cross import PTCGPartitionDataset, compute_topk_correct
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
    ctx = request.get("context", 0) or 0

    # Context-aware multi-select count
    if ctx in (1, 2, 5, 6, 7):
        k = mx
    elif ctx in (8, 26, 27, 29, 30):
        k = mn
    else:
        k = max(mn, 1 if mx >= 1 else 0)
    k = max(mn, min(k, mx, len(opt_indices)))

    order = torch.argsort(opt_scores, descending=True).tolist()
    selected = [int(i) for i in order[:k]]
    if choice not in selected and len(selected) > 0:
        selected[0] = choice

    return selected, token_target, log_prob, value, parsed


def simulate_rollout_match(model_p0, opp_agent, deck0, deck1, device, tracker_p0):
    obs, _ = game.battle_start(deck0, deck1)
    if obs is None:
        return [], -1, 0

    traj = []
    steps = 0
    max_steps = 500

    prev_p0_prizes = None
    prev_p1_prizes = None
    pending_p0_reward = 0.0

    try:
        while steps < max_steps:
            current = obs.get("current", {})
            res = current.get("result", -1)
            if res != -1:
                break

            players = current.get("players", [])
            p0_p = len(players[0].get("prize", [])) if len(players) > 0 else 6
            p1_p = len(players[1].get("prize", [])) if len(players) > 1 else 6
            turn = current.get("turn", 0)

            # In PTCG, player prize pile starts at 6 and decreases when taking prizes.
            # p0 takes prizes when p0's prize pile shrinks!
            # p1 takes prizes when p1's prize pile shrinks!
            if prev_p0_prizes is not None and turn >= 1:
                p0_taken = max(0, prev_p0_prizes - p0_p)
                p0_lost = max(0, prev_p1_prizes - p1_p)
                delta_p0 = float(p0_taken - p0_lost)
                if delta_p0 != 0.0:
                    delta_rew = 0.5 * delta_p0
                    if len(traj) > 0:
                        p, t, lp, v, r = traj[-1]
                        traj[-1] = (p, t, lp, v, r + delta_rew)
                    else:
                        pending_p0_reward += delta_rew

            prev_p0_prizes = p0_p
            prev_p1_prizes = p1_p

            active_seat = current.get("yourIndex", 0)
            if active_seat == 0:
                action, token_tgt, log_p, val, parsed = select_action_for_rollout(
                    model_p0, obs, device, tracker=tracker_p0
                )
                if parsed is not None:
                    step_rew = pending_p0_reward
                    pending_p0_reward = 0.0
                    traj.append((parsed, token_tgt, log_p, val, step_rew))
            else:
                action = opp_agent.act(obs)

            obs = game.battle_select(action)
            steps += 1

        final_res = obs.get("current", {}).get("result", -1)
    finally:
        game.battle_finish()

    return traj, final_res, steps


def train_ppo_phase3(
    bc_checkpoint: str = "ptcg_dual_cross_bc.pt",
    demo_path: str = "parsed_replays_compact.pt",
    output_checkpoint: str = "ptcg_ppo_phase3_model.pt",
    log_file: str = "ppo_phase3_training_log.json",
    total_iterations: int = 20,
    target_steps_per_iter: int = 2000,
    lr: float = 3e-5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Phase 3 PPO Training on device: {device} ({torch.cuda.get_device_name(0)})")

    # Load starting model
    model = PTCGDualCrossNet().to(device)
    if os.path.exists(bc_checkpoint):
        model.load_state_dict(torch.load(bc_checkpoint, map_location=device, weights_only=True))
        print(f"Loaded starting weights from zero-collision BC checkpoint '{bc_checkpoint}'.")
    else:
        print(f"Warning: '{bc_checkpoint}' not found, training from scratch.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    buffer = RolloutBuffer(capacity=target_steps_per_iter + 500, device=device)

    # Load demonstrations for anchor regularization
    print(f"Loading anchor demonstrations from '{demo_path}'...")
    raw_demos = torch.load(demo_path, weights_only=False)
    demo_ds = PTCGPartitionDataset(raw_demos["train"])
    demo_loader = DataLoader(demo_ds, batch_size=256, shuffle=True, drop_last=True)
    demo_iter = iter(demo_loader)

    # Decks
    deck_files = sorted(glob.glob("deck_pool/*.csv") + glob.glob("deck_*.csv"))
    decks = [load_deck(p) for p in deck_files]
    print(f"Loaded {len(decks)} legal tournament decks for multi-archetype rotation.")

    # Historical opponents pool
    league_pool = [copy.deepcopy(model).eval()]
    heuristic_bot = HeuristicAgent()
    random_bot = RandomAgent(seed=seed)

    history = []

    for iteration in range(1, total_iterations + 1):
        model.eval()
        buffer.clear()

        lambda_demo = max(0.05, 0.25 * (1.0 - float(iteration) / float(total_iterations)))

        pbar = tqdm(total=target_steps_per_iter, desc=f"PPO Phase 3 Iter {iteration:02d}/{total_iterations}")
        total_games = 0
        p0_wins = 0

        while buffer.ptr < target_steps_per_iter:
            d0 = random.choice(decks)
            d1 = random.choice(decks)

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
                r = (r_terminal + step_rew) if is_done else step_rew
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
            lam=gae_lambda
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
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

                # Demonstration auxiliary loss anchored on valid decision points
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
                has_pol = d_batch.get("has_policy", torch.ones_like(dtgt, dtype=torch.bool)).to(device)

                if has_pol.any():
                    demo_loss = F.cross_entropy(d_scores[has_pol], dtgt[has_pol])
                    batch_demo_acc = (d_scores[has_pol].argmax(dim=-1) == dtgt[has_pol]).float().mean().item()
                else:
                    demo_loss = torch.tensor(0.0, device=device)
                    batch_demo_acc = 0.0

                total_loss = pi_loss + 0.5 * v_loss - 0.01 * entropy + lambda_demo * demo_loss

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    continue

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = (b["old_log_probs"] - new_log_probs).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > 0.2).float().mean().item()
                    y_true = b["returns"]
                    y_pred = values
                    var_y = torch.var(y_true)
                    exp_var = 1.0 - (torch.var(y_true - y_pred) / (var_y + 1e-8)).item() if var_y > 0 else 0.0

                pi_loss_acc += pi_loss.item()
                val_loss_acc += v_loss.item()
                demo_loss_acc += demo_loss.item()
                demo_acc_sum += batch_demo_acc
                ent_acc += entropy.item()
                kl_acc += approx_kl
                clip_acc += clip_frac
                exp_var_acc += exp_var
                n_updates += 1

        avg_pi = pi_loss_acc / max(n_updates, 1)
        avg_val = val_loss_acc / max(n_updates, 1)
        avg_demo = demo_loss_acc / max(n_updates, 1)
        avg_demo_acc = (demo_acc_sum / max(n_updates, 1)) * 100.0
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

        torch.save(model.state_dict(), output_checkpoint)

        if iteration % 4 == 0:
            league_pool.append(copy.deepcopy(model).eval())
            print(f"Added snapshot to League Pool (size: {len(league_pool)})")

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nPhase 3 PPO Training Completed! Model saved to '{output_checkpoint}'.")
    return model


if __name__ == "__main__":
    train_ppo_phase3(total_iterations=20, target_steps_per_iter=2000)
