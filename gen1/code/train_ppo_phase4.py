# train_ppo_phase4.py
"""
Phase 4 (v2): Clean-Anchored League Self-Play & Metagame-Specialized PPO Fine-Tuning.

Key Upgrades:
1. Warm-started directly from the newly trained Clean BC model (`ptcg_dual_cross_bc.pt`, 63.3% Val Acc@1).
2. Uses `choose_multi_select_count` (combining learned `count_head` with exact PTCG multi-select mechanics)
   during rollout collection.
3. Multi-Task Demonstration Anchor Regularization (`compute_multi_select_policy_loss` + `count_loss` +
   class-balanced critic anchor) to prevent policy drift while improving tactical win-rate via RL.
4. Metagame-Biased Deck Distribution across the top leaderboard decks (Grimmsnarl ex, Alakazam,
   Dragapult ex, Teal Mask Ogerpon / Raging Bolt, Mega Lucario ex).
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
from train_bc_dual_cross import (
    PTCGPartitionDataset,
    compute_multi_select_policy_loss,
)
from ptcg_agents import HeuristicAgent, NNAgent, choose_multi_select_count, count_bounds, apply_tactical_prior_shaping
from cg import game


def load_deck(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def select_action_for_rollout(model, obs: dict, device: torch.device, tracker=None):
    request = obs.get("select") or {}
    options = request.get("option", [])
    if not options:
        return [0], 0, 0.0, 0.0, None
    if len(options) == 1:
        if tracker is not None:
            try:
                tracker.update(obs, request)
            except Exception:
                pass
        return [0], 0, 0.0, 0.0, None

    ctx = request.get("context", 0) or 0
    if ctx == 42:  # Mulligan draw
        for idx, opt in enumerate(options):
            if opt.get("type") == 1:
                return [idx], 0, 0.0, 0.0, None
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
        _, scores, val, _, count_logits, _, action_repr = model(
            g, o, c, e, opt_mask, val_mask, pad_mask, token_types=tt, return_q_and_repr=True
        )
        opt_indices = np.where(parsed["option_mask"])[0]
        opt_scores = scores[0, opt_indices].clone()
        opt_scores = apply_tactical_prior_shaping(obs, request, opt_scores)
        opt_r = action_repr[0, opt_indices]
        opt_cids = torch.from_numpy(parsed["card_ids"][opt_indices]).to(device)
        c_logit = float(count_logits[0].item())

        dist = torch.distributions.Categorical(logits=opt_scores)
        choice = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(choice, device=device)).item()
        token_target = int(opt_indices[choice])
        value = val.item()

    k = choose_multi_select_count(obs, request, opt_scores, count_logit=c_logit, mode="auto")
    k = min(k, len(opt_indices))
    if k <= 0:
        return [], token_target, log_prob, value, parsed

    if k > 1 and hasattr(model, "select_multi_options_autoregressive"):
        selected = model.select_multi_options_autoregressive(opt_scores, opt_r, opt_cids, k, ctx=ctx)
    else:
        order = torch.argsort(opt_scores, descending=True).tolist()
        selected = [int(i) for i in order[:k]]
    if choice not in selected and len(selected) > 0:
        selected[0] = choice

    return selected, token_target, log_prob, value, parsed


def simulate_rollout_match(model_p0, opp_agent, deck0, deck1, device, tracker_p0, p0_seat=0):
    obs, _ = game.battle_start(deck0, deck1) if p0_seat == 0 else game.battle_start(deck1, deck0)
    if obs is None:
        return [], -1, 0

    traj = []
    steps = 0
    max_steps = 600

    prev_my_prizes = None
    prev_opp_prizes = None
    pending_reward = 0.0

    try:
        while steps < max_steps:
            current = obs.get("current", {})
            res = current.get("result", -1)
            if res != -1:
                break

            players = current.get("players", [])
            my_p = len(players[p0_seat].get("prize", [])) if len(players) > p0_seat else 6
            opp_p = len(players[1 - p0_seat].get("prize", [])) if len(players) > (1 - p0_seat) else 6
            turn = current.get("turn", 0)

            if prev_my_prizes is not None and turn >= 1:
                my_taken = max(0, prev_my_prizes - my_p)
                my_lost = max(0, prev_opp_prizes - opp_p)
                delta_p = float(my_taken - my_lost)
                if delta_p != 0.0:
                    delta_rew = 0.35 * delta_p
                    if len(traj) > 0:
                        p, t, lp, v, r = traj[-1]
                        traj[-1] = (p, t, lp, v, r + delta_rew)
                    else:
                        pending_reward += delta_rew

            prev_my_prizes = my_p
            prev_opp_prizes = opp_p

            active_seat = current.get("yourIndex", 0)
            if active_seat == p0_seat:
                action, token_tgt, log_p, val, parsed = select_action_for_rollout(
                    model_p0, obs, device, tracker=tracker_p0
                )
                if parsed is not None:
                    step_rew = pending_reward
                    pending_reward = 0.0
                    traj.append((parsed, token_tgt, log_p, val, step_rew))
            else:
                action = opp_agent.act(obs)

            obs = game.battle_select(action)
            steps += 1

        final_res = obs.get("current", {}).get("result", -1)
    finally:
        game.battle_finish()

    if final_res == -1:
        return traj, -1, steps
    won = 1 if final_res == p0_seat else 0
    return traj, won, steps


def train_ppo_phase4(
    bc_checkpoint: str = "ptcg_iql_v3.pt",
    demo_path: str = "parsed_replays_compact.pt",
    output_checkpoint: str = "ptcg_selfplay_v3.pt",
    log_file: str = "ppo_selfplay_v3_log.json",
    total_iterations: int = 8,
    target_steps_per_iter: int = 1600,
    lr: float = 1.5e-5,
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
    print(f"=== Starting Clean-Anchored Self-Play League PPO Training on {device} ===")

    model = PTCGDualCrossNet().to(device)
    if os.path.exists(bc_checkpoint):
        model.load_state_dict(torch.load(bc_checkpoint, map_location=device, weights_only=True), strict=False)
        print(f"Warm-started weights from checkpoint '{bc_checkpoint}'.")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    buffer = RolloutBuffer(capacity=target_steps_per_iter + 500, device=device)

    print(f"Loading clean anchor demonstrations from '{demo_path}'...")
    raw_demos = torch.load(demo_path, weights_only=False)
    demo_ds = PTCGPartitionDataset(raw_demos["train"])
    demo_loader = DataLoader(demo_ds, batch_size=256, shuffle=True, drop_last=True)
    demo_iter = iter(demo_loader)

    champ_files = sorted(glob.glob("championship_decks/Champ*.csv"))
    tier1_deck_files = champ_files + [
        "deck_dragapult_evolved.csv",
        "deck_gardevoir.csv",
        "deck_pool/deck_06_4x_Dreepy_4x_Drakloak.csv",
        "deck_pool/deck_04_3x_Teal_Mask_Ogerpon_ex_.csv",
        "deck_pool/deck_01_4x_Mega_Lucario_ex_3x_So.csv",
    ]
    tier1_decks = [load_deck(p) for p in tier1_deck_files if os.path.exists(p)]
    all_deck_files = sorted(glob.glob("championship_decks/*.csv") + glob.glob("deck_pool/*.csv") + glob.glob("deck_*.csv"))
    general_decks = [load_deck(p) for p in all_deck_files if os.path.exists(p)]
    print(f"Loaded {len(tier1_decks)} Tier-1 focus decks and {len(general_decks)} total pool decks.")

    bc_anchor_model = copy.deepcopy(model).eval()
    league_pool = [bc_anchor_model]
    heuristic_bot = HeuristicAgent()
    history = []

    for iteration in range(1, total_iterations + 1):
        model.eval()
        buffer.clear()

        # Strong anchor weight (0.25 -> 0.15) to preserve 63.3% clean expert imitation while improving via RL
        lambda_demo = max(0.15, 0.25 - 0.10 * (float(iteration) / float(total_iterations)))

        pbar = tqdm(total=target_steps_per_iter, desc=f"PPO Phase 4 Iter {iteration:02d}/{total_iterations}")
        total_games = 0
        p0_wins = 0

        while buffer.ptr < target_steps_per_iter:
            if random.random() < 0.70 and len(tier1_decks) > 0:
                d0 = random.choice(tier1_decks)
                d1 = random.choice(tier1_decks)
            else:
                d0 = random.choice(general_decks)
                d1 = random.choice(general_decks)

            opp_roll = random.random()
            if opp_roll < 0.45:
                opp = NNAgent(model, "self_play", device=device, stochastic=True)
            elif opp_roll < 0.80 and len(league_pool) > 0:
                opp_m = random.choice(league_pool)
                opp = NNAgent(opp_m, "league_past", device=device, stochastic=True)
            else:
                opp = heuristic_bot

            tracker_p0 = CardTracker(d0)
            opp.reset(d1)
            p0_seat = total_games % 2

            traj, won, steps = simulate_rollout_match(model, opp, d0, d1, device, tracker_p0, p0_seat=p0_seat)
            if won == -1 or len(traj) == 0:
                continue

            total_games += 1
            if won == 1:
                p0_wins += 1

            r_terminal = 1.0 if won == 1 else -1.0
            for i, (parsed, token_target, log_p, val, step_rew) in enumerate(traj):
                is_done = (i == len(traj) - 1)
                r = (r_terminal + step_rew) if is_done else step_rew
                buffer.add(parsed, token_target, log_p, val, r, is_done)

            pbar.n = min(buffer.ptr, target_steps_per_iter)
            pbar.refresh()

        pbar.close()

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
                surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * b["advantages"]
                policy_loss = -torch.min(surr1, surr2).mean()

                val_loss = F.mse_loss(values.squeeze(-1), b["returns"])

                try:
                    demo_batch = next(demo_iter)
                except StopIteration:
                    demo_iter = iter(demo_loader)
                    demo_batch = next(demo_iter)

                d_g = demo_batch["global_vec"].to(device, non_blocking=True)
                d_o = demo_batch["obj_features"].to(device, non_blocking=True)
                d_c = demo_batch["card_ids"].to(device, non_blocking=True)
                d_e = demo_batch["evo_ids"].to(device, non_blocking=True)
                d_tt = demo_batch["token_types"].to(device, non_blocking=True)
                d_opt = demo_batch["option_mask"].to(device, non_blocking=True)
                d_val = demo_batch["valid_action_mask"].to(device, non_blocking=True)
                d_pad = demo_batch["padding_mask"].to(device, non_blocking=True)
                d_tgt = demo_batch["target"].to(device, non_blocking=True)
                d_mm = demo_batch["target_multi_mask"].to(device, non_blocking=True)
                d_cnt_r = demo_batch["target_count_ratio"].to(device, non_blocking=True)
                d_has = demo_batch["has_policy"].to(device, non_blocking=True)
                d_has_c = demo_batch["has_count"].to(device, non_blocking=True)

                _, d_scores, _, _, d_cnt_logits = model(
                    d_g, d_o, d_c, d_e, d_opt, d_val, d_pad, token_types=d_tt, return_aux=True
                )
                if d_has.any():
                    demo_loss = compute_multi_select_policy_loss(d_scores[d_has], d_mm[d_has], d_tgt[d_has])
                    d_preds = d_scores[d_has].argmax(dim=-1)
                    pol_mm = d_mm[d_has]
                    hit = pol_mm.gather(1, d_preds.unsqueeze(1)).squeeze(1)
                    demo_acc = (hit | (d_preds == d_tgt[d_has])).float().mean().item()
                else:
                    demo_loss = torch.tensor(0.0, device=device)
                    demo_acc = 0.0

                if d_has_c.any():
                    cnt_loss = F.binary_cross_entropy_with_logits(
                        d_cnt_logits.squeeze(-1)[d_has_c], d_cnt_r[d_has_c]
                    )
                else:
                    cnt_loss = torch.tensor(0.0, device=device)

                total_loss = policy_loss + 0.4 * val_loss - 0.005 * entropy + lambda_demo * (demo_loss + 0.2 * cnt_loss)

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > 0.2).float().mean().item()
                    y_pred = values.squeeze(-1)
                    y_true = b["returns"]
                    var_y = torch.var(y_true)
                    exp_var = 1.0 - torch.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 1e-6 else 0.0

                pi_loss_acc += policy_loss.item()
                val_loss_acc += val_loss.item()
                demo_loss_acc += demo_loss.item()
                ent_acc += entropy.item()
                kl_acc += approx_kl
                clip_acc += clip_frac
                exp_var_acc += exp_var.item() if isinstance(exp_var, torch.Tensor) else float(exp_var)
                demo_acc_sum += demo_acc
                n_updates += 1

        win_rate = (p0_wins / max(1, total_games)) * 100.0
        avg_pi = pi_loss_acc / max(1, n_updates)
        avg_val = val_loss_acc / max(1, n_updates)
        avg_demo = demo_acc_sum / max(1, n_updates) * 100.0
        avg_clip = clip_acc / max(1, n_updates) * 100.0
        avg_ev = exp_var_acc / max(1, n_updates)

        iter_summary = {
            "iteration": iteration,
            "games": total_games,
            "win_rate_pct": round(win_rate, 2),
            "policy_loss": round(avg_pi, 4),
            "value_loss": round(avg_val, 4),
            "demo_acc_pct": round(avg_demo, 2),
            "demo_weight": round(lambda_demo, 3),
            "clip_frac_pct": round(avg_clip, 2),
            "explained_variance": round(avg_ev, 3),
        }
        history.append(iter_summary)

        print(
            f"Iter {iteration:02d}/{total_iterations} | "
            f"Games: {total_games:2d} | "
            f"Win%: {win_rate:4.1f}% | "
            f"PiLoss: {avg_pi:6.4f} | "
            f"ValLoss: {avg_val:6.4f} | "
            f"DemoAcc: {avg_demo:4.1f}% (wt: {lambda_demo:.2f}) | "
            f"Clip: {avg_clip:4.1f}% | "
            f"ExpVar R2: {avg_ev:5.3f}"
        )

        if iteration % 4 == 0:
            snapshot = copy.deepcopy(model).eval()
            if len(league_pool) >= 8:
                league_pool.pop(1)
            league_pool.append(snapshot)

        torch.save(model.state_dict(), output_checkpoint)
        with open(log_file, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nPhase 4 League PPO Training Completed! Model saved to '{output_checkpoint}'.")
    return model


if __name__ == "__main__":
    train_ppo_phase4()
