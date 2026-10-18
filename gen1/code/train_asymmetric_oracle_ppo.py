# train_asymmetric_oracle_ppo.py
"""
PPO Training with Asymmetric "Oracle" Critic in Imperfect Information (Pillar 2).

Directly demonstrates the AlphaStar privileged learning principle:
1. Actor Policy pi(a | s_obs) receives only observable game tokens.
2. Dual Critics:
   - Observable Critic V_obs(s_obs)
   - Privileged Oracle Critic V_oracle(s_obs, s_priv)
3. Tracks convergence, MSE loss, and Explained Variance (R^2) comparison:
   R^2 = 1 - Var(Returns - V) / Var(Returns)
4. Saves publication comparative plot: results_plots/figure8_oracle_vs_observable_critic.png
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
import matplotlib.pyplot as plt
from tqdm import tqdm

SUBMISSION_DIR = os.path.abspath(
    "the-pokemon-company-ptcg-ai-battle-challenge-playground/sample_submission/sample_submission/sample_submission"
)
if SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, SUBMISSION_DIR)

from cg import game
from model_oracle_critic import PTCGAsymmetricOracleNet, PRIVILEGED_DIM
from card_tracker import CardTracker, get_card_type
from feature_parser import parse_game_state_to_tensors
from gae import compute_gae
from match_sim import select_action


def load_deck(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def extract_privileged_vector(obs: dict, my_idx: int, deck_my: list[int], deck_opp: list[int], true_opp_hand: list = None) -> np.ndarray:
    """
    Constructs the 64-dimensional privileged state vector:
    - Opponent's private hand card category distribution (dims 0-9)
    - Opponent's unobserved deck distribution (dims 10-19)
    - Player's complete deck profile (dims 20-29)
    - Opponent's complete deck profile (dims 30-39)
    - True remaining prize/deck counts (dims 40-45)
    """
    priv = np.zeros(PRIVILEGED_DIM, dtype=np.float32)
    current = obs.get("current", {}) or {}
    players = current.get("players", []) or []

    opp_idx = 1 - my_idx
    opp_state = players[opp_idx] if len(players) > opp_idx else {}
    my_state = players[my_idx] if len(players) > my_idx else {}

    # Opponent's true private hand category distribution (dims 0-9)
    opp_hand = true_opp_hand if true_opp_hand is not None else (opp_state.get("hand") or [])
    for c in opp_hand:
        cid = c.get("cardId", c.get("id", 0)) if isinstance(c, dict) else c
        if cid:
            t = get_card_type(int(cid))
            if 0 <= t < 10:
                priv[t] += 1.0 / 10.0

    # My complete deck profile (dims 20-29)
    for cid in deck_my:
        t = get_card_type(int(cid))
        if 0 <= t < 10:
            priv[20 + t] += 1.0 / 60.0

    # Opponent complete deck profile (dims 30-39)
    for cid in deck_opp:
        t = get_card_type(int(cid))
        if 0 <= t < 10:
            priv[30 + t] += 1.0 / 60.0

    # True prize and deck metrics (dims 40-45)
    my_prizes = float(len(my_state.get("prize", [])))
    opp_prizes = float(len(opp_state.get("prize", [])))
    my_deck_cnt = float(my_state.get("deckCount", 50) or 50)
    opp_deck_cnt = float(opp_state.get("deckCount", 50) or 50)

    priv[40] = my_prizes / 6.0
    priv[41] = opp_prizes / 6.0
    priv[42] = (opp_prizes - my_prizes) / 6.0
    priv[43] = my_deck_cnt / 60.0
    priv[44] = opp_deck_cnt / 60.0

    return priv


def simulate_oracle_match(model, deck0: list[int], deck1: list[int], device: torch.device):
    obs, start_data = game.battle_start(deck0, deck1)
    if obs is None:
        return [], -1

    tracker_p0 = CardTracker(deck0)
    tracker_p1 = CardTracker(deck1)
    traj = []
    steps = 0
    max_steps = 400
    last_known_hands = {0: [], 1: []}
    prev_prizes = {0: 6, 1: 6}

    try:
        while steps < max_steps:
            current = obs.get("current", {})
            if current.get("result", -1) != -1:
                break

            seat = current.get("yourIndex", 0)
            request = obs.get("select") or {}
            options = request.get("option", [])

            players = current.get("players", [])
            if len(players) > seat and players[seat].get("hand"):
                last_known_hands[seat] = players[seat]["hand"]

            active_tracker = tracker_p0 if seat == 0 else tracker_p1
            my_deck = deck0 if seat == 0 else deck1
            opp_deck = deck1 if seat == 0 else deck0
            true_opp_hand = last_known_hands.get(1 - seat, [])

            if options and len(options) > 0:
                parsed = parse_game_state_to_tensors(obs, request, tracker=active_tracker)
                priv_vec = extract_privileged_vector(obs, seat, my_deck, opp_deck, true_opp_hand=true_opp_hand)

                g = torch.from_numpy(parsed["global_vec"]).unsqueeze(0).to(device)
                o = torch.from_numpy(parsed["obj_features"]).unsqueeze(0).to(device)
                c = torch.from_numpy(parsed["card_ids"]).unsqueeze(0).to(device)
                e = torch.from_numpy(parsed["evo_ids"]).unsqueeze(0).to(device)
                tt = torch.from_numpy(parsed["token_types"]).unsqueeze(0).to(device)
                opt_mask = torch.from_numpy(parsed["option_mask"]).unsqueeze(0).to(device)
                val_mask = torch.from_numpy(parsed["valid_action_mask"]).unsqueeze(0).to(device)
                pad_mask = torch.from_numpy(parsed["padding_mask"]).unsqueeze(0).to(device)
                priv_t = torch.from_numpy(priv_vec).unsqueeze(0).to(device)

                with torch.no_grad():
                    probs, scores, v_obs, v_orc = model(
                        g, o, c, e, opt_mask, val_mask, pad_mask,
                        token_types=tt, privileged_features=priv_t, return_oracle_value=True
                    )
                    opt_indices = np.where(parsed["option_mask"])[0]
                    opt_scores = scores[0, opt_indices]
                    opt_probs = F.softmax(opt_scores, dim=-1)

                    dist = torch.distributions.Categorical(opt_probs)
                    choice = dist.sample().item()
                    log_p = dist.log_prob(torch.tensor(choice, device=device)).item()

                    token_target = int(opt_indices[choice])
                    v_obs_val = v_obs.item()
                    v_orc_val = v_orc.item()

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
                my_p_cnt = len(players[seat].get("prize", [])) if len(players) > seat else 6
                opp_p_cnt = len(players[1 - seat].get("prize", [])) if len(players) > (1 - seat) else 6
                my_p_delta = prev_prizes[seat] - my_p_cnt
                opp_p_delta = prev_prizes[1 - seat] - opp_p_cnt
                prev_prizes[seat] = my_p_cnt
                prev_prizes[1 - seat] = opp_p_cnt
                step_reward = float(opp_p_delta) * 0.5 - float(my_p_delta) * 0.5

                traj.append({
                    "seat": seat,
                    "parsed": parsed,
                    "priv": priv_vec,
                    "token_target": token_target,
                    "log_p": log_p,
                    "v_obs": v_obs_val,
                    "v_orc": v_orc_val,
                    "step_reward": step_reward,
                })
                action = selected
            else:
                action = [0]

            obs = game.battle_select(action)
            steps += 1

        final_res = obs.get("current", {}).get("result", -1)
    finally:
        game.battle_finish()

    return traj, final_res


def train_asymmetric_oracle_ppo(
    total_iterations: int = 12,
    target_steps_per_iter: int = 1200,
    lr: float = 4e-5,
    deck_pool_dir: str = "deck_pool",
    warmstart_path: str = "ptcg_ppo_phase2_model.pt",
    output_checkpoint: str = "ptcg_asymmetric_oracle_model.pt",
    log_path: str = "oracle_critic_training_log.json"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Asymmetric Oracle Critic PPO on {device} ({torch.cuda.get_device_name(0)}) ---")

    # Load deck pool
    deck_files = sorted(glob.glob(os.path.join(deck_pool_dir, "deck_*.csv")))[:6]
    if not deck_files:
        deck_files = ["deck_gardevoir.csv", "deck_dragapult.csv", "deck_water.csv"]
    decks = [load_deck(df) for df in deck_files]
    print(f"Loaded {len(decks)} competitive archetypes for Oracle training.")

    # Initialize model
    model = PTCGAsymmetricOracleNet().to(device)
    if os.path.exists(warmstart_path):
        print(f"Warm-starting Base Net from {warmstart_path}...")
        base_state = torch.load(warmstart_path, map_location=device, weights_only=True)
        model.base_net.load_state_dict(base_state, strict=False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4
    )

    history = []

    for iteration in range(1, total_iterations + 1):
        model.eval()
        buffer_steps = []
        pbar = tqdm(total=target_steps_per_iter, desc=f"Oracle Critic Iter {iteration:02d}/{total_iterations}")

        while len(buffer_steps) < target_steps_per_iter:
            d0 = random.choice(decks)
            d1 = random.choice(decks)

            traj, result = simulate_oracle_match(model, d0, d1, device)
            if result not in (0, 1) or len(traj) == 0:
                continue

            # Compute discounted returns incorporating intermediate prize deltas
            for target_seat in (0, 1):
                seat_steps = [item for item in traj if item["seat"] == target_seat]
                running_return = 1.0 if result == target_seat else -1.0
                gamma = 0.99
                for item in reversed(seat_steps):
                    r_t = item["step_reward"]
                    running_return = r_t + gamma * running_return
                    adv = running_return - item["v_orc"]
                    buffer_steps.append({
                        "parsed": item["parsed"],
                        "priv": item["priv"],
                        "action": item["token_target"],
                        "log_p": item["log_p"],
                        "ret": running_return,
                        "adv": adv,
                    })

            pbar.n = min(len(buffer_steps), target_steps_per_iter)
            pbar.refresh()

        pbar.close()

        # Training Phase
        model.train()
        loss_obs_acc = 0.0
        loss_orc_acc = 0.0
        y_true_all = []
        y_obs_all = []
        y_orc_all = []
        n_updates = 0

        # Create mini-batches
        random.shuffle(buffer_steps)
        batch_size = 128
        for i in range(0, len(buffer_steps), batch_size):
            batch = buffer_steps[i:i + batch_size]
            if len(batch) < 16:
                continue

            g = torch.from_numpy(np.stack([b["parsed"]["global_vec"] for b in batch])).to(device)
            o = torch.from_numpy(np.stack([b["parsed"]["obj_features"] for b in batch])).to(device)
            c = torch.from_numpy(np.stack([b["parsed"]["card_ids"] for b in batch])).to(device)
            e = torch.from_numpy(np.stack([b["parsed"]["evo_ids"] for b in batch])).to(device)
            tt = torch.from_numpy(np.stack([b["parsed"]["token_types"] for b in batch])).to(device)
            opt_mask = torch.from_numpy(np.stack([b["parsed"]["option_mask"] for b in batch])).to(device)
            val_mask = torch.from_numpy(np.stack([b["parsed"]["valid_action_mask"] for b in batch])).to(device)
            pad_mask = torch.from_numpy(np.stack([b["parsed"]["padding_mask"] for b in batch])).to(device)
            priv = torch.from_numpy(np.stack([b["priv"] for b in batch])).to(device)

            actions = torch.tensor([b["action"] for b in batch], dtype=torch.long, device=device)
            old_log_probs = torch.tensor([b["log_p"] for b in batch], dtype=torch.float32, device=device)
            returns = torch.tensor([b["ret"] for b in batch], dtype=torch.float32, device=device)
            advantages = torch.tensor([b["adv"] for b in batch], dtype=torch.float32, device=device)

            probs, scores, v_obs, v_orc = model(
                g, o, c, e, opt_mask, val_mask, pad_mask,
                token_types=tt, privileged_features=priv, return_oracle_value=True
            )

            # Policy Loss
            dist = torch.distributions.Categorical(logits=scores)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            pi_loss = -torch.min(ratio * advantages, torch.clamp(ratio, 0.8, 1.2) * advantages).mean()

            # Dual Critic Losses
            loss_obs = F.mse_loss(v_obs, returns)
            loss_orc = F.mse_loss(v_orc, returns)

            total_loss = pi_loss + 0.3 * loss_obs + 0.3 * loss_orc

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            loss_obs_acc += loss_obs.item()
            loss_orc_acc += loss_orc.item()
            y_true_all.extend(returns.cpu().numpy())
            y_obs_all.extend(v_obs.detach().cpu().numpy())
            y_orc_all.extend(v_orc.detach().cpu().numpy())
            n_updates += 1

        avg_loss_obs = loss_obs_acc / max(n_updates, 1)
        avg_loss_orc = loss_orc_acc / max(n_updates, 1)

        y_t = np.array(y_true_all)
        y_o = np.array(y_obs_all)
        y_orc = np.array(y_orc_all)
        var_y = np.var(y_t)

        r2_obs = float(1.0 - np.var(y_t - y_o) / var_y) if var_y > 1e-6 else 0.0
        r2_orc = float(1.0 - np.var(y_t - y_orc) / var_y) if var_y > 1e-6 else 0.0

        print(
            f"Iter {iteration:02d}/{total_iterations} | "
            f"Obs MSE: {avg_loss_obs:.4f} (R2: {r2_obs:+.3f}) | "
            f"Oracle MSE: {avg_loss_orc:.4f} (R2: {r2_orc:+.3f}) | "
            f"Oracle Advantage: +{r2_orc - r2_obs:+.3f} R2"
        )

        history.append({
            "iteration": iteration,
            "obs_mse": avg_loss_obs,
            "oracle_mse": avg_loss_orc,
            "obs_r2": r2_obs,
            "oracle_r2": r2_orc,
            "r2_gap": r2_orc - r2_obs
        })

        torch.save(model.state_dict(), output_checkpoint)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Generate Publication Figure 8
    plot_oracle_vs_observable(history)
    print(f"\nSaved checkpoint to '{output_checkpoint}' and log to '{log_path}'.")
    return model


def plot_oracle_vs_observable(history: list, output_path: str = "results_plots/figure8_oracle_vs_observable_critic.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    iters = [x["iteration"] for x in history]
    obs_mse = [x["obs_mse"] for x in history]
    orc_mse = [x["oracle_mse"] for x in history]
    obs_r2 = [x["obs_r2"] for x in history]
    orc_r2 = [x["oracle_r2"] for x in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: MSE Loss Convergence
    ax1.plot(iters, obs_mse, label="Observable Critic V(s_obs)", color="#e74c3c", marker="o", linewidth=2.2)
    ax1.plot(iters, orc_mse, label="Asymmetric Oracle Critic V(s_obs, s_priv)", color="#27ae60", marker="s", linewidth=2.2)
    ax1.set_xlabel("PPO Iteration", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Value MSE Loss", fontsize=11, fontweight="bold")
    ax1.set_title("Critic Value Estimation Loss: Observable vs. Privileged", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Panel B: Explained Variance R^2
    ax2.plot(iters, obs_r2, label="Observable Critic R²", color="#e74c3c", marker="o", linewidth=2.2)
    ax2.plot(iters, orc_r2, label="Asymmetric Oracle Critic R²", color="#27ae60", marker="s", linewidth=2.2)
    ax2.axhline(y=0.0, color="gray", linestyle=":", alpha=0.6)
    ax2.set_xlabel("PPO Iteration", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Explained Variance (R²)", fontsize=11, fontweight="bold")
    ax2.set_title("Variance Reduction in Value Prediction: R² = 1 - Var(R - V) / Var(R)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("AlphaStar Privileged Learning: Asymmetric Oracle Critic vs. Standard Critic (PTCGDualCrossNet)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved publication figure to '{output_path}'.")


if __name__ == "__main__":
    train_asymmetric_oracle_ppo(total_iterations=10, target_steps_per_iter=1200)
