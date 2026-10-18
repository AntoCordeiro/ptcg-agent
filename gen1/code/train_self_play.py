# train_self_play.py
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from model import PTCGNet
from rollout_buffer import RolloutBuffer
from gae import compute_gae
from feature_parser import parse_game_state_to_tensors

def select_action_for_rollout(model, obs_dict, device, stochastic: bool = True):
    request = obs_dict.get("select")
    if not request or "option" not in request:
        return [0], 0, 0.0, 0.0, None

    options = request.get("option", [])
    if len(options) == 0:
        return [0], 0, 0.0, 0.0, None

    parsed = parse_game_state_to_tensors(obs_dict, request)

    g = torch.from_numpy(parsed["global_vec"]).unsqueeze(0).to(device)
    o = torch.from_numpy(parsed["obj_features"]).unsqueeze(0).to(device)
    c = torch.from_numpy(parsed["card_ids"]).unsqueeze(0).to(device)
    e = torch.from_numpy(parsed["evo_ids"]).unsqueeze(0).to(device)
    opt_mask = torch.from_numpy(parsed["option_mask"]).unsqueeze(0).to(device)
    val_mask = torch.from_numpy(parsed["valid_action_mask"]).unsqueeze(0).to(device)
    pad_mask = torch.from_numpy(parsed["padding_mask"]).unsqueeze(0).to(device)

    with torch.no_grad():
        probs, scores, val = model(g, o, c, e, opt_mask, val_mask, pad_mask)
        
        opt_indices = np.where(parsed["option_mask"])[0]
        # Gather probabilities across legal option slots
        token_probs = probs[0, opt_indices]
        token_probs = token_probs / (token_probs.sum() + 1e-8)
        
        if stochastic:
            dist = torch.distributions.Categorical(token_probs)
            choice = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(choice, device=device)).item()
        else:
            choice = int(torch.argmax(token_probs).item())
            log_prob = 0.0

        token_target = int(opt_indices[choice])
        value = val.item()

    return [choice], token_target, log_prob, value, parsed

def run_self_play(
    total_iterations: int = 50,
    rollout_steps: int = 2048,
    lr: float = 1e-4,
    bc_checkpoint: str = "ptcg_bc_checkpoint.pt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting PPO Self-Play on {device} ({torch.cuda.get_device_name(0)})...")

    model = PTCGNet().to(device)
    if os.path.exists(bc_checkpoint):
        print(f"Loaded warm-start weights from {bc_checkpoint}")
        model.load_state_dict(torch.load(bc_checkpoint, map_location=device, weights_only=True))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    buffer = RolloutBuffer(capacity=rollout_steps, device=device)

    for iteration in range(1, total_iterations + 1):
        buffer.clear()
        collected_steps = 0
        games_played = 0

        # Rollout collection loop
        pbar = tqdm(total=rollout_steps, desc=f"Iteration {iteration}/{total_iterations}")
        while not buffer.is_full():
            # Run one game trajectory
            # Collect transitions (s, a, log_p, v, r, done) into buffer
            # Simulated step addition:
            pbar.update(min(64, rollout_steps - buffer.ptr))
            buffer.ptr = min(buffer.capacity, buffer.ptr + 64)
            collected_steps = buffer.ptr

        pbar.close()

        # Compute GAE across episodic segments
        advantages, returns = compute_gae(
            buffer.rewards[:buffer.ptr],
            buffer.values[:buffer.ptr],
            buffer.dones[:buffer.ptr]
        )
        buffer.advantages[:buffer.ptr] = advantages
        buffer.returns[:buffer.ptr] = returns

        # PPO Update
        model.train()
        pi_loss_acc, val_loss_acc = 0.0, 0.0
        n_batches = 0

        for batch in buffer.get_batches(batch_size=256):
            g = batch["global_vec"]
            o = batch["obj_features"]
            c = batch["card_ids"]
            e = batch["evo_ids"]
            opt_mask = batch["option_mask"]
            val_mask = batch["valid_action_mask"]
            pad_mask = batch["padding_mask"]
            actions = batch["actions"]
            old_log_probs = batch["old_log_probs"]
            adv = batch["advantages"]
            ret = batch["returns"]

            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            probs, scores, values = model(g, o, c, e, opt_mask, val_mask, pad_mask)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(values, ret)
            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            pi_loss_acc += policy_loss.item()
            val_loss_acc += value_loss.item()
            n_batches += 1

        print(f"Iter {iteration:02d} | Pi Loss: {pi_loss_acc/n_batches:.4f} | Val Loss: {val_loss_acc/n_batches:.4f}")
        torch.save(model.state_dict(), "ptcg_ppo_latest.pt")

if __name__ == "__main__":
    run_self_play()