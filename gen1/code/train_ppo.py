# train_ppo.py
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from model import PTCGNet
from rollout_buffer import RolloutBuffer
from gae import compute_gae
from feature_parser import parse_game_state_to_tensors

# Fixed Deck List (Dragapult ex list)
DEFAULT_DECK = [
    7, 7, 7, 7, 112, 112, 112, 112, 646, 646, 646, 646, 
    1086, 1086, 1086, 1086, 1152, 1152, 1152, 1152, 1219, 1219, 
    1219, 1219, 1227, 1227, 1227, 1227, 1259, 1259, 1259, 1259, 
    647, 647, 647, 648, 648, 648, 1079, 1079, 1079, 1097, 1097, 
    1097, 104, 104, 860, 860, 1161, 1161, 1182, 1182, 1080, 1231,
    7, 7, 112, 112, 646, 646
]

def ppo_update(model, optimizer, buffer, clip_eps=0.2, c_val=0.5, c_ent=0.01, epochs=3, batch_size=256):
    model.train()
    total_pi_loss, total_val_loss = 0.0, 0.0
    batches_run = 0

    for _ in range(epochs):
        for batch in buffer.get_batches(batch_size):
            g = batch["global_vec"]
            o = batch["obj_features"]
            c = batch["card_ids"]
            e = batch["evo_ids"]
            opt_mask = batch["option_mask"]
            val_mask = batch["valid_action_mask"]
            pad_mask = batch["padding_mask"]
            actions = batch["actions"]
            old_log_probs = batch["old_log_probs"]
            advantages = batch["advantages"]
            returns = batch["returns"]

            # Normalize advantages per minibatch
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            probs, scores, values = model(g, o, c, e, opt_mask, val_mask, pad_mask)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Policy Loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value Loss (MSE with 0.5 loss coefficient)
            value_loss = nn.functional.mse_loss(values, returns)

            total_loss = policy_loss + c_val * value_loss - c_ent * entropy

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_pi_loss += policy_loss.item()
            total_val_loss += value_loss.item()
            batches_run += 1

    return total_pi_loss / max(batches_run, 1), total_val_loss / max(batches_run, 1)

def run_self_play_training(
    total_iterations: int = 100,
    steps_per_iter: int = 2048,
    lr: float = 1e-4,
    bc_checkpoint: str = "ptcg_bc_checkpoint.pt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing PPO Self-Play on {device}...")

    # Load agent network warm-started from BC
    model = PTCGNet().to(device)
    if os.path.exists(bc_checkpoint):
        print(f"Warm-starting policy and value heads from {bc_checkpoint}...")
        model.load_state_dict(torch.load(bc_checkpoint, map_location=device, weights_only=True))
    else:
        print("No BC checkpoint found. Training from scratch.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    buffer = RolloutBuffer(capacity=steps_per_iter, device=device)

    print("Setup ready.")
    return model, optimizer, buffer

if __name__ == "__main__":
    model, optimizer, buffer = run_self_play_training()