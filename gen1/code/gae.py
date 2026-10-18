# gae.py
import torch

def compute_gae(
    rewards: torch.Tensor,       # [T]
    values: torch.Tensor,        # [T]
    dones: torch.Tensor,         # [T]
    next_value: float = 0.0,
    gamma: float = 0.99,
    lam: float = 0.95,
):
    """
    Computes Generalized Advantage Estimation (GAE) and discounted targets.
    """
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32)
    last_gae = 0.0
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t].float()
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t].float()
            next_val = values[t + 1]
            
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae
        
    returns = advantages + values
    return advantages, returns