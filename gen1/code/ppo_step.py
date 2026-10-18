# ppo_step.py
import torch
import torch.nn.functional as F

def compute_ppo_loss(
    model,
    batch,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float = 0.2,
    c_val: float = 0.5,
    c_ent: float = 0.01,
):
    """
    PPO policy and value loss computation matching Table 7 parameters.
    """
    g = batch["global_vec"]
    o = batch["obj_features"]
    c = batch["card_ids"]
    e = batch["evo_ids"]
    opt_mask = batch["option_mask"]
    val_mask = batch["valid_action_mask"]
    pad_mask = batch["padding_mask"]
    actions = batch["selected_action"]
    
    probs, scores, values = model(g, o, c, e, opt_mask, val_mask, pad_mask)
    dist = torch.distributions.Categorical(probs)
    
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()
    
    # Ratio and clipped surrogate objective
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value loss (MSE with 0.5 loss coefficient)
    value_loss = F.mse_loss(values, returns)
    
    total_loss = policy_loss + c_val * value_loss - c_ent * entropy
    return total_loss, policy_loss.item(), value_loss.item()