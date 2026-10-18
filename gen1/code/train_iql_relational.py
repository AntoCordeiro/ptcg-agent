# train_iql_relational.py
"""
Offline Implicit Q-Learning (IQL) + Relational Pairwise-Bias & Autoregressive Pointer Training
(`train_iql_relational.py` — Approaches A & C).

Key Innovations:
1. Warm-starts `PTCGDualCrossNet` v2 from `ptcg_wise_ensemble.pt` (`strict=False`), preserving
   the +249.4 Elo baseline at epoch 0 via zero-initialized residual gates.
2. Differential Learning Rates:
   - Rapidly trains the new Relational Pairwise-Bias (`rel_bias_proj`), Option-Set Self-Attention
     (`opt_self_attn`), IQL Action-Value Critic (`q_head`), and Conditional Pointer Synergy (`pointer_proj`)
     at `lr=5e-4`, while gently adapting the pre-trained trunk at `lr=6e-5`.
3. Offline IQL + Advantage-Weighted Regression (AWR, Approach C):
   - Trains `q_head` Q(s, a) on empirical outcome + prize returns and contrastive action margins.
   - Trains `value_head` V(s) with asymmetric expectile loss (tau = 0.75) + class-balanced outcome supervision.
   - Weights policy imitation by `w_AWR = clip(exp(beta * Advantage), 0.6, 2.8)`.
4. Conditional Multi-Select Pointer Co-Selection Loss (Approach A3):
   - Directly supervises `pointer_proj` and `pointer_gate` on multi-select subsets (|S*| >= 2)
     so selecting option a_1 boosts the conditional score of synergistic co-selected options a_2..a_k.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_dual_cross import PTCGDualCrossNet
from train_bc_dual_cross import PTCGPartitionDataset


def expectile_loss(diff: torch.Tensor, tau: float = 0.75) -> torch.Tensor:
    """Asymmetric L2 expectile loss from Implicit Q-Learning (Kostrikov et al., ICLR 2022)."""
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return (weight * (diff ** 2)).mean()


def compute_awr_and_pointer_losses(
    logits: torch.Tensor,          # [B, S_obj]
    q_values: torch.Tensor,        # [B, S_obj]
    action_repr: torch.Tensor,     # [B, S_obj, d_model]
    v_pred: torch.Tensor,          # [B]
    target_idx: torch.Tensor,      # [B]
    target_multi_mask: torch.Tensor,  # [B, S_obj]
    valid_mask: torch.Tensor,      # [B, S_obj]
    has_policy: torch.Tensor,      # [B]
    returns: torch.Tensor,         # [B] in [-1, 1]
    model: PTCGDualCrossNet,
    beta_awr: float = 1.5,
):
    """
    Computes:
    1. IQL Critic loss L_Q (return regression + contrastive margin on chosen vs unchosen actions)
    2. Advantage-Weighted Regression policy loss L_AWR
    3. Autoregressive Pointer co-selection loss L_ptr on multi-select steps (|S*| >= 2)
    """
    device = logits.device
    B, S_obj = logits.shape

    # --- 1. IQL Q-Critic Supervision ---
    # For winner trajectories (has_policy == True), supervise Q(s, a*) -> returns[b] (> 0)
    # and encourage Q(s, a*) > Q(s, a_unchosen) + 0.25.
    # For loser trajectories (has_policy == False), supervise mean/max Q(s, .) -> returns[b] (< 0).
    q_loss = torch.tensor(0.0, device=device)
    win_mask = has_policy & (returns > 0)
    lose_mask = ~has_policy

    if win_mask.any():
        q_win = q_values[win_mask]                # [B_w, S_obj]
        m_win = target_multi_mask[win_mask]       # [B_w, S_obj]
        v_win = valid_mask[win_mask]              # [B_w, S_obj]
        r_win = returns[win_mask]                 # [B_w]

        # Fallback to primary target_idx if multi_mask is empty
        t_win = target_idx[win_mask]
        fallback_m = F.one_hot(t_win.clamp(0, S_obj - 1), num_classes=S_obj).bool()
        pos_mask = torch.where(m_win.sum(dim=-1, keepdim=True) > 0, m_win, fallback_m) & v_win
        neg_mask = v_win & (~pos_mask)

        # Mean Q on chosen actions
        pos_cnt = pos_mask.float().sum(dim=-1).clamp(min=1.0)
        q_pos_mean = (q_win * pos_mask.float()).sum(dim=-1) / pos_cnt
        q_loss_pos = F.smooth_l1_loss(q_pos_mean, r_win)

        # Contrastive margin on unchosen valid actions
        has_neg = neg_mask.any(dim=-1)
        if has_neg.any():
            q_neg_masked = q_win[has_neg].masked_fill(~neg_mask[has_neg], -1e4)
            q_neg_hardest = q_neg_masked.max(dim=-1).values
            margin_err = F.relu(0.25 - (q_pos_mean[has_neg] - q_neg_hardest))
            q_loss_pos = q_loss_pos + 0.5 * margin_err.mean()
        q_loss = q_loss + q_loss_pos

    if lose_mask.any():
        q_lose = q_values[lose_mask]
        v_lose = valid_mask[lose_mask]
        r_lose = returns[lose_mask]
        l_cnt = v_lose.float().sum(dim=-1).clamp(min=1.0)
        q_lose_mean = (q_lose * v_lose.float()).sum(dim=-1) / l_cnt
        q_loss = q_loss + F.smooth_l1_loss(q_lose_mean, r_lose)

    # --- 2. Advantage-Weighted Regression (AWR) Policy Loss & Pointer Synergy Loss ---
    if not has_policy.any():
        zero = torch.tensor(0.0, device=device)
        return zero, q_loss, zero, 0, 0, 0

    logits_p = logits[has_policy]
    q_p = q_values[has_policy].detach()
    repr_p = action_repr[has_policy]
    v_p = v_pred[has_policy].detach()
    targets_p = target_idx[has_policy]
    multi_p = target_multi_mask[has_policy]
    valid_p = valid_mask[has_policy]
    ret_p = returns[has_policy]

    log_probs = F.log_softmax(logits_p, dim=-1)
    log_probs_safe = torch.where(valid_p, log_probs, torch.zeros_like(log_probs))

    # Multi-select target distribution
    multi_valid = multi_p & valid_p
    multi_counts = multi_valid.float().sum(dim=-1, keepdim=True)
    one_hot_primary = F.one_hot(targets_p.clamp(0, S_obj - 1), num_classes=S_obj).float()
    target_dist = torch.where(
        multi_counts > 0,
        multi_valid.float() / multi_counts.clamp(min=1.0),
        one_hot_primary
    )
    per_sample_ce = -(target_dist * log_probs_safe).sum(dim=-1)  # [B_p]

    # Compute IQL Advantage A(s, a*) = 0.6*(R - V(s)) + 0.4*(Q(s, a*) - V(s))
    q_star = (q_p * target_dist).sum(dim=-1)
    adv = 0.6 * (ret_p - v_p) + 0.4 * (q_star - v_p)
    awr_weights = torch.exp(beta_awr * adv).clamp(0.6, 2.8)
    awr_weights = awr_weights / awr_weights.mean().clamp(min=1e-6)

    awr_policy_loss = (awr_weights * per_sample_ce).mean()

    # --- 3. Conditional Multi-Select Pointer Synergy Loss (Approach A3) ---
    is_multi_step = (multi_counts.squeeze(-1) >= 2.0)
    ptr_loss = torch.tensor(0.0, device=device)
    if is_multi_step.any():
        m_logits = logits_p[is_multi_step]          # [B_m, S_obj]
        m_repr = repr_p[is_multi_step]              # [B_m, S_obj, d_model]
        m_first = targets_p[is_multi_step]          # [B_m]
        m_mask = multi_valid[is_multi_step].clone() # [B_m, S_obj]
        m_val = valid_p[is_multi_step].clone()      # [B_m, S_obj]

        B_m = m_logits.shape[0]
        batch_arange = torch.arange(B_m, device=device)
        first_clamped = m_first.clamp(0, S_obj - 1)

        # Mask out the first selected option a_1 to predict remaining co-selected options a_2..a_k
        m_mask[batch_arange, first_clamped] = False
        m_val[batch_arange, first_clamped] = False

        rem_counts = m_mask.float().sum(dim=-1, keepdim=True)
        valid_rem = (rem_counts.squeeze(-1) >= 1.0) & m_val.any(dim=-1)
        if valid_rem.any():
            first_repr = m_repr[batch_arange, first_clamped]  # [B_m, d_model]
            proj_first = model.pointer_proj(first_repr).unsqueeze(1)  # [B_m, 1, d_model]
            synergy_scores = (m_repr * proj_first).sum(dim=-1) / (model.d_model ** 0.5)  # [B_m, S_obj]

            cond_logits = m_logits + torch.tanh(model.pointer_gate) * synergy_scores
            cond_logits = cond_logits.masked_fill(~m_val, -1e9)

            cond_log_probs = F.log_softmax(cond_logits[valid_rem], dim=-1)
            cond_log_probs_safe = torch.where(m_val[valid_rem], cond_log_probs, torch.zeros_like(cond_log_probs))
            rem_dist = m_mask[valid_rem].float() / rem_counts[valid_rem].clamp(min=1.0)
            ptr_loss = -(rem_dist * cond_log_probs_safe).sum(dim=-1).mean()

    # Metrics
    with torch.no_grad():
        top1_preds = torch.argmax(logits_p, dim=-1)
        batch_idx = torch.arange(logits_p.size(0), device=device)
        in_multi = multi_valid[batch_idx, top1_preds]
        acc1 = ((top1_preds == targets_p) | in_multi).sum().item()

        k = min(3, logits_p.size(-1))
        _, top3_preds = torch.topk(logits_p, k=k, dim=-1)
        acc3 = (top3_preds == targets_p.unsqueeze(-1)).any(dim=-1).sum().item()
        total = int(has_policy.sum().item())

    return awr_policy_loss, q_loss, ptr_loss, acc1, acc3, total


def train_iql_relational(
    dataset_path: str = "parsed_replays_compact.pt",
    init_ckpt: str = "ptcg_wise_ensemble.pt",
    save_path: str = "ptcg_iql_relational.pt",
    epochs: int = 12,
    batch_size: int = 256,
    lr_trunk: float = 6e-5,
    lr_new: float = 5e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Approach A + C Training (Relational Pointer + IQL) on {device} ===")

    raw = torch.load(dataset_path, map_location="cpu", weights_only=False)
    train_ds = PTCGPartitionDataset(raw["train"])
    val_ds = PTCGPartitionDataset(raw["val"])
    del raw

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = PTCGDualCrossNet(dropout=0.12).to(device)
    if os.path.exists(init_ckpt):
        model.load_state_dict(torch.load(init_ckpt, map_location=device, weights_only=True), strict=False)
        print(f"Warm-started trunk weights from '{init_ckpt}' (zero-initialized relational/pointer/IQL gates).")

    new_param_names = {
        "rel_bias_proj", "opt_self_norm", "opt_self_attn", "opt_self_gate",
        "q_head", "q_logit_gate", "pointer_proj", "pointer_gate"
    }
    new_params, trunk_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in new_param_names):
            new_params.append(param)
        else:
            trunk_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": trunk_params, "lr": lr_trunk, "weight_decay": 1e-4},
        {"params": new_params, "lr": lr_new, "weight_decay": 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    mse_loss_fn = nn.SmoothL1Loss()
    bce_logits_fn = nn.BCEWithLogitsLoss()

    best_val_acc1 = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        t_pol, t_q, t_ptr, t_val, t_cnt = 0.0, 0.0, 0.0, 0.0, 0.0
        c1, c3, tot_pol, steps = 0, 0, 0, 0

        pbar = tqdm(train_loader, desc=f"IQL+Rel Epoch {epoch:02d}/{epochs:02d} [Train]", leave=False)
        for batch in pbar:
            g = batch["global_vec"].to(device, non_blocking=True)
            o = batch["obj_features"].to(device, non_blocking=True)
            c = batch["card_ids"].to(device, non_blocking=True)
            e = batch["evo_ids"].to(device, non_blocking=True)
            tt = batch["token_types"].to(device, non_blocking=True)
            opt_mask = batch["option_mask"].to(device, non_blocking=True)
            val_mask = batch["valid_action_mask"].to(device, non_blocking=True)
            pad_mask = batch["padding_mask"].to(device, non_blocking=True)

            target_idx = batch["target"].to(device, non_blocking=True)
            target_multi = batch["target_multi_mask"].to(device, non_blocking=True)
            target_cnt_ratio = batch["target_count_ratio"].to(device, non_blocking=True)
            has_cnt = batch["has_count"].to(device, non_blocking=True)
            target_v = batch["target_value"].to(device, non_blocking=True)
            target_m = batch["target_margin"].to(device, non_blocking=True)
            has_pol = batch["has_policy"].to(device, non_blocking=True)

            composite_returns = (0.65 * target_v + 0.35 * target_m).clamp(-1.0, 1.0)

            optimizer.zero_grad()
            _, logits, v_pred, m_pred, count_logit, q_values, action_repr = model(
                g, o, c, e, opt_mask, val_mask, pad_mask, token_types=tt, return_q_and_repr=True
            )

            valid_combined = opt_mask & val_mask
            awr_loss, q_loss, ptr_loss, acc1, acc3, n_p = compute_awr_and_pointer_losses(
                logits, q_values, action_repr, v_pred, target_idx, target_multi,
                valid_combined, has_pol, composite_returns, model
            )

            # IQL Expectile Value Loss (tau = 0.75) + Class-Balanced Outcome Supervision
            with torch.no_grad():
                q_valid_masked = q_values.masked_fill(~valid_combined, -1.0)
                q_max = q_valid_masked.max(dim=-1).values
            exp_v_loss = expectile_loss(q_max - v_pred, tau=0.75)

            pos_m = target_v >= 0
            neg_m = target_v < 0
            v_sup = (
                0.5 * (mse_loss_fn(v_pred[pos_m], target_v[pos_m]) + mse_loss_fn(v_pred[neg_m], target_v[neg_m]))
                if (pos_m.any() and neg_m.any()) else mse_loss_fn(v_pred, target_v)
            )
            v_loss = 0.5 * v_sup + 0.5 * exp_v_loss
            m_loss = mse_loss_fn(m_pred, target_m)
            cnt_loss = bce_logits_fn(count_logit[has_cnt], target_cnt_ratio[has_cnt]) if has_cnt.any() else torch.tensor(0.0, device=device)

            loss = awr_loss + 0.6 * q_loss + 0.35 * ptr_loss + 0.35 * v_loss + 0.20 * m_loss + 0.25 * cnt_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            t_pol += awr_loss.item()
            t_q += q_loss.item()
            t_ptr += ptr_loss.item()
            t_val += v_loss.item()
            t_cnt += cnt_loss.item()
            c1 += acc1
            c3 += acc3
            tot_pol += n_p
            steps += 1
            pbar.set_postfix({"AWR": f"{awr_loss.item():.3f}", "Q": f"{q_loss.item():.3f}", "Acc1": f"{100.0 * c1 / max(1, tot_pol):.1f}%"})

        scheduler.step()

        # Validation
        model.eval()
        vc1, vc3, v_tot_pol, v_steps = 0, 0, 0, 0
        v_pol_loss, v_q_loss, v_ptr_loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                g = batch["global_vec"].to(device, non_blocking=True)
                o = batch["obj_features"].to(device, non_blocking=True)
                c = batch["card_ids"].to(device, non_blocking=True)
                e = batch["evo_ids"].to(device, non_blocking=True)
                tt = batch["token_types"].to(device, non_blocking=True)
                opt_mask = batch["option_mask"].to(device, non_blocking=True)
                val_mask = batch["valid_action_mask"].to(device, non_blocking=True)
                pad_mask = batch["padding_mask"].to(device, non_blocking=True)

                target_idx = batch["target"].to(device, non_blocking=True)
                target_multi = batch["target_multi_mask"].to(device, non_blocking=True)
                target_v = batch["target_value"].to(device, non_blocking=True)
                target_m = batch["target_margin"].to(device, non_blocking=True)
                has_pol = batch["has_policy"].to(device, non_blocking=True)

                composite_returns = (0.65 * target_v + 0.35 * target_m).clamp(-1.0, 1.0)
                _, logits, v_pred, _, _, q_values, action_repr = model(
                    g, o, c, e, opt_mask, val_mask, pad_mask, token_types=tt, return_q_and_repr=True
                )
                valid_combined = opt_mask & val_mask
                awr_l, q_l, ptr_l, acc1, acc3, n_p = compute_awr_and_pointer_losses(
                    logits, q_values, action_repr, v_pred, target_idx, target_multi,
                    valid_combined, has_pol, composite_returns, model
                )
                v_pol_loss += awr_l.item()
                v_q_loss += q_l.item()
                v_ptr_loss += ptr_l.item()
                vc1 += acc1
                vc3 += acc3
                v_tot_pol += n_p
                v_steps += 1

        tr_acc1 = 100.0 * c1 / max(1, tot_pol)
        va_acc1 = 100.0 * vc1 / max(1, v_tot_pol)
        va_acc3 = 100.0 * vc3 / max(1, v_tot_pol)
        g_set = float(torch.tanh(model.opt_self_gate).item())
        g_q = float(torch.tanh(model.q_logit_gate).item())
        g_ptr = float(torch.tanh(model.pointer_gate).item())

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Acc@1: {tr_acc1:5.2f}% | "
            f"Val Acc@1: {va_acc1:5.2f}% (Acc@3: {va_acc3:5.2f}%) | "
            f"QLoss: {v_q_loss / max(1, v_steps):.4f} | PtrLoss: {v_ptr_loss / max(1, v_steps):.4f} | "
            f"Gates[Set={g_set:+.3f}, Q={g_q:+.3f}, Ptr={g_ptr:+.3f}]"
        )

        history.append({
            "epoch": epoch,
            "train_acc1": tr_acc1,
            "val_acc1": va_acc1,
            "val_acc3": va_acc3,
            "val_q_loss": v_q_loss / max(1, v_steps),
            "val_ptr_loss": v_ptr_loss / max(1, v_steps),
            "gates": {"opt_self": g_set, "q_logit": g_q, "pointer": g_ptr},
        })

        if va_acc1 >= best_val_acc1:
            best_val_acc1 = va_acc1
            torch.save(model.state_dict(), save_path)

    with open("iql_relational_training_log.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n=== Training Complete! Best Val Acc@1: {best_val_acc1:.2f}% -> saved to '{save_path}' ===")


if __name__ == "__main__":
    train_iql_relational()
