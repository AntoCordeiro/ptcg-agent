# train_bc_dual_cross.py
"""
Multi-Task Behavioral Cloning Training for Dual-Encoder Cross-Attention Architecture (PTCGDualCrossNet).

Trains on strictly aligned, high-Elo expert replay trajectories with:
1. Multi-Select Set Cross-Entropy Loss: supervises ALL selected options in multi-card prompts
   (e.g. Buddy-Buddy Poffin, Ultra Ball, Grimmsnarl ex Punk Up) alongside single-select decisions.
2. Multi-Select Count Ratio Supervision: trains `turn_horizon_head` (as a sigmoid count-ratio head)
   to predict the exact fraction of [minCount, maxCount] chosen by top players.
3. Class-Balanced Value & Prize-Margin Critic Supervision: weights positive (winner) and negative (loser)
   trajectories equally so V(s) in [-1, 1] and Delta P(s) have zero bias.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from model_dual_cross import PTCGDualCrossNet


class PTCGPartitionDataset(Dataset):
    def __init__(self, partition_data: dict):
        self.data = partition_data

    def __len__(self):
        return len(self.data["target_indices"])

    def __getitem__(self, idx):
        item = {
            "global_vec": self.data["global_vec"][idx].float(),
            "obj_features": self.data["obj_features"][idx].float(),
            "card_ids": self.data["card_ids"][idx].long(),
            "evo_ids": self.data["evo_ids"][idx].long(),
            "option_mask": self.data["option_mask"][idx].bool(),
            "valid_action_mask": self.data["valid_action_mask"][idx].bool(),
            "padding_mask": self.data["padding_mask"][idx].bool(),
            "target": self.data["target_indices"][idx].long(),
        }
        if "token_types" in self.data:
            item["token_types"] = self.data["token_types"][idx].long()
        else:
            item["token_types"] = torch.zeros_like(item["card_ids"])

        if "values" in self.data:
            item["target_value"] = self.data["values"][idx].float()
        else:
            item["target_value"] = torch.tensor(1.0, dtype=torch.float32)

        if "prize_margins" in self.data:
            item["target_margin"] = self.data["prize_margins"][idx].float()
        else:
            item["target_margin"] = item["target_value"]

        if "has_policy_target" in self.data:
            item["has_policy"] = self.data["has_policy_target"][idx].bool()
        else:
            item["has_policy"] = torch.tensor(True, dtype=torch.bool)

        if "target_multi_mask" in self.data:
            item["target_multi_mask"] = self.data["target_multi_mask"][idx].bool()
        else:
            mm = torch.zeros_like(item["option_mask"])
            t_idx = item["target"].item()
            if 0 <= t_idx < len(mm):
                mm[t_idx] = True
            item["target_multi_mask"] = mm

        if "target_count_ratios" in self.data:
            item["target_count_ratio"] = self.data["target_count_ratios"][idx].float()
        else:
            item["target_count_ratio"] = torch.tensor(1.0, dtype=torch.float32)

        if "has_count_target" in self.data:
            item["has_count"] = self.data["has_count_target"][idx].bool()
        else:
            item["has_count"] = torch.tensor(False, dtype=torch.bool)

        return item


def compute_topk_correct(scores: torch.Tensor, targets: torch.Tensor, k: int = 3) -> int:
    """Computes number of instances where target is within top-k highest scores."""
    if scores.numel() == 0 or targets.numel() == 0:
        return 0
    _, topk_indices = torch.topk(scores, min(k, scores.size(-1)), dim=-1)
    targets_exp = targets.unsqueeze(-1).expand_as(topk_indices)
    return (topk_indices == targets_exp).any(dim=-1).sum().item()


def compute_multi_select_policy_loss(scores: torch.Tensor, multi_mask: torch.Tensor, single_targets: torch.Tensor) -> torch.Tensor:
    """
    Computes soft cross-entropy over all chosen options in `multi_mask`.
    Falls back to standard cross-entropy if a row has no bits set in `multi_mask`.
    """
    log_probs = F.log_softmax(scores, dim=-1)
    mask_f = multi_mask.float()
    counts = mask_f.sum(dim=-1, keepdim=True)
    has_multi = (counts.squeeze(-1) > 0)

    if has_multi.all():
        target_dist = mask_f / counts.clamp(min=1.0)
        return -(target_dist * log_probs).sum(dim=-1).mean()
    else:
        return F.cross_entropy(scores, single_targets)


def compute_balanced_mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes class-balanced MSE so positive (winner) and negative (loser) states have equal total weight."""
    pos_mask = (targets >= 0)
    neg_mask = (targets < 0)
    n_pos = pos_mask.sum().item()
    n_neg = neg_mask.sum().item()
    if n_pos > 0 and n_neg > 0:
        loss_pos = F.mse_loss(preds[pos_mask], targets[pos_mask])
        loss_neg = F.mse_loss(preds[neg_mask], targets[neg_mask])
        return 0.5 * (loss_pos + loss_neg)
    return F.mse_loss(preds, targets)


def train_bc_dual_cross(
    data_path: str = "parsed_replays_compact.pt",
    epochs: int = 18,
    batch_size: int = 256,
    lr: float = 8e-4,
    save_path: str = "ptcg_dual_cross_bc.pt",
    log_path: str = "bc_dual_cross_training_log.json"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading replay dataset from '{data_path}'...")
    raw_data = torch.load(data_path, weights_only=False)

    if "train" in raw_data and "val" in raw_data:
        train_ds = PTCGPartitionDataset(raw_data["train"])
        val_ds = PTCGPartitionDataset(raw_data["val"])
    else:
        raise ValueError("Dataset does not contain 'train' and 'val' partitions.")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False
    )

    model = PTCGDualCrossNet().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Initialized PTCGDualCrossNet | Trainable params: {total_params:,} | Frozen text params: {frozen_params:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"Training on {len(train_ds)} steps, validating on {len(val_ds)} steps on {device}...")

    metrics_history = []
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_val_loss, train_correct, train_top3_correct = 0.0, 0.0, 0, 0
        train_total, train_pol_total = 0, 0

        for b in train_loader:
            g = b["global_vec"].to(device, non_blocking=True)
            o = b["obj_features"].to(device, non_blocking=True)
            c = b["card_ids"].to(device, non_blocking=True)
            e = b["evo_ids"].to(device, non_blocking=True)
            tt = b["token_types"].to(device, non_blocking=True)
            opt_mask = b["option_mask"].to(device, non_blocking=True)
            val_mask = b["valid_action_mask"].to(device, non_blocking=True)
            pad_mask = b["padding_mask"].to(device, non_blocking=True)
            targets = b["target"].to(device, non_blocking=True)
            multi_mask = b["target_multi_mask"].to(device, non_blocking=True)
            target_vals = b["target_value"].to(device, non_blocking=True)
            target_margins = b["target_margin"].to(device, non_blocking=True)
            target_cnt_ratios = b["target_count_ratio"].to(device, non_blocking=True)
            has_pol = b["has_policy"].to(device, non_blocking=True)
            has_cnt = b["has_count"].to(device, non_blocking=True)

            optimizer.zero_grad()
            _, scores, val_preds, margin_preds, count_logits = model(
                g, o, c, e, opt_mask, val_mask, pad_mask, token_types=tt, return_aux=True
            )

            # 1. Class-Balanced Value & Prize-Margin Loss over ALL states
            val_loss = compute_balanced_mse(val_preds.squeeze(-1), target_vals)
            margin_loss = compute_balanced_mse(margin_preds.squeeze(-1), target_margins)

            # 2. Multi-Select Policy Loss over non-forced winner decisions
            if has_pol.any():
                policy_loss = compute_multi_select_policy_loss(
                    scores[has_pol], multi_mask[has_pol], targets[has_pol]
                )
                preds = scores[has_pol].argmax(dim=-1)
                # Count as top-1 correct if argmax is ANY of the valid chosen options in multi_mask!
                pol_multi = multi_mask[has_pol]
                chosen_hit = pol_multi.gather(1, preds.unsqueeze(1)).squeeze(1)
                train_correct += (chosen_hit | (preds == targets[has_pol])).sum().item()
                train_top3_correct += compute_topk_correct(scores[has_pol], targets[has_pol], k=3)
                train_pol_total += has_pol.sum().item()
            else:
                policy_loss = torch.tensor(0.0, device=device)

            # 3. Multi-Select Count Ratio Loss
            if has_cnt.any():
                count_loss = F.binary_cross_entropy_with_logits(
                    count_logits.squeeze(-1)[has_cnt], target_cnt_ratios[has_cnt]
                )
            else:
                count_loss = torch.tensor(0.0, device=device)

            loss = policy_loss + 0.4 * val_loss + 0.2 * margin_loss + 0.2 * count_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = g.size(0)
            train_loss += policy_loss.item() * bs
            train_val_loss += val_loss.item() * bs
            train_total += bs

        scheduler.step()

        # Validation
        model.eval()
        val_loss, val_v_loss, val_correct, val_top3_correct = 0.0, 0.0, 0, 0
        val_total, val_pol_total = 0, 0

        with torch.no_grad():
            for b in val_loader:
                g = b["global_vec"].to(device)
                o = b["obj_features"].to(device)
                c = b["card_ids"].to(device)
                e = b["evo_ids"].to(device)
                tt = b["token_types"].to(device)
                opt_mask = b["option_mask"].to(device)
                val_mask = b["valid_action_mask"].to(device)
                pad_mask = b["padding_mask"].to(device)
                targets = b["target"].to(device)
                multi_mask = b["target_multi_mask"].to(device)
                target_vals = b["target_value"].to(device)
                has_pol = b["has_policy"].to(device)

                _, scores, val_preds = model(
                    g, o, c, e, opt_mask, val_mask, pad_mask, token_types=tt
                )

                v_loss = compute_balanced_mse(val_preds.squeeze(-1), target_vals)

                if has_pol.any():
                    policy_loss = compute_multi_select_policy_loss(
                        scores[has_pol], multi_mask[has_pol], targets[has_pol]
                    )
                    preds = scores[has_pol].argmax(dim=-1)
                    pol_multi = multi_mask[has_pol]
                    chosen_hit = pol_multi.gather(1, preds.unsqueeze(1)).squeeze(1)
                    val_correct += (chosen_hit | (preds == targets[has_pol])).sum().item()
                    val_top3_correct += compute_topk_correct(scores[has_pol], targets[has_pol], k=3)
                    val_pol_total += has_pol.sum().item()
                else:
                    policy_loss = torch.tensor(0.0, device=device)

                bs = g.size(0)
                val_loss += policy_loss.item() * bs
                val_v_loss += v_loss.item() * bs
                val_total += bs

        train_acc1 = 100.0 * train_correct / max(train_pol_total, 1)
        train_acc3 = 100.0 * train_top3_correct / max(train_pol_total, 1)
        val_acc1 = 100.0 * val_correct / max(val_pol_total, 1)
        val_acc3 = 100.0 * val_top3_correct / max(val_pol_total, 1)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Policy Loss: {train_loss / train_total:.4f} | Val MSE: {train_val_loss / train_total:.4f} | "
            f"Val Policy Loss: {val_loss / val_total:.4f} | Val MSE: {val_v_loss / val_total:.4f} | "
            f"Train Acc@1: {train_acc1:.2f}% | Val Acc@1: {val_acc1:.2f}% | Val Acc@3: {val_acc3:.2f}%"
        )

        metrics_history.append({
            "epoch": epoch,
            "train_loss": train_loss / train_total,
            "train_acc1": train_acc1,
            "train_acc3": train_acc3,
            "val_loss": val_loss / val_total,
            "val_acc1": val_acc1,
            "val_acc3": val_acc3
        })

        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best Dual-Cross BC checkpoint to {save_path} (Val Acc@1: {val_acc1:.2f}%)")

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"Saved Dual-Cross BC training log to {log_path}")


if __name__ == "__main__":
    train_bc_dual_cross(epochs=18, batch_size=256, lr=8e-4)
