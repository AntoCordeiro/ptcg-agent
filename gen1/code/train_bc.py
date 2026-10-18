# train_bc.py
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from model import PTCGNet

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
            
        return item

def compute_topk_correct(scores: torch.Tensor, targets: torch.Tensor, k: int = 3) -> int:
    """Computes number of instances where target is within top-k highest scores."""
    _, topk_indices = torch.topk(scores, min(k, scores.size(-1)), dim=-1)
    targets_exp = targets.unsqueeze(-1).expand_as(topk_indices)
    return (topk_indices == targets_exp).any(dim=-1).sum().item()

def train_bc(
    data_path: str = "parsed_replays_compact.pt",
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 8e-4,
    save_path: str = "ptcg_bc_checkpoint.pt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading replay dataset from '{data_path}'...")
    raw_data = torch.load(data_path, weights_only=False)

    if "train" in raw_data and "val" in raw_data:
        train_ds = PTCGPartitionDataset(raw_data["train"])
        val_ds = PTCGPartitionDataset(raw_data["val"])
    else:
        # Fallback for monolithic dataset
        from torch.utils.data import random_split
        full_ds = PTCGPartitionDataset(raw_data)
        v_size = int(len(full_ds) * 0.1)
        t_size = len(full_ds) - v_size
        train_ds, val_ds = random_split(full_ds, [t_size, v_size])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0
    )

    model = PTCGNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    print(f"Training on {len(train_ds)} steps, validating on {len(val_ds)} steps on {device} ({torch.cuda.get_device_name(0)})...")

    metrics_history = []
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_top3_correct, train_total = 0.0, 0, 0, 0

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
            target_vals = b["target_value"].to(device, non_blocking=True)

            optimizer.zero_grad()
            _, scores, val_preds = model(
                g, o, c, e, opt_mask, val_mask, pad_mask, token_types=tt
            )

            policy_loss = ce_loss_fn(scores, targets)
            val_loss = mse_loss_fn(val_preds, target_vals)
            loss = policy_loss + 0.2 * val_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = g.size(0)
            train_loss += policy_loss.item() * bs
            preds = scores.argmax(dim=-1)
            train_correct += (preds == targets).sum().item()
            train_top3_correct += compute_topk_correct(scores, targets, k=3)
            train_total += bs

        scheduler.step()

        # Validation
        model.eval()
        val_loss, val_correct, val_top3_correct, val_total = 0.0, 0, 0, 0
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

                _, scores, _ = model(
                    g, o, c, e, opt_mask, val_mask, pad_mask, token_types=tt
                )
                
                loss_step = ce_loss_fn(scores, targets)
                bs = g.size(0)
                val_loss += loss_step.item() * bs
                preds = scores.argmax(dim=-1)
                val_correct += (preds == targets).sum().item()
                val_top3_correct += compute_topk_correct(scores, targets, k=3)
                val_total += bs

        train_acc1 = (train_correct / train_total) * 100.0
        train_acc3 = (train_top3_correct / train_total) * 100.0
        val_acc1 = (val_correct / val_total) * 100.0
        val_acc3 = (val_top3_correct / val_total) * 100.0
        avg_train_loss = train_loss / train_total
        avg_val_loss = val_loss / val_total
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:02d}/{epochs} | LR: {current_lr:.1e} | "
            f"Train Loss: {avg_train_loss:.3f} | Train Acc@1: {train_acc1:.1f}% Acc@3: {train_acc3:.1f}% | "
            f"Val Loss: {avg_val_loss:.3f} | Val Acc@1: {val_acc1:.1f}% Acc@3: {val_acc3:.1f}%"
        )

        metrics_history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_acc1": train_acc1,
            "train_acc3": train_acc3,
            "val_loss": avg_val_loss,
            "val_acc1": val_acc1,
            "val_acc3": val_acc3,
        })

        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            torch.save(model.state_dict(), save_path)

    with open("bc_training_log.json", "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, indent=2)

    print(f"\nTraining Complete! Best Val Acc@1: {best_val_acc:.2f}%. Saved weights to '{save_path}'.")
    return best_val_acc

if __name__ == "__main__":
    train_bc("parsed_replays_compact.pt", epochs=15, batch_size=256, lr=8e-4)