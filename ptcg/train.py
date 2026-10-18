"""Supervised pre-training on replays: policy + value + count + belief heads.

python -m ptcg.train --name full --epochs 12
python -m ptcg.train --name nobelief --no-belief
Outputs experiments/train_<name>/{best.pt,last.pt,log.jsonl,config.json}
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ptcg.model import PTCGNet  # noqa: E402
from ptcg.features import ARRAY_KEYS  # noqa: E402
from ptcg.cards import N_CARD  # noqa: E402

LABEL_KEYS = ["target", "k", "kmin", "kmax", "value", "hand", "hand_ok", "seltype", "my_deck_drag", "turn"]


def load_split(shard_dir, split, max_samples=None):
    files = sorted(glob.glob(os.path.join(shard_dir, f"{split}_*.npz")))
    parts = {k: [] for k in ARRAY_KEYS + LABEL_KEYS + ["pool", "opp_hand_count", "elo", "ep", "player"]}
    n = 0
    for f in files:
        z = np.load(f)
        for k in parts:
            parts[k].append(z[k])
        n += len(z["k"])
        if max_samples and n >= max_samples:
            break
    data = {k: torch.from_numpy(np.concatenate(v)) for k, v in parts.items()}
    return data


def batch_at(data, idx, device):
    return {k: v[idx].to(device, non_blocking=True) for k, v in data.items()}


def hand_multihot(hand, B, device):
    t = torch.zeros(B, N_CARD, device=device)
    t.scatter_(1, hand, 1.0)
    t[:, 0] = 0.0
    return t


def losses(model, b, args):
    out = model(b)
    logits = out["logits"].float()
    tgt = b["target"].float()
    logp = F.log_softmax(logits, -1)
    pol = -(tgt * logp).sum(-1) / tgt.sum(-1).clamp(min=1)
    w = torch.where(b["value"] > 0, 1.0, args.loser_weight).float()
    l_pol = (pol * w).sum() / w.sum()
    l_val = F.mse_loss(out["value"].float(), b["value"].float())
    multi = b["kmax"] > b["kmin"]
    ratio = (b["k"] - b["kmin"]).float() / (b["kmax"] - b["kmin"]).clamp(min=1).float()
    l_cnt = (F.binary_cross_entropy_with_logits(out["count"].float(), ratio, reduction="none") * multi).sum() / multi.sum().clamp(min=1)
    total = l_pol + args.w_value * l_val + args.w_count * l_cnt
    l_bel = torch.zeros((), device=logits.device)
    if "belief" in out:
        ok = b["hand_ok"].bool()
        if ok.any():
            mh = hand_multihot(b["hand"][ok], int(ok.sum()), logits.device)
            l_bel = F.binary_cross_entropy_with_logits(out["belief"][ok].float(), mh, reduction="sum") / ok.sum()
            total = total + args.w_belief * l_bel
    single = tgt.sum(-1) == 1
    top1 = (logits.argmax(-1) == tgt.argmax(-1)) & single
    return total, {"pol": l_pol.item(), "val": l_val.item(), "cnt": l_cnt.item(), "bel": l_bel.item(),
                   "top1": top1.sum().item(), "n_single": single.sum().item()}


@torch.no_grad()
def evaluate(model, data, args, device):
    model.eval()
    n = len(data["k"])
    agg = {}
    nb = 0
    for i in range(0, n, 1024):
        idx = torch.arange(i, min(n, i + 1024))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, m = losses(model, batch_at(data, idx, device), args)
        for k, v in m.items():
            agg[k] = agg.get(k, 0) + v
        nb += 1
    res = {k: agg[k] / nb for k in ("pol", "val", "cnt", "bel")}
    res["top1"] = agg["top1"] / max(1, agg["n_single"])
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="full")
    ap.add_argument("--shards", default="data/shards")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--ff", type=int, default=512)
    ap.add_argument("--no-belief", action="store_true")
    ap.add_argument("--w-value", type=float, default=0.5)
    ap.add_argument("--w-count", type=float, default=0.2)
    ap.add_argument("--w-belief", type=float, default=0.05)
    ap.add_argument("--loser-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--gpu-data", action="store_true", help="keep the training set on the GPU (no CPU bottleneck)")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(ROOT, "experiments", f"train_{args.name}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    t0 = time.time()
    train = load_split(args.shards, "train", args.max_train)
    val = load_split(args.shards, "val")
    print(f"loaded train={len(train['k'])} val={len(val['k'])} in {time.time()-t0:.0f}s", flush=True)
    if args.gpu_data and device.type == "cuda":
        train = {k: v.to(device) for k, v in train.items()}
        print("training data moved to GPU", flush=True)
    cfg = dict(d=args.d, layers=args.layers, heads=args.heads, ff=args.ff, belief=not args.no_belief)
    model = PTCGNet(**cfg).to(device)
    print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))
    n = len(train["k"])
    steps_per_epoch = n // args.bs
    total = steps_per_epoch * args.epochs
    warm = min(1000, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    log = open(os.path.join(out_dir, "log.jsonl"), "w")
    best = 1e9
    step = 0
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=train["k"].device)
        agg = {}
        for bi in range(steps_per_epoch):
            idx = perm[bi * args.bs:(bi + 1) * args.bs]
            b = batch_at(train, idx, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, m = losses(model, b, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            for k, v in m.items():
                agg[k] = agg.get(k, 0) + v
            if bi % 200 == 0:
                print(f"ep {ep} step {bi}/{steps_per_epoch} loss {loss.item():.4f} pol {m['pol']:.3f} val {m['val']:.3f} bel {m['bel']:.3f} {time.time()-t0:.0f}s", flush=True)
        tr = {k: agg[k] / steps_per_epoch for k in ("pol", "val", "cnt", "bel")}
        tr["top1"] = agg["top1"] / max(1, agg["n_single"])
        va = evaluate(model, val, args, device)
        rec = {"epoch": ep, "train": tr, "val": va, "time": time.time() - t0, "lr": sched.get_last_lr()[0]}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print("EPOCH", json.dumps(rec), flush=True)
        ck = {"model": model.state_dict(), "config": cfg, "epoch": ep, "val": va}
        torch.save(ck, os.path.join(out_dir, "last.pt"))
        if va["pol"] < best:
            best = va["pol"]
            torch.save(ck, os.path.join(out_dir, "best.pt"))
    print("DONE best val pol", best, flush=True)


if __name__ == "__main__":
    main()
