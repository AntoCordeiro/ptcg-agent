"""Value fine-tuning on self-play outcomes (keeps the policy anchored on replays).

python -m ptcg.train_value --init experiments/train_full_s0/best.pt --name full_s0_v --epochs 3
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
from ptcg.train import load_split, batch_at, losses, evaluate  # noqa: E402
from ptcg.features import ARRAY_KEYS  # noqa: E402


def load_selfplay(d):
    files = sorted(glob.glob(os.path.join(d, "sp_*.npz")))
    keys = ARRAY_KEYS + ["value", "ep", "turn", "target"]
    parts = {k: [] for k in keys}
    for f in files:
        z = np.load(f)
        for k in keys:
            parts[k].append(z[k])
    data = {k: torch.from_numpy(np.concatenate(v)) for k, v in parts.items()}
    is_val = (data["ep"] % 10) == 0
    tr = {k: v[~is_val] for k, v in data.items()}
    va = {k: v[is_val] for k, v in data.items()}
    return tr, va


@torch.no_grad()
def value_eval(model, data, device):
    model.eval()
    n = len(data["value"]); se = 0.0; acc = 0; m = 0
    for i in range(0, n, 2048):
        idx = torch.arange(i, min(n, i + 2048))
        b = batch_at(data, idx, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            v = model(b)["value"].float()
        y = b["value"].float()
        se += ((v - y) ** 2).sum().item()
        nz = y != 0
        acc += ((v[nz] > 0) == (y[nz] > 0)).sum().item(); m += nz.sum().item()
    model.train()
    return {"mse": se / max(1, n), "acc": acc / max(1, m), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--selfplay", default="data/selfplay")
    ap.add_argument("--shards", default="data/shards")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--w-value", type=float, default=0.25)     # replay value weight
    ap.add_argument("--w-count", type=float, default=0.2)
    ap.add_argument("--w-belief", type=float, default=0.05)
    ap.add_argument("--loser-weight", type=float, default=0.5)
    ap.add_argument("--w-sp", type=float, default=1.0)
    ap.add_argument("--w-sp-pol", type=float, default=0.0, help=">0: expert iteration (imitate search visit distributions)")
    args = ap.parse_args()
    dev = torch.device("cuda")
    out_dir = os.path.join(ROOT, "experiments", f"train_{args.name}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    ck = torch.load(os.path.join(ROOT, args.init), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = PTCGNet(**cfg).to(dev)
    model.load_state_dict(ck["model"])
    t0 = time.time()
    sp_tr, sp_va = load_selfplay(os.path.join(ROOT, args.selfplay))
    rp_tr = load_split(args.shards, "train")
    rp_va = load_split(args.shards, "val")
    print(f"selfplay train {len(sp_tr['value'])} val {len(sp_va['value'])}; replay {len(rp_tr['k'])} ({time.time()-t0:.0f}s)", flush=True)
    base = {"sp_val": value_eval(model, sp_va, dev), "replay_val": evaluate(model, rp_va, args, dev)}
    print("BASE", json.dumps(base), flush=True)
    log = open(os.path.join(out_dir, "log.jsonl"), "w")
    log.write(json.dumps({"epoch": -1, **base}) + "\n")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    n_sp = len(sp_tr["value"]); n_rp = len(rp_tr["k"])
    spe = n_sp // args.bs
    total = spe * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    best = base["sp_val"]["mse"]
    torch.save({"model": model.state_dict(), "config": cfg}, os.path.join(out_dir, "best.pt"))
    for ep in range(args.epochs):
        perm = torch.randperm(n_sp)
        for bi in range(spe):
            sb = batch_at(sp_tr, perm[bi * args.bs:(bi + 1) * args.bs], dev)
            rb = batch_at(rp_tr, torch.randint(0, n_rp, (args.bs,)), dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                l_rp, m = losses(model, rb, args)
                so = model(sb)
                v = so["value"].float()
                l_sp = F.mse_loss(v, sb["value"].float())
                loss = l_rp + args.w_sp * l_sp
                if args.w_sp_pol > 0:
                    lp = F.log_softmax(so["logits"].float(), -1)
                    t = sb["target"].float()
                    l_sp_pol = (-(t * lp).sum(-1) / t.sum(-1).clamp(min=1e-6)).mean()
                    loss = loss + args.w_sp_pol * l_sp_pol
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if bi % 200 == 0:
                print(f"ep {ep} {bi}/{spe} sp_mse {l_sp.item():.3f} pol {m['pol']:.3f} {time.time()-t0:.0f}s", flush=True)
        rec = {"epoch": ep, "sp_val": value_eval(model, sp_va, dev), "replay_val": evaluate(model, rp_va, args, dev)}
        print("EPOCH", json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()
        torch.save({"model": model.state_dict(), "config": cfg}, os.path.join(out_dir, "last.pt"))
        if rec["sp_val"]["mse"] < best:
            best = rec["sp_val"]["mse"]
            torch.save({"model": model.state_dict(), "config": cfg}, os.path.join(out_dir, "best.pt"))
    print("DONE best sp val mse", best, flush=True)


if __name__ == "__main__":
    main()
