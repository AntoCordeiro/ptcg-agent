"""Build a Kaggle submission folder + tar.gz.

python submission/build.py --ckpt experiments/train_full_s0_v/best.pt --deck decks/dragapult.csv \
    --config "{'sims': 100000, 'worlds': 4, 'use_overage': True, 'belief': 'learned'}" --out submission/out_v1
"""
import argparse
import ast
import os
import shutil
import tarfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["__init__.py", "cards.py", "features.py", "model.py", "agents.py", "mcts.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--deck", default="decks/dragapult.csv")
    ap.add_argument("--config", default="{'sims': 100000, 'worlds': 4, 'use_overage': True, 'belief': 'learned'}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = ast.literal_eval(args.config)
    out = os.path.join(ROOT, args.out)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "ptcgnet"))
    for f in FILES:
        shutil.copy(os.path.join(ROOT, "ptcg", f), os.path.join(out, "ptcgnet", f))
    shutil.copytree(os.path.join(ROOT, "cg"), os.path.join(out, "cg"), ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(os.path.join(ROOT, "decks"), os.path.join(out, "decks"))
    shutil.copy(os.path.join(ROOT, args.deck), os.path.join(out, "deck.csv"))
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    torch.save({"model": ck["model"], "config": ck["config"]}, os.path.join(out, "weights.pt"))
    tpl = open(os.path.join(ROOT, "submission", "main_template.py")).read()
    open(os.path.join(out, "main.py"), "w").write(tpl.replace("__CONFIG__", repr(cfg)))
    tar_path = out.rstrip("/\\") + ".tar.gz"
    with tarfile.open(tar_path, "w:gz") as t:
        for name in sorted(os.listdir(out)):
            t.add(os.path.join(out, name), arcname=name)
    print("built", out, "->", tar_path, os.path.getsize(tar_path) / 1e6, "MB")


if __name__ == "__main__":
    main()
