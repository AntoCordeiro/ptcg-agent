# package_summission.py
"""
Automated packaging script for Kaggle Submission.
Packages the state-of-the-art Dual-Cross Asymmetric Attention Architecture (PTCGDualCrossNet),
Demonstration-Anchored PPO weights, 384-dimensional dense semantic card text embeddings,
and the Nash-dominant Gardevoir deck archetype into a self-contained submission.tar.gz.
"""

import os
import shutil
import tarfile

STAGING_DIR = "staging_submission"
ARCHIVE_NAME = "submission.tar.gz"

if os.path.exists(STAGING_DIR):
    shutil.rmtree(STAGING_DIR)
os.makedirs(STAGING_DIR, exist_ok=True)

MAIN_PY_CONTENT = """import os
import sys
import torch
import numpy as np

# In Kaggle's evaluation environment, agents run from /kaggle_simulations/agent/
KAGGLE_DIR = "/kaggle_simulations/agent"
if os.path.exists(KAGGLE_DIR):
    BASE_DIR = KAGGLE_DIR
else:
    BASE_DIR = os.getcwd()

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from model_dual_cross import PTCGDualCrossNet
from feature_parser import parse_game_state_to_tensors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model weights
MODEL_PATH = os.path.join(BASE_DIR, "weights.pt")
model = PTCGDualCrossNet(card_text_path=os.path.join(BASE_DIR, "card_text_embeddings_384.npy")).to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

def read_deck_csv() -> list[int]:
    deck_path = os.path.join(BASE_DIR, "deck.csv")
    with open(deck_path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]

def agent(obs_dict: dict) -> list[int]:
    request = obs_dict.get("select")
    
    # Step 0/1: Deck submission
    if request is None or request.get("type") == 9:
        return read_deck_csv()

    options = request.get("option", [])
    if not options:
        return [0]

    max_count = request.get("maxCount", 1) or 1
    min_count = request.get("minCount", 1) or 1

    parsed = parse_game_state_to_tensors(obs_dict, request)

    g = torch.from_numpy(parsed["global_vec"]).unsqueeze(0).to(DEVICE)
    o = torch.from_numpy(parsed["obj_features"]).unsqueeze(0).to(DEVICE)
    c = torch.from_numpy(parsed["card_ids"]).unsqueeze(0).to(DEVICE)
    e = torch.from_numpy(parsed["evo_ids"]).unsqueeze(0).to(DEVICE)
    tt = torch.from_numpy(parsed["token_types"]).unsqueeze(0).to(DEVICE)
    opt_mask = torch.from_numpy(parsed["option_mask"]).unsqueeze(0).to(DEVICE)
    val_mask = torch.from_numpy(parsed["valid_action_mask"]).unsqueeze(0).to(DEVICE)
    pad_mask = torch.from_numpy(parsed["padding_mask"]).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs, _, _ = model(g, o, c, e, opt_mask, val_mask, pad_mask, token_types=tt)
        opt_indices = np.where(parsed["option_mask"])[0]
        token_probs = probs[0, opt_indices].cpu().numpy()
        ranked_choices = np.argsort(-token_probs)

    selected = [int(idx) for idx in ranked_choices[:max_count]]
    return selected if len(selected) >= min_count else [0]
"""

with open(os.path.join(STAGING_DIR, "main.py"), "w", encoding="utf-8") as f:
    f.write(MAIN_PY_CONTENT)
print("Wrote production staging/main.py (PTCGDualCrossNet)")

# 2. Copy dependencies
files_to_copy = {
    "model_dual_cross.py": "model_dual_cross.py",
    "features_spec.py": "features_spec.py",
    "feature_parser.py": "feature_parser.py",
    "card_static_table.py": "card_static_table.py",
    "card_static_table.npy": "card_static_table.npy",
    "card_text_embeddings_384.npy": "card_text_embeddings_384.npy",
    "ptcg_dual_cross_ppo_anchored.pt": "weights.pt",
    "deck_gardevoir.csv": "deck.csv",
}

for src, dst_name in files_to_copy.items():
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing required file: {src}")
    shutil.copyfile(src, os.path.join(STAGING_DIR, dst_name))
    print(f"Copied {src} -> {dst_name}")

# 3. Create submission.tar.gz
with tarfile.open(ARCHIVE_NAME, "w:gz") as tar:
    for item in os.listdir(STAGING_DIR):
        full_path = os.path.join(STAGING_DIR, item)
        tar.add(full_path, arcname=item)

archive_size_mb = os.path.getsize(ARCHIVE_NAME) / (1024 * 1024)
print(f"\nCreated '{ARCHIVE_NAME}' ({archive_size_mb:.2f} MB)")
print("Validation: Archive content listing:")
with tarfile.open(ARCHIVE_NAME, "r:gz") as tar:
    for member in tar.getmembers():
        print(f"  {member.name:<32} {member.size / 1024:>8.1f} KB")

print("\nPackage ready for official submission!")