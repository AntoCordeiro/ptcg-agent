# create_kaggle_submission.py
"""
Builds and verifies the official Kaggle submission package:
- Combines Confidence-Gated Full-Turn Lookahead Search (`SearchAgent` v2.1)
  with the Clean-Anchored Phase 4 League PPO Model (`ptcg_ppo_phase4_model.pt`).
- Uses `choose_multi_select_count` (combining learned `count_head` with exact PTCG multi-select rules
  for Grimmsnarl ex's Punk Up, Raging Bolt ex's Bellowing Thunder, Buddy-Buddy Poffin, etc.).
- Bundles the metagame-dominant Marnie's Grimmsnarl ex / Munkidori 60-card deck (`deck_dragapult_evolved.csv`).
- Includes pure CPU/GPU hybrid execution with failsafe policy fallback.
"""

import os
import sys
import shutil
import tarfile

STAGING_DIR = os.path.abspath("kaggle_submission")
ARCHIVE_NAME = os.path.abspath("submission.tar.gz")

if os.path.exists(STAGING_DIR):
    shutil.rmtree(STAGING_DIR)
os.makedirs(STAGING_DIR, exist_ok=True)

MAIN_PY = '''# main.py - Kaggle Competition Agent Entry Point
import os
import sys
import torch
import numpy as np

# Container paths: Kaggle runs agents from /kaggle_simulations/agent/
KAGGLE_DIR = "/kaggle_simulations/agent"
if os.path.exists(KAGGLE_DIR):
    BASE_DIR = KAGGLE_DIR
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure BASE_DIR is first in sys.path
while BASE_DIR in sys.path:
    sys.path.remove(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from model_dual_cross import PTCGDualCrossNet
from feature_parser import parse_game_state_to_tensors
from card_tracker import CardTracker
from ptcg_agents import choose_multi_select_count, count_bounds, apply_tactical_prior_shaping
from search_agent import SearchAgent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join(BASE_DIR, "weights.pt")
DECK_PATH = os.path.join(BASE_DIR, "deck.csv")

model = None
search_agent = None
live_tracker = None
cached_deck = None


def init_agent():
    global model, search_agent
    if model is None:
        model = PTCGDualCrossNet().to(DEVICE).eval()
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True), strict=False)
        model.eval()
    if search_agent is None:
        try:
            search_agent = SearchAgent(
                model_path=MODEL_PATH,
                device=DEVICE,
                num_determinizations=2,
                c_puct=0.45,
                search_all_main_turns=True,
                name="search_kaggle"
            )
        except Exception:
            search_agent = None


def read_deck_csv() -> list:
    global cached_deck
    if cached_deck is not None:
        return list(cached_deck)
    deck_file = DECK_PATH
    if not os.path.exists(deck_file):
        deck_file = os.path.join(BASE_DIR, "deck.csv")
    with open(deck_file, "r", encoding="utf-8") as f:
        cached_deck = [int(line.strip()) for line in f if line.strip()]
    return list(cached_deck)


def direct_policy_act(obs: dict, request: dict) -> list:
    """Failsafe direct policy inference with learned + domain-aware multi-select count and pointer synergy."""
    global model, live_tracker
    options = request.get("option", [])
    if not options:
        return [0]
    mn, mx = count_bounds(request)
    if len(options) == 1:
        return [0] if mn >= 1 else [0]

    ctx = request.get("context", 0) or 0
    if ctx == 42:  # MULLIGAN: always draw extra cards
        for idx, opt in enumerate(options):
            if opt.get("type") == 1:
                return [idx]
        return [0]

    parsed = parse_game_state_to_tensors(obs, request, tracker=live_tracker)

    g = torch.from_numpy(parsed["global_vec"]).unsqueeze(0).to(DEVICE)
    o = torch.from_numpy(parsed["obj_features"]).unsqueeze(0).to(DEVICE)
    c = torch.from_numpy(parsed["card_ids"]).unsqueeze(0).to(DEVICE)
    e = torch.from_numpy(parsed["evo_ids"]).unsqueeze(0).to(DEVICE)
    tt = torch.from_numpy(parsed["token_types"]).unsqueeze(0).to(DEVICE)
    opt_mask = torch.from_numpy(parsed["option_mask"]).unsqueeze(0).to(DEVICE)
    val_mask = torch.from_numpy(parsed["valid_action_mask"]).unsqueeze(0).to(DEVICE)
    pad_mask = torch.from_numpy(parsed["padding_mask"]).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        _, scores, _, _, count_logits, _, action_repr = model(
            g, o, c, e, opt_mask, val_mask, pad_mask, token_types=tt, return_q_and_repr=True
        )

    opt_indices = np.where(parsed["option_mask"])[0]
    opt_scores = scores[0, opt_indices].clone()
    opt_r = action_repr[0, opt_indices]
    opt_cids = torch.from_numpy(parsed["card_ids"][opt_indices]).to(DEVICE)
    c_logit = float(count_logits[0].item())

    opt_scores = apply_tactical_prior_shaping(obs, request, opt_scores)

    k = choose_multi_select_count(obs, request, opt_scores, count_logit=c_logit, mode="auto")
    k = min(k, len(opt_indices))
    if k <= 0:
        return []

    if k > 1 and hasattr(model, "select_multi_options_autoregressive"):
        selected = model.select_multi_options_autoregressive(opt_scores, opt_r, opt_cids, k, ctx=ctx)
    else:
        order = torch.argsort(opt_scores, descending=True).tolist()
        selected = [int(i) for i in order[:k]]
    return selected if len(selected) >= mn else [0]


def agent(obs_dict: dict) -> list:
    """Kaggle submission entry point."""
    global live_tracker, search_agent
    init_agent()

    request = obs_dict.get("select")

    # Step 0: Deck submission (Kaggle environment calls agent with no select request or maxCount == 60)
    if request is None or request.get("maxCount") == 60:
        deck = read_deck_csv()
        live_tracker = CardTracker(deck)
        if search_agent is not None:
            search_agent.reset(deck)
        return deck

    # Multi-game episode start (Game 2 / Game 3 coin flip choice ctx == 41): reset trackers cleanly
    if request.get("context") == 41:
        deck = read_deck_csv()
        live_tracker = CardTracker(deck)
        if search_agent is not None:
            search_agent.reset(deck)

    options = request.get("option", [])
    if not options:
        return [0]

    # Delegate all gameplay decisions to SearchAgent v4
    if search_agent is not None:
        try:
            if search_agent.tracker is None:
                search_agent.reset(read_deck_csv())
            return search_agent.act(obs_dict)
        except Exception:
            pass

    if live_tracker is None:
        live_tracker = CardTracker(read_deck_csv())
    return direct_policy_act(obs_dict, request)
'''

with open(os.path.join(STAGING_DIR, "main.py"), "w", encoding="utf-8") as f:
    f.write(MAIN_PY)
print("Created kaggle_submission/main.py")

best_ckpt = (
    "ptcg_wise_v3.pt" if os.path.exists("ptcg_wise_v3.pt")
    else ("ptcg_iql_v3.pt" if os.path.exists("ptcg_iql_v3.pt") else "ptcg_iql_relational.pt")
)

# 2. Copy source files
files_to_copy = {
    "championship_decks/best_championship_deck.csv": "deck.csv",
    best_ckpt: "weights.pt",
    "model_dual_cross.py": "model_dual_cross.py",
    "features_spec.py": "features_spec.py",
    "feature_parser.py": "feature_parser.py",
    "card_static_table.py": "card_static_table.py",
    "card_static_table.npy": "card_static_table.npy",
    "card_text_embeddings_384.npy": "card_text_embeddings_384.npy",
    "card_text_embeddings.npy": "card_text_embeddings.npy",
    "card_text_encoder.py": "card_text_encoder.py",
    "card_tracker.py": "card_tracker.py",
    "ptcg_agents.py": "ptcg_agents.py",
    "search_agent.py": "search_agent.py",
    "archetypes.json": "archetypes.json",
}

for src, dst in files_to_copy.items():
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing required file: {src}")
    shutil.copyfile(src, os.path.join(STAGING_DIR, dst))
    print(f"Copied {src} -> {dst}")

# 3. Copy cg engine folder
src_cg = os.path.abspath("the-pokemon-company-ptcg-ai-battle-challenge-playground/sample_submission/sample_submission/sample_submission/cg")
dst_cg = os.path.join(STAGING_DIR, "cg")
if os.path.exists(dst_cg):
    shutil.rmtree(dst_cg)
shutil.copytree(src_cg, dst_cg)
# Clean up any cached pyc files
for root, dirs, files in os.walk(dst_cg):
    for f in files:
        if f.endswith(".pyc"):
            os.remove(os.path.join(root, f))
    for d in list(dirs):
        if d == "__pycache__":
            shutil.rmtree(os.path.join(root, d))
print("Copied clean cg/ engine directory (including libcg.so, libcg-arm64.so, cg.dll)")

# 4. Create submission.tar.gz
if os.path.exists(ARCHIVE_NAME):
    os.remove(ARCHIVE_NAME)

with tarfile.open(ARCHIVE_NAME, "w:gz") as tar:
    for item in os.listdir(STAGING_DIR):
        full_path = os.path.join(STAGING_DIR, item)
        tar.add(full_path, arcname=item)

size_mb = os.path.getsize(ARCHIVE_NAME) / (1024 * 1024)
print(f"\nSuccessfully generated archive: '{ARCHIVE_NAME}' ({size_mb:.2f} MB)")

# 5. Verify archive contents
print("\nArchive contents:")
with tarfile.open(ARCHIVE_NAME, "r:gz") as tar:
    for member in tar.getmembers():
        print(f"  {member.name:<35} {member.size / 1024:>8.1f} KB")

print("\nPackaging complete!")
