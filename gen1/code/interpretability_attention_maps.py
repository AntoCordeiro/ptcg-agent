# interpretability_attention_maps.py
"""
Neural Interpretability & Cross-Attention Attribution (Pillar 1 / XAI).

Extracts and visualizes the asymmetric cross-attention weight matrices:
    A = Softmax(Q_options * K_state^T / sqrt(d_k))
from PTCGDualCrossNet across representative competitive game decisions:
1. Combat Action (Attack selection attending to enemy Active HP and Weakness)
2. Resource Setup (Energy attachment attending to active/bench energy requirements)
3. Disruption / Search (Supporter/Item card search attending to target board entities)

Generates publication-quality attention attribution heatmaps (300 DPI):
    results_plots/figure7_attention_attribution_heatmaps.png
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

SUBMISSION_DIR = os.path.abspath(
    "the-pokemon-company-ptcg-ai-battle-challenge-playground/sample_submission/sample_submission/sample_submission"
)
if SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, SUBMISSION_DIR)

from cg import game
from model_dual_cross import PTCGDualCrossNet
from card_tracker import CardTracker, get_card_name, get_card_type
from feature_parser import parse_game_state_to_tensors
from match_sim import select_action


def load_deck(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def extract_entity_labels(obs: dict, parsed: dict) -> list[str]:
    """
    Constructs readable entity labels corresponding to each position in state_seq:
    Pos 0: Global State Token
    Pos 1..S_obj: Game entities (Active, Bench, Hand, Stadium)
    """
    labels = ["GLOBAL (Turn/Prizes/Rules)"]
    current = obs.get("current", {}) or {}
    my_idx = current.get("yourIndex", 0)
    players = current.get("players", []) or []
    my_state = players[my_idx] if len(players) > my_idx else {}
    opp_state = players[1 - my_idx] if len(players) > (1 - my_idx) else {}

    # My Active
    active_my = my_state.get("active") or []
    if active_my:
        p = active_my[0] if isinstance(active_my, list) else active_my
        cid = p.get("cardId", p.get("id", 0)) if isinstance(p, dict) else 0
        hp = p.get("hp", 0) if isinstance(p, dict) else 0
        labels.append(f"My Active: {get_card_name(cid)[:12]} ({hp}HP)")
    else:
        labels.append("My Active: (Empty)")

    # My Bench (up to 5)
    bench_my = my_state.get("bench") or []
    for b_idx, b_item in enumerate(bench_my[:5]):
        p = b_item[0] if isinstance(b_item, list) and b_item else b_item
        cid = p.get("cardId", p.get("id", 0)) if isinstance(p, dict) else 0
        hp = p.get("hp", 0) if isinstance(p, dict) else 0
        labels.append(f"My Bench {b_idx+1}: {get_card_name(cid)[:10]} ({hp}HP)")

    # Opponent Active
    active_opp = opp_state.get("active") or []
    if active_opp:
        p = active_opp[0] if isinstance(active_opp, list) else active_opp
        cid = p.get("cardId", p.get("id", 0)) if isinstance(p, dict) else 0
        hp = p.get("hp", 0) if isinstance(p, dict) else 0
        labels.append(f"Opp Active: {get_card_name(cid)[:12]} ({hp}HP)")
    else:
        labels.append("Opp Active: (Empty)")

    # Opponent Bench (up to 5)
    bench_opp = opp_state.get("bench") or []
    for b_idx, b_item in enumerate(bench_opp[:5]):
        p = b_item[0] if isinstance(b_item, list) and b_item else b_item
        cid = p.get("cardId", p.get("id", 0)) if isinstance(p, dict) else 0
        hp = p.get("hp", 0) if isinstance(p, dict) else 0
        labels.append(f"Opp Bench {b_idx+1}: {get_card_name(cid)[:10]} ({hp}HP)")

    # Hand cards (up to 7)
    hand_my = my_state.get("hand") or []
    for h_idx, c in enumerate(hand_my[:7]):
        cid = c.get("cardId", c.get("id", 0)) if isinstance(c, dict) else c
        labels.append(f"Hand: {get_card_name(cid)[:12]}")

    # Stadium
    stadium = current.get("stadium") or {}
    if stadium:
        stadium_dict = stadium[0] if isinstance(stadium, list) and stadium else stadium
        if isinstance(stadium_dict, dict):
            cid = stadium_dict.get("cardId", stadium_dict.get("id", 0))
            labels.append(f"Stadium: {get_card_name(cid)[:12]}")
        else:
            labels.append("Stadium: (Active)")
    else:
        labels.append("Stadium: (None)")

    # Deck Belief Token
    labels.append("Deck Belief (Prize Prob)")

    # Pad or truncate to match total active state tokens
    return labels


def record_case_studies(model, device: torch.device):
    deck0 = load_deck("deck_gardevoir.csv")
    deck1 = load_deck("deck_dragapult.csv")

    obs, start_data = game.battle_start(deck0, deck1)
    if obs is None:
        raise RuntimeError("Failed to start battle for interpretability extraction")

    tracker_p0 = CardTracker(deck0)
    tracker_p1 = CardTracker(deck1)

    case_studies = []
    steps = 0
    max_steps = 300

    try:
        while steps < max_steps and len(case_studies) < 3:
            current = obs.get("current", {})
            if current.get("result", -1) != -1:
                break

            seat = current.get("yourIndex", 0)
            active_tracker = tracker_p0 if seat == 0 else tracker_p1

            request = obs.get("select") or {}
            options = request.get("option", [])
            req_type = request.get("type", 0)

            if options and len(options) > 1 and seat == 0:
                parsed = parse_game_state_to_tensors(obs, request, tracker=active_tracker)
                g = torch.from_numpy(parsed["global_vec"]).unsqueeze(0).to(device)
                o = torch.from_numpy(parsed["obj_features"]).unsqueeze(0).to(device)
                c = torch.from_numpy(parsed["card_ids"]).unsqueeze(0).to(device)
                e = torch.from_numpy(parsed["evo_ids"]).unsqueeze(0).to(device)
                tt = torch.from_numpy(parsed["token_types"]).unsqueeze(0).to(device)
                opt_mask = torch.from_numpy(parsed["option_mask"]).unsqueeze(0).to(device)
                val_mask = torch.from_numpy(parsed["valid_action_mask"]).unsqueeze(0).to(device)
                pad_mask = torch.from_numpy(parsed["padding_mask"]).unsqueeze(0).to(device)

                with torch.no_grad():
                    probs, scores, val, attn = model(
                        g, o, c, e, opt_mask, val_mask, pad_mask,
                        token_types=tt, return_attention=True
                    )
                    opt_indices = np.where(parsed["option_mask"])[0]
                    chosen_opt_idx = int(torch.argmax(scores[0, opt_indices]).item())
                    chosen_global_token = opt_indices[chosen_opt_idx]

                # Identify decision category
                opt_obj = options[chosen_opt_idx] if chosen_opt_idx < len(options) else {}
                opt_type = opt_obj.get("type", -1) if isinstance(opt_obj, dict) else -1

                decision_category = None
                if opt_type == 13 and not any(cs["category"] == "Combat (Attack)" for cs in case_studies):
                    decision_category = "Combat (Attack)"
                elif opt_type in (8, 7) and not any(cs["category"] == "Resource Setup (Attach/Play)" for cs in case_studies):
                    decision_category = "Resource Setup (Attach/Play)"
                elif (req_type in (3, 4, 5, 8, 11) or opt_type in (9, 10)) and not any(cs["category"] == "Search & Disruption" for cs in case_studies):
                    decision_category = "Search & Disruption"

                if decision_category:
                    # Extract attention weights for the chosen action query: shape [num_heads, num_keys]
                    attn_matrix = attn[0, :, chosen_global_token, :].cpu().numpy()
                    entity_labels = extract_entity_labels(obs, parsed)
                    valid_keys_count = min(len(entity_labels), attn_matrix.shape[1])

                    case_studies.append({
                        "category": decision_category,
                        "step": steps,
                        "chosen_action": f"Option {chosen_opt_idx} (Type {opt_type})",
                        "attn_weights": attn_matrix[:, :valid_keys_count],
                        "entity_labels": entity_labels[:valid_keys_count],
                        "win_val": float(val.item())
                    })
                    print(f"Captured interpretability case study: '{decision_category}' at step {steps}")

            # Advance battle
            from agents import NNAgent
            nn_agent = NNAgent(model, "nn", device=device)
            action = nn_agent.act(obs)
            obs = game.battle_select(action)
            steps += 1
    finally:
        game.battle_finish()

    return case_studies


def plot_attention_heatmaps(case_studies: list, output_path: str = "results_plots/figure7_attention_attribution_heatmaps.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    num_cases = len(case_studies)
    if num_cases == 0:
        print("Warning: No case studies recorded, creating synthetic demonstration")
        return

    fig, axes = plt.subplots(num_cases, 1, figsize=(14, 4.2 * num_cases))
    if num_cases == 1:
        axes = [axes]

    for idx, (cs, ax) in enumerate(zip(case_studies, axes)):
        attn_heads = cs["attn_weights"] # [num_heads, valid_keys]
        mean_attn = np.mean(attn_heads, axis=0) # [valid_keys]
        labels = cs["entity_labels"]

        # Ensure label count matches keys
        num_keys = min(len(labels), len(mean_attn))
        mean_attn = mean_attn[:num_keys]
        labels = labels[:num_keys]

        # Normalize for visualization
        if mean_attn.sum() > 0:
            mean_attn = mean_attn / mean_attn.sum()

        x = np.arange(num_keys)
        # Highlight top 3 attended entities
        top3_indices = np.argsort(-mean_attn)[:3]
        colors = ["#e74c3c" if i in top3_indices else "#3498db" for i in range(num_keys)]

        bars = ax.bar(x, mean_attn * 100.0, color=colors, edgecolor="black", width=0.6, alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9, fontweight="bold")
        ax.set_ylabel("Attention Weight (%)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Case {idx+1}: {cs['category']} | Action: {cs['chosen_action']} | Critic V(s) = {cs['win_val']:+.2f}",
            fontsize=12, fontweight="bold"
        )
        ax.set_ylim(0, max(mean_attn * 100.0) * 1.25)
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)

        # Annotate top peaks
        for i in top3_indices:
            val = mean_attn[i] * 100.0
            ax.annotate(f"{val:.1f}%",
                        xy=(i, val + 0.5),
                        ha="center", va="bottom", fontsize=9, fontweight="bold", color="#c0392b")

    plt.suptitle("Explainable AI: Neural Cross-Attention Attribution to Board Entities (PTCGDualCrossNet)",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\nSaved publication figure to '{output_path}'.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Neural Interpretability & Cross-Attention Attribution on {device} ---")

    model = PTCGDualCrossNet().to(device)
    ckpt = "ptcg_ppo_phase2_model.pt" if os.path.exists("ptcg_ppo_phase2_model.pt") else "ptcg_capstone_league_model.pt"
    print(f"Loading checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    case_studies = record_case_studies(model, device)
    plot_attention_heatmaps(case_studies)


if __name__ == "__main__":
    main()
