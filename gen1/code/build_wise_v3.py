# build_wise_v3.py
"""
Weight-Space Ensemble Fine-Tuning (WiSE-FT v3) over:
- theta_iql_v3      (`ptcg_iql_v3.pt`: Approaches A + B + C trained on the new 1300-1346+ Elo replays)
- theta_selfplay_v3 (`ptcg_selfplay_v3.pt`: Online Self-Play League PPO across 1340+ Elo championship decks)
- theta_iql_rel_v2  (`ptcg_iql_relational.pt`: Approaches A + C v2 checkpoint)

Evaluates candidate interpolations across both the Ogerpon/Hydrapple championship deck and the
Dragapult ex / Grimmsnarl ex top-leaderboard matchups, saving the #1 weight-space ensemble to `ptcg_wise_v3.pt`.
"""

import json
import os
import sys
import torch

sys.path.insert(0, os.path.abspath("."))
SUBMISSION_DIR = os.path.abspath(
    "the-pokemon-company-ptcg-ai-battle-challenge-playground/sample_submission/sample_submission/sample_submission"
)
if SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, SUBMISSION_DIR)

from cg import game
from model_dual_cross import PTCGDualCrossNet
from ptcg_agents import NNAgent, load_nn_agent


def load_deck(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [int(l.strip()) for l in f if l.strip()]


def play_series(ag_a, ag_b, deck_a, deck_b, n_games=16) -> float:
    wins = 0.0
    for g in range(n_games):
        seat_a = g % 2
        if seat_a == 0:
            obs, _ = game.battle_start(deck_a, deck_b)
            ag_a.reset(deck_a)
            ag_b.reset(deck_b)
            seats = {0: ag_a, 1: ag_b}
        else:
            obs, _ = game.battle_start(deck_b, deck_a)
            ag_b.reset(deck_b)
            ag_a.reset(deck_a)
            seats = {0: ag_b, 1: ag_a}

        steps = 0
        while steps < 800:
            cur = obs.get("current") or {}
            if cur.get("result", -1) != -1:
                break
            s = cur.get("yourIndex", 0)
            act = seats[s].act(obs)
            obs = game.battle_select(act)
            steps += 1
        res = (obs.get("current") or {}).get("result", -1)
        game.battle_finish()
        if res == seat_a:
            wins += 1.0
        elif res == -1:
            wins += 0.5
    return wins / n_games


def blend_state_dicts(sd_a, sd_b, sd_c, w_a: float, w_b: float, w_c: float):
    out = {}
    for k in sd_a.keys():
        va = sd_a[k]
        vb = sd_b.get(k, va)
        vc = sd_c.get(k, va)
        if torch.is_floating_point(va) and va.shape == vb.shape == vc.shape:
            out[k] = (w_a * va.float() + w_b * vb.float() + w_c * vc.float()).to(va.dtype)
        else:
            out[k] = va
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd_iql3 = torch.load("ptcg_iql_v3.pt", map_location=device, weights_only=True)
    sd_sp3 = torch.load("ptcg_selfplay_v3.pt", map_location=device, weights_only=True) if os.path.exists("ptcg_selfplay_v3.pt") else sd_iql3
    sd_iql2 = torch.load("ptcg_iql_relational.pt", map_location=device, weights_only=True)

    d_champ = load_deck("championship_decks/best_championship_deck.csv")
    d_grimm = load_deck("deck_dragapult_evolved.csv")

    opp_ref = load_nn_agent("iql_rel_ref", "ptcg_iql_relational.pt", device=device)

    candidates = [
        ("iql_v3_pure",          1.00, 0.00, 0.00),
        ("sp_v3_pure",           0.00, 1.00, 0.00),
        ("wise_60iql3_25sp_15v2", 0.60, 0.25, 0.15),
        ("wise_50iql3_30sp_20v2", 0.50, 0.30, 0.20),
        ("wise_45iql3_20sp_35v2", 0.45, 0.20, 0.35),
    ]

    best_wr = -1.0
    best_sd = None
    best_tag = ""
    results = []

    print("=== Evaluating WiSE-FT v3 Weight-Space Candidates (32 games each vs iql_rel_v2) ===")
    for tag, w_iql3, w_sp3, w_iql2 in candidates:
        sd = blend_state_dicts(sd_iql3, sd_sp3, sd_iql2, w_iql3, w_sp3, w_iql2)
        m = PTCGDualCrossNet().to(device).eval()
        m.load_state_dict(sd, strict=False)
        ag = NNAgent(m, tag, device=device)

        wr_champ = play_series(ag, opp_ref, d_champ, d_champ, n_games=16)
        wr_grimm = play_series(ag, opp_ref, d_grimm, d_grimm, n_games=16)
        avg_wr = 0.5 * (wr_champ + wr_grimm)
        print(f"  {tag:<24} ({w_iql3:.2f}, {w_sp3:.2f}, {w_iql2:.2f}) -> Champ Mirror: {wr_champ*100:5.1f}% | Grimm Mirror: {wr_grimm*100:5.1f}% | Combined: {avg_wr*100:5.1f}%")
        results.append({
            "tag": tag,
            "weights": [w_iql3, w_sp3, w_iql2],
            "wr_champ_pct": round(wr_champ * 100.0, 2),
            "wr_grimm_pct": round(wr_grimm * 100.0, 2),
            "combined_wr_pct": round(avg_wr * 100.0, 2),
        })
        if avg_wr > best_wr:
            best_wr = avg_wr
            best_sd = sd
            best_tag = tag

    torch.save(best_sd, "ptcg_wise_v3.pt")
    with open("results/wise_v3_sweep.json", "w", encoding="utf-8") as f:
        json.dump({"winner": best_tag, "best_wr_pct": round(best_wr * 100.0, 2), "candidates": results}, f, indent=2)
    print(f"\nSaved #1 WiSE-FT v3 ensemble ('{best_tag}', {best_wr*100:.1f}% Combined WR vs iql_rel_v2) to 'ptcg_wise_v3.pt'!")


if __name__ == "__main__":
    main()
