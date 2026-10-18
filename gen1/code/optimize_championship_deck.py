# optimize_championship_deck.py
"""
Approach E: Top-1200+ Elo Replay Deck Reconstruction & Championship Deck Tournament.

1. Scans the top 250 highest-Elo replays in `data/replays/manifest.csv`.
2. Reconstructs the exact 60-card decks played by the top leaderboard competitors,
   tracking win-rate and peak Elo across the top-Elo replay pool.
3. Identifies the signature archetype of each deck using `card_static_table`.
4. Runs a multi-round seat-balanced tournament under our new `ptcg_iql_relational.pt`
   agent to select the highest-performing 60-card deck for `submission.tar.gz`.
"""

import csv
import json
import os
import sys
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))
SUBMISSION_DIR = os.path.abspath(
    "the-pokemon-company-ptcg-ai-battle-challenge-playground/sample_submission/sample_submission/sample_submission"
)
if SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, SUBMISSION_DIR)

from cg import game
from ptcg_agents import load_nn_agent

ARCHETYPE_SIGNATURES = [
    (661, "Marnie's Grimmsnarl ex"),
    (121, "Dragapult ex"),
    (75, "Alakazam"),
    (142, "Raging Bolt ex"),
    (96, "Teal Mask Ogerpon ex"),
    (150, "Hydrapple ex"),
    (1079, "Rare Candy Evo"),
]


def identify_deck_name(deck: list[int]) -> str:
    counts = defaultdict(int)
    for c in deck:
        counts[int(c)] += 1
    tags = []
    if counts[648] > 0 or counts[647] > 0:
        tags.append("Grimmsnarl_ex")
    if counts[150] > 0:
        tags.append("Hydrapple_ex")
    if counts[85] > 0:
        tags.append("Meganium")
    if counts[661] > 0:
        tags.append("RagingBolt_ex")
    if counts[96] > 0:
        tags.append("Ogerpon_ex")
    if counts[604] > 0 or counts[603] > 0:
        tags.append("Dragapult_ex")
    if counts[247] > 0 or counts[246] > 0:
        tags.append("Alakazam")
    if counts[283] > 0:
        tags.append("MegaLucario_ex")
    if counts[200] > 0 or counts[198] > 0:
        tags.append("WaterBox_Abomasnow")
    if counts[115] > 0:
        tags.append("Crustle")
    if counts[594] > 0 and "Munkidori" not in tags:
        tags.append("Munkidori")
    return "_".join(tags[:2]) if tags else f"TopElo_Deck_{deck[0]}"


def extract_top_elo_decks(manifest_path="data/replays/manifest.csv", top_n_replays=400):
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                score = float(r.get("avg_score") or 0.0)
            except ValueError:
                score = 0.0
            rows.append((score, r["episode_id"]))
    rows.sort(reverse=True)

    deck_stats = {}
    replays_scanned = 0

    for score, ep_id in rows[:top_n_replays]:
        p = os.path.join("data/replays", f"{ep_id}.json")
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        steps = data.get("steps") or []
        if len(steps) < 3:
            continue

        rewards = data.get("rewards") or [0, 0]
        winner = 0 if (rewards[0] or 0) > (rewards[1] or 0) else (1 if (rewards[1] or 0) > (rewards[0] or 0) else -1)
        if winner == -1:
            continue

        replays_scanned += 1
        for p_idx in (0, 1):
            act = steps[1][p_idx].get("action")
            if isinstance(act, list) and len(act) == 60 and all(isinstance(x, int) for x in act):
                sig = tuple(sorted(int(x) for x in act))
                if sig not in deck_stats:
                    deck_stats[sig] = {
                        "deck": [int(x) for x in act],
                        "wins": 0,
                        "games": 0,
                        "max_elo": score,
                        "name": identify_deck_name(act),
                    }
                deck_stats[sig]["games"] += 1
                deck_stats[sig]["max_elo"] = max(deck_stats[sig]["max_elo"], score)
                if p_idx == winner:
                    deck_stats[sig]["wins"] += 1

    print(f"Scanned {replays_scanned} top-Elo replays (up to {rows[0][0]:.1f} Elo); found {len(deck_stats)} unique 60-card lists.")
    return deck_stats


def play_pair(ag0, ag1, d0, d1, n_games=16):
    wins0 = 0.0
    for g in range(n_games):
        seat0 = g % 2
        if seat0 == 0:
            obs, _ = game.battle_start(d0, d1)
            ag0.reset(d0)
            ag1.reset(d1)
            seats = {0: ag0, 1: ag1}
        else:
            obs, _ = game.battle_start(d1, d0)
            ag1.reset(d1)
            ag0.reset(d0)
            seats = {0: ag1, 1: ag0}
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
        if res == seat0:
            wins0 += 1.0
        elif res == -1:
            wins0 += 0.5
    return wins0 / n_games


def main():
    os.makedirs("championship_decks", exist_ok=True)
    deck_stats = extract_top_elo_decks()

    # Select top candidate decks (at least 4 games in top-400 replays, sorted by win-rate & Elo)
    candidates = []
    for sig, st in deck_stats.items():
        if st["games"] >= 4 and st["wins"] >= 3:
            wr = st["wins"] / st["games"]
            score_metric = wr * 100.0 + (st["max_elo"] - 1250.0) * 0.25 + min(st["games"], 25) * 0.8
            candidates.append((score_metric, wr, st))

    candidates.sort(key=lambda x: x[0], reverse=True)

    # Include current baseline Grimmsnarl ex deck + top 5 distinct replay championship decks
    with open("deck_dragapult_evolved.csv", "r", encoding="utf-8") as f:
        base_grimmsnarl = [int(l.strip()) for l in f if l.strip()]

    tourney_decks = [("Current_Grimmsnarl_ex", base_grimmsnarl)]
    seen_sigs = {tuple(sorted(base_grimmsnarl))}

    print("\n=== Top Championship Decks Extracted from 1300-1346+ Elo Replays ===")
    for idx, (sc, wr, st) in enumerate(candidates):
        sig = tuple(sorted(st["deck"]))
        if sig in seen_sigs:
            print(f"  [Matches Current Deck] {st['name']}: {st['wins']}W/{st['games']}G ({wr*100:.1f}% WR, Peak Elo {st['max_elo']:.1f})")
            continue
        if len(tourney_decks) >= 6:
            break
        seen_sigs.add(sig)
        tag = f"Champ{len(tourney_decks)}_{st['name']}"
        path = os.path.join("championship_decks", f"{tag}.csv")
        with open(path, "w", encoding="utf-8") as f:
            for cid in st["deck"]:
                f.write(f"{cid}\n")
        tourney_decks.append((tag, st["deck"]))
        print(f"  {tag:<32}: {st['wins']:2d}W/{st['games']:2d}G ({wr*100:5.1f}% Replay WR, Peak Elo {st['max_elo']:.1f}) -> saved to {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = (
        "ptcg_wise_v3.pt" if os.path.exists("ptcg_wise_v3.pt")
        else ("ptcg_iql_v3.pt" if os.path.exists("ptcg_iql_v3.pt") else "ptcg_iql_relational.pt")
    )
    print(f"\n=== Running Championship Deck Tournament under '{ckpt}' (16 games/pair) ===")
    ag_a = load_nn_agent("a", ckpt, device=device)
    ag_b = load_nn_agent("b", ckpt, device=device)

    n = len(tourney_decks)
    mat = np.full((n, n), 0.5)
    for i in range(n):
        for j in range(i + 1, n):
            wr = play_pair(ag_a, ag_b, tourney_decks[i][1], tourney_decks[j][1], n_games=16)
            mat[i, j] = wr
            mat[j, i] = 1.0 - wr
            print(f"  {tourney_decks[i][0]:<32} vs {tourney_decks[j][0]:<32}: {wr*100:5.1f}% - {(1-wr)*100:5.1f}%")

    print("\n=== Final Championship Deck Standings ===")
    standings = []
    for i, (name, d_list) in enumerate(tourney_decks):
        avg_wr = (np.sum(mat[i]) - 0.5) / max(1, n - 1)
        standings.append((avg_wr, name, d_list))
    standings.sort(reverse=True)
    for rank, (avg_wr, name, _) in enumerate(standings, 1):
        print(f"  #{rank} {name:<32}: {avg_wr*100:5.1f}% Overall Win-Rate")

    best_wr, best_name, best_deck = standings[0]
    with open("championship_decks/best_championship_deck.csv", "w", encoding="utf-8") as f:
        for cid in best_deck:
            f.write(f"{cid}\n")
    with open("championship_decks/tournament_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_used": ckpt,
            "winner_deck": best_name,
            "winner_avg_wr_pct": round(best_wr * 100.0, 2),
            "standings": [{"name": nm, "avg_wr_pct": round(w * 100.0, 2)} for w, nm, _ in standings],
        }, f, indent=2)
    print(f"\nSaved #1 winning deck ('{best_name}', {best_wr*100:.1f}% WR) to 'championship_decks/best_championship_deck.csv'.")


if __name__ == "__main__":
    main()
