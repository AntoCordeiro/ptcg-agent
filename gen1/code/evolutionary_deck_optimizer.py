# evolutionary_deck_optimizer.py
"""
Evolutionary Deck Optimization via Genetic Algorithm & Neural Arena Simulation (Phase 3).

Optimizes 60-card tournament deck lists against the authentic 12-archetype metagame:
1. Genotype: 60-card multiset conforming to official PTCG deck rules.
2. Mutation Operators:
   - Staple Tech Card Swaps (Boss's Orders, Counter Catcher, Super Rod, Enhanced Hammer, etc.)
   - Trainer-to-Energy and Search Ratio tuning.
3. Fitness Evaluation:
   - GPU-accelerated arena battle simulation against benchmark metagame archetypes.
   - Evaluates Balanced Win Rate (BWR) and Prize Margin across seat-alternating matches.
4. Outputs the evolved deck and logs card diff mutations.
"""

import os
import sys
import copy
import json
import glob
import random
from collections import Counter
from typing import List, Tuple, Dict
import numpy as np
import torch
from tqdm import tqdm

SUBMISSION_DIR = os.path.abspath(
    "the-pokemon-company-ptcg-ai-battle-challenge-playground/sample_submission/sample_submission/sample_submission"
)
if SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, SUBMISSION_DIR)

from cg import game
from model_dual_cross import PTCGDualCrossNet
from card_tracker import (
    CardTracker, get_card_type, get_card_name,
    TYPE_BASIC_POKEMON, TYPE_BASIC_ENERGY, TYPE_SUPPORTER, TYPE_ITEM, TYPE_TOOL
)
from match_sim import select_action


def load_deck(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def save_deck(deck: List[int], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for cid in sorted(deck):
            f.write(f"{cid}\n")


# Competitive Staple Card Catalog for Mutation Search
STAPLE_TECH_CATALOG = [
    1079,  # Rare Candy (Item)
    1086,  # Buddy-Buddy Poffin (Item)
    1121,  # Ultra Ball (Item)
    1152,  # Pok Pad (Item)
    1097,  # Night Stretcher (Item / Recovery)
    1129,  # Sacred Ash / Super Rod equivalent
    1081,  # Enhanced Hammer (Item / Energy Disruption)
    1122,  # Pokgear 3.0 (Item)
    1137,  # Tool Scrapper (Item)
    1080,  # Unfair Stamp (Ace Spec Item)
    1182,  # Boss's Orders (Supporter / Gust)
    1184,  # Lana's Aid (Supporter / Recovery)
    1225,  # Hilda (Supporter)
    1227,  # Lillie's Determination (Supporter)
    1231,  # Dawn (Supporter)
    1197,  # Xerosic's Machinations (Supporter)
    1229,  # Wally's Compassion (Supporter)
    1174,  # Air Balloon (Tool)
    19,    # Telepath Psychic Energy (Special Energy)
    13,    # Enriching Energy (Special Energy)
    11,    # Mist Energy (Special Energy)
    5,     # Basic {P} Energy
    7,     # Basic {D} Energy
    3,     # Basic {W} Energy
]


from card_static_table import get_card_info_map

def is_valid_deck(deck: List[int]) -> bool:
    """Enforces official Pokémon TCG rulebook deck construction rules (Rulebook pp. 21, 37)."""
    if len(deck) != 60:
        return False
    info_map = get_card_info_map()
    
    # 1. At least one Basic Pokémon
    has_basic = any(get_card_type(cid) == TYPE_BASIC_POKEMON for cid in deck)
    if not has_basic:
        return False
        
    # 2. Maximum 1 ACE SPEC card in the entire deck
    ace_spec_count = sum(1 for cid in deck if info_map.get(cid, {}).get("aceSpec", False))
    if ace_spec_count > 1:
        return False

    # 3. Maximum 4 cards with the same Card Name (excluding Basic Energy)
    name_counts = Counter()
    for cid in deck:
        if get_card_type(cid) == TYPE_BASIC_ENERGY:
            continue
        c_name = info_map.get(cid, {}).get("name") or get_card_name(cid)
        name_counts[c_name] += 1
        if name_counts[c_name] > 4:
            return False

    return True


def mutate_deck(base_deck: List[int], mutation_rate: float = 0.5) -> List[int]:
    """
    Applies legal card mutation: swaps 1 to 2 non-core cards with staple alternatives,
    strictly obeying 4-of-a-name and 1-ACE-SPEC rules.
    """
    deck = list(base_deck)
    info_map = get_card_info_map()

    # Protected core cards: Basic Pokémon should not be deleted to prevent bricking
    swappable_indices = [
        i for i, cid in enumerate(deck)
        if get_card_type(cid) not in (TYPE_BASIC_POKEMON,)
    ]

    if not swappable_indices:
        return deck

    # Number of cards to mutate (1 or 2)
    num_mutations = 1 if random.random() > 0.3 else 2

    for _ in range(num_mutations):
        idx_to_replace = random.choice(swappable_indices)
        old_card = deck[idx_to_replace]

        # Candidate selection preserving validity
        candidates = []
        for c in STAPLE_TECH_CATALOG:
            candidate_deck = list(deck)
            candidate_deck[idx_to_replace] = c
            if is_valid_deck(candidate_deck):
                candidates.append(c)

        if not candidates:
            continue

        new_card = random.choice(candidates)
        deck[idx_to_replace] = new_card

    return deck if is_valid_deck(deck) else base_deck


def evaluate_deck_fitness(
    candidate_deck: List[int],
    benchmark_opponents: List[Tuple[str, List[int]]],
    model: torch.nn.Module,
    device: torch.device,
    games_per_opponent: int = 8
) -> Tuple[float, float, float]:
    """
    Evaluates candidate deck in fast battle simulation against benchmark opponents.
    Returns: (Fitness, BalancedWinRate, AvgPrizeMargin)
    """
    wins_as_p0 = 0
    games_as_p0 = 0
    wins_as_p1 = 0
    games_as_p1 = 0
    prize_diffs = []

    for opp_name, opp_deck in benchmark_opponents:
        for g_idx in range(games_per_opponent):
            cand_is_p0 = (g_idx % 2 == 0)
            d0 = candidate_deck if cand_is_p0 else opp_deck
            d1 = opp_deck if cand_is_p0 else candidate_deck

            obs, start_data = game.battle_start(d0, d1)
            if obs is None:
                continue

            tracker_p0 = CardTracker(d0)
            tracker_p1 = CardTracker(d1)
            steps = 0

            try:
                while steps < 400:
                    current = obs.get("current", {})
                    res = current.get("result", -1)
                    if res != -1:
                        break

                    seat = current.get("yourIndex", 0)
                    if seat == 0:
                        action, _, _, _, _ = select_action(
                            model, obs, device, stochastic=False, tracker=tracker_p0
                        )
                    else:
                        action, _, _, _, _ = select_action(
                            model, obs, device, stochastic=False, tracker=tracker_p1
                        )

                    obs = game.battle_select(action)
                    steps += 1

                final_res = obs.get("current", {}).get("result", -1)
                players = obs.get("current", {}).get("players", [])
                p0_left = len(players[0].get("prize", [])) if len(players) > 0 else 0
                p1_left = len(players[1].get("prize", [])) if len(players) > 1 else 0

                if final_res in (0, 1):
                    cand_won = (final_res == 0 and cand_is_p0) or (final_res == 1 and not cand_is_p0)
                    if cand_is_p0:
                        games_as_p0 += 1
                        if cand_won:
                            wins_as_p0 += 1
                        prize_diffs.append(p1_left - p0_left)
                    else:
                        games_as_p1 += 1
                        if cand_won:
                            wins_as_p1 += 1
                        prize_diffs.append(p0_left - p1_left)
            finally:
                game.battle_finish()

    wr0 = (wins_as_p0 / max(games_as_p0, 1)) * 100.0
    wr1 = (wins_as_p1 / max(games_as_p1, 1)) * 100.0
    balanced_wr = 0.5 * (wr0 + wr1)
    avg_margin = float(np.mean(prize_diffs)) if prize_diffs else 0.0

    # Fitness combines win rate with margin bonus
    fitness = balanced_wr + 0.1 * avg_margin
    return fitness, balanced_wr, avg_margin


def compute_card_diff(deck_a: List[int], deck_b: List[int]) -> Tuple[List[str], List[str]]:
    ca = Counter(deck_a)
    cb = Counter(deck_b)
    added = []
    removed = []

    for cid in set(list(ca.keys()) + list(cb.keys())):
        diff = cb[cid] - ca[cid]
        name = get_card_name(cid)
        if diff > 0:
            added.append(f"+{diff}x {name} (ID {cid})")
        elif diff < 0:
            removed.append(f"{diff}x {name} (ID {cid})")

    return added, removed


def run_evolutionary_deck_optimization(
    base_deck_path: str = "deck_dragapult.csv",
    generations: int = 4,
    population_size: int = 6,
    output_path: str = "deck_dragapult_evolved.csv",
    log_path: str = "deck_dragapult_evolution_log.json"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Launching Evolutionary Deck Optimization on {device} ({torch.cuda.get_device_name(0)}) ---")

    # Load neural agent
    model = PTCGDualCrossNet().to(device)
    ckpt = "ptcg_ppo_phase4_model.pt" if os.path.exists("ptcg_ppo_phase4_model.pt") else "ptcg_ppo_phase3_model.pt"
    print(f"Using trained agent checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    # Base candidate deck
    base_deck = load_deck(base_deck_path)
    print(f"Base Deck: {base_deck_path} ({len(base_deck)} cards)")

    # Benchmark opponents from core metagame and harvested deck pool
    benchmark_opponents = []
    core_opponents = ["deck_gardevoir.csv", "deck_water.csv", "deck_pool/deck_01_4x_Mega_Lucario_ex_3x_So.csv"]
    for pf in core_opponents:
        if os.path.exists(pf):
            benchmark_opponents.append((os.path.basename(pf), load_deck(pf)))
            print(f"Loaded benchmark opponent: {os.path.basename(pf)}")

    # Evaluate baseline fitness
    print("\nEvaluating baseline deck performance...")
    base_fit, base_wr, base_margin = evaluate_deck_fitness(
        base_deck, benchmark_opponents, model, device, games_per_opponent=8
    )
    print(f"Baseline Deck -> Fitness: {base_fit:.2f} | Balanced Win Rate: {base_wr:.1f}% | Avg Margin: {base_margin:+.2f}")

    # Initialize population: 1 base deck + (P-1) mutated variants
    population = [list(base_deck)]
    for _ in range(population_size - 1):
        population.append(mutate_deck(base_deck, mutation_rate=0.7))

    best_deck_overall = list(base_deck)
    best_fitness_overall = base_fit
    best_wr_overall = base_wr

    evolution_history = [{
        "generation": 0,
        "best_fitness": base_fit,
        "best_win_rate": base_wr,
        "best_margin": base_margin,
        "deck": list(base_deck)
    }]

    for gen in range(1, generations + 1):
        print(f"\n=================== Generation {gen}/{generations} ===================")
        fitness_results = []

        for p_idx, candidate in enumerate(population):
            fit, wr, margin = evaluate_deck_fitness(
                candidate, benchmark_opponents, model, device, games_per_opponent=6
            )
            fitness_results.append((fit, wr, margin, candidate))
            print(f"  Candidate {p_idx+1:02d}: Fitness = {fit:6.2f} | BWR = {wr:5.1f}% | Margin = {margin:+4.2f}")

        # Sort population by fitness (descending)
        fitness_results.sort(key=lambda x: x[0], reverse=True)
        top_fit, top_wr, top_margin, top_deck = fitness_results[0]

        if top_fit > best_fitness_overall:
            best_fitness_overall = top_fit
            best_wr_overall = top_wr
            best_deck_overall = list(top_deck)
            print(f"  >>> NEW BEST DECK DISCOVERED! Fitness: {top_fit:.2f} (BWR: {top_wr:.1f}%) <<<")

        evolution_history.append({
            "generation": gen,
            "best_fitness": top_fit,
            "best_win_rate": top_wr,
            "best_margin": top_margin,
            "deck": list(top_deck)
        })

        # Selection & Breeding for next generation:
        # Elitism: top 2 candidates survive unchanged
        next_gen = [list(fitness_results[0][3]), list(fitness_results[1][3])]

        # Fill remaining slots with mutations of the top performers
        while len(next_gen) < population_size:
            parent = random.choice([fitness_results[0][3], fitness_results[1][3], fitness_results[2][3]])
            next_gen.append(mutate_deck(parent, mutation_rate=0.6))

        population = next_gen

    # Save the evolved champion deck
    save_deck(best_deck_overall, output_path)
    print(f"\nEvolutionary Optimization Completed! Best deck saved to '{output_path}'.")

    # Card Diff Analysis
    added, removed = compute_card_diff(base_deck, best_deck_overall)
    print("\n================ EVOLUTIONARY CARD MUTATION DIFF ================")
    print(f"Fitness Improvement: {base_fit:.2f} -> {best_fitness_overall:.2f} (+{best_fitness_overall - base_fit:+.2f})")
    print(f"Balanced Win Rate  : {base_wr:.1f}% -> {best_wr_overall:.1f}% (+{best_wr_overall - base_wr:+.1f}%)")
    print("\nCards Added:")
    for a in added:
        print(f"  [+] {a}")
    print("Cards Removed:")
    for r in removed:
        print(f"  [-] {r}")

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(evolution_history, f, indent=2)
    print(f"Logged evolution history to '{log_path}'.")

    return best_deck_overall


if __name__ == "__main__":
    run_evolutionary_deck_optimization(
        base_deck_path="deck_dragapult.csv",
        generations=3,
        population_size=6,
        output_path="deck_dragapult_evolved.csv",
        log_path="deck_dragapult_evolution_log.json"
    )
