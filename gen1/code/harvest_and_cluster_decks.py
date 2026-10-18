# harvest_and_cluster_decks.py
"""
Multi-Deck Pool Harvester & Archetype Clustering ("Decks Are All You Need").

Harvests authentic competitive 60-card decks from 4,603 Kaggle tournament replays,
validates tournament legality, clusters them via card-frequency embeddings, and
selects the Top 12 diverse metagame archetypes for PPO self-play and evolutionary search.
"""

import os
import sys
import glob
import json
from collections import Counter
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from features_spec import NUM_CARD_IDS
from card_tracker import (
    _init_card_metadata, get_card_type, get_card_name,
    TYPE_BASIC_POKEMON, TYPE_BASIC_ENERGY, TYPE_STAGE1_POKEMON, TYPE_STAGE2_POKEMON
)


def validate_deck_legality(deck: List[int]) -> bool:
    """Verifies that a 60-card deck conforms to official PTCG deck construction rules."""
    if len(deck) != 60:
        return False

    counts = Counter(deck)

    # Must contain at least 1 Basic Pokmon
    has_basic = any(get_card_type(cid) == TYPE_BASIC_POKEMON for cid in counts)
    if not has_basic:
        return False

    # Max 4 copies rule (except Basic Energy)
    for cid, cnt in counts.items():
        if get_card_type(cid) != TYPE_BASIC_ENERGY and cnt > 4:
            return False

    return True


def harvest_decks_from_replays(replays_dir: str = "data/replays") -> List[Tuple[List[int], int]]:
    """
    Parses all replay JSON files in replays_dir and returns unique valid decks
    along with their observed frequency in competitive ladder play.
    """
    _init_card_metadata()
    replay_files = glob.glob(os.path.join(replays_dir, "*.json"))
    print(f"Scanning {len(replay_files)} match replays for authentic competitive decks...")

    deck_frequencies = Counter()

    for fpath in replay_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            steps = data.get("steps", [])
            if len(steps) > 1:
                # Step 1 contains the 60-card submission actions for both players
                for agent_step in steps[1]:
                    act = agent_step.get("action")
                    if isinstance(act, list) and len(act) == 60:
                        deck_tuple = tuple(sorted(int(x) for x in act))
                        deck_frequencies[deck_tuple] += 1
        except Exception:
            continue

    print(f"Discovered {len(deck_frequencies)} distinct 60-card configurations.")

    # Filter for tournament legality
    legal_decks = []
    for deck_tuple, freq in deck_frequencies.items():
        deck_list = list(deck_tuple)
        if validate_deck_legality(deck_list):
            legal_decks.append((deck_list, freq))

    print(f"Verified {len(legal_decks)} legal tournament-ready decks.")
    return legal_decks


def cluster_and_save_deck_pool(
    legal_decks: List[Tuple[List[int], int]],
    n_clusters: int = 12,
    output_dir: str = "deck_pool"
):
    os.makedirs(output_dir, exist_ok=True)

    # Build frequency-weighted feature vectors for each deck
    vectors = np.zeros((len(legal_decks), 1500), dtype=np.float32)
    weights = []

    for i, (deck, freq) in enumerate(legal_decks):
        for cid in deck:
            if 0 <= cid < 1500:
                vectors[i, cid] += 1.0
        weights.append(freq)

    weights = np.array(weights, dtype=np.float32)

    # K-Means clustering across deck card compositions
    print(f"\nClustering {len(legal_decks)} decks into {n_clusters} representative metagame archetypes...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(vectors, sample_weight=weights)

    archetype_manifest = []

    for k in range(n_clusters):
        cluster_indices = np.where(cluster_labels == k)[0]
        if len(cluster_indices) == 0:
            continue

        # Find the medoid deck (closest to cluster center) weighted by real-world ladder frequency
        center = kmeans.cluster_centers_[k]
        distances = np.linalg.norm(vectors[cluster_indices] - center, axis=1)
        cluster_freqs = weights[cluster_indices]

        # Score = distance / sqrt(frequency) -> prefers central, popular competitive decks
        scores = distances / np.sqrt(np.maximum(cluster_freqs, 1.0))
        best_local_idx = cluster_indices[np.argmin(scores)]
        representative_deck, rep_freq = legal_decks[best_local_idx]
        total_cluster_matches = int(cluster_freqs.sum())

        # Identify signature Pokmon
        counts = Counter(representative_deck)
        pokemon_names = []
        for cid, cnt in counts.most_common():
            if get_card_type(cid) in (TYPE_BASIC_POKEMON, TYPE_STAGE1_POKEMON, TYPE_STAGE2_POKEMON):
                name = get_card_name(cid)
                if not any(name in p for p in pokemon_names):
                    pokemon_names.append(f"{cnt}x {name}")
            if len(pokemon_names) >= 3:
                break

        archetype_title = ", ".join(pokemon_names[:2]) if pokemon_names else "Toolbox"
        clean_name = archetype_title.replace(" ", "_").replace("'", "").replace(":", "").replace(",", "")
        deck_filename = f"deck_{k+1:02d}_{clean_name[:24]}.csv"
        deck_path = os.path.join(output_dir, deck_filename)

        with open(deck_path, "w", encoding="utf-8") as f:
            for cid in representative_deck:
                f.write(f"{cid}\n")

        manifest_entry = {
            "cluster_id": k + 1,
            "filename": deck_filename,
            "archetype_title": archetype_title,
            "signature_cards": pokemon_names,
            "ladder_match_count": total_cluster_matches,
            "unique_variants_in_cluster": len(cluster_indices)
        }
        archetype_manifest.append(manifest_entry)

        print(f"  [Cluster {k+1:02d}] {archetype_title:<36} | {total_cluster_matches:>4} matches ({len(cluster_indices):>2} variants) -> {deck_filename}")

    manifest_path = os.path.join(output_dir, "archetype_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(archetype_manifest, f, indent=2)

    print(f"\nAll {n_clusters} archetype decks saved to '{output_dir}/' with manifest '{manifest_path}'.")
    return archetype_manifest


def main():
    decks = harvest_decks_from_replays("data/replays")
    cluster_and_save_deck_pool(decks, n_clusters=12, output_dir="deck_pool")


if __name__ == "__main__":
    main()
