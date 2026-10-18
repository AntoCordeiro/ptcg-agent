# card_tracker.py
"""
Bayesian Belief State & Card Tracker for Imperfect-Information Turn-Based Strategy (Track 5).

Solves the Partial Observability (POMDP) of hidden Deck & Prize cards in Pokémon TCG:
1. Tracks exact multiset subtraction of observed cards:
   Unobserved(c) = InitialDeck(c) - (Hand + Active + Bench + Discard + LostZone)
2. Computes Bayesian posterior probabilities:
   P(c in Deck) = N_deck / (N_deck + N_prize)
   P(c in Prize) = N_prize / (N_deck + N_prize)
3. Detects Deck-Search Actions to convert probabilistic priors into deterministic certainties:
   Cards observed during search -> P(c in Deck) = 1.0
   Unobserved cards missing from search -> P(c in Prize) = 1.0 (Confirmed Prized!)
4. Produces a 32-dimensional Card Category & Resource Belief Vector for the neural policy.
"""

import os
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


# 9 Canonical Card Types
TYPE_UNKNOWN = 0
TYPE_BASIC_POKEMON = 1
TYPE_STAGE1_POKEMON = 2
TYPE_STAGE2_POKEMON = 3
TYPE_SUPPORTER = 4
TYPE_ITEM = 5
TYPE_TOOL = 6
TYPE_STADIUM = 7
TYPE_BASIC_ENERGY = 8
TYPE_SPECIAL_ENERGY = 9
NUM_TYPES = 10

# Precomputed Card Metadata Cache
_METADATA_LOADED = False
_CARD_TYPE_MAP: Dict[int, int] = {}
_CARD_NAME_MAP: Dict[int, str] = {}
_CARD_HP_MAP: Dict[int, float] = {}


def _init_card_metadata():
    global _METADATA_LOADED, _CARD_TYPE_MAP, _CARD_NAME_MAP, _CARD_HP_MAP
    if _METADATA_LOADED:
        return

    csv_path = "the-pokemon-company-ptcg-ai-battle-challenge-playground/EN_Card_Data_R2_full.csv"
    if not os.path.exists(csv_path):
        _METADATA_LOADED = True
        return

    import pandas as pd
    df = pd.read_csv(csv_path)
    stage_col = "Stage (Pok\u00e9mon)/Type (Energy and Trainer)"
    if stage_col not in df.columns:
        # Fallback for encoding variations
        for c in df.columns:
            if "Stage" in c:
                stage_col = c
                break

    for _, row in df.drop_duplicates("Card ID").iterrows():
        cid = int(row["Card ID"])
        name = str(row.get("Card Name", ""))
        stage_str = str(row.get(stage_col, ""))
        hp = float(row.get("HP", 0) or 0)

        _CARD_NAME_MAP[cid] = name
        _CARD_HP_MAP[cid] = hp

        # Classify into 9 canonical types
        if "Basic Pok" in stage_str or stage_str == "Basic Pok\u00e9mon":
            t = TYPE_BASIC_POKEMON
        elif "Stage 1" in stage_str:
            t = TYPE_STAGE1_POKEMON
        elif "Stage 2" in stage_str:
            t = TYPE_STAGE2_POKEMON
        elif "Supporter" in stage_str:
            t = TYPE_SUPPORTER
        elif "Item" in stage_str:
            t = TYPE_ITEM
        elif "Tool" in stage_str:
            t = TYPE_TOOL
        elif "Stadium" in stage_str:
            t = TYPE_STADIUM
        elif "Basic Energy" in stage_str:
            t = TYPE_BASIC_ENERGY
        elif "Special Energy" in stage_str:
            t = TYPE_SPECIAL_ENERGY
        else:
            t = TYPE_UNKNOWN

        _CARD_TYPE_MAP[cid] = t

    _METADATA_LOADED = True


def get_card_type(card_id: int) -> int:
    _init_card_metadata()
    return _CARD_TYPE_MAP.get(int(card_id), TYPE_UNKNOWN)


def get_card_name(card_id: int) -> str:
    _init_card_metadata()
    return _CARD_NAME_MAP.get(int(card_id), f"Card {card_id}")


_ARCHETYPES = None

def get_archetypes() -> Dict[str, List[int]]:
    global _ARCHETYPES
    if _ARCHETYPES is None:
        _ARCHETYPES = {}
        curr = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(curr, "archetypes.json")
        if not os.path.exists(path):
            path = "archetypes.json"
        if os.path.exists(path):
            import json
            try:
                with open(path, "r", encoding="utf-8") as f:
                    _ARCHETYPES = json.load(f)
            except Exception:
                _ARCHETYPES = {}
    return _ARCHETYPES


class CardTracker:
    def __init__(self, initial_deck: List[int]):
        _init_card_metadata()
        self.initial_deck = list(initial_deck)
        self.initial_counts = Counter(initial_deck)
        self.total_cards = len(initial_deck)

        # Certainty tracking from deck search inspections
        self.confirmed_prized_cards: Counter = Counter()
        self.confirmed_deck_cards: Counter = Counter()
        self.has_inspected_deck = False

        # Current turn dynamic tracking
        self.unobserved_counts: Counter = Counter(initial_deck)
        self.deck_probabilities: Dict[int, float] = {}
        self.prize_probabilities: Dict[int, float] = {}

        # Opponent archetype and belief tracking
        self.opponent_visible: Counter = Counter()
        self.inferred_archetype: Optional[str] = None
        self.last_turn: int = 0

    def reset(self, initial_deck: Optional[List[int]] = None):
        if initial_deck is not None:
            self.initial_deck = list(initial_deck)
            self.initial_counts = Counter(initial_deck)
            self.total_cards = len(initial_deck)
        self.confirmed_prized_cards.clear()
        self.confirmed_deck_cards.clear()
        self.has_inspected_deck = False
        self.unobserved_counts = Counter(self.initial_deck)
        self.deck_probabilities.clear()
        self.prize_probabilities.clear()
        self.opponent_visible.clear()
        self.inferred_archetype = None
        self.last_turn = 0

    def update(self, obs: dict, request: Optional[dict] = None):
        """
        Scans all visible card zones and computes Bayesian belief state.
        Automatically detects new games in multi-game series (Game 2 / Game 3) and resets state.
        """
        current = obs.get("current", {}) or {}
        my_idx = current.get("yourIndex", 0)
        players = current.get("players", []) or []
        if len(players) <= my_idx:
            return

        my_state = players[my_idx] or {}
        cur_turn = int(current.get("turn", 0) or 0)
        ctx = int((request or {}).get("context", 0) or 0)
        deck_cnt_raw = int(my_state.get("deckCount", 60) or 0)
        prize_cnt_raw = len(my_state.get("prize") or [])

        # Multi-game episode boundary detection (Game 2 / Game 3 start)
        if (
            ctx in (1, 41)
            or (cur_turn <= 1 and self.last_turn > 1)
            or (deck_cnt_raw >= 50 and prize_cnt_raw == 0)
        ):
            self.confirmed_prized_cards.clear()
            self.confirmed_deck_cards.clear()
            self.has_inspected_deck = False
        self.last_turn = cur_turn

        # 1. Tally all visible card IDs
        visible = Counter()

        # Hand
        hand = my_state.get("hand") or []
        for c in hand:
            cid = c.get("cardId", c.get("id", 0)) if isinstance(c, dict) else c
            if cid:
                visible[int(cid)] += 1

        def tally_pokemon_into(p, dest):
            if not isinstance(p, dict):
                return
            cid = p.get("cardId", p.get("id", 0))
            if cid:
                dest[int(cid)] += 1
            for e in (p.get("energyCards") or p.get("attached") or []):
                e_id = e.get("cardId", e.get("id", 0)) if isinstance(e, dict) else e
                if e_id:
                    dest[int(e_id)] += 1
            for t in (p.get("tools") or []):
                t_id = t.get("cardId", t.get("id", 0)) if isinstance(t, dict) else t
                if t_id:
                    dest[int(t_id)] += 1
            for pre in (p.get("preEvolution") or []):
                pre_id = pre.get("cardId", pre.get("id", 0)) if isinstance(pre, dict) else pre
                if pre_id:
                    dest[int(pre_id)] += 1

        def tally_pokemon(p):
            tally_pokemon_into(p, visible)

        # Active Pokémon & attached cards
        active = my_state.get("active") or []
        active_list = active if isinstance(active, list) else [active]
        for p in active_list:
            tally_pokemon(p)

        # Bench Pokémon & attached cards
        bench = my_state.get("bench") or []
        for b_item in bench:
            b_list = b_item if isinstance(b_item, list) else [b_item]
            for p in b_list:
                tally_pokemon(p)

        # Stadium in play
        stadium = current.get("stadium") or []
        stadium_list = stadium if isinstance(stadium, list) else [stadium]
        for s in stadium_list:
            if isinstance(s, dict) and s.get("playerIndex") == my_idx:
                s_id = s.get("cardId", s.get("id", 0))
                if s_id:
                    visible[int(s_id)] += 1

        # Discard pile
        discard = my_state.get("discard") or []
        for c in discard:
            cid = c.get("cardId", c.get("id", 0)) if isinstance(c, dict) else c
            if cid:
                visible[int(cid)] += 1

        # Lost zone
        lost = my_state.get("lost") or []
        for c in lost:
            cid = c.get("cardId", c.get("id", 0)) if isinstance(c, dict) else c
            if cid:
                visible[int(cid)] += 1


        # 2. Multiset Subtraction: Unobserved cards = Initial - Visible
        self.unobserved_counts.clear()
        for cid, init_count in self.initial_counts.items():
            vis_count = visible[cid]
            unobs = max(0, init_count - vis_count)
            if unobs > 0:
                self.unobserved_counts[cid] = unobs

        # 3. Read physical zone counts
        deck_count = max(0, int(my_state.get("deckCount", 60) or 60))
        prizes_list = my_state.get("prize", [])
        prize_count = len(prizes_list) if isinstance(prizes_list, list) else 6
        total_unobserved = deck_count + prize_count

        # 4. Deterministic Deck-Search Inspection Detection
        # When request presents deck cards in request['deck'], we inspect exactly what's in the deck
        if request is not None and request.get("deck") is not None:
            deck_cards = request.get("deck") or []
            seen_in_search = set()
            for c_obj in deck_cards:
                if isinstance(c_obj, dict):
                    c_id = c_obj.get("cardId", c_obj.get("id", 0))
                    if c_id:
                        seen_in_search.add(int(c_id))

            if len(seen_in_search) > 0:
                self.has_inspected_deck = True
                for cid, count in self.unobserved_counts.items():
                    if cid in seen_in_search:
                        self.confirmed_deck_cards[cid] = count
                    else:
                        # Card is in deck list, unobserved on board, and NOT in deck search -> Confirmed Prized!
                        self.confirmed_prized_cards[cid] = count

        # 5. Compute Bayesian Probabilities
        self.deck_probabilities.clear()
        self.prize_probabilities.clear()

        for cid, unobs_count in self.unobserved_counts.items():
            if self.confirmed_prized_cards[cid] >= unobs_count:
                p_deck = 0.0
                p_prize = 1.0
            elif self.confirmed_deck_cards[cid] >= unobs_count:
                p_deck = 1.0
                p_prize = 0.0
            elif total_unobserved > 0:
                p_deck = float(deck_count) / float(total_unobserved)
                p_prize = float(prize_count) / float(total_unobserved)
            else:
                p_deck = 0.5
                p_prize = 0.5

            self.deck_probabilities[cid] = p_deck
            self.prize_probabilities[cid] = p_prize

        # 6. Track Opponent Visible Cards for Archetype Fingerprinting
        opp_idx = 1 - my_idx
        if len(players) > opp_idx:
            opp_state = players[opp_idx] or {}
            self.opponent_visible.clear()

            opp_active = opp_state.get("active") or []
            opp_active_list = opp_active if isinstance(opp_active, list) else [opp_active]
            for p in opp_active_list:
                tally_pokemon_into(p, self.opponent_visible)

            opp_bench = opp_state.get("bench") or []
            for b_item in opp_bench:
                b_list = b_item if isinstance(b_item, list) else [b_item]
                for p in b_list:
                    tally_pokemon_into(p, self.opponent_visible)

            for c in (opp_state.get("discard") or []):
                cid = c.get("cardId", c.get("id", 0)) if isinstance(c, dict) else c
                if cid:
                    self.opponent_visible[int(cid)] += 1

            for c in (opp_state.get("lost") or []):
                cid = c.get("cardId", c.get("id", 0)) if isinstance(c, dict) else c
                if cid:
                    self.opponent_visible[int(cid)] += 1

    def infer_opponent_unobserved(self, count_needed: int) -> List[int]:
        """
        Infers opponent unobserved cards by matching visible opponent cards
        against the metagame archetype library.
        """
        import random
        archetypes = get_archetypes()
        if not archetypes:
            pool = list(self.initial_deck)
            while len(pool) < count_needed:
                pool.extend(self.initial_deck)
            random.shuffle(pool)
            return pool[:count_needed]

        best_archetype = None
        best_score = -1

        for name, deck_cards in archetypes.items():
            arch_cnt = Counter(deck_cards)
            score = sum(min(arch_cnt[cid], count) for cid, count in self.opponent_visible.items())
            if score > best_score:
                best_score = score
                best_archetype = deck_cards
                self.inferred_archetype = name

        if best_archetype is None or best_score <= 0:
            # Fallback to dominant metagame mix
            best_archetype = archetypes.get("deck_06_4x_Dreepy_4x_Drakloak") or list(archetypes.values())[0]

        arch_counter = Counter(best_archetype)
        unobs_opp = []
        for cid, cnt in arch_counter.items():
            vis = self.opponent_visible.get(cid, 0)
            rem = max(0, cnt - vis)
            unobs_opp.extend([cid] * rem)

        while len(unobs_opp) < count_needed:
            unobs_opp.extend(best_archetype)

        random.shuffle(unobs_opp)
        return unobs_opp[:count_needed]

    def get_belief_vector(self) -> np.ndarray:
        """
        Returns a 32-dimensional Card Category & Resource Belief Vector:
        - [0:9]   Expected remaining count in deck for 9 card types
        - [9:18]  Expected prized count for 9 card types
        - [18:23] Remaining deck composition density ratios
        - [23:29] Key tactical resource availability flags
        - [29:32] Match health indicators
        """
        vec = np.zeros(32, dtype=np.float32)

        type_deck_exp = np.zeros(10, dtype=np.float32)
        type_prize_exp = np.zeros(10, dtype=np.float32)

        has_searchable_basic = 0.0
        has_searchable_evo = 0.0
        has_searchable_energy = 0.0
        has_searchable_supporter = 0.0
        has_searchable_boss = 0.0
        has_searchable_recovery = 0.0

        total_unobserved = sum(self.unobserved_counts.values())

        for cid, count in self.unobserved_counts.items():
            t = get_card_type(cid)
            name = get_card_name(cid).lower()
            p_deck = self.deck_probabilities.get(cid, 0.5)
            p_prize = self.prize_probabilities.get(cid, 0.5)

            exp_deck = float(count) * p_deck
            exp_prize = float(count) * p_prize

            if 0 <= t < 10:
                type_deck_exp[t] += exp_deck
                type_prize_exp[t] += exp_prize

            # Resource availability
            if t == TYPE_BASIC_POKEMON and exp_deck >= 0.5:
                has_searchable_basic = 1.0
            elif t in (TYPE_STAGE1_POKEMON, TYPE_STAGE2_POKEMON) and exp_deck >= 0.5:
                has_searchable_evo = 1.0
            elif t in (TYPE_BASIC_ENERGY, TYPE_SPECIAL_ENERGY) and exp_deck >= 0.5:
                has_searchable_energy = 1.0
            elif t == TYPE_SUPPORTER and exp_deck >= 0.5:
                has_searchable_supporter = 1.0

            # Critical staples (Boss, Stretcher, Rod)
            if "boss" in name and exp_deck >= 0.5:
                has_searchable_boss = 1.0
            if any(k in name for k in ("stretcher", "rod", "ash", "recovery")) and exp_deck >= 0.5:
                has_searchable_recovery = 1.0

        # Dims 0-8: Expected deck counts (normalized by 10.0)
        for i in range(1, 10):
            vec[i - 1] = min(type_deck_exp[i] / 10.0, 1.0)

        # Dims 9-17: Expected prized counts (normalized by 6.0)
        for i in range(1, 10):
            vec[9 + i - 1] = min(type_prize_exp[i] / 6.0, 1.0)

        # Dims 18-22: Deck density ratios
        deck_est = max(1.0, float(sum(type_deck_exp[1:])))
        vec[18] = type_deck_exp[TYPE_BASIC_POKEMON] / deck_est
        vec[19] = (type_deck_exp[TYPE_STAGE1_POKEMON] + type_deck_exp[TYPE_STAGE2_POKEMON]) / deck_est
        vec[20] = (type_deck_exp[TYPE_BASIC_ENERGY] + type_deck_exp[TYPE_SPECIAL_ENERGY]) / deck_est
        vec[21] = type_deck_exp[TYPE_SUPPORTER] / deck_est
        vec[22] = type_deck_exp[TYPE_ITEM] / deck_est

        # Dims 23-28: Tactical availability flags
        vec[23] = has_searchable_basic
        vec[24] = has_searchable_evo
        vec[25] = has_searchable_energy
        vec[26] = has_searchable_supporter
        vec[27] = has_searchable_boss
        vec[28] = has_searchable_recovery

        # Dims 29-31: Health indicators
        vec[29] = 1.0 if deck_est <= 5.0 else 0.0 # Depletion risk
        vec[30] = 1.0 if self.has_inspected_deck else 0.0
        vec[31] = min(float(sum(self.confirmed_prized_cards.values())) / 6.0, 1.0)

        return vec
