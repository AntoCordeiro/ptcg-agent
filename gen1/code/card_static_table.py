# card_static_table.py
import os
import sys
import numpy as np

_TABLE_CACHE = None
_CARD_INFO_CACHE = None
_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_TABLE_PATH = os.path.join(_CURR_DIR, "card_static_table.npy") if os.path.exists(os.path.join(_CURR_DIR, "card_static_table.npy")) else "card_static_table.npy"
CSV_PATH = "the-pokemon-company-ptcg-ai-battle-challenge-playground/EN_Card_Data_R2_full.csv"

def get_card_info_map(num_cards: int = 1500) -> dict:
    """Returns a fast lookup dictionary mapping cardId -> card metadata dict."""
    global _CARD_INFO_CACHE
    if _CARD_INFO_CACHE is not None:
        return _CARD_INFO_CACHE

    info_map = {}
    try:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        if curr_dir not in sys.path:
            sys.path.insert(0, curr_dir)
        sub_path = os.path.abspath("the-pokemon-company-ptcg-ai-battle-challenge-playground/sample_submission/sample_submission/sample_submission")
        if os.path.exists(sub_path) and sub_path not in sys.path:
            sys.path.insert(0, sub_path)
        import cg.api as api
        cards = api.all_card_data()
        name_to_ids = {}
        for c in cards:
            cid = int(c.cardId)
            nm = str(c.name)
            name_to_ids.setdefault(nm, []).append(cid)

        for c in cards:
            cid = int(c.cardId)
            evo_from = str(c.evolvesFrom) if getattr(c, "evolvesFrom", None) else ""
            pre_ids = name_to_ids.get(evo_from, []) if evo_from else []
            info_map[cid] = {
                "name": str(c.name),
                "hp": float(c.hp),
                "retreatCost": int(c.retreatCost),
                "weakness": c.weakness,       # int 1..8 or None
                "resistance": c.resistance,   # int 1..8 or None
                "ex": bool(c.ex),
                "megaEx": bool(c.megaEx),
                "tera": bool(c.tera),
                "aceSpec": bool(c.aceSpec),
                "cardType": int(c.cardType),
                "energyType": int(c.energyType) if c.energyType is not None else 0,
                "basic": bool(c.basic),
                "stage1": bool(c.stage1),
                "stage2": bool(c.stage2),
                "evolvesFrom": evo_from,
                "pre_evo_ids": pre_ids,
                "pre_evo_id": pre_ids[0] if pre_ids else 0,
                "attacks": list(c.attacks) if c.attacks else [],
            }
    except Exception as e:
        pass

    _CARD_INFO_CACHE = info_map
    return info_map

def build_static_table(csv_path: str = CSV_PATH, num_cards: int = 1500) -> np.ndarray:
    table = np.zeros((num_cards, 25), dtype=np.float32)
    if not os.path.exists(csv_path):
        alt_paths = [
            "the-pokemon-company-ptcg-ai-battle-challenge-playground/EN_Card_Data_R2_full.csv",
            "../the-pokemon-company-ptcg-ai-battle-challenge-playground/EN_Card_Data_R2_full.csv",
            "EN_Card_Data_R2_full.csv",
        ]
        for p in alt_paths:
            if os.path.exists(p):
                csv_path = p
                break

    if os.path.exists(csv_path):
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            stages = [
                'basic energy', 'special energy', 'basic pok', 
                'stage 1 pok', 'stage 2 pok', 'item', 
                'tool', 'supporter', 'stadium'
            ]
            energy_types = ['{G}', '{R}', '{W}', '{L}', '{P}', '{F}', '{D}', '{M}', '{C}']

            for _, row in df.iterrows():
                try:
                    cid = int(row['Card ID'])
                    if cid >= num_cards:
                        continue

                    stg = str(row['Stage (Pokémon)/Type (Energy and Trainer)']).lower()
                    for i, s in enumerate(stages):
                        if s in stg:
                            table[cid, i] = 1.0
                            break

                    typ = str(row.get('Type', ''))
                    for j, et in enumerate(energy_types):
                        if et in typ:
                            table[cid, 9 + j] = 1.0

                    hp_str = str(row.get('HP', ''))
                    if hp_str.isdigit():
                        table[cid, 19] = float(hp_str) / 400.0

                    ret_str = str(row.get('Retreat', ''))
                    if ret_str.isdigit():
                        table[cid, 20] = float(ret_str) / 5.0

                    table[cid, 21] = 1.0 if 'pok' in stg else 0.0
                    table[cid, 22] = 1.0 if any(k in stg for k in ['item', 'supporter', 'stadium', 'tool']) else 0.0
                    table[cid, 23] = 1.0 if 'energy' in stg else 0.0
                    rule = row.get('Rule')
                    table[cid, 24] = 1.0 if pd.notna(rule) and str(rule).strip() != 'n/a' else 0.0
                except Exception:
                    pass
        except Exception:
            pass

    # Ensure rule-box flag and stats from cg.api
    info_map = get_card_info_map(num_cards=num_cards)
    for cid, info in info_map.items():
        if cid < num_cards:
            if info["ex"] or info["megaEx"] or info["aceSpec"]:
                table[cid, 24] = 1.0
            if info["hp"] > 0 and table[cid, 19] == 0:
                table[cid, 19] = float(info["hp"]) / 400.0
            if info["retreatCost"] > 0 and table[cid, 20] == 0:
                table[cid, 20] = float(info["retreatCost"]) / 5.0

    return table

def get_card_static_table(num_cards: int = 1500) -> np.ndarray:
    global _TABLE_CACHE
    if _TABLE_CACHE is not None:
        return _TABLE_CACHE

    if os.path.exists(STATIC_TABLE_PATH):
        try:
            _TABLE_CACHE = np.load(STATIC_TABLE_PATH)
            if _TABLE_CACHE.shape == (num_cards, 25):
                return _TABLE_CACHE
        except Exception:
            pass

    _TABLE_CACHE = build_static_table(num_cards=num_cards)
    try:
        np.save(STATIC_TABLE_PATH, _TABLE_CACHE)
    except Exception:
        pass
    return _TABLE_CACHE

if __name__ == "__main__":
    t = get_card_static_table()
    info = get_card_info_map()
    print("Static card table loaded successfully, shape:", t.shape, "Non-zeros:", np.count_nonzero(t.sum(axis=1)))
    print("Card info map cards count:", len(info))

