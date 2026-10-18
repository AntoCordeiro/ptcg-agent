# feature_parser.py
import os
import sys
import numpy as np
from features_spec import (
    GLOBAL_DIM, OBJECT_FEATURE_DIM, MAX_TOKENS, NUM_CARD_IDS,
    TOKEN_TYPE_GLOBAL, TOKEN_TYPE_ACTIVE_MINE, TOKEN_TYPE_BENCH_MINE,
    TOKEN_TYPE_ACTIVE_OPP, TOKEN_TYPE_BENCH_OPP, TOKEN_TYPE_HAND_MINE,
    TOKEN_TYPE_STADIUM, TOKEN_TYPE_OPTION
)
from card_static_table import get_card_static_table, get_card_info_map
from card_text_encoder import get_card_text_embeddings

_CARD_STATIC_TABLE = get_card_static_table(NUM_CARD_IDS)
_CARD_INFO_MAP = get_card_info_map(NUM_CARD_IDS)
_CARD_TEXT_TABLE = get_card_text_embeddings(NUM_CARD_IDS)


def safe_len(obj) -> int:
    """Returns len(obj) if obj is a valid collection, else 0."""
    return len(obj) if obj is not None and isinstance(obj, (list, dict, tuple)) else 0


def safe_int(val, default: int = 0) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except Exception:
        return default


def safe_float(val, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except Exception:
        return default


def safe_get_count(state: dict, count_key: str, list_key: str, default: int = 0) -> float:
    """Tries count_key first; if None or missing, checks list_key length; else default."""
    if not isinstance(state, dict):
        return float(default)
    val = state.get(count_key)
    if val is not None:
        return float(val)
    lst = state.get(list_key)
    if lst is not None and isinstance(lst, (list, tuple)):
        return float(len(lst))
    return float(default)


_ATTACK_DATA_TABLE = None


def get_attack_info(attack_id: int) -> dict:
    global _ATTACK_DATA_TABLE
    if _ATTACK_DATA_TABLE is None:
        try:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            if curr_dir not in sys.path:
                sys.path.insert(0, curr_dir)
            sub_dir = os.path.abspath(
                "the-pokemon-company-ptcg-ai-battle-challenge-playground/sample_submission/sample_submission/sample_submission"
            )
            if os.path.exists(sub_dir) and sub_dir not in sys.path:
                sys.path.insert(0, sub_dir)
            from cg import api
            _ATTACK_DATA_TABLE = {}
            for a in api.all_attack():
                txt = str(a.text or "").lower()
                _ATTACK_DATA_TABLE[int(a.attackId)] = {
                    "name": str(a.name or ""),
                    "text": txt,
                    "damage": float(a.damage or 0),
                    "cost_len": len(a.energies) if a.energies else 0,
                    "energies": [int(e) for e in (a.energies or [])],
                    "has_bench_dmg": 1.0 if ("benched" in txt or "your opponent’s pokémon" in txt or "your opponent's pokémon" in txt) else 0.0,
                    "has_draw": 1.0 if "draw" in txt else 0.0,
                    "has_search": 1.0 if "search your deck" in txt else 0.0,
                    "has_status": 1.0 if any(k in txt for k in ("asleep", "confused", "paralyzed", "poisoned", "burned", "can’t play", "can't play")) else 0.0,
                }
        except Exception:
            _ATTACK_DATA_TABLE = {}
    return _ATTACK_DATA_TABLE.get(
        int(attack_id),
        {"name": "", "text": "", "damage": 0.0, "cost_len": 0, "energies": [], "has_bench_dmg": 0.0, "has_draw": 0.0, "has_search": 0.0, "has_status": 0.0}
    )


def get_attack_damage(attack_id: int) -> float:
    return get_attack_info(attack_id).get("damage", 0.0)


def _extract_pokemon_dict(p_entry) -> dict:
    if not p_entry:
        return {}
    p_dict = p_entry[0] if isinstance(p_entry, list) and len(p_entry) > 0 else p_entry
    return p_dict if isinstance(p_dict, dict) else {}


def _count_pokemon_energies(p_dict: dict) -> tuple[int, np.ndarray]:
    """Returns total energy count and 10-dim energy type histogram [C, G, R, W, L, P, F, D, M, Dragon]."""
    hist = np.zeros(10, dtype=np.float32)
    if not isinstance(p_dict, dict):
        return 0, hist
    energies = p_dict.get("energies")
    if energies is None:
        energies = p_dict.get("energy") or []
    cnt = safe_len(energies)
    if isinstance(energies, list):
        for e in energies:
            e_val = e.get("id", 0) if isinstance(e, dict) else safe_int(e, -1)
            if 0 <= e_val < 10:
                hist[e_val] += 1.0
    return cnt, hist


def compute_dynamic_attack_damage(
    atk_id: int,
    my_state: dict,
    opp_state: dict,
    my_act: dict,
    opp_act: dict,
) -> tuple[float, float, bool]:
    """
    Computes (base_or_dynamic_damage, bonus_bench_damage, ignores_weakness_resistance)
    accounting for metagame scaling attacks (Powerful Hand, Bellowing Thunder, Myriad Leaf Shower, etc.).
    """
    info = get_attack_info(atk_id)
    raw_dmg = float(info.get("damage", 0.0))
    bench_dmg = 0.0
    ignores_wr = False

    my_hand_cnt = safe_get_count(my_state, "handCount", "hand", default=0)
    opp_hand_cnt = safe_get_count(opp_state, "handCount", "hand", default=0)
    my_bench = my_state.get("bench") or []
    opp_bench = opp_state.get("bench") or []
    my_act_e, _ = _count_pokemon_energies(my_act)
    opp_act_e, _ = _count_pokemon_energies(opp_act)

    # Total energies across my board
    total_my_e = my_act_e
    total_my_grass_e = 0
    total_my_psychic_e = 0
    for p in ([my_act] if my_act else []) + (my_bench if isinstance(my_bench, list) else []):
        if isinstance(p, dict):
            _, h = _count_pokemon_energies(p)
            total_my_grass_e += int(h[1])
            total_my_psychic_e += int(h[5])
    if isinstance(my_bench, list):
        for b in my_bench:
            if isinstance(b, dict):
                be, _ = _count_pokemon_energies(b)
                total_my_e += be

    if atk_id == 1072:  # Alakazam - Powerful Hand (2 damage counters per card in hand)
        raw_dmg = 20.0 * my_hand_cnt
        ignores_wr = True
    elif atk_id == 72:  # Raging Bolt ex - Bellowing Thunder (70x basic energy on board)
        raw_dmg = 70.0 * total_my_e
    elif atk_id == 120:  # Teal Mask Ogerpon ex - Myriad Leaf Shower (30 + 30x both active energies)
        raw_dmg = 30.0 + 30.0 * (my_act_e + opp_act_e)
    elif atk_id == 115:  # Dipplin - Do the Wave (20x bench)
        raw_dmg = 20.0 * safe_len(my_bench)
    elif atk_id == 1240:  # Mega Froslass ex - Resentful Refrain (50x opponent hand)
        raw_dmg = 50.0 * opp_hand_cnt
    elif atk_id == 1225:  # Mega Lopunny ex - Gale Thrust (60 + 170 if moved to active this turn)
        raw_dmg = 230.0 if my_act.get("appearThisTurn") else 60.0
    elif atk_id == 371:  # Lillie's Clefairy ex - Full Moon Rondo (20 + 20x all benched)
        raw_dmg = 20.0 + 20.0 * (safe_len(my_bench) + safe_len(opp_bench))
    elif atk_id == 195:  # Hydrapple ex - Syrup Storm (30 + 30x Grass energy)
        raw_dmg = 30.0 + 30.0 * total_my_grass_e
    elif atk_id == 1079:  # Mega Gardevoir ex - Mega Symphonia (50x Psychic energy)
        raw_dmg = 50.0 * total_my_psychic_e
    elif atk_id == 339:  # Alakazam - Psychic (10 + 50x opp active energy)
        raw_dmg = 10.0 + 50.0 * opp_act_e
    elif atk_id == 540:  # Cynthia's Spiritomb - Raging Curse (10x damage counters on bench)
        dmg_cnts = 0.0
        if isinstance(my_bench, list):
            for b in my_bench:
                if isinstance(b, dict):
                    bhp = safe_float(b.get("hp"), 0.0)
                    bmhp = safe_float(b.get("maxHp"), bhp)
                    dmg_cnts += max(0.0, bmhp - bhp) / 10.0
        raw_dmg = 10.0 * dmg_cnts
    elif atk_id == 937:  # Marnie's Grimmsnarl ex - Shadow Bullet (180 + 30 bench)
        bench_dmg = 30.0
    elif atk_id == 154:  # Dragapult ex - Phantom Dive (200 + 60 bench counters)
        bench_dmg = 60.0
    elif atk_id == 183:  # Fezandipiti ex - Cruel Arrow (100 snipe)
        bench_dmg = 100.0
    elif atk_id == 188:  # Kyurem - Trifrost (110 to 3 targets)
        raw_dmg = 110.0
        bench_dmg = 220.0
    elif atk_id == 305:  # Annihilape - Destined Fight (KO active)
        raw_dmg = safe_float(opp_act.get("hp"), 300.0)
        ignores_wr = True

    return raw_dmg, bench_dmg, ignores_wr


def _get_prize_yield(cid: int) -> float:
    info = _CARD_INFO_MAP.get(int(cid), {})
    if info.get("megaEx"):
        return 3.0
    if info.get("ex"):
        return 2.0
    return 1.0 if info.get("hp", 0) > 0 else 0.0


def parse_game_state_to_tensors(obs: dict, request: dict, tracker=None):
    global_vec = np.zeros(GLOBAL_DIM, dtype=np.float32)
    current = obs.get("current") or {}
    request = request or {}

    my_idx = safe_int(current.get("yourIndex"), 0)
    players = current.get("players") or []

    my_state = (players[my_idx] if len(players) > my_idx else None) or {}
    opp_state = (players[1 - my_idx] if len(players) > (1 - my_idx) else None) or {}

    # --- 1. Global Features (Table 1) ---
    turn_num = safe_float(current.get("turn"), 0.0)
    global_vec[0] = turn_num / 40.0
    global_vec[1] = safe_float(current.get("turnActionCount"), 0.0) / 20.0
    global_vec[2] = float(my_idx)
    first_p = safe_int(current.get("firstPlayer"), -1)
    global_vec[3] = 1.0 if first_p == my_idx else 0.0
    global_vec[4] = float(max(0, first_p))
    global_vec[5] = 1.0  # Bias term

    # Turn allowances used (dims 6-9)
    global_vec[6] = 1.0 if current.get("supporterPlayed") else 0.0
    global_vec[7] = 1.0 if current.get("stadiumPlayed") else 0.0
    global_vec[8] = 1.0 if current.get("energyAttached") else 0.0
    global_vec[9] = 1.0 if current.get("retreated") else 0.0

    # Prizes, deck, hand, and discard sizes (dims 10-21)
    my_prizes = safe_get_count(my_state, "prizeCount", "prize", default=6)
    opp_prizes = safe_get_count(opp_state, "prizeCount", "prize", default=6)
    global_vec[10] = my_prizes / 6.0
    global_vec[11] = opp_prizes / 6.0
    global_vec[12] = (opp_prizes - my_prizes) / 6.0

    global_vec[13] = safe_get_count(my_state, "deckCount", "deck", default=60) / 60.0
    global_vec[14] = safe_get_count(opp_state, "deckCount", "deck", default=60) / 60.0
    global_vec[15] = safe_get_count(my_state, "handCount", "hand", default=7) / 24.0
    global_vec[16] = safe_get_count(opp_state, "handCount", "hand", default=7) / 24.0
    global_vec[17] = safe_get_count(my_state, "discardCount", "discard", default=0) / 60.0
    global_vec[18] = safe_get_count(opp_state, "discardCount", "discard", default=0) / 60.0

    bench_my = my_state.get("bench") or []
    bench_opp = opp_state.get("bench") or []
    global_vec[19] = float(safe_len(bench_my)) / 5.0
    global_vec[20] = float(safe_len(bench_opp)) / 5.0
    global_vec[21] = 1.0 if current.get("stadium") else 0.0

    # Status conditions (dims 22-31)
    global_vec[22] = 1.0 if my_state.get("poisoned") else 0.0
    global_vec[23] = 1.0 if my_state.get("burned") else 0.0
    global_vec[24] = 1.0 if my_state.get("asleep") else 0.0
    global_vec[25] = 1.0 if my_state.get("paralyzed") else 0.0
    global_vec[26] = 1.0 if my_state.get("confused") else 0.0

    global_vec[27] = 1.0 if opp_state.get("poisoned") else 0.0
    global_vec[28] = 1.0 if opp_state.get("burned") else 0.0
    global_vec[29] = 1.0 if opp_state.get("asleep") else 0.0
    global_vec[30] = 1.0 if opp_state.get("paralyzed") else 0.0
    global_vec[31] = 1.0 if opp_state.get("confused") else 0.0

    # Remaining overage time (dim 32)
    overage = obs.get("remainingOverageTime", 300)
    global_vec[32] = min(safe_float(overage, 300.0) / 300.0, 1.0)

    # Discard pile breakdown (dims 33-38)
    def summarize_discard(p_st):
        disc = p_st.get("discard") or []
        n_eng, n_sup, n_bas = 0, 0, 0
        if isinstance(disc, list):
            for c in disc:
                cid = c.get("id", c.get("cardId", 0)) if isinstance(c, dict) else (c if isinstance(c, int) else 0)
                info = _CARD_INFO_MAP.get(safe_int(cid), {})
                if info.get("cardType") in (5, 6) or (info.get("basic") and "Energy" in info.get("name", "")):
                    n_eng += 1
                elif info.get("cardType") == 3:  # Supporter
                    n_sup += 1
                elif info.get("basic") and info.get("hp", 0) > 0:
                    n_bas += 1
        return n_eng, n_sup, n_bas

    my_eng, my_sup, my_bas = summarize_discard(my_state)
    opp_eng, opp_sup, opp_bas = summarize_discard(opp_state)
    global_vec[33] = my_eng / 20.0
    global_vec[34] = my_sup / 10.0
    global_vec[35] = my_bas / 15.0
    global_vec[36] = opp_eng / 20.0
    global_vec[37] = opp_sup / 10.0
    global_vec[38] = opp_bas / 15.0

    # Request features (dims 40-45)
    mn_req = safe_int(request.get("minCount"), 1)
    mx_req = safe_int(request.get("maxCount"), 1)
    global_vec[40] = float(mn_req) / 10.0
    global_vec[41] = float(mx_req) / 10.0
    global_vec[42] = 1.0 if mx_req > 1 else 0.0
    global_vec[43] = safe_float(request.get("remainEnergyCost"), 0.0) / 6.0
    global_vec[44] = safe_float(request.get("remainDamageCounter"), 0.0) / 20.0

    # Active effect card being processed (dim 45-49)
    eff = request.get("effect") or request.get("contextCard")
    eff_id = 0
    if isinstance(eff, dict):
        eff_id = safe_int(eff.get("id", eff.get("cardId", 0)), 0)
    elif isinstance(eff, (int, float)):
        eff_id = safe_int(eff, 0)
    global_vec[45] = min(float(eff_id) / 1500.0, 1.0)
    global_vec[46] = 1.0 if mn_req == 0 else 0.0  # Optional selection flag
    global_vec[47] = float(max(0, mx_req - mn_req)) / 10.0

    # One-hot request type (dims 50-67)
    req_type = safe_int(request.get("type"), 0)
    if 0 <= req_type < 18:
        global_vec[50 + req_type] = 1.0

    # One-hot request context (dims 70-119)
    req_context = safe_int(request.get("context"), 0)
    if 0 <= req_context < 50:
        global_vec[70 + req_context] = 1.0

    # Board Summary Features (dims 120-129 & 162-175)
    my_act_dict = _extract_pokemon_dict(my_state.get("active"))
    opp_act_dict = _extract_pokemon_dict(opp_state.get("active"))

    def summarize_board(act_d, bench_l):
        all_p = ([act_d] if act_d else []) + [b for b in (bench_l if isinstance(bench_l, list) else []) if isinstance(b, dict)]
        tot_e, tot_dmg, n_s1, n_s2, n_ex = 0, 0.0, 0, 0, 0
        for p in all_p:
            ec, _ = _count_pokemon_energies(p)
            tot_e += ec
            hp = safe_float(p.get("hp"), 0.0)
            mhp = safe_float(p.get("maxHp"), hp)
            tot_dmg += max(0.0, mhp - hp)
            cid = safe_int(p.get("id", p.get("cardId", 0)), 0)
            inf = _CARD_INFO_MAP.get(cid, {})
            if inf.get("stage1"):
                n_s1 += 1
            if inf.get("stage2"):
                n_s2 += 1
            if inf.get("ex") or inf.get("megaEx"):
                n_ex += 1
        return tot_e, tot_dmg, n_s1, n_s2, n_ex

    my_tot_e, my_tot_dmg, my_s1, my_s2, my_ex = summarize_board(my_act_dict, bench_my)
    opp_tot_e, opp_tot_dmg, opp_s1, opp_s2, opp_ex = summarize_board(opp_act_dict, bench_opp)

    global_vec[120] = safe_float(my_act_dict.get("hp"), 0.0) / 350.0
    global_vec[121] = max(0.0, safe_float(my_act_dict.get("maxHp"), 0.0) - safe_float(my_act_dict.get("hp"), 0.0)) / 300.0
    global_vec[122] = safe_float(opp_act_dict.get("hp"), 0.0) / 350.0
    global_vec[123] = max(0.0, safe_float(opp_act_dict.get("maxHp"), 0.0) - safe_float(opp_act_dict.get("hp"), 0.0)) / 300.0
    global_vec[124] = float(my_tot_e) / 15.0
    global_vec[125] = float(opp_tot_e) / 15.0
    global_vec[126] = my_tot_dmg / 500.0
    global_vec[127] = opp_tot_dmg / 500.0
    global_vec[128] = float(my_s1 + 2 * my_s2) / 10.0
    global_vec[129] = float(opp_s1 + 2 * opp_s2) / 10.0

    # --- 1b. Bayesian Belief State & Card Tracker (dims 130-161) ---
    if tracker is not None:
        try:
            tracker.update(obs, request)
            global_vec[130:162] = tracker.get_belief_vector()
        except Exception:
            pass

    # --- 1c. Approach B: Episodic Event-Log History Encoding (dims 162-181) ---
    logs = obs.get("logs") or []
    if isinstance(logs, list) and len(logs) > 0:
        global_vec[162] = min(float(len(logs)) / 40.0, 1.0)
        my_abil, opp_abil, my_srch, opp_srch, my_draw, opp_draw = 0, 0, 0, 0, 0, 0
        opp_rev_pkmn, opp_rev_sup, opp_rev_item, opp_rev_eng = 0, 0, 0, 0
        for ev in logs[-32:]:
            if not isinstance(ev, dict):
                continue
            ev_p = safe_int(ev.get("playerIndex", ev.get("player", -1)), -1)
            ev_t = safe_int(ev.get("type", 0), 0)
            ev_cid = safe_int(ev.get("cardId", ev.get("id", 0)), 0)
            if ev_t in (10, 4):  # Ability / Skill
                if ev_p == my_idx:
                    my_abil += 1
                elif ev_p >= 0:
                    opp_abil += 1
            elif ev_t in (7, 1):  # Play / Search
                if ev_p == my_idx:
                    my_srch += 1
                elif ev_p >= 0:
                    opp_srch += 1
            elif ev_t == 2:  # Draw
                if ev_p == my_idx:
                    my_draw += 1
                elif ev_p >= 0:
                    opp_draw += 1
            if ev_p == (1 - my_idx):
                ev_cards = ev.get("cards") or ([ev_cid] if ev_cid > 0 else [])
                if isinstance(ev_cards, list):
                    for rc in ev_cards:
                        rc_id = safe_int(rc.get("id", rc.get("cardId", 0)) if isinstance(rc, dict) else rc, 0)
                        if rc_id > 0:
                            r_inf = _CARD_INFO_MAP.get(rc_id, {})
                            if r_inf.get("hp", 0) > 0:
                                opp_rev_pkmn += 1
                            elif r_inf.get("cardType") == 3:
                                opp_rev_sup += 1
                            elif r_inf.get("cardType") in (5, 6):
                                opp_rev_eng += 1
                            else:
                                opp_rev_item += 1
        global_vec[163] = min(my_abil / 6.0, 1.0)
        global_vec[164] = min(opp_abil / 6.0, 1.0)
        global_vec[165] = min(my_srch / 6.0, 1.0)
        global_vec[166] = min(opp_srch / 6.0, 1.0)
        global_vec[167] = min(my_draw / 6.0, 1.0)
        global_vec[168] = min(opp_draw / 6.0, 1.0)
        global_vec[169] = min(opp_rev_pkmn / 6.0, 1.0)
        global_vec[170] = min(opp_rev_sup / 4.0, 1.0)
        global_vec[171] = min(opp_rev_item / 6.0, 1.0)
        global_vec[172] = min(opp_rev_eng / 6.0, 1.0)

    # --- 2. Object Tokens (Table 2 & Table 3) ---
    num_obj = MAX_TOKENS - 1  # 140
    obj_features = np.zeros((num_obj, OBJECT_FEATURE_DIM), dtype=np.float32)
    card_ids = np.zeros(num_obj, dtype=np.int64)
    evo_ids = np.zeros(num_obj, dtype=np.int64)
    token_types = np.zeros(num_obj, dtype=np.int64)
    option_mask = np.zeros(num_obj, dtype=bool)
    valid_action_mask = np.zeros(num_obj, dtype=bool)
    padding_mask = np.ones(MAX_TOKENS, dtype=bool)

    padding_mask[0] = False  # Global token is always active
    token_idx = 0

    def _populate_pokemon_instance_cols(row: np.ndarray, p_dict: dict, is_active: bool, slot_idx: int = 0):
        """Writes detailed live Pokemon state into cols 116..154 of a state or option token."""
        if not isinstance(p_dict, dict) or not p_dict:
            return
        cid = safe_int(p_dict.get("id", p_dict.get("cardId", 0)), 0)
        hp = safe_float(p_dict.get("hp"), 0.0)
        max_hp = max(safe_float(p_dict.get("maxHp"), hp), 1.0)
        dmg = max(0.0, max_hp - hp)
        appear = bool(p_dict.get("appearThisTurn", False))
        e_count, e_hist = _count_pokemon_energies(p_dict)
        tools = p_dict.get("tools") or []
        has_tool = safe_len(tools) > 0
        info = _CARD_INFO_MAP.get(cid, {})
        retreat_cost = float(info.get("retreatCost", 0))

        row[116] = 1.0 if is_active else 0.0
        row[117] = 1.0 if appear else 0.0
        row[118] = hp / 300.0
        row[119] = hp / max_hp
        row[120] = dmg / 300.0
        row[121] = 1.0 if dmg > 0 else 0.0
        row[122] = max_hp / 350.0
        row[123] = float(slot_idx) / 5.0
        row[124] = 1.0 if hp <= 30.0 and hp > 0 else 0.0
        row[125] = 1.0 if hp <= 60.0 and hp > 0 else 0.0
        row[126] = _get_prize_yield(cid) / 3.0
        row[127] = 1.0 if (not appear and turn_num >= 2) else 0.0  # Can evolve this turn
        row[128] = float(e_count) / 10.0
        row[129] = 1.0 if has_tool else 0.0

        # Energy type histogram (cols 130..139)
        row[130:140] = np.minimum(e_hist / 5.0, 1.0)

        # Stage & rule-box flags (cols 140..146)
        row[140] = 1.0 if info.get("basic") else 0.0
        row[141] = 1.0 if info.get("stage1") else 0.0
        row[142] = 1.0 if info.get("stage2") else 0.0
        row[143] = 1.0 if info.get("ex") else 0.0
        row[144] = 1.0 if info.get("megaEx") else 0.0
        row[145] = 1.0 if info.get("tera") else 0.0
        row[146] = retreat_cost / 4.0

        # Primary attack energy deficit (col 147)
        atks = info.get("attacks") or []
        if atks:
            max_cost = max(get_attack_info(aid).get("cost_len", 0) for aid in atks)
            row[147] = float(max(0, max_cost - e_count)) / 4.0
            row[148] = 1.0 if e_count >= max_cost and max_cost > 0 else 0.0

    def add_card_token(
        cid: int,
        evo_id: int,
        is_mine: bool,
        is_in_play: bool,
        is_in_hand: bool,
        is_active: bool,
        token_type_code: int,
        p_dict: dict = None,
        slot_idx: int = 0
    ):
        nonlocal token_idx
        if token_idx >= num_obj:
            return

        cid_clamped = min(max(safe_int(cid, 0), 0), NUM_CARD_IDS - 1)
        card_ids[token_idx] = cid_clamped
        evo_ids[token_idx] = min(max(safe_int(evo_id, 0), 0), NUM_CARD_IDS - 1)
        token_types[token_idx] = token_type_code

        # Role features (dims 0-3)
        obj_features[token_idx, 0] = 1.0 if is_mine else 0.0
        obj_features[token_idx, 1] = 1.0 if is_in_play else 0.0
        obj_features[token_idx, 2] = 0.0  # not option
        obj_features[token_idx, 3] = 1.0 if is_in_hand else 0.0

        if cid_clamped > 0:
            obj_features[token_idx, 10:35] = _CARD_STATIC_TABLE[cid_clamped]
            obj_features[token_idx, 35:99] = _CARD_TEXT_TABLE[cid_clamped]

        if is_in_play and p_dict is not None:
            _populate_pokemon_instance_cols(obj_features[token_idx], p_dict, is_active=is_active, slot_idx=slot_idx)

        padding_mask[token_idx + 1] = False
        token_idx += 1

    def add_pokemon_token(p_entry, is_mine: bool, is_active: bool, token_type_code: int, slot_idx: int = 0):
        p_dict = _extract_pokemon_dict(p_entry)
        if not p_dict:
            return

        c_id = safe_int(p_dict.get("cardId", p_dict.get("id", 0)), 0)
        pre_evo = p_dict.get("preEvolution") or []
        evo_id = 0
        if isinstance(pre_evo, list) and len(pre_evo) > 0:
            last_pre = pre_evo[-1]
            if isinstance(last_pre, dict):
                evo_id = safe_int(last_pre.get("id", last_pre.get("cardId", 0)), 0)
            elif isinstance(last_pre, (int, float)):
                evo_id = int(last_pre)

        add_card_token(
            cid=c_id,
            evo_id=evo_id,
            is_mine=is_mine,
            is_in_play=True,
            is_in_hand=False,
            is_active=is_active,
            token_type_code=token_type_code,
            p_dict=p_dict,
            slot_idx=slot_idx
        )

    # 1. In-Play Pokémon: Our Active + Bench
    add_pokemon_token(my_state.get("active"), is_mine=True, is_active=True, token_type_code=TOKEN_TYPE_ACTIVE_MINE, slot_idx=0)
    if isinstance(bench_my, list):
        for b_i, b in enumerate(bench_my):
            add_pokemon_token(b, is_mine=True, is_active=False, token_type_code=TOKEN_TYPE_BENCH_MINE, slot_idx=b_i)

    # 2. In-Play Pokémon: Opponent Active + Bench
    add_pokemon_token(opp_state.get("active"), is_mine=False, is_active=True, token_type_code=TOKEN_TYPE_ACTIVE_OPP, slot_idx=0)
    if isinstance(bench_opp, list):
        for b_i, b in enumerate(bench_opp):
            add_pokemon_token(b, is_mine=False, is_active=False, token_type_code=TOKEN_TYPE_BENCH_OPP, slot_idx=b_i)

    # Build fast lookup of our in-play Pokemon and hand cards for evolution & duplicate synergy
    my_board_by_cid = {}
    for p_entry in ([my_state.get("active")] + (bench_my if isinstance(bench_my, list) else [])):
        pd = _extract_pokemon_dict(p_entry)
        if pd:
            pcid = safe_int(pd.get("cardId", pd.get("id", 0)), 0)
            if pcid > 0:
                my_board_by_cid.setdefault(pcid, []).append(pd)

    hand_my = my_state.get("hand") or []
    my_hand_counts = {}
    my_hand_pre_evos = set()
    if isinstance(hand_my, list):
        for card_item in hand_my:
            if isinstance(card_item, dict):
                h_cid = safe_int(card_item.get("id", card_item.get("cardId", 0)), 0)
                if h_cid > 0:
                    my_hand_counts[h_cid] = my_hand_counts.get(h_cid, 0) + 1
                    for pid in (_CARD_INFO_MAP.get(h_cid, {}).get("pre_evo_ids") or []):
                        my_hand_pre_evos.add(pid)

    # 3. Hand Cards (Our private cards in hand)
    if isinstance(hand_my, list):
        for h_i, card_item in enumerate(hand_my):
            if not isinstance(card_item, dict):
                continue
            c_id = safe_int(card_item.get("id", card_item.get("cardId", 0)), 0)
            c_pre_ids = _CARD_INFO_MAP.get(c_id, {}).get("pre_evo_ids") or []
            h_evo_id = 0
            for pid in c_pre_ids:
                if pid in my_board_by_cid:
                    h_evo_id = pid
                    break
            if h_evo_id == 0 and c_pre_ids:
                h_evo_id = c_pre_ids[0]
            add_card_token(
                cid=c_id,
                evo_id=h_evo_id,
                is_mine=True,
                is_in_play=False,
                is_in_hand=True,
                is_active=False,
                token_type_code=TOKEN_TYPE_HAND_MINE,
                slot_idx=h_i
            )

    # 4. Stadium in play
    stadium_entry = current.get("stadium") or []
    s_dict = stadium_entry[0] if isinstance(stadium_entry, list) and len(stadium_entry) > 0 else stadium_entry
    s_id = 0
    if isinstance(s_dict, dict):
        s_id = safe_int(s_dict.get("id", s_dict.get("cardId", 0)), 0)
        s_owner = safe_int(s_dict.get("playerIndex"), my_idx)
        add_card_token(
            cid=s_id,
            evo_id=0,
            is_mine=(s_owner == my_idx),
            is_in_play=True,
            is_in_hand=False,
            is_active=False,
            token_type_code=TOKEN_TYPE_STADIUM
        )

    # Helper to look up an in-play Pokemon dict by (player_idx, area_code, slot_idx)
    def _get_in_play_pokemon(p_idx: int, area_c: int, slot_i: int) -> dict:
        p_st = (players[p_idx] if 0 <= p_idx < len(players) else None) or {}
        if area_c == 4:  # ACTIVE
            return _extract_pokemon_dict(p_st.get("active"))
        elif area_c == 5:  # BENCH
            b_list = p_st.get("bench") or []
            if isinstance(b_list, list) and 0 <= slot_i < len(b_list):
                return _extract_pokemon_dict(b_list[slot_i])
        return {}

    # 5. Legal Options from select['option']
    options = request.get("option") or []
    if isinstance(options, list):
        for opt_order_i, opt in enumerate(options):
            if token_idx >= num_obj:
                break
            if not isinstance(opt, dict):
                continue

            opt_type = safe_int(opt.get("type"), 0)
            raw_p = opt.get("playerIndex")
            opt_player = my_idx if raw_p is None else safe_int(raw_p, my_idx)
            area_code = safe_int(opt.get("area"), 0)
            idx_code = safe_int(opt.get("index"), 0)
            c_id = safe_int(opt.get("cardId", opt.get("id", 0)), 0)
            atk_id = safe_int(opt.get("attackId"), 0)
            in_play_area = safe_int(opt.get("inPlayArea"), 0)
            in_play_idx = safe_int(opt.get("inPlayIndex"), 0)

            target_pkmn_dict = {}
            target_is_active = False
            target_slot_idx = 0
            target_is_mine = (opt_player == my_idx)

            # 1. Resolve source / action card ID when not explicitly serialized
            if not c_id:
                if opt_type in (7, 8, 9):  # PLAY, ATTACH, EVOLVE
                    if area_code in (2, 0):  # Hand
                        hand = my_state.get("hand") or []
                        if 0 <= idx_code < len(hand) and isinstance(hand[idx_code], dict):
                            c_id = safe_int(hand[idx_code].get("id", hand[idx_code].get("cardId", 0)), 0)
                    elif area_code == 3:  # Discard
                        disc = my_state.get("discard") or []
                        if 0 <= idx_code < len(disc) and isinstance(disc[idx_code], dict):
                            c_id = safe_int(disc[idx_code].get("id", disc[idx_code].get("cardId", 0)), 0)
                    elif area_code == 1:  # Deck (e.g. TM Evolution, Punk Up, Salvatore)
                        deck_cards = request.get("deck") or request.get("cards") or []
                        if 0 <= idx_code < len(deck_cards) and isinstance(deck_cards[idx_code], dict):
                            c_id = safe_int(deck_cards[idx_code].get("id", deck_cards[idx_code].get("cardId", 0)), 0)
                elif opt_type in (12, 13):  # RETREAT or ATTACK from active Pokemon
                    c_id = safe_int(my_act_dict.get("id", my_act_dict.get("cardId", 0)), 0)
                    target_pkmn_dict = my_act_dict
                    target_is_active = True
                    target_is_mine = True
                elif opt_type in (10, 11):  # ABILITY or DISCARD from in-play (Active=4, Bench=5, Stadium=7)
                    if area_code in (4, 5):
                        target_pkmn_dict = _get_in_play_pokemon(opt_player, area_code, idx_code)
                        target_is_active = (area_code == 4)
                        target_slot_idx = idx_code if area_code == 5 else 0
                        c_id = safe_int(target_pkmn_dict.get("id", target_pkmn_dict.get("cardId", 0)), 0)
                    elif area_code == 7:  # Stadium Ability (e.g. Spikemuth Gym!)
                        if isinstance(s_dict, dict):
                            c_id = safe_int(s_dict.get("id", s_dict.get("cardId", 0)), 0)
                elif opt_type in (4, 5, 6):  # TOOL_CARD (4), ENERGY_CARD (5), ENERGY (6)
                    p_host = _get_in_play_pokemon(opt_player, area_code, idx_code)
                    target_pkmn_dict = p_host
                    target_is_active = (area_code == 4)
                    target_slot_idx = idx_code if area_code == 5 else 0
                    if p_host:
                        if opt_type == 4:
                            tools = p_host.get("tools") or []
                            t_idx = safe_int(opt.get("toolIndex"), 0)
                            if 0 <= t_idx < len(tools):
                                t_card = tools[t_idx]
                                c_id = safe_int(t_card.get("id", t_card.get("cardId", 0)) if isinstance(t_card, dict) else t_card, 0)
                        elif opt_type == 5:  # ENERGY_CARD
                            energy_cards = p_host.get("energyCards") or []
                            e_idx = safe_int(opt.get("energyIndex"), 0)
                            if 0 <= e_idx < len(energy_cards):
                                e_card = energy_cards[e_idx]
                                c_id = safe_int(e_card.get("id", e_card.get("cardId", 0)) if isinstance(e_card, dict) else e_card, 0)
                        elif opt_type == 6:  # ENERGY unit
                            energy_cards = p_host.get("energyCards") or []
                            e_idx = safe_int(opt.get("energyIndex"), 0)
                            if 0 <= e_idx < len(energy_cards):
                                e_card = energy_cards[e_idx]
                                c_id = safe_int(e_card.get("id", e_card.get("cardId", 0)) if isinstance(e_card, dict) else e_card, 0)
                            if not c_id:
                                c_id = safe_int(p_host.get("id", p_host.get("cardId", 0)), 0)
                elif opt_type == 3:  # CARD target
                    p_target = (players[opt_player] if 0 <= opt_player < len(players) else None) or {}
                    if area_code == 1:  # Deck search
                        deck_cards = request.get("deck") or request.get("cards") or []
                        if 0 <= idx_code < len(deck_cards) and isinstance(deck_cards[idx_code], dict):
                            c_id = safe_int(deck_cards[idx_code].get("id", deck_cards[idx_code].get("cardId", 0)), 0)
                    elif area_code == 2:  # Hand
                        hand = p_target.get("hand") or []
                        if 0 <= idx_code < len(hand) and isinstance(hand[idx_code], dict):
                            c_id = safe_int(hand[idx_code].get("id", hand[idx_code].get("cardId", 0)), 0)
                    elif area_code == 3:  # Discard
                        disc = p_target.get("discard") or []
                        if 0 <= idx_code < len(disc) and isinstance(disc[idx_code], dict):
                            c_id = safe_int(disc[idx_code].get("id", disc[idx_code].get("cardId", 0)), 0)
                    elif area_code in (4, 5):  # Active or Bench Pokemon!
                        target_pkmn_dict = _get_in_play_pokemon(opt_player, area_code, idx_code)
                        target_is_active = (area_code == 4)
                        target_slot_idx = idx_code if area_code == 5 else 0
                        c_id = safe_int(target_pkmn_dict.get("id", target_pkmn_dict.get("cardId", 0)), 0)
                    elif area_code == 6:  # Prize card
                        prizes = p_target.get("prize") or []
                        if 0 <= idx_code < len(prizes) and isinstance(prizes[idx_code], dict):
                            c_id = safe_int(prizes[idx_code].get("id", prizes[idx_code].get("cardId", 0)), 0)
                    elif area_code == 7:  # Stadium
                        if isinstance(s_dict, dict):
                            c_id = safe_int(s_dict.get("id", s_dict.get("cardId", 0)), 0)
                    elif area_code == 12:  # Looking
                        looking = current.get("looking") or request.get("cards") or []
                        if 0 <= idx_code < len(looking) and isinstance(looking[idx_code], dict):
                            c_id = safe_int(looking[idx_code].get("id", looking[idx_code].get("cardId", 0)), 0)

            cid_clamped = min(max(int(c_id), 0), NUM_CARD_IDS - 1)
            card_ids[token_idx] = cid_clamped
            token_types[token_idx] = TOKEN_TYPE_OPTION

            # Role features
            obj_features[token_idx, 0] = 1.0 if opt_player == my_idx else 0.0
            obj_features[token_idx, 2] = 1.0  # is_option

            if cid_clamped > 0:
                obj_features[token_idx, 10:35] = _CARD_STATIC_TABLE[cid_clamped]
                obj_features[token_idx, 35:99] = _CARD_TEXT_TABLE[cid_clamped]

            # Option action details (dims 268-288)
            if 0 <= opt_type < 17:
                obj_features[token_idx, 268 + opt_type] = 1.0
            obj_features[token_idx, 286] = 1.0 if opt_player == my_idx else 0.0
            obj_features[token_idx, 287] = float(area_code) / 10.0
            obj_features[token_idx, 288] = float(idx_code) / 10.0

            # 2. Target Pokemon Binding for ATTACH (8) and EVOLVE (9) + In-Play targets
            if opt_type in (8, 9) and in_play_area in (4, 5):
                target_pkmn_dict = _get_in_play_pokemon(my_idx, in_play_area, in_play_idx)
                target_is_active = (in_play_area == 4)
                target_slot_idx = in_play_idx if in_play_area == 5 else 0
                target_is_mine = True

            if target_pkmn_dict:
                # Populate live instance features (cols 116..148) directly onto the option token!
                _populate_pokemon_instance_cols(
                    obj_features[token_idx],
                    target_pkmn_dict,
                    is_active=target_is_active,
                    slot_idx=target_slot_idx
                )
                target_cid = safe_int(target_pkmn_dict.get("id", target_pkmn_dict.get("cardId", 0)), 0)
                t_hp = safe_float(target_pkmn_dict.get("hp"), 0.0)
                t_mhp = max(safe_float(target_pkmn_dict.get("maxHp"), t_hp), 1.0)
                t_dmg = max(0.0, t_mhp - t_hp)
                t_e_cnt, _ = _count_pokemon_energies(target_pkmn_dict)
                t_has_tool = safe_len(target_pkmn_dict.get("tools", [])) > 0

                obj_features[token_idx, 290] = 1.0 if target_is_active else 0.0
                obj_features[token_idx, 291] = float(target_slot_idx) / 5.0
                obj_features[token_idx, 292] = t_hp / 300.0
                obj_features[token_idx, 293] = float(t_e_cnt) / 10.0
                obj_features[token_idx, 294] = 1.0 if t_has_tool else 0.0
                obj_features[token_idx, 295] = t_dmg / 300.0

                # Tactical context flags on target Pokemon (cols 311..320)
                obj_features[token_idx, 311] = 1.0 if t_dmg > 0 else 0.0
                obj_features[token_idx, 312] = t_hp / t_mhp
                obj_features[token_idx, 313] = 1.0 if target_is_mine else 0.0
                obj_features[token_idx, 314] = 1.0 if (0 < t_hp <= 10.0) else 0.0
                obj_features[token_idx, 315] = 1.0 if (0 < t_hp <= 20.0) else 0.0
                obj_features[token_idx, 316] = 1.0 if (0 < t_hp <= 30.0) else 0.0
                obj_features[token_idx, 317] = 1.0 if (0 < t_hp <= 60.0) else 0.0
                obj_features[token_idx, 318] = _get_prize_yield(target_cid) / 3.0

                evo_ids[token_idx] = min(max(int(target_cid), 0), NUM_CARD_IDS - 1)
            elif cid_clamped > 0:
                # Deck / Hand / Discard option (e.g. Spikemuth Gym, Poke Pad, Bug Catching Set, Night Stretcher):
                # Bind pre_evo_id into evo_ids so Relational Attention (opt_evolves_from_state) fires!
                c_info = _CARD_INFO_MAP.get(cid_clamped, {})
                c_pre_ids = c_info.get("pre_evo_ids") or []
                matched_board_pid = 0
                can_evolve_now = False
                for pid in c_pre_ids:
                    if pid in my_board_by_cid:
                        matched_board_pid = pid
                        for b_pkmn in my_board_by_cid[pid]:
                            if (not b_pkmn.get("appearThisTurn", False) and turn_num >= 2) or (s_id == 1261 and c_info.get("energyType") == 1):
                                can_evolve_now = True
                        break
                if matched_board_pid > 0:
                    evo_ids[token_idx] = min(max(int(matched_board_pid), 0), NUM_CARD_IDS - 1)
                    obj_features[token_idx, 326] = 1.0  # Evolves a Pokemon currently on our board!
                    obj_features[token_idx, 327] = 1.0 if can_evolve_now else 0.0  # Can evolve immediately this turn!
                elif c_pre_ids:
                    evo_ids[token_idx] = min(max(int(c_pre_ids[0]), 0), NUM_CARD_IDS - 1)
                if cid_clamped in my_hand_pre_evos:
                    obj_features[token_idx, 328] = 1.0  # Pre-evolution of a card already in our hand!
                existing_copies = len(my_board_by_cid.get(cid_clamped, [])) + my_hand_counts.get(cid_clamped, 0)
                obj_features[token_idx, 329] = min(float(existing_copies) / 4.0, 1.0)

            # 3. Combat Physics for ATTACK options (dims 289, 296-305, 321-326)
            if opt_type == 13 and atk_id:
                atk_info = get_attack_info(atk_id)
                raw_dmg, bench_dmg, ignores_wr = compute_dynamic_attack_damage(
                    atk_id, my_state, opp_state, my_act_dict, opp_act_dict
                )

                opp_cid = safe_int(opp_act_dict.get("id", opp_act_dict.get("cardId", 0)), 0)
                opp_hp = safe_float(opp_act_dict.get("hp"), 0.0)
                opp_card_info = _CARD_INFO_MAP.get(opp_cid, {})

                my_cid = safe_int(my_act_dict.get("id", my_act_dict.get("cardId", 0)), 0)
                my_card_info = _CARD_INFO_MAP.get(my_cid, {})
                attacker_type = safe_int(my_card_info.get("energyType"), 0)

                # Official Rules Weakness (x2) & Resistance (-30)
                is_weakness = (not ignores_wr) and (opp_card_info.get("weakness") == attacker_type and attacker_type > 0)
                is_res = (not ignores_wr) and (opp_card_info.get("resistance") == attacker_type and attacker_type > 0)

                eff_dmg = raw_dmg * (2.0 if is_weakness else 1.0)
                if is_res and raw_dmg > 0:
                    eff_dmg = max(0.0, eff_dmg - 30.0)

                is_lethal = 1.0 if (eff_dmg >= opp_hp and opp_hp > 0) else 0.0
                prize_yield = _get_prize_yield(opp_cid) if is_lethal else 0.0

                obj_features[token_idx, 289] = raw_dmg / 300.0
                obj_features[token_idx, 296] = raw_dmg / 300.0
                obj_features[token_idx, 297] = 1.0 if is_weakness else 0.0
                obj_features[token_idx, 298] = 1.0 if is_res else 0.0
                obj_features[token_idx, 299] = eff_dmg / 300.0
                obj_features[token_idx, 300] = is_lethal
                obj_features[token_idx, 301] = prize_yield / 3.0
                obj_features[token_idx, 302] = 1.0 if opp_card_info.get("tera") else 0.0
                obj_features[token_idx, 303] = bench_dmg / 200.0
                obj_features[token_idx, 304] = float(atk_info.get("cost_len", 0)) / 5.0
                obj_features[token_idx, 305] = atk_info.get("has_bench_dmg", 0.0)
                obj_features[token_idx, 321] = atk_info.get("has_draw", 0.0)
                obj_features[token_idx, 322] = atk_info.get("has_search", 0.0)
                obj_features[token_idx, 323] = atk_info.get("has_status", 0.0)
                obj_features[token_idx, 324] = 1.0 if (is_lethal and prize_yield >= my_prizes) else 0.0  # Game-winning attack!

            # 4. Extra option fields (dims 306-310)
            if opt_type == 0:  # NUMBER
                num_val = safe_float(opt.get("number"), 0.0)
                obj_features[token_idx, 306] = num_val / 10.0
                obj_features[token_idx, 325] = 1.0 if num_val == 0.0 else 0.0
            if opt.get("energyIndex") is not None:
                obj_features[token_idx, 307] = safe_float(opt.get("energyIndex"), 0.0) / 10.0
            if opt.get("count") is not None:
                obj_features[token_idx, 308] = safe_float(opt.get("count"), 0.0) / 10.0
            if opt.get("toolIndex") is not None:
                obj_features[token_idx, 309] = safe_float(opt.get("toolIndex"), 0.0) / 10.0
            if opt.get("specialConditionType") is not None:
                obj_features[token_idx, 310] = safe_float(opt.get("specialConditionType"), 0.0) / 5.0

            option_mask[token_idx] = True
            valid_action_mask[token_idx] = True
            padding_mask[token_idx + 1] = False

            token_idx += 1

    return {
        "global_vec": global_vec,
        "obj_features": obj_features,
        "card_ids": card_ids,
        "evo_ids": evo_ids,
        "token_types": token_types,
        "option_mask": option_mask,
        "valid_action_mask": valid_action_mask,
        "padding_mask": padding_mask,
    }