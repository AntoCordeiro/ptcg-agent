# search_agent.py
"""
Tactical Lookahead & Verification Search Agent (v3 - Zero-Noise Neural-Guided Search).

Design Philosophy:
In imperfect-information card games (PTCG), rolling out full turns with a fast heuristic sub-policy
over 2 random deck shuffles introduces determinization and rollout noise that can override the
high-accuracy Dual-Encoder Cross-Attention policy network (`PTCGDualCrossNet`).

SearchAgent v3 uses the C++ forward simulator (`cg.api.search_begin` / `search_step`) strictly where
forward physics are exact and deterministic, while preserving the learned top-Elo macro policy:
1. Immediate Lethal Win Detection (`result == y_idx`):
   - Whenever a legal ATTACK is available in MAIN phase, simulates the attack line in C++.
   - If the attack immediately wins the game (takes final prizes / KOs opponent's last Pokemon),
     executes it immediately. Otherwise, lets the neural policy complete all pre-attack setup
     (Evolve, Abilities, Supporters, Energy Attachment) before attacking.
2. Self-Loss / Deck-Out Trap Veto (`result == 1 - y_idx`):
   - Verifies the top neural candidate in C++ before execution when deck size is low or risk exists;
     if an action causes an immediate loss (e.g. self-deck-out or emptying board), vetoes it.
3. Deterministic Combat & Target Lookahead:
   - On close-tie tactical decisions (`ATTACK` selection `sel_type == 6`, `ACTIVE` promotion `ctx == 3`,
     bench snipe / damage counter placement `ctx in (13, 14)`, or close `MAIN` phase ties where
     `p(top1) - p(top2) < 0.12`), evaluates the exact post-action C++ board state with batched
     `PTCGDualCrossNet` Value + Prize-Margin heads.
"""

import math
import os
import sys
from dataclasses import asdict
import random
import time
from typing import List, Optional, Tuple
import numpy as np
import torch

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)
SUBMISSION_DIR = os.path.abspath(
    "the-pokemon-company-ptcg-ai-battle-challenge-playground/sample_submission/sample_submission/sample_submission"
)
if os.path.exists(SUBMISSION_DIR) and SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, SUBMISSION_DIR)

from cg import api
from card_tracker import CardTracker, get_card_type, TYPE_BASIC_POKEMON
from card_static_table import get_card_info_map
from feature_parser import (
    parse_game_state_to_tensors,
    get_attack_info,
)
from ptcg_agents import choose_multi_select_count, count_bounds, apply_tactical_prior_shaping
from model_dual_cross import PTCGDualCrossNet

_CARD_INFO = get_card_info_map(1500)


def _estimate_attack_damage_obj(aid: int, me, opp) -> float:
    """Computes effective attack damage (including dynamic scaling attacks) on C++ Player objects."""
    atk = get_attack_info(int(aid))
    base_dmg = float(atk.get("damage", 0) or 0)
    if base_dmg > 0:
        return base_dmg

    my_act = (me.active[0] if (me and me.active and len(me.active) > 0) else None)
    opp_act = (opp.active[0] if (opp and opp.active and len(opp.active) > 0) else None)
    my_bench = (me.bench or []) if me else []
    opp_bench = (opp.bench or []) if opp else []

    if aid == 1072:  # Alakazam: Powerful Hand (20x cards in hand)
        h_cnt = float(getattr(me, "handCount", 0) or len(getattr(me, "hand", None) or []))
        return 20.0 * h_cnt
    if aid == 72:  # Raging Bolt ex: Bellowing Thunder (70x energy attached to your Pokemon)
        tot_e = sum(len(getattr(p, "energies", None) or []) for p in ([my_act] if my_act else []) + list(my_bench) if p is not None)
        return 70.0 * tot_e
    if aid == 120:  # Teal Mask Ogerpon ex: Myriad Leaf Shower (30 + 30x active energies)
        my_e = len(getattr(my_act, "energies", None) or []) if my_act else 0
        opp_e = len(getattr(opp_act, "energies", None) or []) if opp_act else 0
        return 30.0 + 30.0 * (my_e + opp_e)
    if aid == 115:  # Do the Wave (20x benched)
        return 20.0 * len([p for p in my_bench if p is not None])
    if aid == 1240:  # Resentful Refrain (20x discard Pokemon)
        disc = getattr(me, "discard", None) or []
        p_disc = sum(1 for c in disc if _CARD_INFO.get(int(getattr(c, "id", 0) or 0), {}).get("hp", 0) > 0)
        return 20.0 * p_disc
    if aid == 1225:  # Gale Thrust
        return 150.0
    if aid == 371:  # Full Moon Rondo
        return 20.0 + 20.0 * (len([p for p in my_bench if p is not None]) + len([p for p in opp_bench if p is not None]))
    if aid == 195:  # Hydrapple ex: Syrup Storm
        tot_e = sum(len(getattr(p, "energies", None) or []) for p in ([my_act] if my_act else []) + list(my_bench) if p is not None)
        return 30.0 + 30.0 * tot_e
    if aid == 1079:  # Mega Gardevoir ex: Mega Symphonia
        tot_e = sum(len(getattr(p, "energies", None) or []) for p in ([my_act] if my_act else []) + list(my_bench) if p is not None)
        return 50.0 * tot_e
    if aid == 540:  # Raging Curse
        if my_act:
            dmg_self = max(0.0, float((my_act.maxHp or 0) - (my_act.hp or 0)))
            return min(350.0, 20.0 * (dmg_self / 10.0))
        return 80.0
    if aid == 188:  # Trifrost
        return 330.0
    if aid == 305:  # Destined Fight
        return 140.0
    return base_dmg


def _total_board_damage(p_state) -> float:
    """Computes total damage currently sitting on a player's in-play Pokemon."""
    if not p_state:
        return 0.0
    dmg = 0.0
    act = getattr(p_state, "active", None) or (p_state.get("active") if isinstance(p_state, dict) else []) or []
    bench = getattr(p_state, "bench", None) or (p_state.get("bench") if isinstance(p_state, dict) else []) or []
    all_p = (act if isinstance(act, list) else [act]) + (bench if isinstance(bench, list) else [bench])
    for p in all_p:
        if p is None:
            continue
        hp = float(getattr(p, "hp", None) or (p.get("hp", 0) if isinstance(p, dict) else 0) or 0)
        mhp = float(getattr(p, "maxHp", None) or (p.get("maxHp", hp) if isinstance(p, dict) else hp) or hp)
        dmg += max(0.0, mhp - hp)
    return dmg


class SearchAgent:
    """
    SearchAgent v4 (Approach D): Batched Neural Value + Q-Critic 2-Ply Lookahead & Verification Search.
    """
    name = "search_agent"

    def __init__(
        self,
        model_path: str = "ptcg_wise_v3.pt",
        device: Optional[torch.device] = None,
        num_determinizations: int = 2,
        c_puct: float = 0.35,
        search_all_main_turns: bool = True,
        name: str = "search_v4",
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_determinizations = num_determinizations
        self.c_puct = c_puct
        self.search_all_main_turns = search_all_main_turns
        self.name = name

        self.model = PTCGDualCrossNet().to(self.device).eval()
        for candidate_path in (
            model_path,
            "ptcg_wise_v3.pt",
            "ptcg_iql_v3.pt",
            "ptcg_iql_relational.pt",
            "ptcg_wise_ensemble.pt",
            "ptcg_ppo_phase4_model.pt",
            "ptcg_dual_cross_bc.pt",
        ):
            if candidate_path and os.path.exists(candidate_path):
                self.model.load_state_dict(torch.load(candidate_path, map_location=self.device, weights_only=True), strict=False)
                break
        self.tracker: Optional[CardTracker] = None
        self.deck: List[int] = []

    def reset(self, deck: List[int]):
        self.deck = list(deck)
        self.tracker = CardTracker(deck)

    def _sample_determinization(self, obs: dict) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[int]]:
        cur = obs.get("current") or {}
        y_idx = cur.get("yourIndex", 0)
        players = cur.get("players", [{}, {}])
        p = players[y_idx]
        opp = players[1 - y_idx]

        deck_count = p.get("deckCount", 0) or 0
        prize_count = len(p.get("prize", []))
        opp_deck = opp.get("deckCount", 0) or 0
        opp_prize = len(opp.get("prize", []))
        opp_hand = opp.get("handCount", 0) or 0

        unobs = []
        if self.tracker:
            for cid, cnt in self.tracker.unobserved_counts.items():
                unobs.extend([cid] * cnt)
        else:
            unobs = list(self.deck)
        while len(unobs) < (deck_count + prize_count):
            unobs.extend(self.deck)
        random.shuffle(unobs)

        your_deck = unobs[:deck_count]
        your_prize = unobs[deck_count:deck_count + prize_count]

        total_opp_needed = opp_deck + opp_prize + opp_hand
        if self.tracker:
            opp_pool = self.tracker.infer_opponent_unobserved(total_opp_needed)
        else:
            opp_pool = list(self.deck) if self.deck else unobs
            while len(opp_pool) < total_opp_needed:
                opp_pool.extend(self.deck or unobs)
            random.shuffle(opp_pool)

        od = opp_pool[:opp_deck]
        op = opp_pool[opp_deck:opp_deck + opp_prize]
        oh = opp_pool[opp_deck + opp_prize:opp_deck + opp_prize + opp_hand]

        opp_active_cards = opp.get("active") or []
        opp_active_param = []
        if opp_active_cards and (opp_active_cards[0] is None or opp_active_cards[0] == 0):
            basics = [c for c in opp_pool if get_card_type(c) == TYPE_BASIC_POKEMON]
            opp_active_param = [basics[0]] if basics else [opp_pool[0]]

        return your_deck, your_prize, od, op, oh, opp_active_param

    @torch.no_grad()
    def _evaluate_states_batched(self, obs_dicts: List[dict], y_idx: int) -> List[float]:
        """Evaluates a batch of SearchState observations in a single forward pass."""
        if not obs_dicts:
            return []
        g_list, o_list, c_list, e_list, tt_list, opt_list, val_list, pad_list = [], [], [], [], [], [], [], []
        signs = []
        for d in obs_dicts:
            req = d.get("select") or {}
            cur = d.get("current") or {}
            seat = cur.get("yourIndex", y_idx)
            signs.append(1.0 if seat == y_idx else -1.0)
            parsed = parse_game_state_to_tensors(d, req, tracker=None)
            g_list.append(parsed["global_vec"])
            o_list.append(parsed["obj_features"])
            c_list.append(parsed["card_ids"])
            e_list.append(parsed["evo_ids"])
            tt_list.append(parsed["token_types"])
            opt_list.append(parsed["option_mask"])
            val_list.append(parsed["valid_action_mask"])
            pad_list.append(parsed["padding_mask"])

        t_stack = lambda arrs: torch.from_numpy(np.stack(arrs, axis=0)).to(self.device)
        _, _, vals, margins, _ = self.model(
            t_stack(g_list), t_stack(o_list), t_stack(c_list), t_stack(e_list),
            t_stack(opt_list), t_stack(val_list), t_stack(pad_list),
            token_types=t_stack(tt_list), return_aux=True
        )
        v_np = vals.squeeze(-1).cpu().numpy()
        m_np = margins.squeeze(-1).cpu().numpy()
        if v_np.ndim == 0:
            v_np = np.array([float(v_np)])
            m_np = np.array([float(m_np)])
        return [float(signs[i] * (1.2 * v_np[i] + 0.4 * m_np[i])) for i in range(len(obs_dicts))]

    @torch.no_grad()
    def _get_network_priors(self, obs: dict, req: dict):
        """Returns (action_probs, action_logits, q_vals, count_logit, opt_repr, opt_cids) from neural net."""
        parsed = parse_game_state_to_tensors(obs, req, tracker=self.tracker)
        t = lambda k: torch.from_numpy(parsed[k]).unsqueeze(0).to(self.device)

        _, logits, _, _, count_logits, q_vals, action_repr = self.model(
            t("global_vec"), t("obj_features"), t("card_ids"), t("evo_ids"),
            t("option_mask"), t("valid_action_mask"), t("padding_mask"),
            token_types=t("token_types"), return_q_and_repr=True
        )
        opt_idx = np.where(parsed["option_mask"])[0]
        s_t = logits[0, opt_idx].clone()
        s_t = apply_tactical_prior_shaping(obs, req, s_t)
        s = s_t.cpu().numpy()
        p = torch.softmax(s_t, dim=-1).cpu().numpy()
        q = q_vals[0, opt_idx].cpu().numpy()
        c_logit = float(count_logits[0].item())
        opt_r = action_repr[0, opt_idx]
        opt_cids = torch.from_numpy(parsed["card_ids"][opt_idx]).to(self.device)
        return p, s, q, c_logit, opt_r, opt_cids

    def _resolve_combat_substep(self, c_obs, y_idx: int) -> List[int]:
        """Resolves immediate attack/target sub-steps (e.g. attack selection, snipe target, energy discard)."""
        c_cur = c_obs.current
        c_sel = c_obs.select
        s_options = c_sel.option or []
        n_opts = len(s_options)
        if n_opts == 0:
            return [0]

        min_c = c_sel.minCount if c_sel.minCount is not None else 1
        max_c = c_sel.maxCount if c_sel.maxCount is not None else 1
        max_c = min(max_c, n_opts)
        min_c = min(min_c, max_c)

        s_type = c_sel.type
        ctx = c_sel.context if c_sel.context is not None else 0
        me = c_cur.players[y_idx]
        opp = c_cur.players[1 - y_idx]
        opp_act = (opp.active[0] if (opp.active and len(opp.active) > 0) else None)

        if s_type == 6:  # ATTACK selection
            best_i = max(range(n_opts), key=lambda i: _estimate_attack_damage_obj(s_options[i].attackId or 0, me, opp))
            return [best_i]
        if s_type == 9:  # YES/NO
            for i, o in enumerate(s_options):
                if o.type == 1:
                    return [i]
            return [0]
        if s_type == 8:  # COUNT
            best_i = max(range(n_opts), key=lambda i: s_options[i].number or 0)
            return [best_i]

        if ctx == 26 and min_c == 0 and max_c >= 2:  # Bellowing Thunder energy discard
            opp_hp = float(opp_act.hp if opp_act else 210)
            k_choice = min(max_c, max(1, int(math.ceil(opp_hp / 70.0))))
        elif ctx in (13, 14):  # Bench snipe / damage placement
            k_choice = max(min_c, 1)
            scored = []
            for i, o in enumerate(s_options):
                sc = 0.0
                p_owner = o.playerIndex if o.playerIndex is not None else y_idx
                if p_owner != y_idx:
                    t_p = opp_act if o.area == 4 else (opp.bench[o.index] if (o.area == 5 and o.index is not None and 0 <= o.index < len(opp.bench or [])) else None)
                    if t_p is not None:
                        hp = float(t_p.hp or 100)
                        if hp <= 30:
                            sc += 20.0  # Immediate KO!
                        elif hp <= 60:
                            sc += 8.0
                        sc += max(0.0, (200.0 - hp) / 100.0)
                scored.append((sc, -i, i))
            scored.sort(reverse=True)
            return [idx for _, _, idx in scored[:k_choice]]
        elif ctx in (8, 26, 27, 29, 30):
            k_choice = min_c
        else:
            k_choice = max(min_c, 1 if max_c >= 1 else 0)

        k_choice = max(min_c, min(k_choice, max_c, n_opts))
        return list(range(k_choice))

    def act(self, obs: dict) -> List[int]:
        req = obs.get("select") or {}
        options = req.get("option", [])
        if not options:
            return [0]
        if len(options) == 1:
            if self.tracker:
                try:
                    self.tracker.update(obs, req)
                except Exception:
                    pass
            mn, _ = count_bounds(req)
            return [0] if mn >= 1 else [0]

        sel_type = req.get("type", -1)
        ctx = req.get("context", 0) or 0

        # Mulligan prompt: always draw extra cards
        if ctx == 42:
            for idx, opt in enumerate(options):
                if opt.get("type") == 1:
                    return [idx]
            return [0]

        # 1. Get Neural Policy Priors, IQL Q-Critic & Multi-Select Count
        opt_r, opt_cids = None, None
        try:
            priors, net_scores, q_vals, count_logit, opt_r, opt_cids = self._get_network_priors(obs, req)
        except Exception:
            priors = np.ones(len(options), dtype=np.float32) / len(options)
            net_scores = np.zeros(len(options), dtype=np.float32)
            q_vals = np.zeros(len(options), dtype=np.float32)
            count_logit = 0.0

        attack_indices = [idx for idx, opt in enumerate(options) if opt.get("type") == 13] if sel_type == 0 else []

        s_tensor = torch.from_numpy(net_scores).to(self.device)
        k = choose_multi_select_count(
            obs, req, s_tensor, count_logit=count_logit, mode="auto"
        )
        k = min(k, len(options))
        if k <= 0:
            return []

        if k > 1 and opt_r is not None and hasattr(self.model, "select_multi_options_autoregressive"):
            return self.model.select_multi_options_autoregressive(s_tensor, opt_r, opt_cids, k, ctx=ctx)

        top_order = np.argsort(-net_scores).tolist()
        best_neural_action = [int(i) for i in top_order[:k]]

        # Only invoke C++ simulator on single-choice decisions (k == 1) where tactical verification applies
        if k != 1:
            return best_neural_action

        cur = obs.get("current") or {}
        y_idx = cur.get("yourIndex", 0)
        players = cur.get("players", [{}, {}])
        me_dict = players[y_idx]
        opp_dict = players[1 - y_idx]
        my_deck_cnt = int(me_dict.get("deckCount", 20) or 0)

        top1_idx = int(top_order[0])
        top2_idx = int(top_order[1]) if len(top_order) > 1 else top1_idx
        top1_prob = float(priors[top1_idx])
        top2_prob = float(priors[top2_idx]) if len(top_order) > 1 else 0.0

        # Check if we need C++ verification / 2-ply lookahead (Approach D):
        # Case A: Legal ATTACK exists in MAIN phase -> check if attacking immediately wins the game (lethal!)
        # Case B: Low deck count (<= 8) -> verify top1_idx doesn't cause self-deck-out loss
        # Case C: Combat/target choice (sel_type == 6 or ctx in (3, 13, 14, 15)) or close deterministic MAIN tie
        need_lethal_check = (sel_type == 0 and len(attack_indices) > 0 and top1_idx not in attack_indices)
        need_deckout_check = (sel_type == 0 and my_deck_cnt <= 8)
        need_combat_tiebreak = (
            ((sel_type == 6 or (sel_type == 1 and ctx in (3, 13, 14, 15))) and (top1_prob - top2_prob < 0.30))
            or (sel_type == 0 and (top1_prob - top2_prob < 0.15) and options[top1_idx].get("type") in (8, 9, 10, 12, 13))
        )

        if not (need_lethal_check or need_deckout_check or need_combat_tiebreak):
            return best_neural_action

        try:
            obs_obj = api.to_observation_class(obs)
            yd, yp, od, op, oh, oa = self._sample_determinization(obs)
            root = api.search_begin(obs_obj, yd, yp, od, op, oh, opponent_active=oa)
        except Exception:
            return best_neural_action

        try:
            # Case A: Immediate Lethal Attack Check
            if need_lethal_check:
                for atk_idx in attack_indices:
                    created = []
                    try:
                        node = api.search_step(root.searchId, [atk_idx])
                        created.append(node)
                        for _ in range(4):
                            c_cur = node.observation.current
                            if not c_cur or c_cur.result != -1 or c_cur.yourIndex != y_idx:
                                break
                            c_sel = node.observation.select
                            if not c_sel or not c_sel.option or c_sel.type == 0:
                                break
                            sub_act = self._resolve_combat_substep(node.observation, y_idx)
                            node = api.search_step(node.searchId, sub_act)
                            created.append(node)
                        if node.observation.current and node.observation.current.result == y_idx:
                            return [int(atk_idx)]
                    except Exception:
                        pass
                    finally:
                        for n in reversed(created):
                            try:
                                api.search_release(n.searchId)
                            except Exception:
                                pass

            # Case B: Self-Loss / Deck-Out Veto
            if need_deckout_check:
                for cand_idx in top_order[:4]:
                    created = []
                    try:
                        node = api.search_step(root.searchId, [int(cand_idx)])
                        created.append(node)
                        c_cur = node.observation.current
                        if c_cur and c_cur.result == (1 - y_idx):
                            continue  # Veto self-loss action!
                        return [int(cand_idx)]
                    except Exception:
                        return [int(cand_idx)]
                    finally:
                        for n in reversed(created):
                            try:
                                api.search_release(n.searchId)
                            except Exception:
                                pass

            # Case C (Approach D): Batched Neural Value + Q-Critic 2-Ply Lookahead
            if need_combat_tiebreak:
                initial_my_prizes = len(me_dict.get("prize", []))
                initial_opp_dmg = _total_board_damage(opp_dict)
                eval_candidates = []
                eval_obs_dicts = []
                eval_base_scores = []

                for cand_idx in top_order[:min(4, len(top_order))]:
                    if priors[cand_idx] < 0.10:
                        continue
                    created = []
                    try:
                        node = api.search_step(root.searchId, [int(cand_idx)])
                        created.append(node)
                        for _ in range(3):
                            c_cur = node.observation.current
                            if not c_cur or c_cur.result != -1 or c_cur.yourIndex != y_idx:
                                break
                            c_sel = node.observation.select
                            if not c_sel or not c_sel.option or c_sel.type == 0:
                                break
                            sub_act = self._resolve_combat_substep(node.observation, y_idx)
                            node = api.search_step(node.searchId, sub_act)
                            created.append(node)
                        c_cur = node.observation.current
                        if c_cur and c_cur.result == y_idx:
                            return [int(cand_idx)]
                        if c_cur and c_cur.result == (1 - y_idx):
                            continue
                        c_self = c_cur.players[y_idx] if c_cur else None
                        c_opp = c_cur.players[1 - y_idx] if c_cur else None
                        prizes_taken = (initial_my_prizes - len(c_self.prize)) if c_self else 0
                        dmg_dealt = (_total_board_damage(c_opp) - initial_opp_dmg) if c_opp else 0.0
                        base_sc = (
                            float(priors[cand_idx])
                            + 0.20 * float(q_vals[cand_idx])
                            + 2.0 * float(prizes_taken)
                            + 0.5 * (dmg_dealt / 100.0)
                        )
                        try:
                            post_dict = asdict(node.observation)
                            eval_obs_dicts.append(post_dict)
                        except Exception:
                            eval_obs_dicts.append(obs)
                        eval_candidates.append(int(cand_idx))
                        eval_base_scores.append(base_sc)
                    except Exception:
                        pass
                    finally:
                        for n in reversed(created):
                            try:
                                api.search_release(n.searchId)
                            except Exception:
                                pass

                if eval_candidates:
                    try:
                        post_vals = self._evaluate_states_batched(eval_obs_dicts, y_idx)
                    except Exception:
                        post_vals = [0.0] * len(eval_candidates)
                    best_cand = eval_candidates[0]
                    best_val = -1e9
                    for c_i, b_sc, p_v in zip(eval_candidates, eval_base_scores, post_vals):
                        tot_sc = b_sc + 0.30 * float(p_v)
                        if tot_sc > best_val:
                            best_val = tot_sc
                            best_cand = c_i
                    return [int(best_cand)]

        finally:
            try:
                api.search_release(root.searchId)
                api.search_end()
            except Exception:
                pass

        return best_neural_action
