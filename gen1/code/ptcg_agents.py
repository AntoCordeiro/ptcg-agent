# ptcg_agents.py
"""
Unified agent library used by the evaluation ladder, RL training, search agents, and Kaggle submission.
Renamed to ptcg_agents.py to prevent namespace collision with kaggle_environments.envs.lux_ai_s3.agents.
"""

import math
import os
import sys
import random
import numpy as np
import torch

SUBMISSION_DIR = os.path.abspath(
    "the-pokemon-company-ptcg-ai-battle-challenge-playground/sample_submission/sample_submission/sample_submission"
)
if SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, SUBMISSION_DIR)

# OptionType constants (cg.api.OptionType)
OPT_NUMBER, OPT_YES, OPT_NO, OPT_CARD = 0, 1, 2, 3
OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_ABILITY = 7, 8, 9, 10
OPT_DISCARD, OPT_RETREAT, OPT_ATTACK, OPT_END = 11, 12, 13, 14
# SelectContext constants
CTX_MULLIGAN = 42

_ATTACK_DAMAGE = None


def attack_damage_table() -> dict:
    global _ATTACK_DAMAGE
    if _ATTACK_DAMAGE is None:
        from cg import api
        _ATTACK_DAMAGE = {a.attackId: a.damage for a in api.all_attack()}
    return _ATTACK_DAMAGE


def count_bounds(request: dict):
    options = request.get("option", [])
    mn = request.get("minCount")
    mx = request.get("maxCount")
    mn = 1 if mn is None else int(mn)
    mx = 1 if mx is None else int(mx)
    mx = min(mx, len(options))
    mn = min(mn, mx)
    return mn, mx


def choose_multi_select_count(
    obs: dict,
    request: dict,
    opt_scores: torch.Tensor,
    count_logit: float = 0.0,
    mode: str = "auto"
) -> int:
    """
    Determines the optimal number of options k in [minCount, maxCount] to select.
    Combines exact PTCG rulebook mechanics with the learned multi-select count head.
    """
    options = request.get("option", [])
    mn, mx = count_bounds(request)
    if mx <= mn:
        return mn
    if mode == "max":
        return mx
    if mode == "min":
        return max(mn, 1 if mx >= 1 else 0)

    ctx = request.get("context", 0) or 0

    # Learned count prediction from count_logit (trained via BCE on top-Elo replays)
    try:
        pred_ratio = 1.0 / (1.0 + math.exp(-float(count_logit)))
    except Exception:
        pred_ratio = 1.0
    learned_k = int(round(mn + pred_ratio * (mx - mn)))
    learned_k = max(mn, min(mx, learned_k))

    # 1. Board / Hand acquisition (SETUP_BENCH=2, TO_BENCH=5, TO_FIELD=6, TO_HAND=7):
    # Always acquire maximum beneficial cards (e.g. 2 basics on Buddy-Buddy Poffin, 2 cards on Dawn/Crispin)
    if ctx in (1, 2, 5, 6, 7):
        return mx

    # 2. Energy acceleration from deck/discard (ATTACH_FROM=21, ATTACH_TO=22):
    # Critical fix for Marnie's Grimmsnarl ex "Punk Up" (min=0, max=5) & Gardevoir ex / Glass Trumpet!
    if ctx in (21, 22):
        # Count how many energy slots our board actually needs or use learned_k (at least 2-4 when available)
        cur = obs.get("current") or {}
        y_idx = cur.get("yourIndex", 0)
        players = cur.get("players") or [{}, {}]
        me = (players[y_idx] if len(players) > y_idx else None) or {}
        bench = me.get("bench") or []
        # Grimmsnarl ex needs 2, plus 1 for Munkidori / second Grimmsnarl -> 3 to 5 energies
        min_desired = min(mx, max(2, 2 + min(2, len(bench))))
        return max(mn, min(mx, max(learned_k, min_desired)))

    # 3. Variable Energy Discard Attack (DISCARD_ENERGY_CARD=26 with min=0, e.g. Raging Bolt ex Bellowing Thunder 70x):
    if ctx == 26 and mn == 0 and mx >= 2:
        cur = obs.get("current") or {}
        y_idx = cur.get("yourIndex", 0)
        players = cur.get("players") or [{}, {}]
        opp = (players[1 - y_idx] if len(players) > (1 - y_idx) else None) or {}
        opp_act = opp.get("active") or []
        opp_p = opp_act[0] if (isinstance(opp_act, list) and len(opp_act) > 0 and isinstance(opp_act[0], dict)) else {}
        opp_hp = float(opp_p.get("hp", 210) or 210)
        k_ko = int(math.ceil(opp_hp / 70.0))
        return max(1, min(mx, k_ko))

    # 4. Mandatory resource costs (DISCARD=8, DISCARD_TOOL=27, DISCARD_CARD_OR_ATTACHED=29, DISCARD_ENERGY=30):
    if ctx in (8, 26, 27, 29, 30):
        return mn if mn >= 1 else learned_k

    # 5. Default: use learned count head (at least 1 when mx >= 1 unless strongly negative)
    return max(mn, min(mx, max(learned_k, 1 if mx >= 1 else 0)))


def apply_tactical_prior_shaping(obs: dict, req: dict, s: torch.Tensor) -> torch.Tensor:
    """
    Applies general, rulebook-grounded tactical shaping to option logits `s`:
    1. MAIN phase (sel_type == 0): never pass turn (OPT_END) when a legal attack (OPT_ATTACK) is available.
    2. Energy Acceleration Target Distribution (ctx == 21, area in (4, 5)):
       Prevents over-stacking 3-5 energies onto an already fully-powered attacker when another board
       Pokemon (e.g. benched Munkidori needing 1 Darkness energy for Adrena-Brain, or a secondary attacker)
       still needs energy.
    3. Deck Search Evolution Completion (ctx in (6, 7), area == 1):
       Boosts deck-search candidates that directly evolve a ready-to-evolve board Pokemon when that
       evolution card is not already in hand.
    """
    from card_static_table import get_card_info_map
    card_info = get_card_info_map(1500)

    options = req.get("option", []) or []
    sel_type = req.get("type", -1)
    ctx = req.get("context", 0) or 0

    # 1. Never end turn when an attack is available
    if sel_type == 0 and any(opt.get("type") == OPT_ATTACK for opt in options):
        for idx, opt in enumerate(options):
            if opt.get("type") == OPT_END and idx < len(s):
                s[idx] = -1e9

    cur = obs.get("current") or {}
    y_idx = cur.get("yourIndex", 0)
    players = cur.get("players") or [{}, {}]
    me = (players[y_idx] if len(players) > y_idx else None) or {}
    cur_turn = int(cur.get("turn", 0) or 0)

    # 2. Smart Energy Distribution in ctx == 21 (ATTACH_FROM target selection on board)
    if ctx == 21 and len(options) > 1 and all(opt.get("area") in (4, 5) for opt in options):
        act_list = me.get("active") or []
        my_act = act_list[0] if (isinstance(act_list, list) and len(act_list) > 0 and isinstance(act_list[0], dict)) else None
        my_bench = [b for b in (me.get("bench") or []) if isinstance(b, dict)]

        target_stats = []
        for idx, opt in enumerate(options):
            area = opt.get("area", 0)
            b_idx = opt.get("index", 0) or 0
            p_obj = my_act if area == 4 else (my_bench[b_idx] if (area == 5 and 0 <= b_idx < len(my_bench)) else None)
            if not isinstance(p_obj, dict):
                target_stats.append((0, 2, 0))
                continue
            cid = int(p_obj.get("cardId", p_obj.get("id", 0)) or 0)
            e_cnt = len(p_obj.get("energies") or p_obj.get("energyCards") or p_obj.get("attached") or [])
            c_meta = card_info.get(cid, {})
            max_cost = int(c_meta.get("max_attack_cost", 2) or 2)
            # Munkidori (594) needs 1 energy for Adrena-Brain; pre-evolutions of 2-cost attackers need 2
            needed = 1 if cid == 594 else max(2, max_cost)
            target_stats.append((e_cnt, needed, cid))

        any_unpowered = any(e_cnt < needed for e_cnt, needed, _ in target_stats)
        if any_unpowered:
            for idx, (e_cnt, needed, cid) in enumerate(target_stats):
                if idx >= len(s):
                    continue
                if e_cnt >= needed:
                    # Penalize dumping redundant energy onto an already fully-powered Pokemon
                    s[idx] = s[idx] - 3.5 * float(e_cnt - needed + 1)
                elif cid == 594 and e_cnt == 0:
                    # Prioritize giving Munkidori its first energy once primary attacker is powered
                    s[idx] = s[idx] + 1.5

    # 3. Deck Search Evolution Completion in ctx in (6, 7) (TO_FIELD / TO_HAND from Deck)
    if ctx in (6, 7) and len(options) > 1:
        act_list = me.get("active") or []
        my_act = act_list if isinstance(act_list, list) else [act_list]
        my_bench = me.get("bench") or []
        board_p = [p for p in (my_act + my_bench) if isinstance(p, dict)]

        ready_board_cids = set()
        for p in board_p:
            p_cid = int(p.get("cardId", p.get("id", 0)) or 0)
            p_turn = p.get("playedTurn", -1)
            can_evo = (p_turn is not None and int(p_turn) < cur_turn) and not bool(p.get("evolvedThisTurn", False))
            if p_cid > 0 and can_evo:
                ready_board_cids.add(p_cid)

        if ready_board_cids:
            hand_cids = {
                int(c.get("cardId", c.get("id", 0)) or 0)
                for c in (me.get("hand") or [])
                if isinstance(c, dict)
            }
            for idx, opt in enumerate(options):
                if idx >= len(s) or opt.get("area") != 1:
                    continue
                o_cid = int(opt.get("cardId", opt.get("id", 0)) or 0)
                if o_cid <= 0 or o_cid in hand_cids:
                    continue
                pre_ids = card_info.get(o_cid, {}).get("pre_evo_ids") or []
                if any(pid in ready_board_cids for pid in pre_ids):
                    s[idx] = s[idx] + 1.25

    return s


class RandomAgent:
    name = "random"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def reset(self, deck):
        pass

    def act(self, obs):
        req = obs.get("select") or {}
        options = req.get("option", [])
        if not options:
            return [0]
        mn, mx = count_bounds(req)
        k = self.rng.randint(mn, mx)
        return self.rng.sample(range(len(options)), k)


class HeuristicAgent:
    """Develop board (evolve > attach > play > ability), then attack, else end turn."""
    name = "heuristic"
    MAIN_PRIORITY = [OPT_EVOLVE, OPT_ATTACH, OPT_PLAY, OPT_ABILITY, OPT_ATTACK, OPT_END]

    def reset(self, deck):
        pass

    def act(self, obs):
        req = obs.get("select") or {}
        options = req.get("option", [])
        if not options:
            return [0]
        mn, mx = count_bounds(req)
        types = [o.get("type", -1) for o in options]
        sel_type = req.get("type", -1)
        ctx = req.get("context", -1)

        if sel_type == 0:  # MAIN
            for t in self.MAIN_PRIORITY:
                if t in types:
                    return [types.index(t)]
            return [0]
        if sel_type == 6:  # ATTACK select -> highest damage
            dmg = attack_damage_table()
            best = max(range(len(options)), key=lambda i: dmg.get(options[i].get("attackId"), 0))
            return [best]
        if sel_type == 9:  # YES/NO
            want = OPT_NO if ctx == CTX_MULLIGAN else OPT_YES
            return [types.index(want)] if want in types else [0]
        if sel_type == 8:  # COUNT -> largest number
            best = max(range(len(options)), key=lambda i: options[i].get("number") or 0)
            return [best]
        k = max(mn, min(mx, max(1, mn)))
        return list(range(k))


class NNAgent:
    def __init__(self, model, name: str, device=None, stochastic: bool = False,
                 multi: str = "auto", use_tracker: bool = True):
        from card_tracker import CardTracker  # noqa: F401
        self.model = model.eval()
        self.name = name
        self.device = device or next(model.parameters()).device
        self.stochastic = stochastic
        self.multi = multi
        self.use_tracker = use_tracker
        self.tracker = None

    def reset(self, deck):
        from card_tracker import CardTracker
        self.tracker = CardTracker(deck) if self.use_tracker else None

    @torch.no_grad()
    def act(self, obs):
        from feature_parser import parse_game_state_to_tensors
        req = obs.get("select") or {}
        options = req.get("option", [])
        if not options:
            return [0]
        if len(options) == 1:
            if self.tracker is not None:
                try:
                    self.tracker.update(obs, req)
                except Exception:
                    pass
            mn, _ = count_bounds(req)
            return [0] if mn >= 1 else [0]

        ctx = req.get("context", 0) or 0
        # Mulligan extra draw: always accept extra free cards
        if ctx == 42:
            for idx, opt in enumerate(options):
                if opt.get("type") == OPT_YES:
                    return [idx]
            return [0]

        parsed = parse_game_state_to_tensors(obs, req, tracker=self.tracker)
        t = lambda k: torch.from_numpy(parsed[k]).unsqueeze(0).to(self.device)
        _, scores, _, _, count_logits, _, action_repr = self.model(
            t("global_vec"), t("obj_features"), t("card_ids"), t("evo_ids"),
            t("option_mask"), t("valid_action_mask"), t("padding_mask"),
            token_types=t("token_types"), return_q_and_repr=True
        )
        opt_idx = np.where(parsed["option_mask"])[0]
        s = scores[0, opt_idx].clone()
        opt_r = action_repr[0, opt_idx]
        opt_cids = torch.from_numpy(parsed["card_ids"][opt_idx]).to(self.device)
        c_logit = float(count_logits[0].item())

        s = apply_tactical_prior_shaping(obs, req, s)

        k = choose_multi_select_count(obs, req, s, count_logit=c_logit, mode=self.multi)
        k = min(k, len(opt_idx))
        if k <= 0:
            return []

        if self.stochastic:
            p = torch.softmax(s, dim=-1)
            chosen = torch.multinomial(p, num_samples=max(k, 1), replacement=False)
            order = chosen.tolist()
            return [int(i) for i in order[:k]]

        if k > 1 and hasattr(self.model, "select_multi_options_autoregressive"):
            return self.model.select_multi_options_autoregressive(s, opt_r, opt_cids, k, ctx=ctx)

        order = torch.argsort(s, descending=True).tolist()
        return [int(i) for i in order[:k]]


def load_nn_agent(name: str, ckpt: str, device, **kw) -> NNAgent:
    from model_dual_cross import PTCGDualCrossNet
    model = PTCGDualCrossNet().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=False)
    return NNAgent(model, name, device=device, **kw)


CHECKPOINTS = {
    "bc": "ptcg_dual_cross_bc.pt",
    "ppo": "ptcg_dual_cross_ppo.pt",
    "ppo_anchored": "ptcg_dual_cross_ppo_anchored.pt",
    "capstone": "ptcg_capstone_league_model.pt",
    "ppo_phase2": "ptcg_ppo_phase2_model.pt",
    "ppo_phase3": "ptcg_ppo_phase3_model.pt",
    "ppo_phase4": "ptcg_ppo_phase4_model.pt",
    "wise": "ptcg_wise_ensemble.pt",
    "iql_rel": "ptcg_iql_relational.pt",
    "iql_v3": "ptcg_iql_v3.pt",
    "selfplay_v3": "ptcg_selfplay_v3.pt",
    "wise_v3": "ptcg_wise_v3.pt",
    "old_submission": "old_submission_weights.pt",
}


def make_agent(spec: str, device, seed=None):
    """spec examples: 'random', 'heuristic', 'bc', 'ppo_phase4', 'wise', 'iql_rel', 'wise_v3', 'search_v4'."""
    base, _, variant = spec.partition(":")
    if base == "random":
        return RandomAgent(seed)
    if base == "heuristic":
        return HeuristicAgent()
    if base in ("search", "mcts", "search_agent", "search_phase4", "search_phase3", "search_bc", "search_wise", "search_v3", "search_iql", "search_v4"):
        from search_agent import SearchAgent
        det = int(variant) if variant and variant.isdigit() else 2
        if base == "search_bc":
            ckpt = "ptcg_dual_cross_bc.pt"
        elif base == "search_phase3":
            ckpt = "ptcg_ppo_phase3_model.pt"
        elif base == "search_v4" and os.path.exists("ptcg_wise_v3.pt"):
            ckpt = "ptcg_wise_v3.pt"
        elif base in ("search_iql", "search_v4") and os.path.exists("ptcg_iql_v3.pt"):
            ckpt = "ptcg_iql_v3.pt"
        elif base in ("search_iql", "search_v4") and os.path.exists("ptcg_iql_relational.pt"):
            ckpt = "ptcg_iql_relational.pt"
        elif base in ("search_wise", "search_v3") and os.path.exists("ptcg_wise_ensemble.pt"):
            ckpt = "ptcg_wise_ensemble.pt"
        else:
            ckpt = "ptcg_ppo_phase4_model.pt" if os.path.exists("ptcg_ppo_phase4_model.pt") else "ptcg_dual_cross_bc.pt"
        return SearchAgent(model_path=ckpt, device=device, num_determinizations=det, name=spec)
    if base in CHECKPOINTS:
        agent = load_nn_agent(spec, CHECKPOINTS[base], device, multi=variant or "auto")
        return agent
    if os.path.exists(base):  # direct checkpoint path
        return load_nn_agent(spec, base, device, multi=variant or "auto")
    raise ValueError(f"unknown agent spec {spec}")
