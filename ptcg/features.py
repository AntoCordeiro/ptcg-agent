"""Observation -> fixed-size numpy tensors (entity + option tokens).

Design rules (lessons from the legacy code):
  * one function for every caller (replay parser, self-play, MCTS, Kaggle), so
    there is no train/test skew;
  * accepts both the raw JSON dict and `dataclasses.asdict(Observation)` (fields
    may be missing or None in either form; never trust a default);
  * no per-card hand-written rules: only raw state + static card table ids.

Token layout
  card tokens  [MAX_CARDS]: in-play Pokemon (both players), my hand, stadium,
                            looking cards, deck cards shown by the selection,
                            and any other card referenced by an option
  option tokens [MAX_OPTS]: one per legal option; each points to <=2 card tokens
  bags [2, MAX_BAG]        : my / opponent discard pile ids (mean-pooled in model)
  global vector            : scalar game state + selection header
"""
import numpy as np

from .cards import N_ETYPE

MAX_CARDS = 64
MAX_OPTS = 80
MAX_BAG = 48
N_CTX = 64
N_SELTYPE = 12
N_OPTTYPE = 17

# zones for card tokens
Z_ACTIVE, Z_BENCH, Z_HAND, Z_STADIUM, Z_LOOKING, Z_DECKVIEW, Z_DISCARD, Z_PRIZE, Z_OTHER = range(9)
N_ZONE = 9
AREA_TO_ZONE = {1: Z_DECKVIEW, 2: Z_HAND, 3: Z_DISCARD, 4: Z_ACTIVE, 5: Z_BENCH, 6: Z_PRIZE, 7: Z_STADIUM, 12: Z_LOOKING}

F_CARD = N_ZONE + 1 + 6 + N_ETYPE + 4 + 5 + 1   # = 38
F_OPT = N_OPTTYPE + 4 + N_ETYPE + 2             # = 35
F_GLOBAL = 8 + 2 * 12 + N_SELTYPE + 6           # = 50


def _g(d, k, default=None):
    if d is None:
        return default
    v = d.get(k, default) if isinstance(d, dict) else getattr(d, k, default)
    return default if v is None else v


def _pokemon_feats(p, zone, owner_is_me, bench_idx, status=None):
    f = np.zeros(F_CARD, np.float32)
    f[zone] = 1.0
    f[N_ZONE] = 1.0 if owner_is_me else 0.0
    o = N_ZONE + 1
    if p is not None:
        hp = float(_g(p, "hp", 0)); mhp = float(_g(p, "maxHp", 0))
        ens = _g(p, "energies", []) or []
        f[o + 0] = hp / 300.0
        f[o + 1] = mhp / 300.0
        f[o + 2] = (mhp - hp) / 300.0
        f[o + 3] = len(ens) / 5.0
        f[o + 4] = len(_g(p, "tools", []) or []) / 2.0
        f[o + 5] = len(_g(p, "preEvolution", []) or []) / 2.0
        for e in ens:
            if 0 <= int(e) < N_ETYPE:
                f[o + 6 + int(e)] += 0.25
        o2 = o + 6 + N_ETYPE
        f[o2 + 0] = 1.0 if _g(p, "appearThisTurn", False) else 0.0
        f[o2 + 1] = bench_idx / 8.0
        f[o2 + 2] = 1.0 if hp <= 30 else 0.0
        f[o2 + 3] = 1.0 if hp <= 90 else 0.0
        if status is not None:
            for i, k in enumerate(("poisoned", "burned", "asleep", "paralyzed", "confused")):
                f[o2 + 4 + i] = 1.0 if _g(status, k, False) else 0.0
    f[-1] = 1.0  # "is a real token"
    return f


def _plain_feats(zone, owner_is_me, idx=0):
    f = np.zeros(F_CARD, np.float32)
    f[zone] = 1.0
    f[N_ZONE] = 1.0 if owner_is_me else 0.0
    f[N_ZONE + 1 + 6 + N_ETYPE + 1] = min(idx, 60) / 60.0
    f[-1] = 1.0
    return f


class Encoded(dict):
    pass


def encode(obs):
    """Encode one observation (dict). Returns an `Encoded` dict of numpy arrays.

    Also returns `n_opt` (true number of options) and `opt_valid` mask; options
    beyond MAX_OPTS are dropped (the agent never picks them; this is rare).
    """
    cur = _g(obs, "current", {}) or {}
    sel = _g(obs, "select", {}) or {}
    me = int(_g(cur, "yourIndex", 0))
    opp = 1 - me
    players = _g(cur, "players", [{}, {}]) or [{}, {}]
    P = [players[0] or {}, players[1] or {}]

    card_ids = np.zeros(MAX_CARDS, np.int64)
    card_f = np.zeros((MAX_CARDS, F_CARD), np.float32)
    key2tok = {}
    n = [0]

    def add(cid, feats, keys=()):
        if n[0] >= MAX_CARDS:
            return -1
        i = n[0]
        card_ids[i] = int(cid or 0)
        card_f[i] = feats
        for k in keys:
            key2tok[k] = i
        n[0] += 1
        return i

    # in-play Pokemon
    for pi in (me, opp):
        ps = P[pi]
        act = _g(ps, "active", []) or []
        if len(act) > 0:
            a = act[0]
            add(_g(a, "id", 0) if a is not None else 0,
                _pokemon_feats(a, Z_ACTIVE, pi == me, 0, status=ps), keys=[(4, pi, 0)])
        for bi, b in enumerate(_g(ps, "bench", []) or []):
            add(_g(b, "id", 0), _pokemon_feats(b, Z_BENCH, pi == me, bi + 1), keys=[(5, pi, bi)])
    # my hand
    for hi, c in enumerate(_g(P[me], "hand", []) or []):
        add(_g(c, "id", 0), _plain_feats(Z_HAND, True, hi), keys=[(2, me, hi)])
    # stadium
    for si, c in enumerate(_g(cur, "stadium", []) or []):
        owner = int(_g(c, "playerIndex", me))
        add(_g(c, "id", 0), _plain_feats(Z_STADIUM, owner == me), keys=[(7, 0, si), (7, 1, si)])
    # looking cards
    for li, c in enumerate(_g(cur, "looking", []) or []):
        owner = int(_g(c, "playerIndex", me)) if c is not None else me
        add(_g(c, "id", 0) if c is not None else 0, _plain_feats(Z_LOOKING, owner == me, li),
            keys=[(12, 0, li), (12, 1, li)])
    # deck cards revealed by the selection
    sdeck = _g(sel, "deck", []) or []
    for di, c in enumerate(sdeck):
        owner = int(_g(c, "playerIndex", me))
        add(_g(c, "id", 0), _plain_feats(Z_DECKVIEW, owner == me, di), keys=[(1, owner, di), (1, 1 - owner, di)])

    def resolve(area, pidx, idx):
        """Token for (area, player, index); create one lazily for discard/prize refs."""
        if area is None:
            return -1
        area = int(area); pidx = me if pidx is None else int(pidx); idx = int(idx or 0)
        k = (area, pidx, idx)
        if k in key2tok:
            return key2tok[k]
        cid = 0
        if area == 3:
            lst = _g(P[pidx], "discard", []) or []
            cid = _g(lst[idx], "id", 0) if idx < len(lst) and lst[idx] is not None else 0
        elif area == 6:
            lst = _g(P[pidx], "prize", []) or []
            cid = _g(lst[idx], "id", 0) if idx < len(lst) and lst[idx] is not None else 0
        elif area == 2 and pidx != me:
            cid = 0
        return add(cid, _plain_feats(AREA_TO_ZONE.get(area, Z_OTHER), pidx == me, idx), keys=[k])

    def pokemon_at(area, pidx, idx):
        pidx = me if pidx is None else int(pidx)
        ps = P[pidx]
        if area == 4:
            act = _g(ps, "active", []) or []
            return act[0] if act else None
        if area == 5:
            b = _g(ps, "bench", []) or []
            return b[idx] if idx is not None and idx < len(b) else None
        return None

    # options
    opts = _g(sel, "option", []) or []
    n_opt = len(opts)
    opt_f = np.zeros((MAX_OPTS, F_OPT), np.float32)
    opt_attack = np.zeros(MAX_OPTS, np.int64)
    opt_card = np.zeros(MAX_OPTS, np.int64)
    opt_ref = np.full((MAX_OPTS, 2), -1, np.int64)
    opt_mask = np.zeros(MAX_OPTS, bool)
    for oi, op in enumerate(opts[:MAX_OPTS]):
        t = int(_g(op, "type", 0))
        f = opt_f[oi]
        f[min(t, N_OPTTYPE - 1)] = 1.0
        o = N_OPTTYPE
        f[o + 0] = float(_g(op, "number", 0)) / 10.0
        f[o + 1] = float(_g(op, "count", 0)) / 4.0
        f[o + 2] = oi / 40.0
        f[o + 3] = 1.0
        area = _g(op, "area"); pidx = _g(op, "playerIndex"); idx = _g(op, "index")
        if t == 7:      # PLAY: index within my hand
            area, pidx = 2, me
        r1 = -1
        if area is not None:
            r1 = resolve(area, pidx, idx)
        r2 = -1
        ipa = _g(op, "inPlayArea")
        if ipa is not None:
            r2 = resolve(ipa, me, _g(op, "inPlayIndex", 0))
        opt_ref[oi] = (r1, r2)
        if t == 13:
            opt_attack[oi] = int(_g(op, "attackId", 0))
        if t == 15:
            opt_card[oi] = int(_g(op, "cardId", 0))
        if t in (4, 5, 6) and area is not None:
            pk = pokemon_at(int(area), pidx, idx)
            if pk is not None:
                if t == 4:
                    tl = _g(pk, "tools", []) or []
                    ti = int(_g(op, "toolIndex", 0))
                    opt_card[oi] = _g(tl[ti], "id", 0) if ti < len(tl) else 0
                else:
                    ec = _g(pk, "energyCards", []) or []
                    ei = int(_g(op, "energyIndex", 0))
                    if t == 5 and ei < len(ec):
                        opt_card[oi] = _g(ec[ei], "id", 0)
                    ens = _g(pk, "energies", []) or []
                    if ei < len(ens) and 0 <= int(ens[ei]) < N_ETYPE:
                        f[o + 4 + int(ens[ei])] = 1.0
        if t == 16:
            f[o + 4 + N_ETYPE] = (int(_g(op, "specialConditionType", 0)) + 1) / 5.0
        f[o + 5 + N_ETYPE] = 1.0 if (t == 1) else 0.0
        opt_mask[oi] = True

    # discard bags
    bags = np.zeros((2, MAX_BAG), np.int64)
    for j, pi in enumerate((me, opp)):
        d = _g(P[pi], "discard", []) or []
        ids = [int(_g(c, "id", 0)) for c in d if c is not None][-MAX_BAG:]
        bags[j, :len(ids)] = ids

    # global
    g = np.zeros(F_GLOBAL, np.float32)
    g[0] = float(_g(cur, "turn", 0)) / 20.0
    g[1] = float(_g(cur, "turnActionCount", 0)) / 10.0
    g[2] = float(bool(_g(cur, "supporterPlayed", False)))
    g[3] = float(bool(_g(cur, "stadiumPlayed", False)))
    g[4] = float(bool(_g(cur, "energyAttached", False)))
    g[5] = float(bool(_g(cur, "retreated", False)))
    g[6] = 1.0 if int(_g(cur, "firstPlayer", -1)) == me else 0.0
    g[7] = 1.0 if len(_g(cur, "stadium", []) or []) > 0 else 0.0
    o = 8
    for pi in (me, opp):
        ps = P[pi]
        g[o + 0] = float(_g(ps, "deckCount", 0)) / 60.0
        g[o + 1] = float(_g(ps, "handCount", 0)) / 20.0
        g[o + 2] = len(_g(ps, "prize", []) or []) / 6.0
        g[o + 3] = float(_g(ps, "benchMax", 5)) / 8.0
        g[o + 4] = len(_g(ps, "bench", []) or []) / 8.0
        g[o + 5] = len(_g(ps, "discard", []) or []) / 60.0
        for i, k in enumerate(("poisoned", "burned", "asleep", "paralyzed", "confused")):
            g[o + 6 + i] = float(bool(_g(ps, k, False)))
        g[o + 11] = 1.0 if len(_g(ps, "active", []) or []) > 0 else 0.0
        o += 12
    st = int(_g(sel, "type", 0))
    g[o + min(st, N_SELTYPE - 1)] = 1.0
    o += N_SELTYPE
    g[o + 0] = float(_g(sel, "minCount", 0)) / 10.0
    g[o + 1] = float(_g(sel, "maxCount", 0)) / 10.0
    g[o + 2] = float(_g(sel, "remainDamageCounter", 0)) / 20.0
    g[o + 3] = float(_g(sel, "remainEnergyCost", 0)) / 5.0
    g[o + 4] = n_opt / 50.0
    g[o + 5] = 1.0

    eff = _g(sel, "effect"); ctxc = _g(sel, "contextCard")
    return Encoded(
        card_ids=card_ids, card_f=card_f, card_mask=(np.arange(MAX_CARDS) < n[0]),
        opt_f=opt_f, opt_attack=opt_attack, opt_card=opt_card, opt_ref=opt_ref, opt_mask=opt_mask,
        bags=bags, glob=g,
        ctx=np.int64(min(int(_g(sel, "context", 0)), N_CTX - 1)),
        eff_card=np.int64(_g(eff, "id", 0) if eff else 0),
        ctx_card=np.int64(_g(ctxc, "id", 0) if ctxc else 0),
        n_opt=n_opt,
    )


ARRAY_KEYS = ["card_ids", "card_f", "card_mask", "opt_f", "opt_attack", "opt_card", "opt_ref",
              "opt_mask", "bags", "glob", "ctx", "eff_card", "ctx_card"]


def stack(encs):
    """Batch a list of Encoded dicts into numpy arrays."""
    return {k: np.stack([e[k] for e in encs]) for k in ARRAY_KEYS}
