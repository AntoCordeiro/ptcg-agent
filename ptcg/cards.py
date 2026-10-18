"""Static card / attack tables built from the simulator's own card database.

Everything here comes from `cg.api.all_card_data()` / `all_attack()`, so the same
code works locally, in self-play and inside the Kaggle container.
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from cg import api  # noqa: E402

N_CARD = 1500      # card-id vocabulary (max id in the game is 1431)
N_ATTACK = 1800    # attack-id vocabulary (max id 1755)
N_ETYPE = 12

_cards = api.all_card_data()
_attacks = api.all_attack()

CARD_INFO = {c.cardId: c for c in _cards}
ATTACK_INFO = {a.attackId: a for a in _attacks}
NAME_TO_IDS = {}
for c in _cards:
    NAME_TO_IDS.setdefault(c.name, []).append(c.cardId)


def _onehot(i, n):
    v = np.zeros(n, np.float32)
    if i is not None and 0 <= int(i) < n:
        v[int(i)] = 1.0
    return v


def _card_vec(c):
    return np.concatenate([
        _onehot(c.cardType, 7),
        [c.hp / 300.0, c.retreatCost / 4.0],
        _onehot(c.weakness, N_ETYPE),
        _onehot(c.resistance, N_ETYPE),
        _onehot(c.energyType, N_ETYPE),
        [float(c.basic), float(c.stage1), float(c.stage2), float(c.ex), float(c.megaEx),
         float(c.tera), float(c.aceSpec), len(c.attacks) / 3.0, len(c.skills) / 3.0],
    ]).astype(np.float32)


F_CARD_STATIC = 7 + 2 + 3 * N_ETYPE + 9
CARD_TABLE = np.zeros((N_CARD, F_CARD_STATIC), np.float32)
for _c in _cards:
    if _c.cardId < N_CARD:
        CARD_TABLE[_c.cardId] = _card_vec(_c)

F_ATTACK_STATIC = 2 + N_ETYPE
ATTACK_TABLE = np.zeros((N_ATTACK, F_ATTACK_STATIC), np.float32)
for _a in _attacks:
    if _a.attackId < N_ATTACK:
        cost = np.zeros(N_ETYPE, np.float32)
        for e in _a.energies:
            if 0 <= int(e) < N_ETYPE:
                cost[int(e)] += 1
        ATTACK_TABLE[_a.attackId] = np.concatenate([[_a.damage / 300.0, len(_a.energies) / 5.0], cost / 3.0])


def is_basic_pokemon(cid):
    c = CARD_INFO.get(cid)
    return c is not None and c.cardType == 0 and c.basic


def is_pokemon(cid):
    c = CARD_INFO.get(cid)
    return c is not None and c.cardType == 0
