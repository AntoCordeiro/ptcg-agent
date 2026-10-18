"""PTCGNet: entity/option transformer with policy, value, count and belief heads.

Sequence = [global, my-discard bag, opp-discard bag, card tokens..., option tokens...]
Option tokens are built from their own features plus the *input* embeddings of
the card tokens they point to (ref1 = acted card, ref2 = in-play target), then
everything is mixed by a standard pre-norm transformer encoder.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cards import CARD_TABLE, ATTACK_TABLE, N_CARD, N_ATTACK, F_CARD_STATIC, F_ATTACK_STATIC
from .features import F_CARD, F_OPT, F_GLOBAL, N_CTX, MAX_CARDS, MAX_OPTS


class PTCGNet(nn.Module):
    def __init__(self, d=192, layers=4, heads=6, ff=512, dropout=0.1, belief=True):
        super().__init__()
        self.d = d
        self.use_belief = belief
        self.register_buffer("card_table", torch.from_numpy(CARD_TABLE), persistent=False)
        self.register_buffer("attack_table", torch.from_numpy(ATTACK_TABLE), persistent=False)
        self.card_emb = nn.Embedding(N_CARD, d)
        self.card_static = nn.Linear(F_CARD_STATIC, d)
        self.attack_emb = nn.Embedding(N_ATTACK, d)
        self.attack_static = nn.Linear(F_ATTACK_STATIC, d)
        self.card_in = nn.Linear(F_CARD, d)
        self.opt_in = nn.Linear(F_OPT, d)
        self.ref1 = nn.Linear(d, d, bias=False)
        self.ref2 = nn.Linear(d, d, bias=False)
        self.glob_in = nn.Linear(F_GLOBAL, d)
        self.ctx_emb = nn.Embedding(N_CTX, d)
        self.type_emb = nn.Embedding(5, d)  # 0 global, 1 bag-me, 2 bag-opp, 3 card, 4 option
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.pol = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.val = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.cnt = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        if belief:
            self.bel = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, N_CARD))
            nn.init.constant_(self.bel[-1].bias, -5.0)   # prior: a given card is rarely in hand

    def card_vec(self, ids):
        return self.card_emb(ids) + self.card_static(self.card_table[ids])

    def forward(self, b):
        B = b["card_ids"].shape[0]
        d = self.d
        cards = self.card_in(b["card_f"].float()) + self.card_vec(b["card_ids"]) + self.type_emb.weight[3]
        pad = torch.cat([cards.new_zeros(B, 1, d), cards], 1)          # index 0 = "no ref"
        r = b["opt_ref"] + 1
        g1 = torch.gather(pad, 1, r[..., 0:1].expand(-1, -1, d))
        g2 = torch.gather(pad, 1, r[..., 1:2].expand(-1, -1, d))
        oc = b["opt_card"]
        oa = b["opt_attack"]
        opts = (self.opt_in(b["opt_f"].float()) + self.ref1(g1) + self.ref2(g2)
                + self.card_vec(oc) * (oc > 0).unsqueeze(-1)
                + (self.attack_emb(oa) + self.attack_static(self.attack_table[oa])) * (oa > 0).unsqueeze(-1)
                + self.type_emb.weight[4])
        bag = b["bags"]
        bv = self.card_vec(bag) * (bag > 0).unsqueeze(-1)
        bv = bv.sum(2) / (bag > 0).sum(2, keepdim=True).clamp(min=1)
        bags = bv + self.type_emb.weight[1:3]
        ec, cc = b["eff_card"], b["ctx_card"]
        glob = (self.glob_in(b["glob"].float()) + self.ctx_emb(b["ctx"])
                + self.card_vec(ec) * (ec > 0).unsqueeze(-1) + self.card_vec(cc) * (cc > 0).unsqueeze(-1)
                + self.type_emb.weight[0])
        x = torch.cat([glob.unsqueeze(1), bags, cards, opts], 1)
        valid = torch.cat([torch.ones(B, 3, dtype=torch.bool, device=x.device),
                           b["card_mask"].bool(), b["opt_mask"].bool()], 1)
        h = self.norm(self.encoder(x, src_key_padding_mask=~valid))
        hg = h[:, 0]
        ho = h[:, 3 + b["card_ids"].shape[1]:]       # works with trimmed (dynamic-length) batches
        om = b["opt_mask"].bool()
        logits = self.pol(ho).squeeze(-1).masked_fill(~om, -1e9)
        value = torch.tanh(self.val(hg).squeeze(-1))
        mo = (ho * om.unsqueeze(-1)).sum(1) / om.sum(1, keepdim=True).clamp(min=1)
        count = self.cnt(torch.cat([hg, mo], -1)).squeeze(-1)
        out = {"logits": logits, "value": value, "count": count}
        if self.use_belief:
            out["belief"] = self.bel(hg)
        return out


def to_torch(batch_np, device):
    return {k: torch.as_tensor(v, device=device) for k, v in batch_np.items()}


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = PTCGNet(**ck.get("config", {}))
    m.load_state_dict(ck["model"], strict=True)
    m.to(device).eval()
    return m
