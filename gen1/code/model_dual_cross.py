# model_dual_cross.py
"""
Relational Pairwise-Bias & Set-Context Dual-Encoder Architecture with IQL Action-Critic
and Conditional Pointer Synergy (`PTCGDualCrossNet` v2 - Approaches A & C).

Key Architectural Innovations:
1. State-Action Decoupling (Dual-Encoder):
   - Game state entities (Active, Bench, Opponent, Hand, Stadium) undergo Pre-LN self-attention
     without distortion from candidate action tokens.
2. Pairwise Symbolic Relational Cross-Attention (Approach A1 - AlphaFold2 / Graphormer style):
   - Computes an 8-dim exact relational tensor R[b, i, j] between every Candidate Option i and
     every State Entity j (`is_global`, `same_card`, `opt_evolves_from_state`, `opt_evolves_into_state`,
     `same_evo_family`, `state_is_my_active`, `state_is_my_bench`, `state_is_opp_pokemon`) and injects
     per-head additive attention biases directly into the Cross-Attention logits.
3. Option-Set Self-Attention (Approach A2):
   - Candidate options attend to one another via gated Pre-LN Multi-Head Self-Attention before scoring,
     enabling context-sensitive comparisons across the legal action menu.
4. Conditional Multi-Select Pointer Synergy & IQL Action-Value Critic Q(s, a) (Approaches A3 & C):
   - Predicts per-option Action-Value Q(s, a) in [-1, 1] alongside policy logits pi(a | s) and
     pairwise option synergy embeddings for multi-select subset selection.
   - Zero-initialized residual gates allow exact warm-starting from v1 checkpoints (`strict=False`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from features_spec import (
    GLOBAL_DIM, OBJECT_INPUT_DIM, CARD_EMBED_DIM,
    EVO_EMBED_DIM, TEXT_RAW_DIM, TEXT_PROJ_DIM, NUM_CARD_IDS, D_MODEL, NUM_TOKEN_TYPES,
    TOKEN_TYPE_GLOBAL, TOKEN_TYPE_ACTIVE_MINE, TOKEN_TYPE_BENCH_MINE,
    TOKEN_TYPE_ACTIVE_OPP, TOKEN_TYPE_BENCH_OPP, TOKEN_TYPE_OPTION
)
from card_text_encoder import get_card_text_embeddings_384

NUM_RELATIONS = 8


class PTCGDualCrossNet(nn.Module):
    def __init__(
        self,
        num_cards: int = NUM_CARD_IDS,
        card_emb_dim: int = CARD_EMBED_DIM,
        evo_emb_dim: int = EVO_EMBED_DIM,
        text_raw_dim: int = TEXT_RAW_DIM,
        text_proj_dim: int = TEXT_PROJ_DIM,
        d_model: int = D_MODEL,
        nhead: int = 8,
        dim_ff: int = 512,
        num_layers: int = 3,
        num_token_types: int = NUM_TOKEN_TYPES,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # 1. Learnable embeddings
        self.card_embedding = nn.Embedding(num_cards, card_emb_dim, padding_idx=0)
        self.evo_embedding = nn.Embedding(num_cards, evo_emb_dim, padding_idx=0)
        self.token_type_embedding = nn.Embedding(num_token_types, d_model)

        # 2. Dense Semantic Card Text Embeddings (100% variance, 384-dim uncompressed)
        text_weights = torch.from_numpy(get_card_text_embeddings_384()).float()
        self.card_text_embedding = nn.Embedding.from_pretrained(text_weights, freeze=True)

        # Task-specific non-linear projection: 384 -> 128 -> 64
        self.text_proj = nn.Sequential(
            nn.Linear(text_raw_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, text_proj_dim),
        )

        # 3. Dual Encoders
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.state_entity_encoder = nn.Sequential(
            nn.Linear(OBJECT_INPUT_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.action_option_encoder = nn.Sequential(
            nn.Linear(OBJECT_INPUT_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # 4. State Trunk: Pre-LN TransformerEncoder strictly over Game State tokens
        state_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.state_transformer = nn.TransformerEncoder(state_layer, num_layers=num_layers)

        # 5. Asymmetric Relational Cross-Attention Trunk (Approach A1)
        self.cross_norm_q = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_dropout = nn.Dropout(dropout)

        # Pairwise Symbolic Relation Bias Projection (zero-initialized for exact warm-start)
        self.rel_bias_proj = nn.Linear(NUM_RELATIONS, nhead, bias=False)
        nn.init.zeros_(self.rel_bias_proj.weight)

        self.cross_norm_ffn = nn.LayerNorm(d_model)
        self.cross_ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

        # 5b. Option-Set Self-Attention Block (Approach A2: Options attend to competing Options)
        self.opt_self_norm = nn.LayerNorm(d_model)
        self.opt_self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.opt_self_gate = nn.Parameter(torch.zeros(1))

        # 6. Action Scoring Head & IQL Action-Value Critic Q(s, a) (Approach C)
        self.action_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.q_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Tanh(),
        )
        self.q_logit_gate = nn.Parameter(torch.zeros(1))

        # 6b. Conditional Multi-Select Pointer Synergy Projection (Approach A3)
        self.pointer_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.pointer_proj.weight)
        self.pointer_gate = nn.Parameter(torch.zeros(1))

        # 7. Multi-Task Auxiliary Heads
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Tanh(),
        )

        self.prize_margin_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        self.turn_horizon_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def _compute_pairwise_relation_bias(
        self,
        card_ids: torch.Tensor,          # [B, S_obj]
        evo_ids: torch.Tensor,           # [B, S_obj]
        obj_types: torch.Tensor,         # [B, S_obj]
        state_key_padding_mask: torch.Tensor,  # [B, 1 + S_obj] (True = masked)
    ) -> torch.Tensor:
        """
        Computes exact symbolic pairwise relation bias between every Option token i in [0..S_obj-1]
        and every State token j in [0..S_obj] (where j=0 is Global and j=1..S_obj are State Objects).
        Returns float attention mask of shape [B * nhead, S_obj, 1 + S_obj].
        """
        B, S_obj = card_ids.shape
        device = card_ids.device

        # Expand card_ids and evo_ids for pairwise comparison [B, S_obj, S_obj]
        ci = card_ids.unsqueeze(2)  # [B, S_obj, 1]
        cj = card_ids.unsqueeze(1)  # [B, 1, S_obj]
        ei = evo_ids.unsqueeze(2)   # [B, S_obj, 1]
        ej = evo_ids.unsqueeze(1)   # [B, 1, S_obj]
        tj = obj_types.unsqueeze(1).expand(B, S_obj, S_obj)  # [B, S_obj, S_obj]

        same_card = ((ci == cj) & (ci > 0)).float()
        opt_evolves_from_state = ((ei == cj) & (ei > 0)).float()
        opt_evolves_into_state = ((ci == ej) & (ci > 0)).float()
        same_evo_family = ((ei == ej) & (ei > 0)).float()
        state_is_my_active = (tj == TOKEN_TYPE_ACTIVE_MINE).float()
        state_is_my_bench = (tj == TOKEN_TYPE_BENCH_MINE).float()
        state_is_opp = ((tj == TOKEN_TYPE_ACTIVE_OPP) | (tj == TOKEN_TYPE_BENCH_OPP)).float()
        zeros_obj = torch.zeros_like(same_card)

        # Stack object-to-object relations: [B, S_obj, S_obj, 8]
        rel_obj = torch.stack([
            zeros_obj,               # 0: is_global (0 for object tokens)
            same_card,               # 1: exact same card ID
            opt_evolves_from_state,  # 2: option evolves from board/hand Pokemon j
            opt_evolves_into_state,  # 3: option is pre-evolution of hand/board Pokemon j
            same_evo_family,         # 4: option and state token share evolution line
            state_is_my_active,      # 5: state token is my Active Pokemon
            state_is_my_bench,       # 6: state token is my Bench Pokemon
            state_is_opp,            # 7: state token is opponent Pokemon
        ], dim=-1)

        # Global token relation at j=0: [B, S_obj, 1, 8]
        rel_glob = torch.zeros((B, S_obj, 1, NUM_RELATIONS), dtype=torch.float32, device=device)
        rel_glob[:, :, 0, 0] = 1.0  # is_global = 1

        rel_full = torch.cat([rel_glob, rel_obj], dim=2)  # [B, S_obj, 1 + S_obj, 8]
        bias_heads = self.rel_bias_proj(rel_full)         # [B, S_obj, 1 + S_obj, nhead]
        bias_heads = bias_heads.permute(0, 3, 1, 2).contiguous()  # [B, nhead, S_obj, 1 + S_obj]

        # Fold state_key_padding_mask into float attention bias so masked state tokens receive -1e9
        pad_expanded = state_key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, 1 + S_obj]
        bias_heads = bias_heads.masked_fill(pad_expanded, -1e9)
        return bias_heads.view(B * self.nhead, S_obj, 1 + S_obj)

    def forward(
        self,
        global_features: torch.Tensor,       # [B, 191]
        object_features: torch.Tensor,       # [B, S_obj, 366]
        card_ids: torch.Tensor,              # [B, S_obj]
        evo_ids: torch.Tensor,               # [B, S_obj]
        option_mask: torch.Tensor,           # [B, S_obj]
        valid_action_mask: torch.Tensor,     # [B, S_obj]
        padding_mask: torch.Tensor = None,   # [B, 1 + S_obj]
        token_types: torch.Tensor = None,    # [B, S_obj]
        return_aux: bool = False,
        return_attention: bool = False,
        return_q_and_repr: bool = False,
    ):
        B, S_obj, _ = object_features.shape
        device = global_features.device

        # 1. Common Object Embeddings (Static + Text + Card ID + Evo ID)
        c_ids_clamped = card_ids.clamp(0, self.card_embedding.num_embeddings - 1)
        e_ids_clamped = evo_ids.clamp(0, self.evo_embedding.num_embeddings - 1)
        c_emb = self.card_embedding(c_ids_clamped)  # [B, S_obj, 64]
        e_emb = self.evo_embedding(e_ids_clamped)   # [B, S_obj, 64]

        raw_text_384 = self.card_text_embedding(c_ids_clamped)  # [B, S_obj, 384]
        t_emb = self.text_proj(raw_text_384)                    # [B, S_obj, 64]

        raw_obj = torch.cat([object_features, c_emb, e_emb, t_emb], dim=-1)  # [B, S_obj, 558]

        # Token type embeddings
        if token_types is not None:
            obj_types = token_types.clamp(0, self.token_type_embedding.num_embeddings - 1)
        else:
            obj_types = torch.where(
                option_mask,
                torch.tensor(TOKEN_TYPE_OPTION, device=device),
                torch.tensor(1, device=device)
            )
        type_emb_obj = self.token_type_embedding(obj_types)  # [B, S_obj, d_model]

        # 2. Dual Encoder Projections
        state_obj_tok = self.state_entity_encoder(raw_obj) + type_emb_obj    # [B, S_obj, d_model]
        action_opt_tok = self.action_option_encoder(raw_obj) + type_emb_obj  # [B, S_obj, d_model]

        # Global token
        global_tok = self.global_encoder(global_features).unsqueeze(1)  # [B, 1, d_model]
        glob_type = torch.full((B, 1), TOKEN_TYPE_GLOBAL, dtype=torch.long, device=device)
        global_tok = global_tok + self.token_type_embedding(glob_type)

        # Assemble Game State sequence: [Global, State Objects]
        state_seq = torch.cat([global_tok, state_obj_tok], dim=1)  # [B, 1 + S_obj, d_model]

        # 3. State Attention Masking
        if padding_mask is not None:
            obj_pad_mask = padding_mask[:, 1:]  # [B, S_obj]
        else:
            obj_pad_mask = torch.zeros((B, S_obj), dtype=torch.bool, device=device)

        state_mask_obj = obj_pad_mask | option_mask  # True = mask out (invalid or is an option)
        glob_pad = torch.zeros((B, 1), dtype=torch.bool, device=device)
        state_key_padding_mask = torch.cat([glob_pad, state_mask_obj], dim=1)  # [B, 1 + S_obj]

        # 4. Contextualize Game State Entities via State Transformer
        state_ctx = self.state_transformer(
            state_seq,
            src_key_padding_mask=state_key_padding_mask
        )  # [B, 1 + S_obj, d_model]

        state_global = state_ctx[:, 0, :]  # [B, d_model]

        # 5. Pairwise Relational Cross-Attention (Approach A1)
        opt_q = self.cross_norm_q(action_opt_tok)  # [B, S_obj, d_model]
        rel_attn_mask = self._compute_pairwise_relation_bias(
            c_ids_clamped, e_ids_clamped, obj_types, state_key_padding_mask
        )

        if return_attention:
            cross_out, attn_weights = self.cross_attn(
                query=opt_q,
                key=state_ctx,
                value=state_ctx,
                attn_mask=rel_attn_mask,
                need_weights=True,
                average_attn_weights=False,
            )
        else:
            cross_out, _ = self.cross_attn(
                query=opt_q,
                key=state_ctx,
                value=state_ctx,
                attn_mask=rel_attn_mask,
                need_weights=False,
            )
            attn_weights = None

        action_repr = action_opt_tok + self.cross_dropout(cross_out)
        action_repr = action_repr + self.cross_ffn(self.cross_norm_ffn(action_repr))  # [B, S_obj, d_model]

        # 5b. Option-Set Self-Attention (Approach A2: Candidate options attend to competing options)
        if torch.abs(self.opt_self_gate).item() > 1e-8 or self.training:
            opt_normed = self.opt_self_norm(action_repr)
            # Ensure at least position 0 is unmasked per batch item to prevent NaN in softmax
            not_opt_mask = ~option_mask.clone()
            all_masked = not_opt_mask.all(dim=1)
            if all_masked.any():
                not_opt_mask[all_masked, 0] = False
            opt_self_out, _ = self.opt_self_attn(
                query=opt_normed,
                key=opt_normed,
                value=opt_normed,
                key_padding_mask=not_opt_mask,
                need_weights=False,
            )
            action_repr = action_repr + torch.tanh(self.opt_self_gate) * opt_self_out

        # 6. Action Scoring + IQL Action-Value Critic Q(s, a)
        state_glob_expanded = state_global.unsqueeze(1).expand(-1, S_obj, -1)
        action_pair = torch.cat([action_repr, state_glob_expanded], dim=-1)  # [B, S_obj, d_model * 2]
        raw_logits = self.action_scorer(action_pair).squeeze(-1)             # [B, S_obj]
        q_values = self.q_head(action_pair).squeeze(-1)                      # [B, S_obj] in [-1, 1]

        # Fuse learned IQL Q-advantage into logits via learnable gate
        action_logits = raw_logits + torch.tanh(self.q_logit_gate) * (2.5 * q_values)

        # Mask invalid actions
        valid_mask = option_mask & valid_action_mask
        action_logits = action_logits.masked_fill(~valid_mask, -1e9)
        probs = F.softmax(action_logits, dim=-1)

        # 7. Multi-Task Auxiliary Heads (Pooled State Representation)
        state_valid_weights = (~state_key_padding_mask).unsqueeze(-1).float()  # [B, 1 + S_obj, 1]
        state_pooled = (state_ctx * state_valid_weights).sum(dim=1) / state_valid_weights.sum(dim=1).clamp(min=1.0)
        aux_input = torch.cat([state_global, state_pooled], dim=-1)  # [B, d_model * 2]

        value = self.value_head(aux_input).squeeze(-1)                # [B]
        prize_margin = self.prize_margin_head(aux_input).squeeze(-1)  # [B]
        turn_horizon = self.turn_horizon_head(aux_input).squeeze(-1)  # [B]

        if return_q_and_repr:
            return probs, action_logits, value, prize_margin, turn_horizon, q_values, action_repr

        if return_attention:
            if return_aux:
                return probs, action_logits, value, prize_margin, turn_horizon, attn_weights
            return probs, action_logits, value, attn_weights

        if return_aux:
            return probs, action_logits, value, prize_margin, turn_horizon
        return probs, action_logits, value

    def select_multi_options_autoregressive(
        self,
        opt_scores: torch.Tensor,     # [N_opt]
        opt_repr: torch.Tensor,       # [N_opt, d_model]
        opt_card_ids: torch.Tensor,   # [N_opt]
        k: int,
        ctx: int = 0,
    ) -> list[int]:
        """
        Autoregressive Pointer Selection (Approach A3):
        Selects k options sequentially, updating candidate scores at each step m > 1
        using the learned conditional pointer synergy head `pointer_proj` and species diversity.
        """
        n_opt = opt_scores.shape[0]
        k = max(0, min(int(k), n_opt))
        if k == 0:
            return []
        if k == 1:
            return [int(torch.argmax(opt_scores).item())]

        selected = []
        used_mask = torch.zeros(n_opt, dtype=torch.bool, device=opt_scores.device)
        ptr_weight = float(torch.tanh(self.pointer_gate).item())

        for step in range(k):
            cur_scores = opt_scores.clone()
            cur_scores = cur_scores.masked_fill(used_mask, -1e9)

            if step > 0 and abs(ptr_weight) > 1e-6:
                chosen_repr = opt_repr[selected].mean(dim=0, keepdim=True)  # [1, d_model]
                synergy = torch.matmul(opt_repr, self.pointer_proj(chosen_repr).squeeze(0)) / (self.d_model ** 0.5)
                cur_scores = cur_scores + ptr_weight * synergy

            best_idx = int(torch.argmax(cur_scores).item())
            selected.append(best_idx)
            used_mask[best_idx] = True

        return selected
