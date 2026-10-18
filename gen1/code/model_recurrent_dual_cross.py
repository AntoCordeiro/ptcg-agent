# model_recurrent_dual_cross.py
"""
Recurrent Cross-Attention Architecture for Sequential POMDP Modeling (Pillar 3).

Extends PTCGDualCrossNet with a Gated Recurrent Unit (GRU) Temporal Memory Core:
    h_t = GRU(z_t, h_{t-1})
where z_t is the pooled contextualized state representation at turn t.

Theoretical Rationale:
In partial-observability card games unfolding over 50-150 turns, a memoryless policy
cannot maintain temporal continuity across turns unless explicitly encoded.
The recurrent hidden memory h_t integrates historical card reveals, opponent tempo shifts,
and resource expenditures across turns, enabling temporal credit assignment.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from features_spec import (
    GLOBAL_DIM, OBJECT_INPUT_DIM, OBJECT_FEATURE_DIM, CARD_EMBED_DIM,
    EVO_EMBED_DIM, TEXT_RAW_DIM, TEXT_PROJ_DIM, D_MODEL, NUM_CARD_IDS,
    NUM_TOKEN_TYPES, TOKEN_TYPE_GLOBAL, TOKEN_TYPE_OPTION
)
from model_dual_cross import PTCGDualCrossNet


class PTCGRecurrentCrossNet(nn.Module):
    def __init__(
        self,
        d_model: int = D_MODEL,
        num_layers: int = 3,
        nhead: int = 8,
        dim_ff: int = 512,
        dropout: float = 0.1,
        rnn_hidden_dim: int = D_MODEL,
    ):
        super().__init__()
        self.d_model = d_model
        self.rnn_hidden_dim = rnn_hidden_dim

        # 1. Base Transformer Encoder & Cross-Attention Trunk
        self.base_net = PTCGDualCrossNet(
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dim_ff=dim_ff,
            dropout=dropout,
        )

        # 2. Recurrent Memory Core: Gated Recurrent Unit (GRU)
        # Ingests the contextualized pooled state representation z_t (d_model)
        self.gru_core = nn.GRUCell(input_size=d_model, hidden_size=rnn_hidden_dim)
        self.gru_norm = nn.LayerNorm(rnn_hidden_dim)

        # 3. Recurrent Action Scoring Head: Conditioned on [Option Repr, GRU Memory]
        self.recurrent_action_scorer = nn.Sequential(
            nn.Linear(d_model + rnn_hidden_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # 4. Recurrent Value Head: V(s_t, h_t)
        self.recurrent_value_head = nn.Sequential(
            nn.Linear(d_model + rnn_hidden_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Tanh(),
        )

    def init_hidden(self, batch_size: int = 1, device: torch.device = None) -> torch.Tensor:
        """Returns initialized zero memory vector [B, rnn_hidden_dim]."""
        dev = device or next(self.parameters()).device
        return torch.zeros(batch_size, self.rnn_hidden_dim, device=dev)

    def forward(
        self,
        global_features: torch.Tensor,
        object_features: torch.Tensor,
        card_ids: torch.Tensor,
        evo_ids: torch.Tensor,
        option_mask: torch.Tensor,
        valid_action_mask: torch.Tensor,
        padding_mask: torch.Tensor = None,
        token_types: torch.Tensor = None,
        hidden_state: torch.Tensor = None,  # [B, rnn_hidden_dim]
    ):
        B, S_obj, _ = object_features.shape
        device = global_features.device

        if hidden_state is None:
            hidden_state = self.init_hidden(B, device)

        # 1. Embeddings & Pre-LN Transformer trunk
        c_clamped = card_ids.clamp(0, self.base_net.card_embedding.num_embeddings - 1)
        e_clamped = evo_ids.clamp(0, self.base_net.evo_embedding.num_embeddings - 1)
        raw_text = self.base_net.card_text_embedding(c_clamped)
        t_emb = self.base_net.text_proj(raw_text)

        raw_obj = torch.cat([
            object_features,
            self.base_net.card_embedding(c_clamped),
            self.base_net.evo_embedding(e_clamped),
            t_emb
        ], dim=-1)

        if token_types is not None:
            obj_types = token_types.clamp(0, self.base_net.token_type_embedding.num_embeddings - 1)
        else:
            obj_types = torch.zeros(card_ids.shape, dtype=torch.long, device=device)
        type_emb = self.base_net.token_type_embedding(obj_types)

        state_obj_tok = self.base_net.state_entity_encoder(raw_obj) + type_emb
        action_opt_tok = self.base_net.action_option_encoder(raw_obj) + type_emb

        global_tok = self.base_net.global_encoder(global_features).unsqueeze(1)
        glob_type = torch.full((B, 1), TOKEN_TYPE_GLOBAL, dtype=torch.long, device=device)
        global_tok = global_tok + self.base_net.token_type_embedding(glob_type)

        state_seq = torch.cat([global_tok, state_obj_tok], dim=1)
        obj_pad = padding_mask[:, 1:] if padding_mask is not None else torch.zeros((B, S_obj), dtype=torch.bool, device=device)
        state_mask = torch.cat([torch.zeros((B, 1), dtype=torch.bool, device=device), obj_pad | option_mask], dim=1)

        state_ctx = self.base_net.state_transformer(state_seq, src_key_padding_mask=state_mask)
        state_global = state_ctx[:, 0, :]

        # 2. Asymmetric Cross-Attention
        opt_q = self.base_net.cross_norm_q(action_opt_tok)
        cross_out, _ = self.base_net.cross_attn(
            query=opt_q, key=state_ctx, value=state_ctx,
            key_padding_mask=state_mask, need_weights=False
        )
        action_repr = action_opt_tok + self.base_net.cross_dropout(cross_out)
        action_repr = action_repr + self.base_net.cross_ffn(self.base_net.cross_norm_ffn(action_repr))

        # 3. Contextualized Pooled Representation
        valid_w = (~state_mask).unsqueeze(-1).float()
        state_pooled = (state_ctx * valid_w).sum(dim=1) / valid_w.sum(dim=1).clamp(min=1.0)
        z_t = state_pooled  # Current state summary token [B, d_model]

        # 4. Recurrent Memory Update
        new_hidden = self.gru_core(z_t, hidden_state) # [B, rnn_hidden_dim]
        new_hidden_norm = self.gru_norm(new_hidden)

        # 5. Temporal Action Scoring
        hidden_expanded = new_hidden_norm.unsqueeze(1).expand(-1, S_obj, -1)
        action_pair = torch.cat([action_repr, hidden_expanded], dim=-1)
        logits = self.recurrent_action_scorer(action_pair).squeeze(-1)

        valid_mask = option_mask & valid_action_mask
        logits = logits.masked_fill(~valid_mask, -1e9)
        probs = F.softmax(logits, dim=-1)

        # 6. Recurrent Value Head
        val_input = torch.cat([state_global, new_hidden_norm], dim=-1)
        value = self.recurrent_value_head(val_input).squeeze(-1)

        return probs, logits, value, new_hidden
