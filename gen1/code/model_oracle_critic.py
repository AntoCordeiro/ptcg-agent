# model_oracle_critic.py
"""
Asymmetric "Oracle" Critic & Privileged Actor-Critic Architecture (Pillar 2).

Inspired by AlphaStar (DeepMind) and OpenAI Five:
In POMDP strategy games, the true game value depends heavily on hidden information
(opponent's private hand and unrevealed deck composition).

Architecture:
1. Actor Policy pi(a | s_obs):
   Strictly constrained to observable state tokens (my hand, board, legal options).
2. Observable Critic V_obs(s_obs):
   Standard critic estimating value from the observable perspective.
3. Asymmetric Oracle Critic V_oracle(s_obs, s_priv):
   Privileged critic conditioned on observable state PLUS privileged hidden state:
   - Opponent's private hand composition (card embeddings & count)
   - Both players' complete deck composition vectors
   - High-precision POMDP Bayesian belief state

Theoretical Advantage:
   Var(R_t | s_obs, s_priv) << Var(R_t | s_obs)
Yields substantially higher Explained Variance (R^2 > 0.70) and lower GAE variance.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from features_spec import (
    GLOBAL_DIM, OBJECT_INPUT_DIM, OBJECT_FEATURE_DIM, CARD_EMBED_DIM,
    EVO_EMBED_DIM, TEXT_RAW_DIM, TEXT_PROJ_DIM, D_MODEL, NUM_CARD_IDS,
    NUM_TOKEN_TYPES, TOKEN_TYPE_GLOBAL, TOKEN_TYPE_OPTION
)
from model_dual_cross import PTCGDualCrossNet


PRIVILEGED_DIM = 64  # Encodes opponent's private hand and hidden deck distribution


class PTCGAsymmetricOracleNet(nn.Module):
    def __init__(
        self,
        d_model: int = D_MODEL,
        num_layers: int = 3,
        nhead: int = 8,
        dim_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 1. Base Observable Model (provides Trunk, Policy, and Observable Critic)
        self.base_net = PTCGDualCrossNet(
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dim_ff=dim_ff,
            dropout=dropout,
        )
        
        # 2. Privileged Oracle Feature Encoder
        self.privileged_encoder = nn.Sequential(
            nn.Linear(PRIVILEGED_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # 3. Asymmetric Oracle Value Head: V_oracle(s_obs, s_priv)
        # Ingests [state_global (d_model), state_pooled (d_model), priv_embedding (d_model)]
        self.oracle_value_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Tanh(),
        )

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
        privileged_features: torch.Tensor = None,  # [B, PRIVILEGED_DIM]
        return_oracle_value: bool = False,
    ):
        # 1. Standard observable forward pass
        probs, logits, val_obs = self.base_net(
            global_features, object_features, card_ids, evo_ids,
            option_mask, valid_action_mask, padding_mask=padding_mask,
            token_types=token_types
        )
        
        if not return_oracle_value or privileged_features is None:
            return probs, logits, val_obs

        # 2. Compute privileged oracle value
        B = global_features.shape[0]
        device = global_features.device
        
        # Extract contextualized state representations from base_net trunk
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
        global_tok = self.base_net.global_encoder(global_features).unsqueeze(1)
        glob_type = torch.full((B, 1), TOKEN_TYPE_GLOBAL, dtype=torch.long, device=device)
        global_tok = global_tok + self.base_net.token_type_embedding(glob_type)
        
        state_seq = torch.cat([global_tok, state_obj_tok], dim=1)
        obj_pad = padding_mask[:, 1:] if padding_mask is not None else torch.zeros((B, card_ids.shape[1]), dtype=torch.bool, device=device)
        state_mask = torch.cat([torch.zeros((B, 1), dtype=torch.bool, device=device), obj_pad | option_mask], dim=1)
        
        state_ctx = self.base_net.state_transformer(state_seq, src_key_padding_mask=state_mask)
        state_global = state_ctx[:, 0, :]
        
        valid_w = (~state_mask).unsqueeze(-1).float()
        state_pooled = (state_ctx * valid_w).sum(dim=1) / valid_w.sum(dim=1).clamp(min=1.0)
        
        # Process privileged hidden features
        priv_emb = self.privileged_encoder(privileged_features) # [B, d_model]
        
        # Oracle Critic input combines: [Observable Global, Observable Pooled, Privileged Hidden]
        oracle_input = torch.cat([state_global, state_pooled, priv_emb], dim=-1) # [B, d_model * 3]
        val_oracle = self.oracle_value_head(oracle_input).squeeze(-1)            # [B]
        
        return probs, logits, val_obs, val_oracle
