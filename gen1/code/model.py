# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from features_spec import (
    GLOBAL_DIM, OBJECT_INPUT_DIM, CARD_EMBED_DIM, 
    EVO_EMBED_DIM, TEXT_RAW_DIM, TEXT_PROJ_DIM, NUM_CARD_IDS, D_MODEL, NUM_TOKEN_TYPES,
    TOKEN_TYPE_GLOBAL, TOKEN_TYPE_OPTION
)
from card_text_encoder import get_card_text_embeddings_384

class PTCGNet(nn.Module):
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
    ):
        super().__init__()
        self.d_model = d_model
        
        # 1. Learnable embeddings
        self.card_embedding = nn.Embedding(num_cards, card_emb_dim, padding_idx=0)
        self.evo_embedding = nn.Embedding(num_cards, evo_emb_dim, padding_idx=0)
        self.token_type_embedding = nn.Embedding(num_token_types, d_model)
        
        # 2. 100% Full-variance uncompressed 384-dim semantic card text embeddings from MiniLM
        text_weights = torch.from_numpy(get_card_text_embeddings_384()).float()
        self.card_text_embedding = nn.Embedding.from_pretrained(text_weights, freeze=True)
        
        # Task-specific non-linear projection: 384 -> 128 -> 64
        self.text_proj = nn.Sequential(
            nn.Linear(text_raw_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, text_proj_dim),
        )
        
        # 3. Token Encoders with LayerNorm
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # Object Encoder: 558 (366 features + 64 card ID + 64 evo ID + 64 text proj) -> 256
        self.object_encoder = nn.Sequential(
            nn.Linear(OBJECT_INPUT_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # 4. Transformer Trunk (Pre-LN, 8 heads, GELU)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Multi-Task Heads
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Tanh(),
        )
        
        self.prize_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        
        # 6. Relational Action Pointer Head
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5
        
        self.relational_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

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
    ):
        B, S_obj, _ = object_features.shape
        device = global_features.device
        
        # 1. Encoders
        global_tok = self.global_encoder(global_features).unsqueeze(1) # [B, 1, d_model]
        
        c_ids_clamped = card_ids.clamp(0, self.card_embedding.num_embeddings - 1)
        e_ids_clamped = evo_ids.clamp(0, self.evo_embedding.num_embeddings - 1)
        c_emb = self.card_embedding(c_ids_clamped) # [B, S_obj, 64]
        e_emb = self.evo_embedding(e_ids_clamped)   # [B, S_obj, 64]
        
        # Full 100% variance 384-dim uncompressed text lookup + task-specific projection
        raw_text_384 = self.card_text_embedding(c_ids_clamped) # [B, S_obj, 384]
        t_emb = self.text_proj(raw_text_384)                   # [B, S_obj, 64]
        
        raw_obj = torch.cat([object_features, c_emb, e_emb, t_emb], dim=-1) # [B, S_obj, 558]
        obj_tokens = self.object_encoder(raw_obj) # [B, S_obj, d_model]
        
        # 2. Token Type Embeddings
        if token_types is not None:
            obj_types = token_types.clamp(0, self.token_type_embedding.num_embeddings - 1)
        else:
            obj_types = torch.where(
                option_mask,
                torch.tensor(TOKEN_TYPE_OPTION, device=device),
                torch.tensor(1, device=device)
            )
        
        glob_types = torch.full((B, 1), TOKEN_TYPE_GLOBAL, dtype=torch.long, device=device)
        all_types = torch.cat([glob_types, obj_types], dim=1) # [B, 1 + S_obj]
        type_emb = self.token_type_embedding(all_types)
        
        all_tokens = torch.cat([global_tok, obj_tokens], dim=1) + type_emb # [B, 1 + S_obj, d_model]
        
        # 3. Transformer Trunk
        ctx = self.transformer(all_tokens, src_key_padding_mask=padding_mask)
        
        ctx_global = ctx[:, 0, :]   # [B, d_model]
        ctx_objects = ctx[:, 1:, :] # [B, S_obj, d_model]
        
        # 4. Value & Auxiliary Heads
        if padding_mask is not None:
            mask_weight = (~padding_mask).unsqueeze(-1).float()
            token_mean = (ctx * mask_weight).sum(dim=1) / mask_weight.sum(dim=1).clamp(min=1.0)
        else:
            token_mean = ctx.mean(dim=1)
            
        pooled_repr = torch.cat([ctx_global, token_mean], dim=-1)
        value = self.value_head(pooled_repr).squeeze(-1)       # [B]
        prize_pred = self.prize_head(pooled_repr).squeeze(-1)  # [B]
        
        # 5. Relational Action Pointer Head
        query = self.query_proj(ctx_global).unsqueeze(1) # [B, 1, d_model]
        keys = self.key_proj(ctx_objects)                # [B, S_obj, d_model]
        dot_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / self.scale # [B, S_obj]
        
        glob_expanded = ctx_global.unsqueeze(1).expand(-1, S_obj, -1)
        rel_pair = torch.cat([ctx_objects, glob_expanded], dim=-1)
        rel_scores = self.relational_scorer(rel_pair).squeeze(-1) # [B, S_obj]
        
        total_scores = dot_scores + rel_scores
        
        valid_mask = option_mask & valid_action_mask
        total_scores = total_scores.masked_fill(~valid_mask, -1e9)
        
        probs = F.softmax(total_scores, dim=-1)
        
        if return_aux:
            return probs, total_scores, value, prize_pred
        return probs, total_scores, value