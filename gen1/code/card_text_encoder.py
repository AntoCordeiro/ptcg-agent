# card_text_encoder.py
import os
import torch
import torch.nn.functional as F
import numpy as np

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_EMBEDDINGS_384_PATH = os.path.join(_CURR_DIR, "card_text_embeddings_384.npy") if os.path.exists(os.path.join(_CURR_DIR, "card_text_embeddings_384.npy")) else "card_text_embeddings_384.npy"
TEXT_EMBEDDINGS_192_PATH = os.path.join(_CURR_DIR, "card_text_embeddings_192.npy") if os.path.exists(os.path.join(_CURR_DIR, "card_text_embeddings_192.npy")) else "card_text_embeddings_192.npy"
TEXT_EMBEDDINGS_64_PATH = os.path.join(_CURR_DIR, "card_text_embeddings.npy") if os.path.exists(os.path.join(_CURR_DIR, "card_text_embeddings.npy")) else "card_text_embeddings.npy"
CSV_PATH = "the-pokemon-company-ptcg-ai-battle-challenge-playground/EN_Card_Data_R2_full.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NUM_CARDS = 1500

_CACHE_384 = None
_CACHE_192 = None
_CACHE_64 = None

def build_card_descriptions(csv_path: str = CSV_PATH) -> dict[int, str]:
    if not os.path.exists(csv_path):
        alt_paths = [
            "the-pokemon-company-ptcg-ai-battle-challenge-playground/EN_Card_Data_R2_full.csv",
            "../the-pokemon-company-ptcg-ai-battle-challenge-playground/EN_Card_Data_R2_full.csv",
            "EN_Card_Data_R2_full.csv",
        ]
        for p in alt_paths:
            if os.path.exists(p):
                csv_path = p
                break

    import pandas as pd
    df = pd.read_csv(csv_path)
    card_texts = {}

    grouped = df.groupby("Card ID")
    for cid, group in grouped:
        cid_int = int(cid)
        if cid_int >= NUM_CARDS:
            continue

        first_row = group.iloc[0]
        name = str(first_row.get("Card Name", "Unknown"))
        stage = str(first_row.get("Stage (Pokémon)/Type (Energy and Trainer)", ""))
        rule = str(first_row.get("Rule", ""))
        hp = str(first_row.get("HP", ""))
        card_type = str(first_row.get("Type", ""))

        moves = []
        for _, r in group.iterrows():
            m_name = str(r.get("Move Name", ""))
            m_cost = str(r.get("Cost", ""))
            m_dmg = str(r.get("Damage", ""))
            m_eff = str(r.get("Effect Explanation", ""))

            if m_name and m_name != "nan":
                m_str = f"Move: {m_name}"
                if m_cost and m_cost != "nan":
                    m_str += f" Cost: {m_cost}"
                if m_dmg and m_dmg != "nan":
                    m_str += f" Damage: {m_dmg}"
                if m_eff and m_eff != "nan":
                    m_str += f" - {m_eff}"
                moves.append(m_str)
            elif m_eff and m_eff != "nan":
                moves.append(f"Effect: {m_eff}")

        header_parts = [name]
        if stage and stage != "nan":
            header_parts.append(f"Category: {stage}")
        if hp and hp != "nan":
            header_parts.append(f"HP: {hp}")
        if card_type and card_type != "nan":
            header_parts.append(f"Type: {card_type}")
        if rule and rule != "nan":
            header_parts.append(f"Rule: {rule}")

        desc = ". ".join(header_parts) + ". " + " ".join(moves)
        card_texts[cid_int] = desc.strip()

    return card_texts

def compute_all_embeddings(csv_path: str = CSV_PATH, batch_size: int = 128):
    from transformers import AutoTokenizer, AutoModel
    from sklearn.decomposition import PCA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Building card text descriptions from '{csv_path}'...")
    card_texts_map = build_card_descriptions(csv_path)

    print(f"Encoding {len(card_texts_map)} cards using '{MODEL_NAME}' on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    cids_ordered = sorted(card_texts_map.keys())
    texts_ordered = [card_texts_map[cid] for cid in cids_ordered]

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts_ordered), batch_size):
            batch_texts = texts_ordered[i:i + batch_size]
            inputs = tokenizer(
                batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(device)
            out = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(out.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            normed = F.normalize(mean_pooled, p=2, dim=1)
            all_embs.append(normed.cpu().numpy())

    raw_matrix_384 = np.vstack(all_embs)
    print("Full 384-dim uncompressed embeddings computed (100% variance).")

    # 1. Save Full 384-dim (100% variance)
    table_384 = np.zeros((NUM_CARDS, 384), dtype=np.float32)
    for idx, cid in enumerate(cids_ordered):
        if cid < NUM_CARDS:
            table_384[cid] = raw_matrix_384[idx]
    np.save(TEXT_EMBEDDINGS_384_PATH, table_384)
    print(f"Saved full-variance embeddings to '{TEXT_EMBEDDINGS_384_PATH}' (100% variance, shape: {table_384.shape}).")

    # 2. PCA 192-dim (95.8% variance)
    pca_192 = PCA(n_components=192, random_state=42)
    reduced_192 = pca_192.fit_transform(raw_matrix_384)
    reduced_192 = reduced_192 / np.maximum(np.linalg.norm(reduced_192, axis=1, keepdims=True), 1e-8)
    var_192 = np.sum(pca_192.explained_variance_ratio_) * 100.0

    table_192 = np.zeros((NUM_CARDS, 192), dtype=np.float32)
    for idx, cid in enumerate(cids_ordered):
        if cid < NUM_CARDS:
            table_192[cid] = reduced_192[idx]
    np.save(TEXT_EMBEDDINGS_192_PATH, table_192)
    print(f"Saved 192-dim embeddings to '{TEXT_EMBEDDINGS_192_PATH}' ({var_192:.2f}% variance, shape: {table_192.shape}).")

    # 3. PCA 64-dim (68.1% variance)
    pca_64 = PCA(n_components=64, random_state=42)
    reduced_64 = pca_64.fit_transform(raw_matrix_384)
    reduced_64 = reduced_64 / np.maximum(np.linalg.norm(reduced_64, axis=1, keepdims=True), 1e-8)
    var_64 = np.sum(pca_64.explained_variance_ratio_) * 100.0

    table_64 = np.zeros((NUM_CARDS, 64), dtype=np.float32)
    for idx, cid in enumerate(cids_ordered):
        if cid < NUM_CARDS:
            table_64[cid] = reduced_64[idx]
    np.save(TEXT_EMBEDDINGS_64_PATH, table_64)
    print(f"Saved 64-dim embeddings to '{TEXT_EMBEDDINGS_64_PATH}' ({var_64:.2f}% variance, shape: {table_64.shape}).")

    return table_384, table_192, table_64

def get_card_text_embeddings_384() -> np.ndarray:
    global _CACHE_384
    if _CACHE_384 is not None:
        return _CACHE_384
    paths_to_check = [
        TEXT_EMBEDDINGS_384_PATH,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "card_text_embeddings_384.npy"),
        "card_text_embeddings_384.npy"
    ]
    for p in paths_to_check:
        if os.path.exists(p):
            try:
                _CACHE_384 = np.load(p)
                return _CACHE_384
            except Exception:
                pass
    return np.zeros((NUM_CARDS, 384), dtype=np.float32)

def get_card_text_embeddings_192() -> np.ndarray:
    global _CACHE_192
    if _CACHE_192 is not None:
        return _CACHE_192
    paths_to_check = [
        TEXT_EMBEDDINGS_192_PATH,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "card_text_embeddings_192.npy"),
        "card_text_embeddings_192.npy"
    ]
    for p in paths_to_check:
        if os.path.exists(p):
            try:
                _CACHE_192 = np.load(p)
                return _CACHE_192
            except Exception:
                pass
    return np.zeros((NUM_CARDS, 192), dtype=np.float32)

def get_card_text_embeddings(dim: int = 64) -> np.ndarray:
    if dim == 384:
        return get_card_text_embeddings_384()
    elif dim == 192:
        return get_card_text_embeddings_192()
    else:
        global _CACHE_64
        if _CACHE_64 is not None:
            return _CACHE_64
        paths_to_check = [
            TEXT_EMBEDDINGS_64_PATH,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "card_text_embeddings.npy"),
            "card_text_embeddings.npy"
        ]
        for p in paths_to_check:
            if os.path.exists(p):
                try:
                    _CACHE_64 = np.load(p)
                    return _CACHE_64
                except Exception:
                    pass

        # Fallback to 384-dim slice if 64-dim file is absent
        emb384 = get_card_text_embeddings_384()
        if emb384 is not None and emb384.shape[1] >= 64:
            _CACHE_64 = emb384[:, :64].copy()
            return _CACHE_64

        _CACHE_64 = np.zeros((NUM_CARDS, 64), dtype=np.float32)
        return _CACHE_64

if __name__ == "__main__":
    compute_all_embeddings()
