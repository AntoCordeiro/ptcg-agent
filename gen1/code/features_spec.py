# features_spec.py

GLOBAL_DIM = 191        # Table 1: Turn/Seat(10) + Prizes/Board(30) + Request(70) + Effects(53) + Logs(27) + 1 pad
OBJECT_FEATURE_DIM = 366 # Table 2: Role(4) + Static(132) + Zone(20) + Instance(112) + Option(98)
CARD_EMBED_DIM = 64     # Learnable card ID embedding
EVO_EMBED_DIM = 64      # Learnable evolution history embedding
TEXT_RAW_DIM = 384      # Full uncompressed MiniLM representation (100% variance)
TEXT_PROJ_DIM = 64      # Learnable task-specific projection from 384-dim semantics
OBJECT_INPUT_DIM = OBJECT_FEATURE_DIM + CARD_EMBED_DIM + EVO_EMBED_DIM + TEXT_PROJ_DIM # 558 dims total

MAX_TOKENS = 141        # Figure 2: 1 Global + up to 140 Object tokens
MAX_OPTIONS = 58        # Up to 58 legal option tokens
NUM_CARD_IDS = 1500     # Covers all 1,431 card IDs safely
D_MODEL = 256

# Token entity type indices for learned positional/type embeddings
TOKEN_TYPE_GLOBAL = 0
TOKEN_TYPE_ACTIVE_MINE = 1
TOKEN_TYPE_BENCH_MINE = 2
TOKEN_TYPE_ACTIVE_OPP = 3
TOKEN_TYPE_BENCH_OPP = 4
TOKEN_TYPE_HAND_MINE = 5
TOKEN_TYPE_STADIUM = 6
TOKEN_TYPE_OPTION = 7
NUM_TOKEN_TYPES = 8