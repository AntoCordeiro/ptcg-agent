# Results (draft — updated as experiments finish)

All online numbers: candidate vs opponent, seat-balanced, Kaggle-like 600 s time bank per agent,
95% Wilson intervals. "Pool" = 6 fixed opponents: legacy v1 (≈650 ladder, Grimmsnarl), legacy latest
(≈500 ladder, Hydrapple), and our BC policy piloting Dragapult / Ogerpon / Clefairy / Kangaskhan-Crustle.

## 1. Legacy system audit (motivation)
| Comparison | Local result | Ladder |
|---|---|---|
| legacy v1 (Grimmsnarl) vs legacy latest (Hydrapple), 350 games | v1 scores **0.317** [0.271, 0.368] (SPRT: H0) | v1 **650+** vs latest **~500** |

Local head-to-head between our own agents ranked them in the *opposite* order to the ladder.

## 2. Offline model quality (held-out test games, 20.5k decisions)
| Metric | Model | Baseline |
|---|---|---|
| Policy top-1 / top-3 | **70.0% / 93.7%** | uniform legal 21.7% |
| Policy NLL | **0.832** | uniform 1.747 |
| MAIN phase with ≥5 options top-1 | **60.7%** | uniform 11.6% |
| Value Brier, replay-only head | 0.2536 | constant 0.2493 (worse than constant: overfit) |
| Value Brier, + self-play fine-tuning | **0.2258** | constant 0.2493 |
| Value ECE | 0.190 → **0.134** | |
| Belief AUROC (opponent hand) | learned **0.752**, learned×count **0.761** | count-based w/ true decklist 0.711, random 0.500 |
| Belief mAP | learned **0.558** | count 0.486, random 0.369 |
| Self-play value MSE (held-out games) | 1.130 → **0.638** (acc 63% → 76%) | |

Belief head as auxiliary loss does not change policy quality: test NLL 0.832 (with) vs 0.831 (without).

## 3. Deck selection for our agent (raw policy, 240 games each vs pool)
| Deck | Score [95% CI] |
|---|---|
| Mega Kangaskhan ex / Crustle | **0.731** [0.672, 0.783] |
| Teal Mask Ogerpon ex | 0.650 [0.588, 0.708] |
| Hydrapple ex | 0.644 [0.581, 0.702] |
| Grimmsnarl ex | 0.592 [0.529, 0.652] |
| Dragapult ex (best deck at top-Elo, 57% there) | 0.494 [0.431, 0.557] |
| Lillie's Clefairy ex | 0.279 [0.226, 0.339] |
| Dragapult/Clefairy | 0.287 [0.234, 0.348] |
| Raging Bolt/Ogerpon | 0.200 [0.154, 0.255] |

## 4. Search
| Agent (Crustle deck) | Pool score [95% CI] |
|---|---|
| raw policy (value-FT net) | 0.750 [0.692, 0.801] |
| MCTS 16 sims, 4 worlds | 0.813 [0.758, 0.857] |
| MCTS 64 sims, 4 worlds | 0.802 [0.747, 0.848] |
| MCTS 256 sims, 4 worlds | **0.873** [0.825, 0.909] |
| **deployed config (time bank, ≤2 s/decision, 4 worlds)** | **0.903** [0.851, 0.938] (180 games; mean 84 s thinking/game, max 158 s, 0 timeouts) |

Head-to-head (mirror, Crustle vs Crustle):
| A vs B | games | score A | Elo |
|---|---|---|---|
| MCTS-64 vs raw policy | 200 | **0.593** [0.523, 0.658] | +65 |
| MCTS-64 learned-belief worlds vs uniform (count-based) worlds | 400 | 0.479 [0.430, 0.528] | −15 [−49, +19], SPRT H0 |
| MCTS-256 learned-belief worlds vs uniform worlds | 208 | 0.517 [0.449, 0.584] | +12 [−35, +59] |
| MCTS-64, 4 worlds vs 1 world (same total simulations) | 300 | **0.393** [0.340, 0.450] | **−75** [−115, −35], SPRT H0 |
| deployed (time bank), 1 world vs 4 worlds | 200 | 0.558 [0.488, 0.625] | +40 [−8, +88] |

Seed robustness of the search gain (seed-1 network, Crustle vs pool): policy 0.792 [0.736, 0.838] → MCTS-64 0.850 [0.799, 0.890]
(seed 0: 0.750 → 0.802).

Interpretation: at our budgets, depth beats averaging over hidden-information samples — one deeper tree
outperforms several shallow determinized trees, which also explains why better opponent-hand sampling
(learned belief) gives no measurable online gain.

## 4b. Expert iteration (1 round: 2,400 MCTS-16 self-play games, visit distributions as targets)
| A vs B (Crustle mirror) | games | score A [95% CI] |
|---|---|---|
| ExIt policy vs base policy | 400 | 0.511 [0.462, 0.560] |
| ExIt + MCTS-64 vs base + MCTS-64 | 304 | 0.467 [0.412, 0.523] |
| ExIt policy vs pool (gauntlet) | 240 | 0.708 [0.648, 0.762] (base policy 0.750) |

Offline: ExIt net replay top-1 69.0% (base 70.1%), value Brier 0.2187 (base 0.2258).
One round with 16-simulation targets did not improve playing strength.

## 4c. Seed reproducibility (offline, test split)
| model | top-1 | NLL | value Brier | belief AUROC |
|---|---|---|---|---|
| seed 0 / + value FT | 70.0 / 70.1 | 0.832 / 0.838 | 0.2536 / 0.2258 | 0.752 |
| seed 1 / + value FT | 70.2 / 70.3 | 0.827 / 0.834 | 0.2435 / 0.2233 | 0.754 |
| no belief head (seed 0) | 70.3 | 0.831 | 0.2282 | — |

## 5. Kaggle submissions
| # | Package | Contents | Ladder |
|---|---|---|---|
| 1 | `submission/sub1_mcts_crustle.tar.gz` | value-FT net + MCTS (4 worlds, ≤2 s/decision) + Crustle | (pending upload) |
| 2 | `submission/sub2_mcts_ogerpon.tar.gz` | same agent + Teal Mask Ogerpon deck | (optional) |
| 3 | `submission/sub3_mcts1w_crustle.tar.gz` | value-FT net + MCTS **1 world** (≤2 s/decision) + Crustle | (pending upload) |
