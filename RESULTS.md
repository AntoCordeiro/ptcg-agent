# Results

Final numbers as reported in `report/main.pdf`. The two agents are named after their networks:
**DualCross** (first agent, imitation + RL, code in `gen1/`) and **PTCGNet** (second agent, network-guided
search, code in `ptcg/`). "Score" = fraction of games won by the candidate (A vs B in head-to-head; .50 = equal).

**Kaggle leaderboard (stabilised ratings, 2026-10-18):** best submission sub4 (PTCGNet, 1 world, expert iteration)
**773.2, 154th of 432**. DualCross: 647.7 (WiSE-FT) / 498.0 (one-step search).

All online numbers: candidate vs opponent, seat-balanced, Kaggle-like 600 s time bank per agent,
95% Wilson intervals. "Pool" (the gauntlet) = 6 fixed opponents: DualCross WiSE-FT (647.7 ladder, Grimmsnarl), DualCross final
(498.0 ladder, Hydrapple), and the PTCGNet policy piloting Dragapult / Ogerpon / Clefairy / Kangaskhan-Crustle.

## 1. DualCross evaluation audit (motivation)
| Comparison | Local result | Ladder |
|---|---|---|
| DualCross WiSE-FT (Grimmsnarl) vs DualCross final (Hydrapple), 350 games | WiSE-FT scores **0.317** [0.271, 0.368] (SPRT: H0) | WiSE-FT **647.7** vs final **498.0** |

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
| deployed (time bank), 1 world vs 4 worlds (run 1) | 200 | 0.558 [0.488, 0.625] | +40 [−8, +88] |
| deployed (time bank), 1 world vs 4 worlds (run 2) | 200 | 0.575 [0.506, 0.641] | +53 [+4, +101] |
| **deployed, 1 vs 4 worlds, pooled** | **400** | **0.566** [0.517, 0.614] | **+46** |
| MCTS-64 with root search over k-of-n card combinations vs policy-greedy k-of-n | 400 | 0.483 [0.434, 0.531] | −12 [−46, +22] |
| MCTS-64 c_puct 0.7 vs 1.25 | 240 | 0.513 [0.450, 0.575] | +9 [−35, +53] |
| MCTS-64 c_puct 2.5 vs 1.25 | 240 | **0.431** [0.370, 0.495] | **−48** [−92, −4], SPRT H0 |

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

### Expert iteration, stronger targets (1,600 MCTS self-play games, 64 sims, 1 world)
| A vs B (Crustle mirror) | games | score A [95% CI] | Elo |
|---|---|---|---|
| ExIt-2 policy vs base policy | 400 | 0.503 [0.454, 0.551] | +2 [−32, +36] |
| **ExIt-2 + MCTS-64 (1 world) vs base + MCTS-64 (1 world)** | **300** | **0.570** [0.513, 0.625] | **+49** [+9, +89] |

| ExIt-3 + MCTS-64 (1 world) vs ExIt-2 + MCTS-64 (1 world) | 300 | 0.502 [0.445, 0.558] | +1 [−38, +40] |

Offline: ExIt-2 replay top-1 68.8%, NLL 0.888, value Brier 0.2210; ExIt-3 68.6%, 0.890, 0.2195.
Round 3 (targets from the ExIt-2 search) brings no further gain: expert iteration plateaus after round 2 at this compute.
With better search targets, expert iteration improves the *search* agent (better priors/values for the tree)
without improving the raw policy — the network moves away from human imitation towards what search needs.

## 4c. Seed reproducibility (offline, test split)
| model | top-1 | NLL | value Brier | belief AUROC |
|---|---|---|---|---|
| seed 0 / + value FT | 70.0 / 70.1 | 0.832 / 0.838 | 0.2536 / 0.2258 | 0.752 |
| seed 1 / + value FT | 70.2 / 70.3 | 0.827 / 0.834 | 0.2435 / 0.2233 | 0.754 |
| no belief head (seed 0) | 70.3 | 0.831 | 0.2282 | — |

## 5. Kaggle submissions (stabilised ladder ratings)
| # | Package | Contents | Ladder |
|---|---|---|---|
| DualCross | (gen1) | WiSE-FT average of IQL + PPO models | **647.7** |
| DualCross | (gen1) | final, with one-step simulator checks | **498.0** |
| 1 | `submission/sub1_mcts_crustle.tar.gz` | PTCGNet (value-FT net) + MCTS (4 worlds, ≤2 s/decision) + Crustle | **736.6** |
| 2 | `submission/sub2_mcts_ogerpon.tar.gz` | same agent + Teal Mask Ogerpon deck | not used in the report |
| 3 | `submission/sub3_mcts1w_crustle.tar.gz` | PTCGNet (value-FT net) + MCTS **1 world** (≤2 s/decision) + Crustle | **768.4** |
| 4 | `submission/sub4_exit2_1w_crustle.tar.gz` | ExIt-2 net + MCTS 1 world + Crustle | **773.2 (best, 154th of 432)** — h2h vs sub3 config: 0.515 [0.446, 0.583], 200 games |
| 5 | `submission/sub5_exit2_1w_3s_lib.tar.gz` | sub4 + time cap 3 s (reserve 60 s, divisor 60) + 120-deck opponent library | **757.5** — h2h vs sub4 config: 0.538 [0.460, 0.613], 160 games; mean 185 s / max 300 s thinking per game |
| 6 | `submission/sub6_exit2_1w_6s_lib.tar.gz` | sub5 with time cap 6 s (reserve 90 s, divisor 35) | not uploaded — h2h vs sub5: 0.491 [0.426, 0.557], 220 games, 4.5 s/move: no gain |

Local prediction vs ladder: 1 world > 4 worlds locally (+46 Elo, 400 games) and on the ladder (768.4 vs 736.6),
unlike the DualCross evaluation, which ranked its two versions in the wrong order.

## 6. Ladder forensics (sub1 + sub3, 162 ladder episodes)
- No errors or timeouts from our agent; the two INVALID episodes were opponents forfeiting.
- Matches are best-of-3; the 600 s time bank **resets every game** (remainingOverageTime returns to 600 at each
  game boundary); with a 2 s cap we used ~350 s per match and never went below ~400 s remaining.
- Score by opponent rating: 600s 0.78 (36 eps), 700s 0.48 (81), 800s 0.36 (25), 900+ 0.00 (7).
- Many ladder opponents play decks outside our 8-deck library -> library expanded to 120 mined decklists (sub5).
- sub4/sub5 forensics (121 more ladder episodes): no errors/timeouts from our agent (INVALIDs are a broken opponent bot);
  sub5 scores 0.654 overall and 0.50 vs 800-rated opponents (sub4: 0.551 / 0.59).
