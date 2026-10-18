# PTCG Agent — Restart Plan (v2, 24-hour edition, for approval)

**Date:** 2026-10-17 (Sat, 20:30 Rome) · **Hard deadline:** Sun 2026-10-18 evening · **Solo project**
**Goals:** (1) a strong Kaggle leaderboard score, (2) a demanding DL exam project (DLAI 2025/26, BYOP, "gaming and AI"), (3) a rigorous evaluation. The repo and the 2-page PDF report come afterwards. This plan produces everything they need: code, numbers, figures.

---

## 0. TL;DR

- **The diagnosis (now confirmed by the ladder).**
  - The last submission (`search_v4 + Hydrapple`) **scores ~500**. The one before it (`wise + Grimmsnarl`, "Rebuild v1") **scores 650+**.
  - Locally, the new one "won 80%, +240 Elo". The local evaluation said the opposite of the truth.
  - So the bottleneck is **evaluation that can't tell signal from noise**, plus **an agent that hardly uses the simulator or its time budget**, not the network architecture.
- **The new system, scoped for 24 hours:**
  - a **compact, clean transformer** trained on high-Elo replays, with a policy head, a value head and an **auxiliary head that predicts the opponent's hidden hand**;
  - **neural-guided determinized MCTS** on the C++ simulator, using the ~600 s time bank;
  - **one deck**, chosen from ladder data.
- **Evaluation:**
  - fixed gauntlet, hundreds of games per pairing, Wilson confidence intervals, SPRT gating before every upload;
  - a held-out test set and offline metrics against simple baselines;
  - an ablation table.
  - Rebuild v1 (650+) is the **reference to beat** and the **safety fallback**.
- **Dropped for time:** expert iteration, PPO, IQL, WiSE averaging, multi-deck generalisation, multiple training seeds for everything. They become "future work" in the report.

---

## 1. Facts that shape the plan (verified)

| Fact | Consequence |
|---|---|
| `actTimeout = 0`, and each agent has a **~600 s overage bank per episode** | That is about **1–3 s per decision**. The current agent uses **~6 ms**. Search can use about 100× more compute than today. |
| `cg.api.search_begin(obs, your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active)` + `search_step` | A full forward model that accepts **guessed hidden cards**. Determinized MCTS is directly possible. |
| Replays contain **both players' observations**, and each player's observation shows its own hand | The opponent's hand is a **free supervised label**. This gives the auxiliary belief head. |
| Ladder: Rebuild v1 **650+**, latest **~500**, despite the "+240 local Elo" | The old local benchmark is invalid. This is a strong motivating finding for the report. |
| Kaggle runtime: **2 vCPU, no GPU** | The network must be small (~1–2 M parameters) and fast on CPU. |
| Machines I can drive: cloud workspace and a VM on your PC, both **2 CPU cores, no GPU, no torch** | **GPU training and large evaluation runs must be launched by you** on your Windows PC (RTX 4070 Ti SUPER). I write single-command scripts; you run them and tell me when they finish. |
| Only the **latest 2 submissions are active** on Kaggle | Every upload deactivates an older one, so uploads must be planned (§6). |

---

## 2. Review of the current work (unchanged from v1, abridged)

**Critical issues**
1. **Evaluation can't separate signal from noise.**
   - 20 games per pair gives a confidence interval of about ±22 percentage points.
   - Deck is confounded with policy (the "80%" was mostly Hydrapple vs Grass-weak Grimmsnarl).
   - Opponents are only our own checkpoints, so local results don't predict the ladder (now proven).
   - Deck choice was circular, there is no held-out test set, and the WiSE weights were tuned on noise.
2. **The simulator and the time bank are almost unused:** only lethal/deck-out checks and close-tie 2-ply lookahead, at 6 ms per move.
3. **Card-specific hand rules sit inside the "DL" agent:** Munkidori `594`, 12 hard-coded attack IDs, energy rules tuned for Grimmsnarl while playing Hydrapple.
4. **Silent fallbacks** (`try/except → None`, `return [0]`) hide failures on Kaggle.
5. **The RL is too small to matter:** PPO ran 12.8k steps (~165 games).
6. **Five stacked additions with no ablations.** A 332-dimensional feature vector with magic offsets caused repeated train/test skew bugs.
7. **No git, 4 copies of the agent code, ~120 flat files.** `parsed_replays.pt` (13.5 GB) came from the buggy parser.

**Kept as knowledge and tests (no code needs to be reused):**
- the observation at `steps[s]` pairs with the action at `steps[s+1]`;
- episodes contain multiple games, so reset at `ctx == 41`;
- the module collision with Kaggle's `agents.py`;
- `asdict(Observation)` differs from JSON;
- multi-select `minCount`/`maxCount` semantics;
- the replay downloader and the 4,953 raw replays;
- the entity/option-token idea;
- the `search_begin` usage pattern.

---

## 3. The new system

```
raw replays ─► tokeniser (cards + legal options as tokens, raw numeric fields only)
                 │
                 ▼
     Transformer encoder (~1–2M params, CPU-fast)
       ├─ policy head   π(a|s)        (option scoring + autoregressive multi-select)
       ├─ value head    V(s)          (win probability)
       └─ belief head   p(card ∈ opp hand | public info)   ← auxiliary, multi-label
                 │
live decision ─► sample K determinizations (opp hand from belief head, rest uniform
                 over unseen cards)  ─► PUCT-MCTS in C++ simulator (prior π, leaf V)
                 ─► sum visit counts over K worlds ─► action
                 time manager: spend on branching decisions, ~0 on forced ones
```

**Why each piece (and why it fits in 24 hours):**

- **A new tokeniser, no handcrafted features.** Each card instance becomes a token: an ID embedding plus raw HP, damage, energy, zone and owner. Each option becomes a token with its type and references to the card tokens it touches. There are no per-card rules, because search handles card semantics by simulating them. It is quick to write, and it removes the class of skew bugs we already hit.
- **A small transformer with three heads in one training run.** The policy and value heads drive search; the belief head is a free extra loss on the same data. One GPU run of about 1–2 hours gives every model we need.
- **Determinized MCTS (PUCT, AlphaZero-style priors and values).** This is the biggest expected strength gain. It turns the unused time bank and the exact simulator into play quality, and it is model-agnostic, so it can also wrap the old Rebuild-v1 network for an ablation.
- **The belief head drives determinization.** This is the DL research contribution. Planned comparisons:
  - **uniform** sampling over unseen cards;
  - **count-based** sampling (deck list minus seen cards);
  - sampling from the **learned belief**.
- **One deck,** picked from ≥1300-Elo replay statistics with confidence intervals. The ladder evidence so far (Grimmsnarl 650+ vs Hydrapple ~500) is confounded with the agent change, so the replay data decides.

**Research questions for the report**
- **RQ1:** Does neural-guided search improve over the raw policy, and how does strength scale with the search budget (a scaling curve)?
- **RQ2:** Does a learned belief model improve determinization over uniform and count-based sampling, measured offline (AUROC, log-likelihood) and online (win rate at a fixed budget)?
- **RQ3:** Does local evaluation predict the ladder? (The old benchmark vs the new one, as a validity analysis.)

---

## 4. Evaluation protocol

- **Offline:**
  - split **by game**, with a separate held-out test split never used for model selection;
  - policy top-1, top-3 and NLL **by decision type**, against uniform-legal and most-frequent baselines;
  - value: Brier score and a calibration plot;
  - belief: AUROC and mAP against uniform and count-based baselines.
- **Online gauntlet:** random, a simple heuristic, **Rebuild v1 (650+ reference)**, the latest submission (~500), our raw policy, and our search agent at several budgets. Decks: ours plus the top 4–5 meta decks, seat-balanced.
  - **Target: 200+ games per key pairing**, or as many as the time allows. Every number is reported with a **Wilson 95% interval**, and a Bradley–Terry ladder with bootstrap intervals.
- **SPRT gate** (H0: +0 Elo, H1: +30 Elo against Rebuild v1) before uploading.
- **Fixed, reported compute:** simulations per move and time per move under a 2-core limit, matching Kaggle.
- **Ladder:** record μ for each upload, and compare it with the local predictions (RQ3).

**Report figures:**
1. search-budget scaling curve;
2. belief AUROC and calibration;
3. ablation table (policy → +search → +belief determinization);
4. local vs ladder.

---

## 5. Clean-up (Phase 0, ~30 min)

- Move everything legacy into `_archive/`. Keep only the **raw replays + manifest**, `cg/`, the card CSVs, and the Rebuild-v1 and latest submission packages (as gauntlet opponents).
- **Delete** `parsed_replays.pt` (13.5 GB, buggy), `parsed_replays_compact.pt` (7.8 GB, replaced by new shards) and `__pycache__`.
- New layout:

```
ptcg/  env/ data/ features/ models/ agents/ train/ eval/
submission/build.py   tests/   configs/   experiments/   _archive/   data/replays/
```

(git and GitHub come later, as you said. The layout is ready for them.)

---

## 6. 24-hour timeline

Times are Rome time and approximate. **[You]** marks a step you launch on your PC.

| When | Work | Output / gate |
|---|---|---|
| **Sat 20:30–22:00** | Clean-up and new layout. Measure simulator speed and per-decision budget. **Deck analysis** from replays. Write the new **replay parser** (with the alignment, multi-game and parity tests). | Deck chosen; parser tested on a few replays. |
| **Sat 22:00–23:30** | Write the **model + training script** (policy, value and belief heads) and the **parallel evaluation harness**. Start baseline gauntlet games (Rebuild v1 vs latest vs heuristic) on the VM. | **[You] start overnight job:** parse the top ~1,500 replays → train (~1–2 h GPU) → offline metrics. |
| **Sat night** | Training runs; baseline evaluation runs. | Checkpoint + offline metrics in the morning. |
| **Sun 08:00–11:00** | Write the **MCTS agent** (determinization, PUCT, time manager, batching), check its correctness against the simulator, and build the **submission package** with an end-to-end Kaggle-style test. | MCTS agent plays full episodes within the time budget. |
| **Sun 11:00–13:00** | SPRT: search agent vs Rebuild v1. **Upload #1** (best agent). | First ladder games accumulate all afternoon. |
| **Sun 13:00–17:00** | Ablations: budget sweep, uniform vs count vs belief determinization, raw policy vs search. Analyse live replays from upload #1 and fix anything broken; **upload #2** if a fix passes SPRT. | Ablation numbers with confidence intervals. |
| **Sun 17:00–19:00** | Final figures, results summary (`RESULTS.md`), clean README. Material ready for the report and the repo. | Deliverables ready. |

**Upload policy (only 2 slots are active):**
- Upload #1 replaces the ~500 agent, so the 650+ Rebuild v1 **stays active as the safety net**.
- Upload #2 replaces Rebuild v1 **only if** #1 is clearly ahead on the ladder, or the local SPRT strongly favours it.

**Fallbacks**
- **If the new network is weak:** run MCTS on top of the old Rebuild-v1 network. This is still a clean "search vs no search" result.
- **If the simulator is too slow for deep search:** use shallow search with value-network leaves, only at branching decisions.
- **If the belief head doesn't help online:** report it as a negative result for RQ2 (the offline metrics still stand).

---

## 7. Approval checklist

1. Approve the scope cut: no expert iteration, PPO, IQL or WiSE; three RQs; one deck.
2. **Can you keep your PC on overnight** and launch a couple of commands in your conda environment (I'll give you copy-paste one-liners)?
3. **How many CPU cores does your PC have?** This sets how many evaluation games we can run in parallel.
