# PTCG Agent — Full Plan (v1, long version — kept as backlog; active plan is PLAN.md)

**Date:** 2026-10-17
**Goals:** (1) a strong Kaggle leaderboard score, (2) a deep-learning project good enough for a demanding exam (DLAI 2025/26, BYOP, topic "gaming and AI"), with (3) a rigorous evaluation of the models.
**Based on:** a review of `HANDOFF_SUMMARY.md`, `PROJECT_JOURNAL.md`, the code in `ptcg_agent/`, the result files, the replay format, the simulator API (`cg/api.py`), and the course guidelines.

---

## 0. TL;DR

- **The diagnosis.** The real bottleneck is **not** the network architecture. It is (a) **evaluation that can't tell signal from noise** and (b) **an agent that hardly uses the simulator or its time budget**. Five "approaches" have been stacked on top of each other and none of them has been shown to help. Every decision has been based on 16–40-game samples against our own models.
- **The new direction.** A deck-specialised agent built from:
  - a **compact entity/option transformer**, pre-trained on high-Elo replays;
  - a **learned belief model** that predicts the opponent's hidden cards;
  - **neural-guided determinized MCTS** on the C++ simulator;
  - improved by **expert iteration**, where search is distilled back into the network.
- **Evaluation comes first.** Before any new model, we build a fixed opponent gauntlet with 200+ games per pairing, confidence intervals, SPRT gating for submissions, a held-out test set and a proper ablation protocol.
- **Clean-up.** We start a git repo (the exam requires one) and move the ~120 flat files into a clean package. Legacy code goes to `_archive/`. Large derived files become candidates for deletion (about 21 GB freed).

---

## 1. Facts that shape the plan (verified during this review)

| Fact | Where | Consequence |
|---|---|---|
| `actTimeout = 0`, and each agent has a **~600 s overage bank per episode** (`remainingOverageTime ≈ 593 s` at step 10) | replay `configuration` / observations | That is roughly **1–3 s per decision** on average. The current agent uses **~6 ms per move**, so >99% of the thinking budget is unused. |
| The simulator exposes `search_begin(obs, your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active)` + `search_step` | `cg/api.py` | We have a full **forward model with determinization support**. This is exactly what Perfect-Information Monte Carlo (PIMC) and Information-Set MCTS (IS-MCTS) need. |
| Replays store **both players' observations** at every step, and the inactive player's observation includes **its own hand** | `data/replays/*.json` | The opponent's hidden hand is a **supervised label**. We can train a **belief network** on it. |
| Kaggle hardware: **2 vCPU, no GPU**, 12 GB RAM | competition FAQ | The network must be small and fast on CPU, and search must be batched. |
| Exam deliverables: a repo link, a **2-page report** (fixed template) and a mandatory AI-use statement | `guidelines 2.pdf` | Depth beats breadth. The report needs **one clear research question + clean ablations**, not nine approaches. |
| Someone else's public repo for this competition reports that a **deck-specific scripted pilot** (Dragapult) reached ~880 μ, while their generic RL+MCTS stayed at ~580 μ | github.com/TomBombadyl/kaggle_pokemon | **Specialising on one deck** is worth a lot. We can also use a scripted pilot as a strong benchmark opponent. |

---

## 2. Review of the current work

### 2.1 Critical issues

1. **Evaluation can't separate signal from noise.** This is the single most important issue.
   - **Sample sizes are too small.** `benchmark_approach_ac.py` plays **20 games per pair**. The 95% confidence interval on a 20-game win rate is about **±22 percentage points**. "search_v4 beat selfplay_v3 55% (11W-9L)" is indistinguishable from 50%. Most rankings in the journal (`+120.7` vs `+116.1` vs `+111.6` Elo) are within noise of each other.
   - **Decks are confounded with policies.** The headline "80% vs Rebuild v1" pits `search_v4 + Hydrapple deck` against `wise + Grimmsnarl deck`. Grimmsnarl is 2× weak to Grass, so the result mostly measures the **deck matchup**, not the agent.
   - **The opponents are self-referential.** All local opponents are our own checkpoints plus random and heuristic bots. A +240 local Elo gain never translated to the ladder, so local evaluation does **not predict** the leaderboard.
   - **Deck choice is circular.** The "best deck" was chosen by our own agent playing 16 games per pair. That measures which deck *our agent* pilots best, with large noise. It does not measure which deck is strongest.
   - **There is no held-out test set.** The best epoch is chosen on validation accuracy and the same number is reported. The WiSE mixing coefficients (0.60/0.25/0.15) were picked from 16-game sweeps, which overfits to noise.
   - **Accuracy has no reference point.** 63% top-1 is never compared with simple baselines (uniform over legal moves, most-frequent action) or broken down by decision type. Many decisions have only 1–3 options, so the number alone means little.

2. **The simulator and the time budget are barely used.** Search runs only for lethal checks, deck-out vetoes, and a 2-ply lookahead on close ties, with 2 determinizations, at 6 ms per move. A ~600 s budget combined with a forward model is the biggest unused asset in the project.

3. **Hand-coded, card-specific patches are mixed into the "deep learning" agent.**
   - `apply_tactical_prior_shaping` hard-codes Munkidori (`cid == 594`) and energy rules tuned for **Grimmsnarl**, but the submitted deck is now **Hydrapple/Ogerpon**.
   - `compute_dynamic_attack_damage` hard-codes 12 attack IDs. The mulligan is always "draw".
   - These rules are brittle, confound every ablation, and weaken the DL story in the report.

4. **Failures are hidden.** `main.py` wraps `SearchAgent` construction in `try/except → None` and falls back to `[0]` in several places. If something breaks on Kaggle, the agent quietly gets weaker instead of crashing. This has already caused two lost submissions (an import collision, and the `playerIndex=None` bug).

5. **The RL is too small to matter.** The self-play PPO ran for **12,800 steps (~165 games)**. That is far too little to change a policy meaningfully. Its "win" in the ladder rests on 20-game samples (see issue 1).

6. **The architecture is complex and nothing has been ablated.**
   - A 332-dimensional hand-engineered feature vector with magic offsets (`[130:162]`, `[162:173]`, `[:, 326:330]`), zero-initialised gates, relational biases, a pointer network, a Q-gate and a three-model weight average.
   - Nobody knows which parts matter.
   - The train/test-skew bugs found so far (step alignment, a disabled tracker, `playerIndex=None`) came from this complexity, and there was no test catching parity between replay JSON and live observations.

7. **Engineering hygiene.**
   - There is **no git** (the exam requires a repo).
   - There are ~120 files at the top level and **four copies** of the agent code (root, `kaggle_submission/`, `submission/`, `staging_submission/`).
   - There are 19 checkpoints and about 20 `*_log.json` files from dead branches.
   - `parsed_replays.pt` (**13.5 GB**) was produced by the **buggy, misaligned parser**. `parsed_replays_compact.pt` (7.8 GB) is one monolithic file that must be reloaded entirely for every run.

### 2.2 Good points we keep

- **Domain knowledge that cost real effort.** We keep these as unit tests and documentation:
  - an observation at `steps[s]` is paired with the action at `steps[s+1]`;
  - Kaggle episodes contain multiple games, so state must reset at `ctx == 41` and on turn rollback;
  - the module-name collision with Kaggle's `agents.py` and the `sys.path[0]` fix;
  - the `asdict(Observation)` vs JSON differences (`playerIndex=None`);
  - multi-select count semantics (`minCount`/`maxCount`, Punk Up, Bellowing Thunder).
- **The replay data pipeline.** `download_top_replays.py` plus the **4,953 replays** with an Elo manifest. The raw JSON is the valuable asset; everything derived from it can be regenerated.
- **The representation idea.** Entities and legal options as tokens, options scored by attending to the board, autoregressive selection for multi-select. This is the right inductive bias for a variable action set. We keep the idea, in a simpler form.
- **The search plumbing.** `search_begin`/`search_step` usage, determinization sampling and batched value evaluation in `search_agent.py`. This is useful reference code for the new MCTS.
- **`CardTracker`.** It is a sensible hand-written belief baseline (deck minus seen cards). It becomes **the baseline the learned belief model is compared against**.
- **Packaging and local verification.** `create_kaggle_submission.py` and `test_kaggle_submission.py` ideas: a local end-to-end run before uploading.
- **Diagnostic habit.** Forensic replay audits of our own live losses found real bugs. We keep this as a standard step after every submission.

### 2.3 Verdict on complexity

The current system **is over-complicated for what it has shown**. Five stacked additions with no ablations, plus rule patches, make it impossible to attribute any gain. It is also architecture-heavy and search-light, when the problem (a stochastic, imperfect-information card game with a fast simulator and a generous time bank) rewards the opposite.

Recommendation: **restart the codebase cleanly** and port only the verified pieces above. Re-evaluate the old checkpoints once in the new harness, so the report can include them as properly measured baselines.

---

## 3. New approach and the reasons for it

### 3.1 Core system: "Neural-guided determinized search with a learned belief model"

```
replays ──► [Encoder: entity/option transformer] ──► policy head π(a|s), value head V(s)
                      │
                      └──► belief head  p(opponent hand / deck composition | public history)
                                   │
 live decision ──► sample K determinizations from belief ──► PUCT-MCTS in C++ simulator
                   (opp hand, opp deck order, own prize cards)    (prior = π, leaf = V)
                                   │
                         aggregate visit counts over K worlds ──► action
                                   │
              self-play games with search ──► Expert Iteration: train π on visit
                                              distributions, V on outcomes (loop)
```

**Why each piece:**

1. **Specialise on one deck.**
   - The action space, card semantics and plans become much narrower, so a small model can learn them well.
   - The best-performing public agents we found are deck specialists.
   - The deck is chosen from **ladder data** (win rate of each archetype in ≥1300-Elo replays, with confidence intervals), not from our own agent's tournaments.
   - A generalisation experiment to a second deck can be optional, for the report.

2. **Use a compact entity/option transformer instead of the current 332-dimensional handcrafted vector.**
   - Each card instance becomes a token: card-ID embedding, initialised from card-text embeddings, plus a few raw numeric fields (HP, damage, energies, zone, owner).
   - Each legal option becomes a token pointing at its card tokens.
   - There are no magic offsets and no per-card damage rules. Search handles the card semantics, by actually simulating them.
   - It has to be small (about 1–3 M parameters) so it can run hundreds of evaluations per second on 2 vCPUs.

3. **Add a learned belief model.** This is the **main DL contribution**.
   - It predicts the opponent's hand contents (multi-label) and deck archetype from the public history.
   - It is trained with the hidden labels found in the replays.
   - It turns imperfect information into a supervised representation-learning problem, and it plugs straight into search, because better determinizations mean better search.
   - Clear baselines exist: uniform sampling over unseen cards, and the current `CardTracker`.
   - This gives the report a sharp research question (§3.2).

4. **Use determinized MCTS (PIMC / IS-MCTS) with PUCT.**
   - The simulator is the most accurate model of the game we will ever have, and the time bank allows real search.
   - The network supplies priors and leaf values, so search depth stays manageable. Using a value network instead of random rollouts is the AlphaZero lesson.
   - The time manager spends more on high-branching turns and almost nothing on forced moves.
   - Determinization has known weaknesses (strategy fusion, non-locality). We control the number of worlds K and report its effect.

5. **Improve with expert iteration (ExIt / AlphaZero-style) instead of PPO.**
   - Search acts as a policy-improvement operator. Training π on visit counts is far more sample-efficient than policy-gradient RL at our compute scale (one RTX 4070 Ti SUPER).
   - It also gives a clean training curve for the report: network strength by ExIt iteration, with and without search.
   - Self-play runs against a **league of the top 5–6 meta decks** from replays, so the agent doesn't overfit to the mirror.

6. **Leave out the old extras for now:** IQL, WiSE averaging, relational biases and the Q-gate.
   - IQL on mixed-quality, mixed-deck data adds complexity that search plus a value head already covers.
   - WiSE can come back later as an optional "model merging" experiment, since that is also a course topic.
   - Each one is added back only if it beats the simpler system in a proper A/B test.

### 3.2 Research questions for the report

- **RQ1 (main):** Does a **learned belief model** improve determinized search over uniform or count-based determinization, (a) in hidden-hand prediction quality and (b) in playing strength at a fixed compute budget?
- **RQ2:** How does playing strength scale with **search budget** (simulations per move) and the **number of determinizations**, and how much of the search gain does **expert iteration** distil back into the raw network?
- **RQ3 (supporting):** What does each part contribute? Behaviour cloning only, + value head, + search, + belief, + ExIt.

---

## 4. Evaluation protocol (built **before** any new model)

**Offline (model level)**
- Data split **by game**, plus a **time-based test split** (the most recent replays), never used for model selection.
- **Policy:** top-1/top-3 accuracy and negative log-likelihood, **broken down by decision type** (main-phase action, target selection, multi-select, setup), next to uniform-legal and most-frequent baselines.
- **Value:** win-prediction Brier score, plus calibration (ECE and reliability diagrams) as a function of turn number.
- **Belief:** per-card AUROC and mAP for the opponent's hand, and log-likelihood of the true hand, against the uniform and `CardTracker` baselines.
- 3 seeds for every reported number; we report mean ± std.

**Online (agent level)**
- **A fixed gauntlet:**
  - random;
  - heuristic;
  - a **scripted reference pilot** (from the public Kaggle notebooks, where licensing allows);
  - the **current `submission.tar.gz`** (frozen old best);
  - the BC-only network.
- Played across the **top 5–6 ladder decks**, seat-balanced.
- **Sample sizes:** at least 200 games per pairing for headline numbers, with Wilson 95% confidence intervals. Bradley–Terry ratings with **bootstrap confidence intervals**.
- **SPRT gating** (as chess engines do: H0 = +0 Elo, H1 = +30 Elo) before any new version is uploaded. This saves the 5-per-day submission quota for real improvements.
- Compute is fixed and reported (simulations per move, wall-clock time on a 2-vCPU-equivalent setting).

**Ladder**
- Submit at milestones and log each μ/σ trajectory.
- Report how well the local gauntlet predicts the ladder, as a **validity check** of the evaluation.

**Planned figures:** search-budget scaling curve; ExIt iteration curve; belief-model calibration and AUROC; ablation table; ladder μ over time.

---

## 5. Folder clean-up and new structure

```
ptcg_agent/
  ptcg/
    env/        simulator wrapper, observation utils (JSON ⇄ Observation parity), deck I/O
    data/       replay download, parsing → sharded datasets (.npz/.pt shards, streamable)
    features/   tokeniser (entities + options)
    models/     encoder, policy/value/belief heads
    agents/     random, heuristic, policy, mcts (+ time manager)
    train/      bc.py, belief.py, exit.py
    eval/       gauntlet.py, bradley_terry.py, sprt.py, offline_metrics.py, plots.py
  submission/   build.py → submission.tar.gz (single source of truth, no duplicated code)
  tests/        step alignment, multi-game reset, obs parity, legal-action invariants,
                per-move latency budget, end-to-end Kaggle-style episode
  configs/      yaml per experiment
  experiments/  run logs/metrics (jsonl) + figures
  report/       LaTeX (course template) + AI-use statement
  _archive/     everything legacy (read-only reference)
  data/replays/ raw replays (git-ignored)
```

**Clean-up steps (nothing deleted without your OK):**
1. `git init`, add a `.gitignore` (data, checkpoints, caches), and make a first commit of the current state as a **"legacy snapshot"** so nothing is lost.
2. Move all old scripts, logs, checkpoints, `submission/`, `staging_submission/`, `kaggle_submission/`, `scratch/` and `results*/` to `_archive/`.
   - Keep the **current `submission.tar.gz`** and `ptcg_wise_v3.pt` plus `ptcg_wise_ensemble.pt` as baselines.
3. **Candidates for deletion (about 21 GB):**
   - `parsed_replays.pt` (13.5 GB, made by the buggy parser);
   - `parsed_replays_compact.pt` (7.8 GB, regenerated as shards);
   - `__pycache__`.
   - Superseded checkpoints are optional to delete.
4. Port the kept pieces into `ptcg/`, **each with a test** that encodes the bug lesson it carries.

---

## 6. Phased roadmap and decision gates

| Phase | Work | Exit gate |
|---|---|---|
| **0. Baseline & measurements** | • Upload the current `submission.tar.gz` now, so it gives us a ladder anchor while we work. • Measure simulator speed (`search_step`/s), network latency on 2 threads, and the real time budget per episode and per decision. • Analyse replays: archetype win rates at ≥1300 Elo with confidence intervals → **pick the deck**. | Measured numbers in the repo; deck chosen with data. |
| **1. Clean-up & skeleton** | • git, new package, archive. • Port the env wrapper, replay parser (streaming shards) and tests. | All tests pass; one random agent plays a full Kaggle-style multi-game episode locally. |
| **2. Evaluation harness** | • Gauntlet, Bradley–Terry, SPRT, offline metrics. • **Re-evaluate the old checkpoints properly** (e.g. is `search_v4` really better than `wise_v3`? do the shaping rules help?). | Old-system numbers with confidence intervals. These are the baselines of the report. |
| **3. Encoder + BC + value** | • New tokeniser and model. • Pre-train on all high-Elo data, then fine-tune on the chosen deck. • Value head. | Beats the old `wise_v3` policy under SPRT; offline metrics above baselines. |
| **4. Belief model** | • Hidden-hand and archetype head. • Compare with uniform and `CardTracker`. | Better AUROC/log-likelihood than both baselines on the test split. |
| **5. Determinized MCTS** | • PUCT with π/V, K determinizations sampled from the belief model, a time manager, batching. • Budget-scaling study. | Large SPRT win over the BC policy; latency safe within the time bank (with margin); **ladder submission**. |
| **6. Expert iteration** | • Self-play with search against the deck league. • Train on visit distributions and outcomes; repeat. | Each iteration gated by SPRT; ladder submissions at milestones. |
| **7. Final evaluation & report** | • Ablations (RQ1–3), 3 seeds, figures. • 2-page report, AI-use statement, clean README. | Deliverables ready. |

**Risk management**
- The **simulator may be too slow** for deep search on 2 vCPUs. Fall back to shallow search with a strong value network, or do heavy search only at key decisions. Phase 0 measures this first.
- If **the belief model gives no gain**, that is still a valid, reportable negative result for RQ1. Strength then comes from search plus ExIt.
- **Kaggle failures:** no silent fallbacks in development builds. The submission build gets **one** top-level guard that logs and falls back to the network policy, and the end-to-end test runs on every build.
- **Time:** Phases 0–3 alone should already give a cleaner and probably stronger agent than today. Phases 5–6 are where the big ladder gains are expected.

---

## 7. Decisions needed from you before we start

1. **Exam session / deadline**, and how many weeks you can put in. This decides whether Phase 6 (ExIt) is in scope.
2. **Deletion:** may I delete `parsed_replays.pt` + `parsed_replays_compact.pt` (~21 GB), or should they only be moved to `_archive/`?
3. **Solo or group project?** This sets the report page budget and the expected scope.
4. **Deck:** OK to let the Phase 0 replay analysis choose the deck (even if it isn't Hydrapple/Ogerpon)?
5. **GitHub:** create a private repo now, to share with the professor and TAs later?
