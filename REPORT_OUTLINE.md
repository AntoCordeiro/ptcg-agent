# 2-page report — outline (DLAI 2025/26, topic: gaming and AI)

**Working title:** *Search beats imitation: neural-guided determinized MCTS with a learned belief model for the Pokémon TCG*

## 1. Introduction (≈0.25 page)
- Pokémon TCG: stochastic, imperfect-information card game, variable legal-action sets, 60-card decks.
- Kaggle PTCG AI Battle Challenge: agents play on a TrueSkill-like ladder; 2 vCPU, ~600 s time bank per episode.
- Prior system (ours): stacked BC/IQL/PPO/WiSE components, ~6 ms/move, ladder 250–650. Local tests
  ranked a ~500-ladder agent above a 650+-ladder agent (0.317 score for the stronger one, n=350), i.e. the
  evaluation was invalid -> motivation for a clean rebuild with rigorous evaluation.
- Contributions: (i) entity/option transformer with policy/value/belief heads trained on 453k expert decisions;
  (ii) value learning from self-play outcomes; (iii) determinized MCTS on the exact simulator with belief-guided
  worlds; (iv) rigorous evaluation (held-out test games, 6-opponent gauntlet, Wilson CIs, SPRT).

## 2. Method (≈0.6 page) — Figure: architecture/pipeline diagram
- Tokenisation: card instances + legal options as tokens; options point to card tokens; no hand-crafted card rules.
- PTCGNet (2.6M params, 4-layer pre-norm transformer, d=192): policy (option scoring + count head for
  multi-select), value, belief (multi-label: which cards are in the opponent's hand).
- Data: top-3000 replays by Elo, alignment obs[s] -> action[s+1], split by game (90/5/5); opponent-hand labels
  come from the opponent's own observation in the replay.
- Value: replay-only value overfits (3k outcomes) -> fine-tune on 8k policy self-play games (AlphaGo SL->value).
- Search: K=4 determinizations (own deck/prizes uniform; opponent decklist = nearest archetype;
  opponent hand ~ belief head or uniform), PUCT with network priors/values, forced and multi-select steps
  played by the policy, root visits summed over worlds; time manager on the Kaggle overage bank.
- Expert iteration (1 round): MCTS self-play visit distributions as policy targets.

## 3. Experiments (≈0.9 page)
- Offline (Table 1): policy top-1 70.0% (uniform 21.7%), MAIN≥5 options 60.7% (11.6%); value Brier
  0.2536 -> 0.2258 (constant 0.2493); belief AUROC 0.752 vs count-based with true decklist 0.711. Two seeds agree.
- Deck selection under our own agent (Table 2): Kangaskhan/Crustle 0.73 vs Dragapult 0.49 — the strongest
  deck among top humans/bots is not the strongest for our agent.
- Search scaling (Figure 1): 0.750 (policy) -> 0.813 (16) -> 0.802 (64) -> 0.873 (256 sims); head-to-head
  MCTS-64 vs its own policy 0.593 [0.523, 0.658].
- Deployed agent (time bank, ≤2 s/decision): 0.903 [0.851, 0.938] vs pool; 0.93 vs the 650+-ladder legacy agent.
- Seed robustness: search gain reproduced on seed 1 (0.792 -> 0.850).
- Belief ablation (Table 3): learned vs uniform worlds 0.479 [0.430, 0.528] at 64 sims (SPRT H0); 0.517 [0.449, 0.584] at 256 sims.
- Determinization count: 4 worlds vs 1 world at equal simulations 0.393 [0.340, 0.450] (−75 Elo): depth > hidden-info
  averaging at this budget — explains the null belief result. Deployed: 1 vs 4 worlds 0.558 [0.488, 0.625].
- Expert iteration (1 round, 16-sim targets): no gain — policy 0.511 [0.462, 0.560], with search 0.467 [0.412, 0.523].
- Ladder: submission(s) μ (fill).

## 4. Discussion & limitations (≈0.25 page)
- Better hidden-information prediction did not translate into win rate at our search depth (strategy fusion /
  shallow trees / only the hand is sampled); belief helps offline but not online.
- Local gauntlet vs ladder validity; pool composition bias; opponents partly share our network.
- Compute limits: one ExIt round with 16-sim targets.

## Figures/tables available (experiments/figures, experiments/results_tables.md)
search_scaling.png, belief_auroc.png, value_calibration.png, policy_by_type.png, training_curves.png

## AI-use statement (separate, not counted)
Describe: Claude (Anthropic) used as a coding/analysis assistant for planning, code generation (data pipeline,
model, search, evaluation), running experiments via a job runner, and drafting; all design decisions,
results verification and final text reviewed by the student. (Adapt to the template's instructions.)
