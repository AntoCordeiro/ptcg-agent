### Policy (held-out test games, experiments/train_full_s0_v/best.pt)

| decision type | n | top-1 | top-3 | NLL | uniform top-1 | uniform NLL |
|---|---|---|---|---|---|---|
| ALL | 20526 | 70.1 | 93.7 | 0.838 | 21.7 | 1.747 |
| MAIN, >=5 options | 9841 | 60.8 | 89.8 | 1.107 | 11.6 | 2.256 |
| Dragapult player | 4412 | 65.8 | 93.3 | 0.887 | 22.2 | 1.708 |
| MAIN | 12445 | 64.8 | 91.8 | 0.982 | 16.0 | 2.025 |
| CARD | 6803 | 75.2 | 95.9 | 0.696 | 27.9 | 1.406 |
| YES_NO | 605 | 96.2 | 100.0 | 0.104 | 50.0 | 0.693 |
| COUNT | 454 | 97.8 | 99.6 | 0.127 | 37.7 | 0.996 |

### Value head (held-out replay test games)

| model | Brier | Brier (constant) | accuracy | ECE |
|---|---|---|---|---|
| replay-only value head | 0.2536 | 0.2493 | 65.5 | 0.1895 |
| + self-play value fine-tuning | 0.2258 | 0.2493 | 67.4 | 0.1336 |

### Belief head (n=10980 test decisions)

| method | AUROC | mAP |
|---|---|---|
| random | 0.500 | 0.369 |
| count-based (true decklist) | 0.711 | 0.486 |
| learned head (no decklist) | 0.752 | 0.558 |
| learned x count | 0.761 | 0.565 |

### Gauntlet (candidate pilots Dragapult; Kaggle-like 600 s time bank)

| candidate | games | pooled score [95% CI] | mean over opponents | per opponent |
|---|---|---|---|---|
| decksweep_clefairy | 240 | 0.279 [0.226, 0.339] | 0.279 | legacy_v1approx@grimmsnarl: 0.33; legacy_latest@hydrapple: 0.03; best.pt@dragapult: 0.38; best.pt@ogerpon: 0.17; best.pt@clefairy: 0.60; best.pt@other_top: 0.17 |
| decksweep_dragapult | 240 | 0.494 [0.431, 0.557] | 0.494 | legacy_v1approx@grimmsnarl: 0.65; legacy_latest@hydrapple: 0.53; best.pt@dragapult: 0.55; best.pt@ogerpon: 0.28; best.pt@clefairy: 0.71; best.pt@other_top: 0.25 |
| decksweep_dragapult_clefairy | 240 | 0.287 [0.234, 0.348] | 0.287 | legacy_v1approx@grimmsnarl: 0.20; legacy_latest@hydrapple: 0.23; best.pt@dragapult: 0.30; best.pt@ogerpon: 0.20; best.pt@clefairy: 0.60; best.pt@other_top: 0.20 |
| decksweep_grimmsnarl | 240 | 0.592 [0.529, 0.652] | 0.592 | legacy_v1approx@grimmsnarl: 0.82; legacy_latest@hydrapple: 0.33; best.pt@dragapult: 0.68; best.pt@ogerpon: 0.38; best.pt@clefairy: 0.78; best.pt@other_top: 0.57 |
| decksweep_hydrapple | 240 | 0.644 [0.581, 0.702] | 0.644 | legacy_v1approx@grimmsnarl: 0.85; legacy_latest@hydrapple: 0.60; best.pt@dragapult: 0.80; best.pt@ogerpon: 0.50; best.pt@clefairy: 0.91; best.pt@other_top: 0.20 |
| decksweep_ogerpon | 240 | 0.650 [0.588, 0.708] | 0.650 | legacy_v1approx@grimmsnarl: 0.90; legacy_latest@hydrapple: 0.72; best.pt@dragapult: 0.70; best.pt@ogerpon: 0.45; best.pt@clefairy: 0.95; best.pt@other_top: 0.17 |
| decksweep_other_top | 240 | 0.731 [0.672, 0.783] | 0.731 | legacy_v1approx@grimmsnarl: 0.75; legacy_latest@hydrapple: 0.85; best.pt@dragapult: 0.68; best.pt@ogerpon: 0.89; best.pt@clefairy: 0.78; best.pt@other_top: 0.45 |
| decksweep_ragingbolt_ogerpon | 240 | 0.200 [0.154, 0.255] | 0.200 | legacy_v1approx@grimmsnarl: 0.25; legacy_latest@hydrapple: 0.15; best.pt@dragapult: 0.15; best.pt@ogerpon: 0.05; best.pt@clefairy: 0.55; best.pt@other_top: 0.05 |
| deployed_sub1_crustle | 180 | 0.903 [0.851, 0.938] | 0.903 | legacy_v1approx@grimmsnarl: 0.93; legacy_latest@hydrapple: 1.00; best.pt@dragapult: 0.90; best.pt@ogerpon: 0.87; best.pt@clefairy: 0.85; best.pt@other_top: 0.87 |
| exit1_policy_crustle | 240 | 0.708 [0.648, 0.762] | 0.708 | legacy_v1approx@grimmsnarl: 0.70; legacy_latest@hydrapple: 0.78; best.pt@dragapult: 0.68; best.pt@ogerpon: 0.82; best.pt@clefairy: 0.90; best.pt@other_top: 0.38 |
| pre_mcts64_ep8 | 120 | 0.450 [0.364, 0.539] | 0.450 | legacy_v1approx@grimmsnarl: 0.55; best.pt@dragapult: 0.57; best.pt@ogerpon: 0.23 |
| pre_policy_ep8 | 120 | 0.450 [0.364, 0.539] | 0.450 | legacy_v1approx@grimmsnarl: 0.62; best.pt@dragapult: 0.47; best.pt@ogerpon: 0.25 |
| pre_policy_ep8_DRAG_std | 240 | 0.450 [0.388, 0.513] | 0.450 | legacy_v1approx@grimmsnarl: 0.68; legacy_latest@hydrapple: 0.38; best.pt@dragapult: 0.40; best.pt@ogerpon: 0.33; best.pt@clefairy: 0.60; best.pt@other_top: 0.33 |
| pre_policy_ep8_OGERPON | 240 | 0.642 [0.579, 0.700] | 0.642 | legacy_v1approx@grimmsnarl: 0.82; legacy_latest@hydrapple: 0.65; best.pt@dragapult: 0.68; best.pt@ogerpon: 0.57; best.pt@clefairy: 0.95; best.pt@other_top: 0.17 |
| scale_16 | 240 | 0.812 [0.758, 0.857] | 0.813 | legacy_v1approx@grimmsnarl: 0.85; legacy_latest@hydrapple: 0.88; best.pt@dragapult: 0.75; best.pt@ogerpon: 0.93; best.pt@clefairy: 0.90; best.pt@other_top: 0.57 |
| scale_256 | 240 | 0.873 [0.825, 0.909] | 0.873 | legacy_v1approx@grimmsnarl: 0.88; legacy_latest@hydrapple: 0.90; best.pt@dragapult: 0.90; best.pt@ogerpon: 0.88; best.pt@clefairy: 0.90; best.pt@other_top: 0.79 |
| seed1_mcts64_crustle | 240 | 0.850 [0.799, 0.890] | 0.850 | legacy_v1approx@grimmsnarl: 0.85; legacy_latest@hydrapple: 0.90; best.pt@dragapult: 0.95; best.pt@ogerpon: 0.95; best.pt@clefairy: 0.85; best.pt@other_top: 0.60 |
| seed1_policy_crustle | 240 | 0.792 [0.736, 0.838] | 0.792 | legacy_v1approx@grimmsnarl: 0.90; legacy_latest@hydrapple: 0.90; best.pt@dragapult: 0.70; best.pt@ogerpon: 0.90; best.pt@clefairy: 0.90; best.pt@other_top: 0.45 |
| v_mcts64_crustle | 240 | 0.802 [0.747, 0.848] | 0.802 | legacy_v1approx@grimmsnarl: 0.82; legacy_latest@hydrapple: 0.90; best.pt@dragapult: 0.80; best.pt@ogerpon: 0.82; best.pt@clefairy: 0.82; best.pt@other_top: 0.64 |
| v_mcts64_ogerpon | 240 | 0.675 [0.613, 0.731] | 0.675 | legacy_v1approx@grimmsnarl: 0.85; legacy_latest@hydrapple: 0.65; best.pt@dragapult: 0.82; best.pt@ogerpon: 0.57; best.pt@clefairy: 0.95; best.pt@other_top: 0.20 |
| v_policy_crustle | 240 | 0.750 [0.692, 0.801] | 0.750 | legacy_v1approx@grimmsnarl: 0.80; legacy_latest@hydrapple: 0.97; best.pt@dragapult: 0.65; best.pt@ogerpon: 0.93; best.pt@clefairy: 0.75; best.pt@other_top: 0.40 |
| v_policy_ogerpon | 240 | 0.621 [0.558, 0.680] | 0.621 | legacy_v1approx@grimmsnarl: 0.88; legacy_latest@hydrapple: 0.65; best.pt@dragapult: 0.68; best.pt@ogerpon: 0.47; best.pt@clefairy: 0.95; best.pt@other_top: 0.10 |

### Search budget scaling (Crustle deck vs pool)

| simulations | games | score [95% CI] |
|---|---|---|
| raw policy | 240 | 0.750 [0.692, 0.801] |
| 16 | 240 | 0.812 [0.758, 0.857] |
| 64 | 240 | 0.802 [0.747, 0.848] |
| 256 | 240 | 0.873 [0.825, 0.909] |
| deployed: time bank, <=2 s/decision | 180 | 0.903 [0.851, 0.938] |

### Head-to-head matches

| A | B | games | score A [95% CI] | Elo diff | SPRT |
|---|---|---|---|---|---|
| mcts:experiments/train_full_s0_v/best.pt:sims=256,worlds=4,belief=learned (other_top.csv) | mcts:experiments/train_full_s0_v/best.pt:sims=256,worlds=4,belief=uniform (other_top.csv) | 208 | 0.517 [0.449, 0.584] | +12 | continue |
| mcts:experiments/train_full_s0_v/best.pt:sims=64,worlds=4,belief=learned (other_top.csv) | mcts:experiments/train_full_s0_v/best.pt:sims=64,worlds=4,belief=uniform (other_top.csv) | 400 | 0.479 [0.430, 0.528] | -15 | H0 accepted (<=0) |
| mcts:experiments/train_full_s0_v/best.pt:sims=100000,worlds=1,use_overage=1,max_time=2.0 (other_top.csv) | mcts:experiments/train_full_s0_v/best.pt:sims=100000,worlds=4,use_overage=1,max_time=2.0 (other_top.csv) | 200 | 0.557 [0.488, 0.625] | +40 | continue |
| mcts:experiments/train_full_s0_v/best.pt:sims=100000,worlds=1,use_overage=1,max_time=2.0 (other_top.csv) | mcts:experiments/train_full_s0_v/best.pt:sims=100000,worlds=4,use_overage=1,max_time=2.0 (other_top.csv) | 200 | 0.575 [0.506, 0.641] | +53 | continue |
| mcts:experiments/train_exit1/last.pt:sims=64,worlds=4 (other_top.csv) | mcts:experiments/train_full_s0_v/best.pt:sims=64,worlds=4 (other_top.csv) | 304 | 0.467 [0.412, 0.523] | -23 | continue |
| policy:experiments/train_exit1/last.pt (other_top.csv) | policy:experiments/train_full_s0_v/best.pt (other_top.csv) | 400 | 0.511 [0.462, 0.560] | +8 | continue |
| mcts:experiments/train_exit2/last.pt:sims=64,worlds=1 (other_top.csv) | mcts:experiments/train_full_s0_v/best.pt:sims=64,worlds=1 (other_top.csv) | 300 | 0.570 [0.513, 0.625] | +49 | continue |
| policy:experiments/train_exit2/last.pt (other_top.csv) | policy:experiments/train_full_s0_v/best.pt (other_top.csv) | 400 | 0.502 [0.454, 0.551] | +2 | continue |
| mcts:experiments/train_full_s0_v/best.pt:sims=64,worlds=4 (other_top.csv) | policy:experiments/train_full_s0_v/best.pt (other_top.csv) | 200 | 0.593 [0.523, 0.658] | +65 | continue |
| mcts:experiments/train_full_s0_v/best.pt:sims=64,worlds=4 (other_top.csv) | mcts:experiments/train_full_s0_v/best.pt:sims=64,worlds=1 (other_top.csv) | 300 | 0.393 [0.340, 0.450] | -75 | H0 accepted (<=0) |
| legacy:baselines/legacy_v1approx (grimmsnarl.csv) | legacy:baselines/legacy_latest (hydrapple.csv) | 400 | 0.310 [0.267, 0.357] | -139 | H0 accepted (<=0) |

