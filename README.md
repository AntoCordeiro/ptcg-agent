# PTCG Agent — neural-guided determinized search for the Pokémon TCG

Kaggle *Pokémon TCG AI Battle Challenge (Playground)* agent + DLAI 2025/26 exam project.

Report: [`report/main.pdf`](report/main.pdf) — *Learning to Play the Pokémon Trading Card Game: Imitation vs
Neural-Guided Search*. Full results: [`RESULTS.md`](RESULTS.md).

Two agents, named after their networks:
- **DualCross** (`gen1/`): relational transformer trained by behavioural cloning on top replays, refined with
  IQL / PPO and weight averaging (WiSE-FT). Best ladder rating 647.7.
- **PTCGNet** (`ptcg/`): small transformer (policy / value / count / belief heads) guiding a determinized
  PUCT tree search on the game simulator, improved by expert iteration. Gauntlet score 0.75 (policy) → 0.90 (search);
  best submission (sub4) **773.2 on the Kaggle leaderboard, 154th of 432**.

## Layout
```
ptcg/            PTCGNet agent (second agent, report Sec. 4)
  cards.py         static card/attack tables from the simulator database
  features.py      observation -> entity/option tokens (single encoder for every caller)
  parse.py         replays -> training shards (obs[s] answered by action[s+1]; opponent-hand labels)
  model.py         PTCGNet: transformer with policy / value / count / belief heads
  train.py         supervised training on replays
  selfplay.py      policy self-play games -> value-head data
  train_value.py   value fine-tuning on self-play outcomes (policy anchored on replays)
  agents.py        random / heuristic / policy / legacy wrappers
  mcts.py          determinized PUCT search on the C++ simulator (belief-guided worlds)
  eval.py          parallel matches, Kaggle time-bank simulation, Wilson CI, SPRT
  gauntlet.py      candidate vs fixed opponent pool
  metrics.py       offline test-split metrics (policy / value / belief)
  report.py        figures + results tables
report/          LaTeX source and PDF of the report
submission/      build.py + main template -> Kaggle tar.gz
decks/           decklists (top-Elo archetype lists mined from replays + legacy decks)
tools/           deck statistics, simulator benchmark, debugging scripts
tests/           regression tests
jobs/            file-based job runner (runs jobs on the Windows GPU machine)
baselines/       legacy submissions used as fixed opponents
gen1/            DualCross agent: code and benchmark results (report Sec. 3)
experiments/     logs, metrics, gauntlet results, figures
```

## Setup
- Python 3.11, PyTorch 2.5 (CUDA optional), numpy, matplotlib.
- The competition engine (`cg/`) and card data are **not** included (the engine license forbids
  redistribution): download them from the Kaggle competition page and place `cg/` at the repo root.
- Replays: `data/replays/*.json` + `data/replays/manifest.csv` (downloaded from Kaggle's episode API).

## Reproduce
```
python -m ptcg.parse --top 3000 --workers 20 --out data/shards
python -m ptcg.train --name full_s0 --epochs 12 --w-value 0.25
python -m ptcg.selfplay --ckpt experiments/train_full_s0/best.pt --games 8000 --workers 16
python -m ptcg.train_value --init experiments/train_full_s0/best.pt --name full_s0_v
python -m ptcg.metrics --ckpt experiments/train_full_s0_v/best.pt --name full_s0_v
python -m ptcg.gauntlet --cand "mcts:experiments/train_full_s0_v/best.pt:sims=64,worlds=4" --tag mcts64
python -m ptcg.report
python submission/build.py --ckpt experiments/train_full_s0_v/best.pt --deck decks/<deck>.csv --out submission/out
```
