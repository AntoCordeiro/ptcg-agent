# PTCG Agent — neural-guided determinized search for the Pokémon TCG

Kaggle *Pokémon TCG AI Battle Challenge (Playground)* agent + DLAI 2025/26 exam project.

## Layout
```
ptcg/            new code base
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
submission/      build.py + main template -> Kaggle tar.gz
decks/           decklists (top-Elo archetype lists mined from replays + legacy decks)
tools/           deck statistics, simulator benchmark, debugging scripts
tests/           regression tests
jobs/            file-based job runner (runs jobs on the Windows GPU machine)
baselines/       legacy submissions used as fixed opponents
_archive/        previous code base (reference only)
experiments/     logs, metrics, gauntlet results, figures
```

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
