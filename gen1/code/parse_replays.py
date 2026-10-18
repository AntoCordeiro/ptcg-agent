# parse_replays.py
"""
High-Elo Replay Parser (v2) with Strict Step-Action Alignment & Multi-Select Supervision.

Fixes the critical Kaggle replay alignment bug:
- In kaggle-environments replays, `steps[s][p_idx]["observation"]` (when `status == "ACTIVE"`)
  is answered by `steps[s + 1][p_idx]["action"]` (NOT `steps[s][p_idx]["action"]`, which is from s-1).
- Maintains a live `CardTracker` per player across each game so training and inference share
  identical Bayesian belief features (`global_vec[130:162]`).
- Prioritizes highest-rated matches from `data/replays/manifest.csv` (avg_score 1100-1210+).
- Extracts full multi-select targets (`target_multi_mask` + `target_count_ratios`) alongside
  primary `target_indices`, calibrated `values`, and `prize_margins`.
"""

import csv
import json
import glob
import os
import random
import numpy as np
import torch
from tqdm import tqdm
from feature_parser import parse_game_state_to_tensors
from card_tracker import CardTracker

REPLAY_DIR = "data/replays"
MANIFEST_FILE = os.path.join(REPLAY_DIR, "manifest.csv")
OUTPUT_FILE = "parsed_replays_compact.pt"

TARGET_MAX_STEPS = 75000
MAX_EPISODES = 2000


def get_sorted_replay_files() -> list[str]:
    """Returns replay file paths prioritized by highest average Elo score from manifest.csv."""
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rows.sort(key=lambda r: float(r.get("avg_score", 0.0)), reverse=True)
        files = []
        for r in rows:
            p = os.path.join(REPLAY_DIR, f"{r['episode_id']}.json")
            if os.path.exists(p):
                files.append(p)
        if files:
            return files
    files = glob.glob(os.path.join(REPLAY_DIR, "*.json"))
    random.seed(42)
    random.shuffle(files)
    return files


def main():
    replay_files = get_sorted_replay_files()
    print(f"Found {len(replay_files)} replays (sorted by highest Elo). Extracting up to {TARGET_MAX_STEPS} strictly aligned steps...")

    train_globals, val_globals = [], []
    train_obj, val_obj = [], []
    train_card_ids, val_card_ids = [], []
    train_evo_ids, val_evo_ids = [], []
    train_token_types, val_token_types = [], []
    train_option_masks, val_option_masks = [], []
    train_valid_masks, val_valid_masks = [], []
    train_padding_masks, val_padding_masks = [], []
    train_targets, val_targets = [], []
    train_multi_masks, val_multi_masks = [], []
    train_count_ratios, val_count_ratios = [], []
    train_has_count, val_has_count = [], []
    train_values, val_values = [], []
    train_prize_margins, val_prize_margins = [], []
    train_has_policy, val_has_policy = [], []

    total_steps = 0
    games_processed = 0

    pbar = tqdm(total=TARGET_MAX_STEPS, desc="Extracting aligned high-Elo steps")

    for file_path in replay_files[:MAX_EPISODES]:
        if total_steps >= TARGET_MAX_STEPS:
            break
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        steps = data.get("steps", [])
        if len(steps) <= 3:
            continue

        # Determine winner index from terminal step rewards
        last_step = steps[-1]
        if len(last_step) < 2:
            continue
        r0 = last_step[0].get("reward", 0)
        r1 = last_step[1].get("reward", 0)
        if r0 == 1:
            winner_idx = 0
        elif r1 == 1:
            winner_idx = 1
        else:
            continue

        # Extract initial decks from step 1 (action submitted in response to step 0)
        deck_0 = steps[1][0].get("action") if len(steps[1]) > 0 else None
        deck_1 = steps[1][1].get("action") if len(steps[1]) > 1 else None
        trackers = {
            0: CardTracker(deck_0) if isinstance(deck_0, list) and len(deck_0) == 60 else None,
            1: CardTracker(deck_1) if isinstance(deck_1, list) and len(deck_1) == 60 else None,
        }

        # Terminal prize margin
        term_obs = (last_step[0].get("observation") or {}).get("current") or {}
        term_players = term_obs.get("players") or [{}, {}]
        p0_left = len((term_players[0] or {}).get("prize") or []) if len(term_players) > 0 else 0
        p1_left = len((term_players[1] or {}).get("prize") or []) if len(term_players) > 1 else 0

        # 90% games for train, 10% games for val (split cleanly by game file)
        is_val_game = (games_processed % 10 == 9)
        games_processed += 1

        # Strictly align observation at step s_idx (s_idx >= 1) with action at step s_idx + 1
        for s_idx in range(1, len(steps) - 1):
            if total_steps >= TARGET_MAX_STEPS:
                break
            curr_step = steps[s_idx]
            next_step = steps[s_idx + 1]
            if len(curr_step) < 2 or len(next_step) < 2:
                continue

            for p_idx in (0, 1):
                if total_steps >= TARGET_MAX_STEPS:
                    break
                p_curr = curr_step[p_idx]
                if p_curr.get("status") != "ACTIVE":
                    continue

                obs = p_curr.get("observation") or {}
                request = obs.get("select")
                if not isinstance(request, dict):
                    continue

                options = request.get("option") or []
                action = next_step[p_idx].get("action")

                # Always update tracker even on forced single-option steps (e.g. deck inspection)
                if len(options) <= 1:
                    if trackers[p_idx] is not None:
                        try:
                            trackers[p_idx].update(obs, request)
                        except Exception:
                            pass
                    continue

                if not isinstance(action, list) or len(action) == 0:
                    if not (isinstance(action, list) and len(action) == 0 and (request.get("minCount", 1) == 0)):
                        continue

                is_winner = (p_idx == winner_idx)
                # Keep 100% of winner decisions and 50% of loser decisions (for balanced critic training)
                if (not is_winner) and ((s_idx + p_idx) % 2 == 1):
                    if trackers[p_idx] is not None:
                        try:
                            trackers[p_idx].update(obs, request)
                        except Exception:
                            pass
                    continue

                # Parse state and options (this also updates trackers[p_idx])
                parsed = parse_game_state_to_tensors(obs, request, tracker=trackers[p_idx])
                opt_indices = np.where(parsed["option_mask"])[0]
                if len(opt_indices) == 0:
                    continue

                # Filter valid chosen option indices
                valid_chosen = [int(a) for a in action if isinstance(a, int) and 0 <= int(a) < len(opt_indices)]

                mn = request.get("minCount")
                mx = request.get("maxCount")
                mn = 1 if mn is None else int(mn)
                mx = 1 if mx is None else int(mx)
                mx = min(mx, len(options))
                mn = min(mn, mx)

                has_policy = False
                token_target = int(opt_indices[0])
                multi_mask = np.zeros(len(parsed["option_mask"]), dtype=bool)
                count_ratio = 1.0
                has_count = False

                if is_winner:
                    if len(valid_chosen) > 0:
                        token_target = int(opt_indices[valid_chosen[0]])
                        for a in valid_chosen:
                            multi_mask[int(opt_indices[a])] = True
                        has_policy = True
                    if mx > mn:
                        chosen_k = min(max(len(valid_chosen), mn), mx)
                        count_ratio = float(chosen_k - mn) / float(max(1, mx - mn))
                        has_count = True

                cur_turn = ((obs.get("current") or {}).get("turn") or 1)
                val_scale = min(1.0, 0.5 + 0.5 * (float(cur_turn) / 14.0))
                val = (1.0 if is_winner else -1.0) * val_scale

                my_rem = p0_left if p_idx == 0 else p1_left
                opp_rem = p1_left if p_idx == 0 else p0_left
                raw_margin = float(opp_rem - my_rem)
                if is_winner and raw_margin <= 0:
                    raw_margin = 1.0
                elif (not is_winner) and raw_margin >= 0:
                    raw_margin = -1.0
                prize_margin = np.clip(raw_margin / 6.0, -1.0, 1.0) * val_scale

                g_t = torch.from_numpy(parsed["global_vec"].astype(np.float16))
                o_t = torch.from_numpy(parsed["obj_features"].astype(np.float16))
                c_t = torch.from_numpy(parsed["card_ids"].astype(np.int16))
                e_t = torch.from_numpy(parsed["evo_ids"].astype(np.int16))
                tt_t = torch.from_numpy(parsed["token_types"].astype(np.int8))
                opt_m = torch.from_numpy(parsed["option_mask"])
                val_m = torch.from_numpy(parsed["valid_action_mask"])
                pad_m = torch.from_numpy(parsed["padding_mask"])
                mm_t = torch.from_numpy(multi_mask)

                if is_val_game:
                    val_globals.append(g_t)
                    val_obj.append(o_t)
                    val_card_ids.append(c_t)
                    val_evo_ids.append(e_t)
                    val_token_types.append(tt_t)
                    val_option_masks.append(opt_m)
                    val_valid_masks.append(val_m)
                    val_padding_masks.append(pad_m)
                    val_targets.append(token_target)
                    val_multi_masks.append(mm_t)
                    val_count_ratios.append(count_ratio)
                    val_has_count.append(has_count)
                    val_values.append(val)
                    val_prize_margins.append(prize_margin)
                    val_has_policy.append(has_policy)
                else:
                    train_globals.append(g_t)
                    train_obj.append(o_t)
                    train_card_ids.append(c_t)
                    train_evo_ids.append(e_t)
                    train_token_types.append(tt_t)
                    train_option_masks.append(opt_m)
                    train_valid_masks.append(val_m)
                    train_padding_masks.append(pad_m)
                    train_targets.append(token_target)
                    train_multi_masks.append(mm_t)
                    train_count_ratios.append(count_ratio)
                    train_has_count.append(has_count)
                    train_values.append(val)
                    train_prize_margins.append(prize_margin)
                    train_has_policy.append(has_policy)

                total_steps += 1
                pbar.update(1)

    pbar.close()

    n_train = len(train_targets)
    n_val = len(val_targets)
    n_train_pol = sum(train_has_policy)
    n_val_pol = sum(val_has_policy)

    print(f"\nExtracted {total_steps} strictly aligned steps across {games_processed} top-Elo games.")
    print(f"Train: {n_train} total ({n_train_pol} winner policy decisions, {sum(train_has_count)} multi-count decisions).")
    print(f"Val:   {n_val} total ({n_val_pol} winner policy decisions, {sum(val_has_count)} multi-count decisions).")
    print(f"Packaging dataset into '{OUTPUT_FILE}'...")

    torch.save({
        "train": {
            "global_vec": torch.stack(train_globals),
            "obj_features": torch.stack(train_obj),
            "card_ids": torch.stack(train_card_ids),
            "evo_ids": torch.stack(train_evo_ids),
            "token_types": torch.stack(train_token_types),
            "option_mask": torch.stack(train_option_masks),
            "valid_action_mask": torch.stack(train_valid_masks),
            "padding_mask": torch.stack(train_padding_masks),
            "target_indices": torch.tensor(train_targets, dtype=torch.long),
            "target_multi_mask": torch.stack(train_multi_masks),
            "target_count_ratios": torch.tensor(train_count_ratios, dtype=torch.float32),
            "has_count_target": torch.tensor(train_has_count, dtype=torch.bool),
            "values": torch.tensor(train_values, dtype=torch.float32),
            "prize_margins": torch.tensor(train_prize_margins, dtype=torch.float32),
            "has_policy_target": torch.tensor(train_has_policy, dtype=torch.bool),
        },
        "val": {
            "global_vec": torch.stack(val_globals),
            "obj_features": torch.stack(val_obj),
            "card_ids": torch.stack(val_card_ids),
            "evo_ids": torch.stack(val_evo_ids),
            "token_types": torch.stack(val_token_types),
            "option_mask": torch.stack(val_option_masks),
            "valid_action_mask": torch.stack(val_valid_masks),
            "padding_mask": torch.stack(val_padding_masks),
            "target_indices": torch.tensor(val_targets, dtype=torch.long),
            "target_multi_mask": torch.stack(val_multi_masks),
            "target_count_ratios": torch.tensor(val_count_ratios, dtype=torch.float32),
            "has_count_target": torch.tensor(val_has_count, dtype=torch.bool),
            "values": torch.tensor(val_values, dtype=torch.float32),
            "prize_margins": torch.tensor(val_prize_margins, dtype=torch.float32),
            "has_policy_target": torch.tensor(val_has_policy, dtype=torch.bool),
        }
    }, OUTPUT_FILE)

    print(f"Done! Dataset saved to '{OUTPUT_FILE}' (~{os.path.getsize(OUTPUT_FILE) / (1024 * 1024):.1f} MB).")


if __name__ == "__main__":
    main()