# metrics_eval.py
"""
Classical Reinforcement Learning & Game-Theoretic Evaluation Suite.

Implements standard academic metrics:
1. Wilson Score Binomial Confidence Intervals (95% CI)
2. Exact Binomial Two-Sided Hypothesis Testing against H0: p = 0.5
3. Explained Variance of the Value Function: R^2 = 1 - Var(y - y_hat) / Var(y)
4. Policy Optimization Diagnostics: Approximate KL Divergence & Clip Fraction
5. Seat & Turn-Order Disaggregated Metrics (Balanced Win Rate, First-Player Advantage)
6. Margin of Victory / Prize Differential Distribution
"""

import math
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy import stats


def compute_wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Computes the Wilson score interval for a binomial proportion.
    Unlike normal approximation (Wald interval), Wilson works reliably even for small n or extreme p.
    Returns: (point_estimate, ci_lower, ci_upper) as percentages in [0, 100].
    """
    if n == 0:
        return 0.0, 0.0, 0.0

    p = k / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2.0)
    z2 = z * z
    n_float = float(n)

    denominator = 1.0 + z2 / n_float
    center = (p + z2 / (2.0 * n_float)) / denominator
    spread = (z / denominator) * math.sqrt((p * (1.0 - p) / n_float) + (z2 / (4.0 * n_float * n_float)))

    ci_lower = max(0.0, center - spread) * 100.0
    ci_upper = min(1.0, center + spread) * 100.0
    point_est = p * 100.0

    return point_est, ci_lower, ci_upper


def compute_binomial_pvalue(k: int, n: int, p_null: float = 0.5) -> float:
    """
    Performs a two-sided exact binomial test for H0: true win rate p == p_null.
    """
    if n == 0:
        return 1.0
    res = stats.binomtest(k, n, p=p_null, alternative="two-sided")
    return float(res.pvalue)


def compute_explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes Explained Variance: R^2 = 1 - Var(y_true - y_pred) / Var(y_true).
    Standard metric in Stable-Baselines3 and CleanRL.
    Interpreted as:
      1.0: Perfect value prediction
      0.0: Value critic is no better than predicting the mean return
      < 0: Value critic is actively worse than a constant mean prediction (indicates critic divergence)
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    var_y = np.var(y_true)
    if var_y == 0 or np.isnan(var_y):
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / var_y)


def compute_approx_kl(new_log_probs: np.ndarray, old_log_probs: np.ndarray) -> float:
    """
    Computes Schulman's approximate KL divergence:
    KL ~= E[(exp(logratio) - 1) - logratio]
    Always >= 0 and numerically well-behaved.
    """
    logratio = new_log_probs - old_log_probs
    ratio = np.exp(logratio)
    approx_kl = np.mean((ratio - 1.0) - logratio)
    return float(approx_kl)


def compute_clip_fraction(new_log_probs: np.ndarray, old_log_probs: np.ndarray, clip_eps: float = 0.2) -> float:
    """
    Computes fraction of policy updates where the surrogate ratio was clipped by [1 - eps, 1 + eps].
    Typical healthy range for PPO: 0.05 to 0.25.
    """
    ratio = np.exp(new_log_probs - old_log_probs)
    clipped = (ratio < (1.0 - clip_eps)) | (ratio > (1.0 + clip_eps))
    return float(np.mean(clipped))


def compute_tournament_statistics(games: List[dict], agent_a_name: str, agent_b_name: str) -> dict:
    """
    Comprehensive tournament match analyzer.
    games is a list of dicts:
      {
         "winner": 0 or 1 (0 = player who went first / seat 0, 1 = player who went second / seat 1),
         "agent_a_is_p0": bool,
         "p0_prizes_left": int,
         "p1_prizes_left": int,
         "steps": int
      }
    """
    valid_games = len(games)
    if valid_games == 0:
        return {}

    a_wins = 0
    b_wins = 0
    a_first_games = 0
    a_first_wins = 0
    a_second_games = 0
    a_second_wins = 0
    prize_deltas_a = []
    steps_list = []

    for g in games:
        a_is_p0 = g["agent_a_is_p0"]
        winner = g["winner"]
        p0_left = g.get("p0_prizes_left", 0)
        p1_left = g.get("p1_prizes_left", 0)
        steps = g.get("steps", 0)
        steps_list.append(steps)

        # In Pokémon TCG, prizes taken = 6 - prizes_left
        p0_taken = 6 - p0_left
        p1_taken = 6 - p1_left
        margin_a = (p0_taken - p1_taken) if a_is_p0 else (p1_taken - p0_taken)
        prize_deltas_a.append(margin_a)

        a_won = (winner == 0 and a_is_p0) or (winner == 1 and not a_is_p0)
        if a_won:
            a_wins += 1
        else:
            b_wins += 1

        if a_is_p0:
            a_first_games += 1
            if a_won:
                a_first_wins += 1
        else:
            a_second_games += 1
            if a_won:
                a_second_wins += 1

    # Wilson 95% CIs
    a_point, a_ci_low, a_ci_high = compute_wilson_score_interval(a_wins, valid_games)
    p_val = compute_binomial_pvalue(a_wins, valid_games)

    # Disaggregated Seat Metrics
    win_as_first = (a_first_wins / max(a_first_games, 1)) * 100.0
    win_as_second = (a_second_wins / max(a_second_games, 1)) * 100.0
    balanced_win_rate = (win_as_first + win_as_second) / 2.0
    seat_bias_impact = win_as_first - win_as_second

    # Margin of Victory
    avg_margin = float(np.mean(prize_deltas_a))
    std_margin = float(np.std(prize_deltas_a))
    sweeps = sum(1 for m in prize_deltas_a if m >= 5)  # Dominant 6-0 or 6-1
    close_games = sum(1 for m in prize_deltas_a if abs(m) <= 1)

    return {
        "agent_a": agent_a_name,
        "agent_b": agent_b_name,
        "n_games": valid_games,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "win_rate": a_point,
        "ci_95": (round(a_ci_low, 1), round(a_ci_high, 1)),
        "p_value": p_val,
        "statistically_significant": p_val < 0.05,
        "win_rate_first": round(win_as_first, 1),
        "win_rate_second": round(win_as_second, 1),
        "balanced_win_rate": round(balanced_win_rate, 1),
        "first_player_advantage": round(seat_bias_impact, 1),
        "avg_prize_margin": round(avg_margin, 2),
        "margin_std": round(std_margin, 2),
        "sweep_rate": round(100.0 * sweeps / valid_games, 1),
        "close_game_rate": round(100.0 * close_games / valid_games, 1),
        "avg_steps": round(float(np.mean(steps_list)), 1),
    }
