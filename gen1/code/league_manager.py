# league_manager.py
"""
Dynamic Elo League Manager for Multi-Deck Pokémon TCG Self-Play (Track 3).

Key Capabilities:
1. Dynamic Elo tracking across all historical checkpoints and archetypes.
2. Competitive Matchmaking: Samples opponents based on logistic win probability (close Elo ratings)
   to ensure training stability and prevent catastrophic forgetting / policy cycling.
3. Automated Leaderboard Logging: Persists league standings to JSON for evaluation and visualization.
"""

import os
import json
import math
import random
import copy
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class LeagueAgent:
    def __init__(self, name: str, model: nn.Module, elo: float = 1200.0, is_trainable: bool = False):
        self.name = name
        self.model = model
        self.elo = elo
        self.is_trainable = is_trainable
        self.matches_played = 0
        self.wins = 0

    def win_rate(self) -> float:
        return (self.wins / max(self.matches_played, 1)) * 100.0


class LeagueManager:
    def __init__(self, k_factor: float = 32.0, log_path: str = "league_leaderboard.json"):
        self.k_factor = k_factor
        self.log_path = log_path
        self.agents: Dict[str, LeagueAgent] = {}

    def register_agent(self, name: str, model: nn.Module, initial_elo: float = 1200.0, is_trainable: bool = False):
        # Store a frozen copy if not trainable
        if not is_trainable:
            model_copy = copy.deepcopy(model).eval()
            for p in model_copy.parameters():
                p.requires_grad = False
        else:
            model_copy = model

        self.agents[name] = LeagueAgent(name, model_copy, elo=initial_elo, is_trainable=is_trainable)
        print(f"[League] Registered agent '{name}' with initial Elo {initial_elo:.1f}")

    def get_expected_score(self, elo_a: float, elo_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

    def update_match_result(self, name_a: str, name_b: str, score_a: float):
        """
        score_a: 1.0 for A win, 0.5 for draw, 0.0 for B win
        """
        if name_a not in self.agents or name_b not in self.agents:
            return

        agent_a = self.agents[name_a]
        agent_b = self.agents[name_b]

        exp_a = self.get_expected_score(agent_a.elo, agent_b.elo)
        exp_b = 1.0 - exp_a
        score_b = 1.0 - score_a

        # Elo update
        agent_a.elo += self.k_factor * (score_a - exp_a)
        agent_b.elo += self.k_factor * (score_b - exp_b)

        agent_a.matches_played += 1
        agent_b.matches_played += 1

        if score_a > 0.5:
            agent_a.wins += 1
        elif score_b > 0.5:
            agent_b.wins += 1

    def sample_opponent(self, active_name: str, pool_names: Optional[List[str]] = None) -> Tuple[str, nn.Module]:
        """
        Samples an opponent from the league using Elo-distance softmax weighting.
        Prefers opponents of comparable skill level to maximize learning gradient.
        """
        if pool_names is None:
            pool = [name for name in self.agents.keys() if name != active_name]
        else:
            pool = [name for name in pool_names if name in self.agents and name != active_name]

        if not pool:
            # Fallback to self
            return active_name, self.agents[active_name].model

        active_elo = self.agents[active_name].elo
        
        # Compute matchmaking weights: higher probability for closer Elo
        weights = []
        for opp_name in pool:
            opp_elo = self.agents[opp_name].elo
            diff = abs(active_elo - opp_elo)
            # Gaussian similarity kernel around active Elo (sigma = 150)
            weight = math.exp(-0.5 * (diff / 150.0) ** 2)
            weights.append(max(weight, 0.05))

        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        chosen_name = random.choices(pool, weights=probs, k=1)[0]
        return chosen_name, self.agents[chosen_name].model

    def get_leaderboard(self) -> List[dict]:
        sorted_agents = sorted(self.agents.values(), key=lambda a: a.elo, reverse=True)
        board = []
        for rank, a in enumerate(sorted_agents, 1):
            board.append({
                "rank": rank,
                "name": a.name,
                "elo": round(a.elo, 1),
                "matches": a.matches_played,
                "wins": a.wins,
                "win_rate": round(a.win_rate(), 1),
            })
        return board

    def save_leaderboard(self):
        board = self.get_leaderboard()
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(board, f, indent=2)
        print(f"[League] Saved leaderboard ({len(board)} agents) to {self.log_path}")

    def print_leaderboard(self):
        board = self.get_leaderboard()
        print("\n" + "="*60)
        print(f"{'Rank':<6}{'Agent Name':<28}{'Elo':<10}{'Matches':<10}{'Win Rate':<10}")
        print("-" * 60)
        for b in board:
            print(f"{b['rank']:<6}{b['name']:<28}{b['elo']:<10.1f}{b['matches']:<10}{b['win_rate']:<9.1f}%")
        print("="*60 + "\n")
