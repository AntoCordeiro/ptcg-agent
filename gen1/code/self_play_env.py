# self_play_env.py
import sys
import os
import json
import torch
import numpy as np

# Add cg path
submission_dir = os.path.abspath(
    "the-pokemon-company-ptcg-ai-battle-challenge-playground/sample_submission/sample_submission/sample_submission"
)
if submission_dir not in sys.path:
    sys.path.insert(0, submission_dir)

from cg.game import Game
from feature_parser import parse_game_state_to_tensors

def load_deck(deck_path: str = None) -> list:
    if deck_path and os.path.exists(deck_path):
        with open(deck_path, "r") as f:
            return [int(line.strip()) for line in f if line.strip()]
    # Default Dragapult ex sample list
    return [
        7, 7, 7, 7, 112, 112, 112, 112, 646, 646, 646, 646, 
        1086, 1086, 1086, 1086, 1152, 1152, 1152, 1152, 1219, 1219, 
        1219, 1219, 1227, 1227, 1227, 1227, 1259, 1259, 1259, 1259, 
        647, 647, 647, 648, 648, 648, 1079, 1079, 1079, 1097, 1097, 
        1097, 104, 104, 860, 860, 1161, 1161, 1182, 1182, 1080, 1231,
        7, 7, 112, 112, 646, 646
    ]

class PTCGSelfPlayEnv:
    def __init__(self, deck_p0=None, deck_p1=None):
        self.deck_p0 = deck_p0 or load_deck()
        self.deck_p1 = deck_p1 or load_deck()
        self.game = None

    def reset(self):
        # Initialize match instance
        self.game = Game()
        # Seed both decks into the game setup
        obs_p0, obs_p1 = self.game.initialize(self.deck_p0, self.deck_p1)
        return obs_p0, obs_p1

    def step(self, player_idx: int, action_indices: list):
        # Step the engine with selected options
        return self.game.step(player_idx, action_indices)