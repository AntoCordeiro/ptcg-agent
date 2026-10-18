import os, sys, random, json, collections
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
torch.set_num_threads(1)
from cg import game
from ptcg.agents import load_deck, RandomAgent, PolicyAgent, HeuristicAgent
from ptcg.model import load_model
m = load_model(sys.argv[1], "cpu")
dd, do = load_deck("decks/dragapult.csv"), load_deck("decks/ogerpon.csv")
def run(a0, a1, d0, d1):
    obs, _ = game.battle_start(d0, d1); steps = 0; acts = collections.Counter(); last = None
    while obs["current"]["result"] < 0 and steps < 4000:
        who = obs["current"]["yourIndex"]; s = obs["select"]
        a = [a0, a1][who].act(obs)
        if who == 0 and s["type"] == 0:
            for i in a: acts[s["option"][i]["type"]] += 1
        last = obs
        obs = game.battle_select(a); steps += 1
    reasons = [lg for lg in (obs.get("logs") or []) if lg.get("type") == 23]
    cur = obs["current"]
    info = dict(result=cur["result"], turn=cur["turn"], steps=steps, reason=reasons[-1].get("reason") if reasons else None,
                prizes=[len(p["prize"]) for p in cur["players"]], deck=[p["deckCount"] for p in cur["players"]], round=cur.get("round"), win=[p.get("win") for p in cur["players"]], main_acts_p0=dict(acts))
    game.battle_finish(); return info
for opp_name, opp in [("random", RandomAgent(do, 1)), ("heuristic", HeuristicAgent(do, 1))]:
    for g in range(4):
        print(opp_name, run(PolicyAgent(m, dd), opp, dd, do), flush=True)
print("random vs random", [run(RandomAgent(dd, g), RandomAgent(do, g + 50), dd, do)["result"] for g in range(20)])
print("heuristic(drag) vs random", [run(HeuristicAgent(dd, g), RandomAgent(do, g + 50), dd, do)["result"] for g in range(20)])
