import os, sys, random, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
torch.set_num_threads(1)
from cg import game, api
from ptcg.agents import load_deck, RandomAgent, PolicyAgent
from ptcg.mcts import MCTSAgent, obs_to_dict, Node
from ptcg.model import load_model
from ptcg.eval import play_game
m = load_model(sys.argv[1], "cpu")
dd, do = load_deck("decks/dragapult.csv"), load_deck("decks/ogerpon.csv")
# 1) policy vs random sanity
res = []
for g in range(6):
    pa, ra = PolicyAgent(m, dd), RandomAgent(do, g)
    seat = g % 2
    r = play_game([pa, ra] if seat == 0 else [ra, pa], [dd, do] if seat == 0 else [do, dd])
    res.append(r["winner"] == seat)
print("policy vs random wins:", res, flush=True)
# 2) inspect a search
rng = random.Random(1)
obs, _ = game.battle_start(dd, do)
ra = RandomAgent(do, 3); pa = PolicyAgent(m, dd)
while True:
    cur = obs["current"]; s = obs["select"]
    if cur["yourIndex"] == 0 and s["type"] == 0 and len(s["option"]) >= 3 and cur["turn"] >= 3:
        break
    a = pa.act(obs) if cur["yourIndex"] == 0 else ra.act(obs)
    obs = game.battle_select(a)
print("root turn", obs["current"]["turn"], "opts", [o["type"] for o in obs["select"]["option"]])
ag = MCTSAgent(m, dd, sims=64, worlds=1, seed=0)
from ptcg.agents import net_eval
p, v, c, bel, _ = net_eval(m, [obs])[0]
print("policy probs", np.round(p, 3), "value", round(v, 3))
w = ag.sample_world(obs, bel)
print("world sizes", [len(x) for x in w], "true opp hand count", obs["current"]["players"][1]["handCount"], "deck", obs["current"]["players"][0]["deckCount"])
st = api.search_begin(api.to_observation_class(obs), *w)
ro = obs_to_dict(st.observation)
print("root same options:", [o["type"] for o in ro["select"]["option"]] == [o["type"] for o in obs["select"]["option"]], "root yourIndex", ro["current"]["yourIndex"])
node = Node(st.searchId, ro)
for d in range(40):
    if node.terminal: print("terminal", node.result); break
    s = node.obs["select"]; cur = node.obs["current"]
    hands = [None if pl.get("hand") is None else len(pl["hand"]) for pl in cur["players"]]
    print(f"d{d} turn {cur['turn']} yourIndex {cur['yourIndex']} seltype {s['type']} ctx {s['context']} nopt {len(s['option'])} min{s['minCount']} max{s['maxCount']} handsVisible {hands} prizes {[len(pl['prize']) for pl in cur['players']]}")
    n = len(s["option"]); k = max(s["minCount"], min(1, s["maxCount"]))
    a = sorted(rng.sample(range(n), k))
    st = api.search_step(node.sid, a); node = Node(st.searchId, obs_to_dict(st.observation))
api.search_end()
ag2 = MCTSAgent(m, dd, sims=128, worlds=2, seed=0)
act = ag2.act(obs); print("mcts action", act, dict(ag2.stats))
