"""Phase 0: simulator throughput + search determinism."""
import os, sys, time, random, json, dataclasses
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cg import game, api
deck_d = json.load(open("experiments/deck_stats_1250.json"))["archetypes"][0]["top_list"]
deck_o = [int(x) for x in open("baselines/deck_grimmsnarl.csv").read().split()]
rng = random.Random(0)
def rand_sel(obs):
    s = obs["select"]; n = len(s["option"]); lo, hi = s["minCount"], s["maxCount"]
    k = rng.randint(lo, hi) if hi > 0 else 0
    return sorted(rng.sample(range(n), k))
t0 = time.time(); steps = 0; games = 0; lens = []
while time.time() - t0 < 20:
    obs, sd = game.battle_start(deck_d, deck_o)
    n = 0
    while obs["current"]["result"] < 0 and n < 3000:
        obs = game.battle_select(rand_sel(obs)); n += 1
    game.battle_finish(); steps += n; games += 1; lens.append(n)
dt = time.time() - t0
print(f"random play: {games} games, {steps/dt:.0f} steps/s, {dt/games*1000:.1f} ms/game, mean len {sum(lens)/len(lens):.0f}")
# search determinism + speed
obs, sd = game.battle_start(deck_d, deck_o)
for _ in range(30):
    obs = game.battle_select(rand_sel(obs))
o = api.to_observation_class(obs); cur = obs["current"]; me = cur["yourIndex"]; P = cur["players"]
def cnt(p, k): return len(p[k]) if p.get(k) is not None else 0
yd = [deck_d[0]] * P[me]["deckCount"]; ypz = [deck_d[0]] * len(P[me]["prize"])
od = [deck_o[0]] * P[1-me]["deckCount"]; opz = [deck_o[0]] * len(P[1-me]["prize"]); oh = [deck_o[0]] * P[1-me]["handCount"]
# use plausible cards: basic pokemon id from deck
st = api.search_begin(o, yd, ypz, od, opz, oh, [646])
root = st.searchId
print("root options", len(st.observation.select.option), "yourIndex", st.observation.current.yourIndex)
res = []
for rep in range(3):
    s1 = api.search_step(root, [0] if st.observation.select.minCount >= 1 or st.observation.select.maxCount>=1 else [])
    res.append(json.dumps(dataclasses.asdict(s1.observation.current), sort_keys=True)[:100000])
print("deterministic same-node step:", len(set(res)) == 1)
t1 = time.time(); k = 0
while time.time() - t1 < 5:
    cur_id = root; ob = st.observation
    for d in range(20):
        sel = ob.select
        if ob.current.result >= 0 or sel is None or len(sel.option) == 0: break
        n = len(sel.option); kk = max(sel.minCount, min(1, sel.maxCount))
        s2 = api.search_step(cur_id, sorted(rng.sample(range(n), kk)))
        cur_id = s2.searchId; ob = s2.observation; k += 1
    api.search_end(); st = api.search_begin(o, yd, ypz, od, opz, oh, [646]); root = st.searchId
print(f"search_step throughput incl. asdict-free: {k/(time.time()-t1):.0f} steps/s")
