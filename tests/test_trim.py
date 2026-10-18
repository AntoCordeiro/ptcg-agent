"""Trimmed (dynamic-length) inference must equal padded inference; also measures speed."""
import os, sys, time, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
torch.set_num_threads(1)
from ptcg.model import load_model, to_torch
from ptcg.features import encode, stack
from ptcg.agents import net_eval
m = load_model("experiments/train_full_s0_crashed_ep8/best.pt")
d = json.load(open("data/replays/104323071.json"))
obs = [p["observation"] for s in d["steps"] for p in s if p.get("status") == "ACTIVE" and (p.get("observation") or {}).get("select") and len(p["observation"]["select"]["option"]) >= 2][:60]
maxdiff = 0
for o in obs:
    p_trim, v_trim, *_ = net_eval(m, [o])[0]
    with torch.no_grad():
        out = m(to_torch(stack([encode(o)]), "cpu"))
    n = len(p_trim)
    lg = out["logits"][0, :n].numpy(); pp = np.exp(lg - lg.max()); pp /= pp.sum()
    maxdiff = max(maxdiff, float(np.abs(pp - p_trim).max()), abs(float(out["value"][0]) - v_trim))
print("max abs diff trimmed vs padded:", maxdiff)
t = time.time()
for o in obs: net_eval(m, [o])
t1 = (time.time() - t) / len(obs)
t = time.time()
with torch.no_grad():
    for o in obs: m(to_torch(stack([encode(o)]), "cpu"))
t2 = (time.time() - t) / len(obs)
t = time.time()
for o in obs: encode(o)
t3 = (time.time() - t) / len(obs)
print(f"ms per eval: trimmed {t1*1000:.2f}  padded {t2*1000:.2f}  encode-only {t3*1000:.2f}")
assert maxdiff < 1e-4
print("OK")
