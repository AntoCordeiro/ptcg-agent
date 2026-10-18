"""Build a large opponent-deck library for MCTS determinization: exact 60-card lists from top replays
and from our ladder opponents, ranked by frequency. Output: decks_lib/NNN_<count>.csv"""
import json, os, csv, collections, glob, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
man = sorted(csv.DictReader(open(os.path.join(ROOT, "data/replays/manifest.csv"))), key=lambda r: -float(r["avg_score"]))
cnt = collections.Counter()
def add(steps):
    for s in steps[:6]:
        for p in s:
            a = p.get("action")
            if isinstance(a, list) and len(a) == 60 and all(isinstance(x, int) for x in a):
                cnt[tuple(sorted(a))] += 1
for r in man[:2500]:
    fp = os.path.join(ROOT, "data/replays", r["episode_id"] + ".json")
    if os.path.exists(fp):
        try: add(json.load(open(fp))["steps"])
        except Exception: pass
for fp in glob.glob(os.path.join(ROOT, "data/ladder/*/*.json")):
    if fp.endswith("episodes.json"): continue
    try:
        st = json.load(open(fp))["steps"]
        for s in st:   # bo3: decks re-submitted each game
            for p in s:
                a = p.get("action")
                if isinstance(a, list) and len(a) == 60: cnt[tuple(sorted(a))] += 3   # ladder decks weighted up
    except Exception: pass
out = os.path.join(ROOT, "decks_lib")
shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
keep = [(d, c) for d, c in cnt.most_common() if c >= 2][:120]
for i, (d, c) in enumerate(keep):
    open(os.path.join(out, f"{i:03d}_{c}.csv"), "w").write("\n".join(map(str, d)) + "\n")
print("distinct decks", len(cnt), "kept", len(keep), "top counts", [c for _, c in keep[:10]])
