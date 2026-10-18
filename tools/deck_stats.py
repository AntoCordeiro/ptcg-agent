"""Archetype win rates in high-Elo replays (Phase 0 deck choice)."""
import csv, json, math, os, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cards = {}
for r in csv.DictReader(open(os.path.join(ROOT, "the-pokemon-company-ptcg-ai-battle-challenge-playground/EN_Card_Data_R2_full.csv"), encoding="utf-8-sig")):
    cid = int(r["Card ID"]); cards.setdefault(cid, r)
def name(c): return cards.get(c, {}).get("Card Name", str(c))
def is_main_poke(c):
    r = cards.get(c, {})
    return r.get("Category") not in ("n/a", None) and ("ex" in r.get("Card Name", "") or r.get("Stage (Pokémon)/Type (Energy and Trainer)") == "Stage 2")
def archetype(deck):
    cnt = collections.Counter(deck)
    mains = sorted({c for c in cnt if is_main_poke(c)}, key=lambda c: (-cnt[c], c))[:2]
    return " / ".join(sorted(name(c) for c in mains)) or "other"
def wilson(w, n, z=1.96):
    if n == 0: return (0, 0, 1)
    p = w / n; d = 1 + z*z/n; c = p + z*z/(2*n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return p, (c-h)/d, (c+h)/d
min_elo = float(sys.argv[1]) if len(sys.argv) > 1 else 1250
man = [r for r in csv.DictReader(open(os.path.join(ROOT, "data/replays/manifest.csv"))) if float(r["avg_score"]) >= min_elo]
stats = collections.defaultdict(lambda: [0, 0]); decks = collections.defaultdict(collections.Counter); h2h = collections.defaultdict(lambda: [0, 0])
teams = collections.defaultdict(collections.Counter)
for r in man:
    fp = os.path.join(ROOT, "data/replays", r["episode_id"] + ".json")
    if not os.path.exists(fp): continue
    d = json.load(open(fp))
    rew = d.get("rewards") or [0, 0]
    dk = [None, None]
    for s in d["steps"][:4]:
        for i, p in enumerate(s):
            a = p.get("action")
            if isinstance(a, list) and len(a) == 60: dk[i] = a
    if None in dk or rew[0] == rew[1]: continue
    arch = [archetype(x) for x in dk]
    for i in (0, 1):
        win = 1 if rew[i] > rew[1-i] else 0
        stats[arch[i]][0] += win; stats[arch[i]][1] += 1
        decks[arch[i]][tuple(sorted(dk[i]))] += 1
        h2h[(arch[i], arch[1-i])][0] += win; h2h[(arch[i], arch[1-i])][1] += 1
        teams[arch[i]][(d.get("info") or {}).get("TeamNames", ["?","?"])[i]] += 1
print(f"replays >= {min_elo} Elo: {len(man)}")
rows = sorted(stats.items(), key=lambda kv: -kv[1][1])
out = []
for a, (w, n) in rows[:15]:
    p, lo, hi = wilson(w, n)
    print(f"{a:55s} n={n:4d} wr={p:.3f} [{lo:.3f},{hi:.3f}]  teams={dict(teams[a].most_common(3))}")
    out.append({"archetype": a, "games": n, "wins": w, "wr": p, "ci": [lo, hi], "top_list": list(decks[a].most_common(1)[0][0]), "teams": dict(teams[a].most_common(5))})
json.dump({"min_elo": min_elo, "archetypes": out, "h2h": {f"{k[0]} || {k[1]}": v for k, v in h2h.items() if v[1] >= 8}}, open(os.path.join(ROOT, "experiments", f"deck_stats_{int(min_elo)}.json"), "w"), indent=1)
