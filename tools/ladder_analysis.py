"""Analyse our Kaggle ladder episodes: opponents, decks, results, time usage."""
import json, os, sys, collections, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cards = {}
for r in csv.DictReader(open("the-pokemon-company-ptcg-ai-battle-challenge-playground/EN_Card_Data_R2_full.csv", encoding="utf-8-sig")):
    cards.setdefault(int(r["Card ID"]), r)
def is_main(c):
    r = cards.get(c, {})
    return r.get("Category") not in ("n/a", None) and ("ex" in r.get("Card Name", "") or r.get("Stage (Pokémon)/Type (Energy and Trainer)") == "Stage 2")
def arch(deck):
    cnt = collections.Counter(deck)
    m = sorted({c for c in cnt if is_main(c)}, key=lambda c: (-cnt[c], c))[:2]
    return " / ".join(sorted(cards[c]["Card Name"] for c in m)) or "other"
for sub in sys.argv[1:]:
    d = json.load(open(f"data/ladder/{sub}/episodes.json"))
    mysub = collections.Counter(a["sub"] for e in d for a in e["agents"]).most_common(1)[0][0]
    rows = []
    for e in d:
        fp = f"data/ladder/{sub}/{e['id']}.json"
        if not os.path.exists(fp) or e["state"] != "COMPLETED":
            continue
        ep = json.load(open(fp))
        ags = e["agents"]
        me = 0 if ags[0]["sub"] == mysub else 1
        if ags[0]["sub"] == ags[1]["sub"]:
            continue  # validation self-play
        opp = 1 - me
        rew = [x if x is not None else -1 for x in ep["rewards"]]
        if None in ep["rewards"]: print("  note: None reward in", e["id"], ep["rewards"], [s.get("status") for s in ep["steps"][-1]])
        res = 1 if rew[me] > rew[opp] else (0 if rew[me] < rew[opp] else 0.5)
        odeck = None; ngames = 0; minrem = 1e9
        for s in ep["steps"]:
            a = s[opp].get("action")
            if isinstance(a, list) and len(a) == 60:
                odeck = a; ngames += 1
            r = (s[me].get("observation") or {}).get("remainingOverageTime")
            if r is not None: minrem = min(minrem, r)
        last = ep["steps"][-1][me].get("observation") or {}
        rows.append(dict(id=e["id"], res=res, opp_name=ep["info"]["TeamNames"][opp], opp_score=ags[opp]["init"] or 0,
                         my_score=ags[me]["init"] or 0, opp_arch=arch(odeck) if odeck else "?", n_decks=ngames, min_rem=minrem))
    rows.sort(key=lambda r: r["id"])
    print(f"===== {sub}: {len(rows)} ladder episodes, score {sum(r['res'] for r in rows)/max(1,len(rows)):.3f}")
    print("final my score", rows[-1]["my_score"] if rows else None, " min remaining bank over all episodes %.0f" % min(r["min_rem"] for r in rows))
    A = collections.defaultdict(list)
    for r in rows: A[r["opp_arch"]].append(r["res"])
    for k, v in sorted(A.items(), key=lambda kv: -len(kv[1])):
        print(f"  vs {k:50s} n={len(v):3d} score={sum(v)/len(v):.2f}")
    B = collections.defaultdict(list)
    for r in rows:
        b = int(r["opp_score"] // 100 * 100); B[b].append(r["res"])
    print("  by opponent rating:", {k: (len(v), round(sum(v)/len(v), 2)) for k, v in sorted(B.items())})
    L = collections.Counter(r["opp_name"] for r in rows if r["res"] == 0)
    print("  most losses to:", L.most_common(6))
