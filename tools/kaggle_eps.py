"""List/download Kaggle episodes (public endpoints). usage: kaggle_eps.py ids <episodeId> | sub <submissionId> [download_dir]"""
import json, sys, os, urllib.request, http.cookiejar
cj = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
op.open("https://www.kaggle.com", timeout=15)
xsrf = next((c.value for c in cj if "XSRF" in c.name.upper()), "")
def call(payload):
    req = urllib.request.Request("https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "X-XSRF-TOKEN": xsrf})
    return json.loads(op.open(req, timeout=20).read().decode())
mode, val = sys.argv[1], int(sys.argv[2])
d = call({"ids": [val]} if mode == "ids" else {"submissionId": val})
eps = d.get("episodes", [])
print(len(eps), "episodes")
out = []
for ep in eps:
    ag = ep.get("agents") or []
    out.append({"id": ep["id"], "state": ep.get("state"), "create": ep.get("createTime"),
                "agents": [{"sub": a.get("submissionId"), "reward": a.get("reward"), "init": a.get("initialScore"), "upd": a.get("updatedScore"), "idx": a.get("index")} for a in ag]})
print(json.dumps(out[:3], indent=0)[:1500])
if len(sys.argv) > 3:
    dd = sys.argv[3]; os.makedirs(dd, exist_ok=True)
    json.dump(out, open(os.path.join(dd, "episodes.json"), "w"), indent=1)
    for e in out:
        fp = os.path.join(dd, f"{e['id']}.json")
        if e["state"] == "COMPLETED" and not os.path.exists(fp):
            try:
                with urllib.request.urlopen(f"https://www.kaggleusercontent.com/episodes/{e['id']}.json", timeout=30) as r:
                    open(fp, "wb").write(r.read())
            except Exception as ex:
                print("dl fail", e["id"], ex)
