# download_top_replays.py
"""
Scrapes the newest 1240-1398+ Elo replays from the live Kaggle PTCG AI Battle Challenge leaderboard
using Kaggle's EpisodeService API and kaggleusercontent.com/episodes/{id}.json, and merges them into
`data/replays/` and `data/replays/manifest.csv`.
"""

import csv
import http.cookiejar
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

REPLAY_DIR = "data/replays"
MANIFEST_PATH = os.path.join(REPLAY_DIR, "manifest.csv")

# Seed submission IDs of top leaderboard teams (1240 - 1398+ Elo)
TOP_SUBMISSION_IDS = [
    55565261,  # Team Luca (1398.1 Elo)
    55559155,  # Team Unown Gradiant (1329.5 Elo)
    55565694,  # Team flg (1324.2 Elo)
    55563046,  # Team Petit Canard (1311.5 Elo)
    55556923,  # Team Petit Canard prev
    55563661,  # Team palsystem (1307.2 Elo)
    55553888,  # Team KawattaTaido (1307.0 Elo)
    55565305,  # Team LumenLiquidity (1297.4 Elo)
    55563809,  # Team Azat Akhtyamov (1295.1 Elo)
    55563984,  # Team e-toppo + kurupical (1251.6 Elo)
    55565623,  # Team AlphaTCG
]


def fetch_episode_metadata():
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    opener.open("https://www.kaggle.com", timeout=10)
    xsrf = next((c.value for c in cj if "XSRF" in c.name.upper()), "")

    all_eps = {}
    for sub_id in TOP_SUBMISSION_IDS:
        try:
            req = urllib.request.Request(
                "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes",
                data=json.dumps({"submissionId": sub_id}).encode("utf-8"),
                headers={"Content-Type": "application/json", "X-XSRF-TOKEN": xsrf},
            )
            with opener.open(req, timeout=12) as r:
                d = json.loads(r.read().decode("utf-8"))
                for ep in d.get("episodes", []):
                    if ep.get("state") == "COMPLETED":
                        all_eps[int(ep["id"])] = ep
        except Exception as e:
            print(f"Warning on sub_id={sub_id}: {e}")

    scored_eps = []
    for eid, ep in all_eps.items():
        agents = ep.get("agents") or []
        if len(agents) != 2:
            continue
        s0 = float(agents[0].get("updatedScore") or agents[0].get("initialScore") or 0.0)
        s1 = float(agents[1].get("updatedScore") or agents[1].get("initialScore") or 0.0)
        if s0 <= 0 or s1 <= 0:
            continue
        avg_s = 0.5 * (s0 + s1)
        min_s = min(s0, s1)
        sum_s = s0 + s1
        scored_eps.append({
            "episode_id": str(eid),
            "create_time": ep.get("createTime", ""),
            "avg_score": avg_s,
            "min_score": min_s,
            "sum_score": sum_s,
            "agent_count": 2,
        })

    scored_eps.sort(key=lambda x: x["avg_score"], reverse=True)
    return scored_eps


def download_one_episode(ep_meta: dict) -> tuple[bool, dict]:
    eid = ep_meta["episode_id"]
    target_path = os.path.join(REPLAY_DIR, f"{eid}.json")
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        ep_meta["size_bytes"] = os.path.getsize(target_path)
        return True, ep_meta

    url = f"https://www.kaggleusercontent.com/episodes/{eid}.json"
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                raw = r.read()
                if len(raw) < 500:
                    return False, ep_meta
                with open(target_path, "wb") as f:
                    f.write(raw)
                ep_meta["size_bytes"] = len(raw)
                return True, ep_meta
        except Exception:
            time.sleep(0.3)
    return False, ep_meta


def main(max_to_download: int = 350):
    os.makedirs(REPLAY_DIR, exist_ok=True)
    existing_rows = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_rows[str(row["episode_id"])] = row

    print(f"Existing manifest episodes: {len(existing_rows)}")
    scored_eps = fetch_episode_metadata()
    print(f"Discovered {len(scored_eps)} completed high-Elo episodes from top leaderboard teams (Peak avg_score: {scored_eps[0]['avg_score']:.1f}).")

    # Select top `max_to_download` episodes not yet downloaded (with avg_score >= 1220)
    to_fetch = [ep for ep in scored_eps if ep["avg_score"] >= 1220.0][:max_to_download]
    print(f"Downloading top {len(to_fetch)} episodes (avg_score {to_fetch[-1]['avg_score']:.1f} .. {to_fetch[0]['avg_score']:.1f})...")

    downloaded = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(download_one_episode, ep) for ep in to_fetch]
        for idx, fu in enumerate(as_completed(futs), 1):
            ok, meta = fu.result()
            if ok:
                downloaded += 1
                existing_rows[str(meta["episode_id"])] = {
                    "episode_id": str(meta["episode_id"]),
                    "create_time": meta["create_time"],
                    "avg_score": f"{meta['avg_score']:.6f}",
                    "min_score": f"{meta['min_score']:.6f}",
                    "sum_score": f"{meta['sum_score']:.6f}",
                    "agent_count": str(meta["agent_count"]),
                    "size_bytes": str(meta.get("size_bytes", 0)),
                }
            if idx % 50 == 0 or idx == len(to_fetch):
                print(f"  Progress: {idx}/{len(to_fetch)} ({downloaded} ok) in {time.time() - t0:.1f}s", flush=True)

    # Sort all manifest rows by avg_score descending and write back
    all_rows = list(existing_rows.values())
    all_rows.sort(key=lambda r: float(r.get("avg_score") or 0.0), reverse=True)
    fieldnames = ["episode_id", "create_time", "avg_score", "min_score", "sum_score", "agent_count", "size_bytes"]
    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\nUpdated '{MANIFEST_PATH}' with {len(all_rows)} total episodes (Top avg_score: {float(all_rows[0]['avg_score']):.1f}).")


if __name__ == "__main__":
    main()
