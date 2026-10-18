"""Figures + RESULTS tables from experiment outputs (skips anything missing).

python -m ptcg.report        -> experiments/figures/*.png, experiments/results_tables.md
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
FIG = os.path.join(EXP, "figures")
os.makedirs(FIG, exist_ok=True)
# validated categorical palette (fixed order), light mode
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.axisbelow": True, "axes.spines.right": False, "lines.linewidth": 2,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
TABLES = []


def jl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def jload(path):
    return json.load(open(path)) if os.path.exists(path) else None


def fig_training():
    runs = {"with belief head": "train_full_s0", "without belief head": "train_nobelief_s0"}
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
    ok = False
    for i, (lab, d) in enumerate(runs.items()):
        log = jl(os.path.join(EXP, d, "log.jsonl"))
        if not log:
            continue
        ok = True
        ep = [r["epoch"] + 1 for r in log]
        ax[0].plot(ep, [r["val"]["pol"] for r in log], color=C[i], label=lab)
        ax[1].plot(ep, [100 * r["val"]["top1"] for r in log], color=C[i], label=lab)
    if not ok:
        plt.close(fig); return
    ax[0].set_title("Validation policy NLL", loc="left"); ax[0].set_xlabel("epoch")
    ax[1].set_title("Validation top-1 accuracy (%)", loc="left"); ax[1].set_xlabel("epoch")
    ax[1].legend(frameon=False)
    fig.savefig(os.path.join(FIG, "training_curves.png")); plt.close(fig)


def fig_policy_by_type(name="full_s0_v"):
    m = jload(os.path.join(EXP, f"metrics_{name}.json")) or jload(os.path.join(EXP, "metrics_full_s0.json"))
    if not m:
        return
    rows = [(k, v) for k, v in m["policy_by_seltype"].items() if v["n"] >= 300]
    rows.sort(key=lambda kv: -kv[1]["n"])
    fig, ax = plt.subplots(figsize=(6, 2.6))
    x = range(len(rows))
    ax.bar([i - 0.2 for i in x], [100 * v["top1"] for _, v in rows], 0.38, color=C[0], label="network")
    ax.bar([i + 0.2 for i in x], [100 * v["uniform_top1"] for _, v in rows], 0.38, color=C[1], label="uniform legal")
    ax.set_xticks(list(x)); ax.set_xticklabels([k for k, _ in rows], rotation=25, ha="right")
    ax.set_title("Test top-1 accuracy by decision type (%)", loc="left"); ax.legend(frameon=False)
    fig.savefig(os.path.join(FIG, "policy_by_type.png")); plt.close(fig)
    t = "| decision type | n | top-1 | top-3 | NLL | uniform top-1 | uniform NLL |\n|---|---|---|---|---|---|---|\n"
    allr = [("ALL", m["policy_all"]), ("MAIN, >=5 options", m.get("policy_main_ge5_options")), ("Dragapult player", m.get("policy_dragapult_player"))] + rows
    for k, v in allr:
        if v:
            t += f"| {k} | {v['n']} | {100*v['top1']:.1f} | {100*v['top3']:.1f} | {v['nll']:.3f} | {100*v['uniform_top1']:.1f} | {v['uniform_nll']:.3f} |\n"
    TABLES.append(("Policy (held-out test games, " + m["ckpt"] + ")", t))


def fig_value():
    cands = [("replay-only value head", "metrics_full_s0.json"), ("+ self-play value fine-tuning", "metrics_full_s0_v.json")]
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    t = "| model | Brier | Brier (constant) | accuracy | ECE |\n|---|---|---|---|---|\n"
    ok = False
    for i, (lab, f) in enumerate(cands):
        m = jload(os.path.join(EXP, f))
        if not m:
            continue
        ok = True
        rel = m["value"]["reliability"]
        ax.plot([r["pred"] for r in rel], [r["obs"] for r in rel], marker="o", ms=4, color=C[i], label=lab)
        v = m["value"]
        t += f"| {lab} | {v['brier']:.4f} | {v['brier_const']:.4f} | {100*v['accuracy']:.1f} | {v['ece']:.4f} |\n"
    if not ok:
        plt.close(fig); return
    ax.set_xlabel("predicted win probability"); ax.set_ylabel("observed win rate")
    ax.set_title("Value calibration (test)", loc="left"); ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(FIG, "value_calibration.png")); plt.close(fig)
    TABLES.append(("Value head (held-out replay test games)", t))


def fig_belief():
    m = jload(os.path.join(EXP, "metrics_full_s0_v.json")) or jload(os.path.join(EXP, "metrics_full_s0.json"))
    if not m or "belief" not in m:
        return
    names = [("random", "random"), ("count_oracle_decklist", "count-based\n(true decklist)"),
             ("learned", "learned head\n(no decklist)"), ("learned_x_count", "learned x count")]
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    vals = [m["belief"][k]["auroc_pooled"] for k, _ in names]
    ax.bar(range(4), vals, color=[MUTED, C[1], C[0], C[2]], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", color=INK, fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels([n for _, n in names], fontsize=7)
    ax.set_ylim(0.4, 1.0); ax.set_title("Opponent-hand prediction AUROC (test)", loc="left")
    fig.savefig(os.path.join(FIG, "belief_auroc.png")); plt.close(fig)
    t = "| method | AUROC | mAP |\n|---|---|---|\n" + "".join(
        f"| {n.replace(chr(10), ' ')} | {m['belief'][k]['auroc_pooled']:.3f} | {m['belief'][k]['mAP']:.3f} |\n" for k, n in names)
    TABLES.append((f"Belief head (n={m['belief']['n_samples']} test decisions)", t))


def gauntlet_table():
    files = sorted(glob.glob(os.path.join(EXP, "gauntlet", "*.summary.json")))
    if not files:
        return
    t = "| candidate | games | pooled score [95% CI] | mean over opponents | per opponent |\n|---|---|---|---|---|\n"
    for f in files:
        s = json.load(open(f))
        p = s["pooled"]
        per = "; ".join(f"{k.split('|')[0].split('/')[-1].strip()[:18]}@{k.split('|')[1].strip()[:-4]}: {v['score']:.2f}" for k, v in s["per_opponent"].items())
        t += f"| {s['tag']} | {p['games']} | {p['score']:.3f} [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}] | {s['mean_score_over_opponents']:.3f} | {per} |\n"
    TABLES.append(("Gauntlet (candidate pilots Dragapult; Kaggle-like 600 s time bank)", t))
    # scaling figure (Crustle deck): raw policy = 0 simulations
    tagmap = {"v_policy_crustle": 0, "scale_16": 16, "v_mcts64_crustle": 64, "scale_256": 256, "scale_1024": 1024,
              "deployed_sub1_crustle": 10**6}
    pts = []
    for tag, sims in tagmap.items():
        s = jload(os.path.join(EXP, "gauntlet", tag + ".summary.json"))
        if s:
            pts.append((sims, s["pooled"]["score"], s["pooled"]["ci95"], s["pooled"]["games"]))
    if len(pts) >= 2:
        pts.sort()
        xs = list(range(len(pts)))
        fig, ax = plt.subplots(figsize=(3.8, 2.6))
        ax.errorbar(xs, [p[1] for p in pts], yerr=[[p[1] - p[2][0] for p in pts], [p[2][1] - p[1] for p in pts]],
                    color=C[0], marker="o", ms=5, capsize=3)
        ax.set_xticks(xs); ax.set_xticklabels(["policy" if p[0] == 0 else ("time bank\n(<=2 s)" if p[0] == 10**6 else str(p[0])) for p in pts])
        ax.set_xlabel("MCTS simulations per decision"); ax.set_ylabel("score vs opponent pool")
        ax.set_title("Search budget scaling (95% CI)", loc="left")
        fig.savefig(os.path.join(FIG, "search_scaling.png")); plt.close(fig)
        t = "| simulations | games | score [95% CI] |\n|---|---|---|\n" + "".join(
            f"| {'raw policy' if p[0] == 0 else ('deployed: time bank, <=2 s/decision' if p[0] == 10**6 else p[0])} | {p[3]} | {p[1]:.3f} [{p[2][0]:.3f}, {p[2][1]:.3f}] |\n" for p in pts)
        TABLES.append(("Search budget scaling (Crustle deck vs pool)", t))


def eval_table():
    files = sorted(glob.glob(os.path.join(EXP, "eval", "*.summary.json")))
    if not files:
        return
    t = "| A | B | games | score A [95% CI] | Elo diff | SPRT |\n|---|---|---|---|---|---|\n"
    for f in files:
        s = json.load(open(f))
        if s["games"] < 20:
            continue
        t += f"| {s['a']} ({os.path.basename(s.get('deck_a') or '')}) | {s['b']} ({os.path.basename(s.get('deck_b') or 'own')}) | {s['games']} | {s['score']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] | {s['elo']:+.0f} | {s['sprt']} |\n"
    TABLES.append(("Head-to-head matches", t))


def main():
    for f in (fig_training, fig_policy_by_type, fig_value, fig_belief, gauntlet_table, eval_table):
        try:
            f()
        except Exception as e:
            print("skip", f.__name__, repr(e))
    with open(os.path.join(EXP, "results_tables.md"), "w", encoding="utf-8") as fh:
        for title, t in TABLES:
            fh.write(f"### {title}\n\n{t}\n")
    print("figures:", os.listdir(FIG))


if __name__ == "__main__":
    main()
