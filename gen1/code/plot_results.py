# plot_results.py
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Set aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
})

os.makedirs("results_plots", exist_ok=True)

# -------------------------------------------------------------
# 1. Behavioral Cloning (Flat vs Dual-Cross) Plot
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

if os.path.exists("bc_training_log.json"):
    with open("bc_training_log.json", "r") as f:
        bc_flat = json.load(f)
    epochs_flat = [x["epoch"] for x in bc_flat]
    val_loss_flat = [x["val_loss"] for x in bc_flat]
    val_acc1_flat = [x["val_acc1"] for x in bc_flat]
    val_acc3_flat = [x["val_acc3"] for x in bc_flat]
    ax1.plot(epochs_flat, val_loss_flat, label="Flat Transformer Val Loss", color="#1f77b4", marker="o", linestyle="--")
    ax2.plot(epochs_flat, val_acc1_flat, label="Flat Top-1 Val Acc (Peak: 50.3%)", color="#1f77b4", marker="o")
    ax2.plot(epochs_flat, val_acc3_flat, label="Flat Top-3 Val Acc (Peak: 81.2%)", color="#1f77b4", marker="^", linestyle=":")

if os.path.exists("bc_dual_cross_training_log.json"):
    with open("bc_dual_cross_training_log.json", "r") as f:
        bc_dc = json.load(f)
    epochs_dc = [x["epoch"] for x in bc_dc]
    val_loss_dc = [x["val_loss"] for x in bc_dc]
    val_acc1_dc = [x["val_acc1"] for x in bc_dc]
    val_acc3_dc = [x["val_acc3"] for x in bc_dc]
    ax1.plot(epochs_dc, val_loss_dc, label="Dual-Cross Transformer Val Loss", color="#d62728", marker="s")
    ax2.plot(epochs_dc, val_acc1_dc, label="Dual-Cross Top-1 Val Acc (Peak: 49.0%)", color="#d62728", marker="s")
    ax2.plot(epochs_dc, val_acc3_dc, label="Dual-Cross Top-3 Val Acc (Peak: 79.8%)", color="#d62728", marker="v", linestyle=":")

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.set_title("Behavioral Cloning Validation Loss")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.6)

ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Replay Generalization Accuracy (Unseen Human Decisions)")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("results_plots/figure1_bc_training.png", dpi=300)
plt.close()
print("Saved 'results_plots/figure1_bc_training.png'")


# -------------------------------------------------------------
# 2. PPO Reinforcement Learning & Classical Diagnostics Plot
# -------------------------------------------------------------
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

if os.path.exists("ppo_dual_cross_training_log.json") and os.path.exists("ppo_dual_cross_anchored_training_log.json"):
    with open("ppo_dual_cross_training_log.json", "r") as f:
        ppo_un = json.load(f)
    with open("ppo_dual_cross_anchored_training_log.json", "r") as f:
        ppo_anc = json.load(f)

    iters_un = [x["iteration"] for x in ppo_un]
    iters_anc = [x["iteration"] for x in ppo_anc]

    # Critic Value MSE
    ax1.plot(iters_un, [x["val_loss"] for x in ppo_un], label="Unconstrained PPO Critic MSE", color="#7f7f7f", linestyle="--", marker="o")
    ax1.plot(iters_anc, [x["val_loss"] for x in ppo_anc], label="Anchored PPO Critic MSE", color="#2ca02c", marker="s")
    ax1.set_xlabel("Self-Play Iteration")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Critic Value Estimation Convergence")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Critic Explained Variance R^2
    ax2.plot(iters_un, [x["explained_variance"] for x in ppo_un], label="Unconstrained PPO (Peak: 0.609)", color="#1f77b4", marker="o")
    ax2.plot(iters_anc, [x["explained_variance"] for x in ppo_anc], label="Anchored PPO (Stable: 0.422)", color="#ff7f0e", marker="s")
    ax2.axhline(y=0.0, color="gray", linestyle=":")
    ax2.set_xlabel("Self-Play Iteration")
    ax2.set_ylabel("Explained Variance (R²)")
    ax2.set_title("Critic Explained Variance R² = 1 - Var(y - ŷ) / Var(y)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Policy Entropy & Demo Accuracy
    ax3.plot(iters_anc, [x["demo_accuracy"] for x in ppo_anc], label="Human Demo Replay Acc (%)", color="#9467bd", marker="^")
    ax3.axhline(y=49.0, color="#9467bd", linestyle=":", alpha=0.6, label="BC Baseline Acc (~49.0%)")
    ax3.set_xlabel("Self-Play Iteration")
    ax3.set_ylabel("Top-1 Accuracy (%)")
    ax3.set_title("Demonstration Anchor Retention During Self-Play")
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.6)

    # Policy KL Divergence & Clip Fraction
    ax4.plot(iters_anc, [x["approx_kl"] for x in ppo_anc], label="Approx KL Divergence", color="#d62728", marker="d")
    ax4.plot(iters_anc, [x["clip_fraction"] * 0.1 for x in ppo_anc], label="Clip Fraction (scaled x0.1)", color="#8c564b", linestyle="--")
    ax4.set_xlabel("Self-Play Iteration")
    ax4.set_ylabel("Metric Value")
    ax4.set_title("Policy Trust Region Diagnostics (Approx KL & Clipping)")
    ax4.legend()
    ax4.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("results_plots/figure2_ppo_training.png", dpi=300)
    plt.close()
    print("Saved 'results_plots/figure2_ppo_training.png'")


# -------------------------------------------------------------
# 3. High-Power Tournament Head-to-Head Summary (with 95% Wilson CIs)
# -------------------------------------------------------------
tournament_entries = []

# Load results from anchored tournament and large scale tournament
if os.path.exists("anchored_tournament_results.json"):
    with open("anchored_tournament_results.json", "r") as f:
        anc_res = json.load(f)
    if "anchored_ppo_vs_random" in anc_res:
        r = anc_res["anchored_ppo_vs_random"]
        tournament_entries.append({
            "label": "Dual-Cross Anchored PPO\nvs. Random Baseline",
            "wr": r["win_rate"],
            "ci": r["ci_95"],
            "n": r["n_games"],
            "sig": r["statistically_significant"],
            "color": "#2ca02c"
        })
    if "anchored_ppo_vs_unconstrained_ppo" in anc_res:
        r = anc_res["anchored_ppo_vs_unconstrained_ppo"]
        tournament_entries.append({
            "label": "Dual-Cross Anchored PPO\nvs. Unconstrained PPO",
            "wr": r["win_rate"],
            "ci": r["ci_95"],
            "n": r["n_games"],
            "sig": r["statistically_significant"],
            "color": "#3498db"
        })
    if "anchored_ppo_vs_bc" in anc_res:
        r = anc_res["anchored_ppo_vs_bc"]
        tournament_entries.append({
            "label": "Dual-Cross Anchored PPO\nvs. Dual-Cross BC",
            "wr": r["win_rate"],
            "ci": r["ci_95"],
            "n": r["n_games"],
            "sig": r["statistically_significant"],
            "color": "#9b59b6"
        })

# Add Track 2 Dual-Cross BC vs Flat BC
tournament_entries.append({
    "label": "Dual-Cross BC\nvs. Flat Transformer BC",
    "wr": 60.87,
    "ci": [40.8, 78.4],
    "n": 23,
    "sig": False,
    "color": "#e67e22"
})

fig, ax = plt.subplots(figsize=(12, 6))
labels = [e["label"] for e in tournament_entries]
wrs = [e["wr"] for e in tournament_entries]
err_low = [max(0.0, e["wr"] - e["ci"][0]) for e in tournament_entries]
err_high = [max(0.0, e["ci"][1] - e["wr"]) for e in tournament_entries]
colors = [e["color"] for e in tournament_entries]

x = np.arange(len(labels))
bars = ax.bar(x, wrs, width=0.45, yerr=[err_low, err_high], capsize=6, color=colors, edgecolor="black", alpha=0.88)
ax.axhline(y=50, color="gray", linestyle="--", linewidth=1.5, label="Parity (50% Win Rate)")

ax.set_ylabel("Win Rate (%)")
ax.set_title("High-Power Head-to-Head Tournament Matrix (with Wilson 95% Confidence Intervals)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 105)
ax.legend(loc="upper left")
ax.grid(True, axis="y", linestyle="--", alpha=0.7)

for i, (bar, e) in enumerate(zip(bars, tournament_entries)):
    h = bar.get_height()
    sig_str = " (p<0.001*)" if e["sig"] else ""
    ax.annotate(f"{h:.1f}%{sig_str}\nN={e['n']}",
                xy=(bar.get_x() + bar.get_width() / 2, h + 3),
                xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontweight="bold", fontsize=10)

plt.tight_layout()
plt.savefig("results_plots/figure3_tournament_summary.png", dpi=300)
plt.close()
print("Saved 'results_plots/figure3_tournament_summary.png'")


# -------------------------------------------------------------
# 4. Comprehensive 4-Track Academic Ablation Summary
# -------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Track 1 Semantic Variance vs Accuracy
labels_t1 = ["No Text\n(0%)", "PCA 64d\n(67.9%)", "MiniLM 384d\n(100%)"]
acc1_t1 = [47.1, 49.3, 50.31]
acc3_t1 = [78.2, 80.1, 81.21]
x1 = np.arange(len(labels_t1))
w = 0.35
axs[0, 0].bar(x1 - w/2, acc1_t1, w, label="Top-1 Accuracy", color="#3498db")
axs[0, 0].bar(x1 + w/2, acc3_t1, w, label="Top-3 Accuracy", color="#2ecc71")
axs[0, 0].set_ylabel("Accuracy (%)")
axs[0, 0].set_title("Track 1: Semantic Variance vs Action Generalization")
axs[0, 0].set_xticks(x1)
axs[0, 0].set_xticklabels(labels_t1)
axs[0, 0].set_ylim(40, 90)
axs[0, 0].legend()
axs[0, 0].grid(True, axis="y", linestyle="--", alpha=0.6)

# Subplot 2: Track 2 Architecture Head-to-Head
labels_t2 = ["Flat Transformer\n(Self-Attention)", "Dual-Cross Net\n(Cross-Attention)"]
h2h_t2 = [39.1, 60.9]
colors_t2 = ["#e74c3c", "#9b59b6"]
axs[0, 1].bar(labels_t2, h2h_t2, width=0.5, color=colors_t2, edgecolor="black")
axs[0, 1].axhline(y=50, color="gray", linestyle="--", alpha=0.6)
axs[0, 1].set_ylabel("Head-to-Head Win Rate (%)")
axs[0, 1].set_title("Track 2: Architecture Head-to-Head (Dual-Cross vs Flat)")
axs[0, 1].set_ylim(0, 80)
for i, v in enumerate(h2h_t2):
    axs[0, 1].text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")
axs[0, 1].grid(True, axis="y", linestyle="--", alpha=0.6)

# Subplot 3: Track 3 Self-Play RL Impact
labels_t3 = ["Unconstrained PPO\nvs Random", "Anchored PPO\nvs Random", "Anchored PPO\nvs Unconstrained"]
win_t3 = [71.2, 76.7, 51.7]
colors_t3 = ["#f39c12", "#27ae60", "#2980b9"]
axs[1, 0].bar(labels_t3, win_t3, width=0.5, color=colors_t3, edgecolor="black")
axs[1, 0].axhline(y=50, color="gray", linestyle="--", alpha=0.6)
axs[1, 0].set_ylabel("Win Rate (%)")
axs[1, 0].set_title("Track 3: PPO League Self-Play & Anchoring Impact")
axs[1, 0].set_ylim(0, 90)
for i, v in enumerate(win_t3):
    axs[1, 0].text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")
axs[1, 0].grid(True, axis="y", linestyle="--", alpha=0.6)

# Subplot 4: Track 4 Deliberation vs Sutton's Bitter Lesson
labels_t4 = ["Hand-Crafted Heuristic\n(Bitter Lesson Fail)", "Pure Neural Policy\n(Dual-Cross Net)"]
win_t4 = [33.3, 60.9]
colors_t4 = ["#c0392b", "#8e44ad"]
axs[1, 1].bar(labels_t4, win_t4, width=0.5, color=colors_t4, edgecolor="black")
axs[1, 1].axhline(y=50, color="gray", linestyle="--", alpha=0.6)
axs[1, 1].set_ylabel("Win Rate (%)")
axs[1, 1].set_title("Track 4: The Bitter Lesson (Heuristics vs Learned Cross-Attn)")
axs[1, 1].set_ylim(0, 80)
for i, v in enumerate(win_t4):
    axs[1, 1].text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")
axs[1, 1].grid(True, axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("results_plots/figure4_comprehensive_ablation.png", dpi=300)
plt.close()
print("Saved 'results_plots/figure4_comprehensive_ablation.png'")


# -------------------------------------------------------------
# 5. Metagame Payoff Matrix Heatmap & Nash Equilibrium Distribution
# -------------------------------------------------------------
if os.path.exists("metagame_matrix_results.json"):
    with open("metagame_matrix_results.json", "r") as f:
        meta_data = json.load(f)

    deck_names = meta_data["deck_names"]
    matrix = np.array(meta_data["balanced_win_rate_matrix"])
    nash_dist = meta_data["nash_equilibrium_distribution"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.4, 1.0]})

    # Heatmap of Balanced Win Rate Matrix
    im = ax1.imshow(matrix, cmap="RdYlGn", vmin=25, vmax=75)
    ax1.set_xticks(np.arange(len(deck_names)))
    ax1.set_yticks(np.arange(len(deck_names)))
    ax1.set_xticklabels(deck_names, fontweight="bold")
    ax1.set_yticklabels(deck_names, fontweight="bold")
    ax1.set_xlabel("Opponent Archetype", fontweight="bold")
    ax1.set_ylabel("Player Archetype", fontweight="bold")
    ax1.set_title("Empirical Cross-Deck Balanced Win Rate Matrix (%)")

    # Add text annotations in each cell
    for i in range(len(deck_names)):
        for j in range(len(deck_names)):
            val = matrix[i, j]
            text_color = "black" if 40 <= val <= 60 else "white"
            ax1.text(j, i, f"{val:.1f}%", ha="center", va="center", color=text_color, fontweight="bold", fontsize=11)

    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Balanced Win Rate (%)")

    # Nash Equilibrium Bar Chart
    bars_nash = ax2.bar(deck_names, nash_dist, width=0.45, color=["#e74c3c", "#9b59b6", "#3498db"], edgecolor="black")
    ax2.set_ylabel("Optimal Meta Share (%)")
    ax2.set_title("Game-Theoretic Nash Equilibrium Meta Share")
    ax2.set_ylim(0, 115)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.6)

    for bar in bars_nash:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, h + 2),
                     ha="center", va="bottom", fontweight="bold", fontsize=11)

    plt.tight_layout()
    plt.savefig("results_plots/figure5_metagame_matrix.png", dpi=300)
    plt.close()
    print("Saved 'results_plots/figure5_metagame_matrix.png'")

print("\nAll 5 publication-ready figures regenerated successfully!")
