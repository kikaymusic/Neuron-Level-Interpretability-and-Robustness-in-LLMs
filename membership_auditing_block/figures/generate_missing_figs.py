"""
generate_missing_figs.py
Genera las 5 figuras del anexo que no produce generate_presentation_figs.py.
Nomenclatura nueva: E1=random, E2=single-layer, E3=four-layer, E4=global.

Output (PDF):
  figs_presentacion/P05_e4_cascade.pdf
  figs_presentacion/P07_e4_vs_e1.pdf
  figs_presentacion/P08_comparative.pdf
  figs_presentacion/P09_summary_dashboard.pdf
  figs_syn+lum/G_summary_panel.pdf
"""

import json, glob, os, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "axes.labelsize":   13,
    "legend.fontsize":  11,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "lines.linewidth":  2.2,
    "lines.markersize": 7,
})

C = {
    "baseline": "#555555",
    "e1":       "#E07B39",   # E1 random
    "e2":       "#2F8EC3",   # E2 single-layer
    "e3":       "#4BAE8A",   # E3 four-layer
    "e4":       "#9B6BC5",   # E4 global
    "red":      "#D9534F",
    "chance":   "#BBBBBB",
}

OUT_P = "../figs_presentacion"
OUT_G = "../figs_syn+lum"
os.makedirs(OUT_P, exist_ok=True)
os.makedirs(OUT_G, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with open("../results/sweep_results_malwspecsys.json") as f:
    mem = json.load(f)

with open("../results/task_utility_malwspecsys.json") as f:
    task = json.load(f)

task_by_p = {r["top_p"]: r for r in task["results"]}
TOP_P     = [r["top_p"] for r in mem["results"]]
x         = np.arange(len(TOP_P))
xlabs     = [str(p) for p in TOP_P]
LAYERS    = list(range(12))
l_labs    = [f"L{l}" for l in LAYERS]

def mem_serie(key):  return [r[key] for r in mem["results"]]
def task_serie(key): return [task_by_p[p][key] for p in TOP_P]
def layer_auc(r, prefix):
    d = json.loads(r[f"{prefix}_auc_per_layer"])
    return [d.get(str(l), np.nan) for l in LAYERS]

# Series (new naming)
e1_auc = mem_serie("e4_rnd_best_auc_def")    # E1 random
e2_auc = mem_serie("e1_p3_best_auc_def")     # E2 single-layer
e3_auc = mem_serie("e2_p4a_best_auc_def")    # E3 four-layer
e4_auc = mem_serie("e3_p4b_best_auc_def")    # E4 global

e1_tpr = mem_serie("e4_rnd_tpr_at_fpr1")
e4_tpr = mem_serie("e3_p4b_tpr_at_fpr1")

e1_f1  = task_serie("e4_rnd_weighted_f1")        # E1 random
e2_f1  = task_serie("e1_p3_weighted_f1")        # E2 single-layer
e3_f1  = task_serie("e2_p4a_weighted_f1")       # E3 four-layer
e4_f1  = task_serie("e3_p4b_weighted_f1")       # E4 global

BASE_AUC = mem["baseline"]["best_auc"]
BASE_F1  = task["baseline_f1"]
BASE_TPR = 0.117

sp = TOP_P.index(0.4)

# E4 per-layer matrix (cascade)
e4_mat = np.array([layer_auc(r, "e3_p4b") for r in mem["results"]])


# ─────────────────────────────────────────────────────────────────────────────
# P05_e4_cascade — cascade collapse + TPR sweet spot
# ─────────────────────────────────────────────────────────────────────────────
SHOW_P = [0.001, 0.05, 0.2, 0.4, 0.65, 0.8]
show_idx = [TOP_P.index(p) for p in SHOW_P]
colors_cascade = plt.cm.Blues(np.linspace(0.35, 0.95, len(SHOW_P)))

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))

for ci, (pi, p_val) in enumerate(zip(show_idx, SHOW_P)):
    ax_l.plot(l_labs, e4_mat[pi], "o-", color=colors_cascade[ci],
              linewidth=1.8, markersize=5, label=f"top_p = {p_val}")
ax_l.axhline(0.5, color=C["chance"], linewidth=1.5, linestyle="--",
             label="Chance (0.5)")
ax_l.set_ylim(0.1, 1.02)
ax_l.set_xlabel("Transformer layer")
ax_l.set_ylabel("MIA AUC")
ax_l.legend(fontsize=9, loc="lower left")

# Annotation at p=0.4: L11 goes below chance
idx_04 = TOP_P.index(0.4)
l11_val = e4_mat[idx_04][11]
ax_l.annotate(f"L11 goes below\nchance at top_p=0.4\n({l11_val:.2f})",
              xy=(11, l11_val), xytext=(8, 0.22),
              fontsize=9, color=C["e4"],
              arrowprops=dict(arrowstyle="->", color=C["e4"], lw=1.2))

# Right panel: TPR@1%FPR
ax_r.plot(x, e4_tpr, "^-", color=C["e4"], label="E4  TPR@1%FPR")
ax_r.axhline(BASE_TPR, color=C["baseline"], linewidth=1.5, linestyle=":",
             alpha=0.8, label=f"Baseline ({BASE_TPR:.3f})")
ax_r.axvline(sp, color=C["e4"], linewidth=1.5, linestyle="--", alpha=0.6)
ax_r.text(sp + 0.15, 0.135, f"top_p = 0.4\nTPR = {e4_tpr[sp]:.3f}",
          fontsize=11, color=C["e4"], fontweight="bold")
ax_r.set_xticks(x); ax_r.set_xticklabels(xlabs, rotation=40, ha="right")
ax_r.set_xlabel("top_p")
ax_r.set_ylabel("TPR at 1% FPR")
ax_r.set_ylim(-0.005, 0.175)
ax_r.legend(loc="upper right")

plt.tight_layout()
plt.savefig(f"{OUT_P}/P05_e4_cascade.pdf", bbox_inches="tight")
plt.close()
print("✓ P05_e4_cascade.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# P07_e4_vs_e1 — E4 (informed) vs E1 (random) at top_p=0.4, 3 panels
# ─────────────────────────────────────────────────────────────────────────────
e4_vals = [e4_auc[sp], e4_tpr[sp], e4_f1[sp]]
e1_vals = [e1_auc[sp], e1_tpr[sp], e1_f1[sp]]
ylims   = [(0.60, 0.92), (0.0, 0.14), (0.70, 1.00)]
stars   = [BASE_AUC, BASE_TPR, BASE_F1]
titles  = ["MIA AUC", "TPR@1%FPR", "Task wt. F1"]

fig, axes = plt.subplots(1, 3, figsize=(13, 6))

bw = 0.32
xi = np.array([0])

for i, ax in enumerate(axes):
    b4 = ax.bar(xi - bw/2, [e4_vals[i]], bw, color=C["e4"],
                label="E4  informed", alpha=0.88)
    b1 = ax.bar(xi + bw/2, [e1_vals[i]], bw, color=C["e1"],
                label="E1  random",   alpha=0.88)
    ax.bar_label(b4, labels=[f"{e4_vals[i]:.3f}"], padding=5,
                 fontsize=13, fontweight="bold", color=C["e4"])
    ax.bar_label(b1, labels=[f"{e1_vals[i]:.3f}"], padding=5,
                 fontsize=13, fontweight="bold", color=C["e1"])
    ax.plot([0], [stars[i]], "*", color="#DAA520", markersize=16, zorder=5,
            clip_on=False)
    ax.set_xticks([]); ax.set_ylim(*ylims[i])
    ax.set_title(titles[i], fontsize=14, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.4, linestyle="--")
    ax.legend(fontsize=11)

# Annotate TPR panel with factor
axes[1].text(0, ylims[1][0] + (ylims[1][1]-ylims[1][0])*0.55,
             "E1 is\n13× higher\n(worse)",
             ha="center", fontsize=11, color=C["e1"], fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec=C["e1"], alpha=0.85))

plt.tight_layout()
plt.savefig(f"{OUT_P}/P07_e4_vs_e1.pdf", bbox_inches="tight")
plt.close()
print("✓ P07_e4_vs_e1.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# P08_comparative — AUC sweep + F1 sweep, all 4 conditions
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for ax, series, ylabel, title, base, ylim in [
    (ax1, [(e1_auc,"E1  Random","D"), (e2_auc,"E2  Single-layer","o"),
           (e3_auc,"E3  Four-layer","s"), (e4_auc,"E4  Global  ★","^")],
     "Best MIA AUC", "MIA AUC across all conditions", BASE_AUC, (0.68, 1.02)),
    (ax2, [(e1_f1,"E1  Random","D"), (e2_f1,"E2  Single-layer","o"),
           (e3_f1,"E3  Four-layer","s"), (e4_f1,"E4  Global  ★","^")],
     "Task weighted F1", "Task utility across all conditions", BASE_F1, (0.0, 1.05)),
]:
    ax.axhline(base, color=C["baseline"], linewidth=1.5, linestyle="--",
               alpha=0.7, label=f"Baseline ({base:.3f})")
    for (data, lbl, mk), col in zip(series, [C["e1"],C["e2"],C["e3"],C["e4"]]):
        ax.plot(x, data, f"{mk}-", color=col, label=lbl)
    ax.axvline(sp, color=C["e4"], linewidth=1.2, linestyle=":", alpha=0.5)
    ax.text(sp + 0.1, ylim[0] + (ylim[1]-ylim[0])*0.05, "sweet\nspot",
            fontsize=9, color=C["e4"], alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(xlabs, rotation=40, ha="right")
    ax.set_xlabel("top_p")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.legend(loc="lower left", ncol=1, fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUT_P}/P08_comparative.pdf", bbox_inches="tight")
plt.close()
print("✓ P08_comparative.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# P09_summary_dashboard — 5-panel key results
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
ax_auc_bar = fig.add_subplot(gs[0, 0])
ax_tpr_bar = fig.add_subplot(gs[0, 1])
ax_f1_bar  = fig.add_subplot(gs[0, 2])
ax_sweep   = fig.add_subplot(gs[1, 0:2])
ax_tpr_sw  = fig.add_subplot(gs[1, 2])

labels_bar = ["Baseline", "E1\n(rand)", "E2\n(L11)", "E3\n(4-lay)", "E4\n(glob)"]
auc_vals   = [BASE_AUC, e1_auc[sp], e2_auc[sp], e3_auc[sp], e4_auc[sp]]
tpr_vals   = [BASE_TPR, e1_tpr[sp], float("nan"), float("nan"), e4_tpr[sp]]
f1_vals    = [BASE_F1,  e1_f1[sp],  e2_f1[sp],   e3_f1[sp],   e4_f1[sp]]
bar_colors = [C["baseline"], C["e1"], C["e2"], C["e3"], C["e4"]]

for ax, vals, ylabel, title, ylim in [
    (ax_auc_bar, auc_vals, "MIA AUC", f"MIA AUC at top_p = 0.4", (0.68, 1.05)),
    (ax_tpr_bar, tpr_vals, "TPR @ 1% FPR", f"Attack precision at top_p = 0.4", (0.0, 0.16)),
    (ax_f1_bar,  f1_vals,  "Task wt. F1", f"Task utility at top_p = 0.4", (0.70, 1.02)),
]:
    bars = ax.bar(labels_bar, vals, color=bar_colors, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=bar.get_facecolor())
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")

# Annotate TPR bar: factor
e4_tpr_val = e4_tpr[sp]
e1_tpr_val = e1_tpr[sp]
if e4_tpr_val > 0:
    factor = e1_tpr_val / e4_tpr_val
    ax_tpr_bar.text(3, e4_tpr_val + 0.005,
                    f"← {factor:.0f}× less\nthan random",
                    fontsize=8, color=C["e4"], fontweight="bold")

# AUC sweep
ax_sweep.axhline(BASE_AUC, color=C["baseline"], linewidth=1.5, linestyle="--",
                 alpha=0.7, label=f"Baseline ({BASE_AUC:.3f})")
ax_sweep.plot(x, e1_auc, "D-", color=C["e1"], label="E1")
ax_sweep.plot(x, e2_auc, "o-", color=C["e2"], label="E2")
ax_sweep.plot(x, e3_auc, "s-", color=C["e3"], label="E3")
ax_sweep.plot(x, e4_auc, "^-", color=C["e4"], label="E4  ★")
ax_sweep.axvline(sp, color=C["e4"], linewidth=1.2, linestyle=":", alpha=0.5)
ax_sweep.set_xticks(x); ax_sweep.set_xticklabels(xlabs, rotation=40, ha="right")
ax_sweep.set_xlabel("top_p"); ax_sweep.set_ylabel("MIA AUC")
ax_sweep.set_title("MIA AUC sweep"); ax_sweep.set_ylim(0.68, 1.02)
ax_sweep.legend(ncol=2, fontsize=10)

# TPR sweep E4 vs E1
ax_tpr_sw.plot(x, e4_tpr, "^-", color=C["e4"], label="E4")
ax_tpr_sw.plot(x, e1_tpr, "D-", color=C["e1"], label="E1")
ax_tpr_sw.axhline(BASE_TPR, color=C["baseline"], linewidth=1.5, linestyle=":",
                  alpha=0.7, label=f"Baseline")
ax_tpr_sw.axvline(sp, color=C["e4"], linewidth=1.2, linestyle=":", alpha=0.5)
ax_tpr_sw.set_xticks(x); ax_tpr_sw.set_xticklabels(xlabs, rotation=40, ha="right")
ax_tpr_sw.set_xlabel("top_p"); ax_tpr_sw.set_ylabel("TPR @ 1% FPR")
ax_tpr_sw.set_title("High-precision attack\nE4 vs E1"); ax_tpr_sw.set_ylim(-0.005, 0.22)
ax_tpr_sw.legend(fontsize=10)

plt.savefig(f"{OUT_P}/P09_summary_dashboard.pdf", bbox_inches="tight")
plt.close()
print("✓ P09_summary_dashboard.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# G_summary_panel — 4-panel academic summary
# ─────────────────────────────────────────────────────────────────────────────
xlabs_pct = [f"{int(p*100)}%" for p in TOP_P]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# (a) Membership AUC
ax = axes[0, 0]
ax.axhline(BASE_AUC, color=C["baseline"], linewidth=1.5, linestyle="--",
           alpha=0.7, label=f"Baseline = {BASE_AUC:.4f}")
ax.plot(x, e1_auc, "D--", color=C["e1"], linewidth=1.8, label="E1 – Random control")
ax.plot(x, e2_auc, "o-",  color=C["e2"], linewidth=1.8, label="E2 – Single-layer (L*)")
ax.plot(x, e3_auc, "s-",  color=C["e3"], linewidth=1.8, label="E3 – Four-layer (L8–L11)")
ax.plot(x, e4_auc, "^-",  color=C["e4"], linewidth=2.2, label="E4 – Global informed")
ax.set_xticks(x); ax.set_xticklabels(xlabs_pct, rotation=40, ha="right", fontsize=9)
ax.set_ylabel("Membership AUC"); ax.set_title("(a) Membership AUC")
ax.set_ylim(0.68, 1.02); ax.legend(fontsize=9, loc="lower left")

# (b) TPR@FPR=1%
ax = axes[0, 1]
ax.axhline(BASE_TPR, color=C["baseline"], linewidth=1.5, linestyle="--",
           label=f"Baseline = {BASE_TPR:.3f}")
ax.plot(x, e4_tpr, "^-",  color=C["e4"], linewidth=2.2, label="E4 – Global informed")
ax.plot(x, e1_tpr, "D--", color=C["e1"], linewidth=1.8, label="E1 – Random control")
ax.set_xticks(x); ax.set_xticklabels(xlabs_pct, rotation=40, ha="right", fontsize=9)
ax.set_ylabel("TPR @ FPR=1%"); ax.set_title("(b) TPR @ FPR=1%")
ax.set_ylim(-0.005, 0.22); ax.legend(fontsize=9)

# (c) Task utility
ax = axes[1, 0]
ax.axhline(BASE_F1, color=C["baseline"], linewidth=1.5, linestyle="--",
           alpha=0.7, label=f"Baseline = {BASE_F1:.4f}")
ax.plot(x, e1_f1, "D--", color=C["e1"], linewidth=1.8, label="E1")
ax.plot(x, e2_f1, "o-",  color=C["e2"], linewidth=1.8, label="E2")
ax.plot(x, e3_f1, "s-",  color=C["e3"], linewidth=1.8, label="E3")
ax.plot(x, e4_f1, "^-",  color=C["e4"], linewidth=2.2, label="E4")
ax.set_xticks(x); ax.set_xticklabels(xlabs_pct, rotation=40, ha="right", fontsize=9)
ax.set_ylabel("Weighted F1 Score"); ax.set_title("(c) Task utility")
ax.set_ylim(0.0, 1.05); ax.legend(fontsize=9, loc="lower left")

# (d) AUC reduction vs baseline
ax = axes[1, 1]
ax.plot(x, [BASE_AUC - v for v in e1_auc], "D--", color=C["e1"],
        linewidth=1.8, label="E1")
ax.plot(x, [BASE_AUC - v for v in e2_auc], "o-",  color=C["e2"],
        linewidth=1.8, label="E2")
ax.plot(x, [BASE_AUC - v for v in e3_auc], "s-",  color=C["e3"],
        linewidth=1.8, label="E3")
ax.plot(x, [BASE_AUC - v for v in e4_auc], "^-",  color=C["e4"],
        linewidth=2.2, label="E4")
ax.axhline(0, color=C["baseline"], linewidth=1.0, linestyle=":", alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(xlabs_pct, rotation=40, ha="right", fontsize=9)
ax.set_ylabel("ΔAUC (reduction)"); ax.set_title("(d) AUC reduction")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT_G}/G_summary_panel.pdf", bbox_inches="tight")
plt.close()
print("✓ G_summary_panel.pdf")

print("\n✅  5 figures generated.")
