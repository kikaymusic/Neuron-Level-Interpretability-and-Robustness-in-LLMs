"""
Presentation figures - clean, minimal, Canva-ready.
One story per figure, no overlapping text, readable at projector size.

Nomenclatura actual (nueva):
  E1 = Random control        (datos: e4_rnd,  task: e1_rnd_weighted_f1)
  E2 = Single-layer (L11)   (datos: e1_p3,   task: e2_single_weighted_f1)
  E3 = Four-layer (L8-L11)  (datos: e2_p4a,  task: e3_multi_weighted_f1)
  E4 = Global informed       (datos: e3_p4b,  task: e4_global_weighted_f1)

Output: figs_presentacion/
  P01_vulnerability_bar.pdf      - baseline AUC vs random chance
  P02_layer_gradient.pdf         - per-layer AUC (where the model leaks)
  P03_e2_layer_shift.pdf         - L11 vs L10 as top_p increases (E2)
  P04_e3_floor.pdf               - E2 vs E3 sweep + L7 floor line
  P05_e4_tpr.pdf                 - E4 TPR@1%FPR sweep
  P06_e4_sweet_spot.pdf          - E4 dual-axis AUC + F1
  P07_e1_bars.pdf                - E4 vs E1 at top_p=0.4 (AUC / TPR / F1)
  P08_all_auc.pdf                - all 4 conditions AUC sweep
  P09_all_f1.pdf                 - all 4 conditions F1 sweep
"""

import json, glob, os, warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Style
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         14,
    "axes.titlesize":    16,
    "axes.titleweight":  "bold",
    "axes.labelsize":    14,
    "legend.fontsize":   13,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "lines.linewidth":   2.4,
    "lines.markersize":  8,
})

C = {
    "baseline": "#555555",
    "chance":   "#BBBBBB",
    "e1":       "#E07B39",   # E1 random - naranja
    "e2":       "#2F8EC3",   # E2 single-layer - azul
    "e3":       "#4BAE8A",   # E3 four-layer - verde
    "e4":       "#9B6BC5",   # E4 global - morado
    "red":      "#D9534F",
}

OUT = "../figs_presentacion"
os.makedirs(OUT, exist_ok=True)

# Load data
with open("../results/sweep_results_malwspecsys.json") as f:
    mem = json.load(f)

with open("../results/task_utility_malwspecsys.json") as f:
    task = json.load(f)

task_by_p = {r["top_p"]: r for r in task["results"]}
TOP_P     = [r["top_p"] for r in mem["results"]]
x         = np.arange(len(TOP_P))
xlabs     = [str(p) for p in TOP_P]

def mem_serie(key):   return [r[key] for r in mem["results"]]
def task_serie(key):  return [task_by_p[p][key] for p in TOP_P]

# Data keys → new experiment labels
e2_auc = mem_serie("e1_p3_best_auc_def")    # E2 single-layer
e3_auc = mem_serie("e2_p4a_best_auc_def")   # E3 four-layer
e4_auc = mem_serie("e3_p4b_best_auc_def")   # E4 global informed
e1_auc = mem_serie("e4_rnd_best_auc_def")   # E1 random control

e4_tpr = mem_serie("e3_p4b_tpr_at_fpr1")   # E4 global TPR
e1_tpr = mem_serie("e4_rnd_tpr_at_fpr1")   # E1 random TPR

e1_f1  = task_serie("e4_rnd_weighted_f1")        # E1 random
e2_f1  = task_serie("e1_p3_weighted_f1")        # E2 single-layer
e3_f1  = task_serie("e2_p4a_weighted_f1")       # E3 four-layer
e4_f1  = task_serie("e3_p4b_weighted_f1")       # E4 global

BASE_AUC = mem["baseline"]["best_auc"]   # 0.9468
BASE_F1  = task["baseline_f1"]            # 0.9214
BASE_TPR = 0.117

# Per-layer baseline
r001  = next(r for r in mem["results"] if r["top_p"] == 0.001)
lbase = {int(k): v for k, v in json.loads(r001["e4_rnd_auc_per_layer"]).items()}
layers  = sorted(lbase)
l_vals  = [lbase[l] for l in layers]
l_labs  = [f"L{l}" for l in layers]

# E2 per-layer: L11 (defended) and L10 (absorbs signal)
e2_l11 = [json.loads(r["e1_p3_auc_per_layer"]).get("11", 0) for r in mem["results"]]
e2_l10 = [json.loads(r["e1_p3_auc_per_layer"]).get("10", 0) for r in mem["results"]]

sp = TOP_P.index(0.4)   # sweet-spot index


# ---
# P01 - Vulnerability bar: chance vs baseline
# ---
fig, ax = plt.subplots(figsize=(6, 5))

bars = ax.bar(["Random\nchance", "Undefended\nmodel"],
              [0.5, BASE_AUC],
              color=[C["chance"], C["red"]],
              width=0.45, edgecolor="white", linewidth=1.5)

ax.bar_label(bars, labels=["0.500", f"{BASE_AUC:.3f}"],
             padding=6, fontsize=14, fontweight="bold")
ax.axhline(0.5, color=C["chance"], linewidth=1.5, linestyle="--")
ax.set_ylim(0, 1.12)
ax.set_ylabel("MIA AUC")
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

plt.tight_layout()
plt.savefig(f"{OUT}/P01_vulnerability_bar.pdf", bbox_inches="tight")
plt.close()
print("✓ P01")


# ---
# P02 - Per-layer AUC gradient  (l* = L11)
# ---
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(l_labs, l_vals, "o-", color=C["red"], zorder=3)
ax.axhline(0.5, color=C["chance"], linewidth=1.5, linestyle="--",
           label="Random chance (0.5)")

# Highlight L11 as l*
lstar = layers.index(11)
ax.plot(l_labs[lstar], l_vals[lstar], "o", color=C["red"],
        markersize=14, zorder=4)
ax.text(lstar - 1.1, l_vals[lstar] + 0.015,
        "L11 = 0.947", fontsize=13,
        color=C["red"], fontweight="bold")

ax.set_ylim(0.45, 1.02)
ax.set_xlabel("Transformer layer")
ax.set_ylabel("MIA AUC")
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.legend(loc="upper left")

plt.tight_layout()
plt.savefig(f"{OUT}/P02_layer_gradient.pdf", bbox_inches="tight")
plt.close()
print("✓ P02")


# ---
# P03 - E2: layer shift  (L11 defended, L10 absorbs signal)
# ---
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(x, e2_l11, "o--", color=C["e2"], label="L11  (defended)")
ax.plot(x, e2_l10, "s-",  color=C["red"], label="L10  (undefended)")
ax.axhline(BASE_AUC, color=C["baseline"], linewidth=1.5, linestyle=":",
           alpha=0.7, label=f"Baseline AUC ({BASE_AUC:.3f})")

ax.set_xticks(x); ax.set_xticklabels(xlabs, rotation=40, ha="right")
ax.set_xlabel("top_p  (fraction of neurons suppressed at L11)")
ax.set_ylabel("MIA AUC")
ax.set_ylim(0.50, 1.02)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.legend(loc="lower left")

plt.tight_layout()
plt.savefig(f"{OUT}/P03_e2_layer_shift.pdf", bbox_inches="tight")
plt.close()
print("✓ P03")


# ---
# P04 - E3: AUC sweep vs E2, with L7 floor
# ---
fig, ax = plt.subplots(figsize=(9, 5))

ax.axhline(BASE_AUC, color=C["baseline"], linewidth=1.5, linestyle=":",
           alpha=0.7, label=f"Baseline ({BASE_AUC:.3f})")
ax.plot(x, e2_auc, "o-",  color=C["e2"], label="E2  (L11 only)")
ax.plot(x, e3_auc, "s-",  color=C["e3"], label="E3  (L8–L11)")
ax.axhline(0.8437, color=C["e3"], linewidth=1.5, linestyle="--",
           label="L7 floor  (0.844)")

ax.set_xticks(x); ax.set_xticklabels(xlabs, rotation=40, ha="right")
ax.set_xlabel("top_p  (fraction of neurons suppressed per layer)")
ax.set_ylabel("Best MIA AUC")
ax.set_ylim(0.80, 1.02)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.legend(loc="lower left")

plt.tight_layout()
plt.savefig(f"{OUT}/P04_e3_floor.pdf", bbox_inches="tight")
plt.close()
print("✓ P04")


# ---
# P05 - E4: TPR@1%FPR sweep
# ---
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(x, e4_tpr, "^-", color=C["e4"], label="E4  TPR@1%FPR")
ax.axhline(BASE_TPR, color=C["baseline"], linewidth=1.5, linestyle=":",
           alpha=0.8, label=f"Baseline ({BASE_TPR:.3f})")
ax.axvline(sp, color=C["e4"], linewidth=1.5, linestyle="--", alpha=0.6)
ax.text(sp + 0.15, 0.135, f"top_p = 0.4\nTPR = {e4_tpr[sp]:.3f}",
        fontsize=12, color=C["e4"], fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(xlabs, rotation=40, ha="right")
ax.set_xlabel("top_p  (fraction of neurons suppressed globally)")
ax.set_ylabel("TPR at 1% FPR")
ax.set_ylim(-0.005, 0.175)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig(f"{OUT}/P05_e4_tpr.pdf", bbox_inches="tight")
plt.close()
print("✓ P05")


# ---
# P06 - E4: dual-axis sweet spot (AUC + F1)
# ---
fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()

ax1.plot(x, e4_auc, "^-", color=C["e4"], label="MIA AUC  (left)")
ax1.axhline(BASE_AUC, color=C["baseline"], linewidth=1.5, linestyle=":",
            alpha=0.6, label=f"Baseline AUC  ({BASE_AUC:.3f})")
ax2.plot(x, e4_f1, "^--", color="#7A4DA0", linewidth=2.0, markersize=7,
         alpha=0.75, label="Task F1  (right)")
ax2.axhline(BASE_F1, color="#7A4DA0", linewidth=1.5, linestyle="-.",
            alpha=0.5, label=f"Baseline F1  ({BASE_F1:.3f})")

ax1.axvline(sp, color=C["e4"], linewidth=1.5, linestyle="--", alpha=0.55)
ax1.text(sp + 0.12, 0.515, "top_p = 0.4", fontsize=12,
         color=C["e4"], fontweight="bold")

ax1.set_xticks(x); ax1.set_xticklabels(xlabs, rotation=40, ha="right")
ax1.set_xlabel("top_p")
ax1.set_ylabel("MIA AUC", color=C["e4"])
ax2.set_ylabel("Task weighted F1", color="#7A4DA0")
ax1.set_ylim(0.50, 1.02)
ax2.set_ylim(0.0, 1.3)
ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax1.spines['top'].set_visible(True)

lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labs1 + labs2, loc="lower left",
           ncol=2, fontsize=11)

plt.tight_layout()
plt.savefig(f"{OUT}/P06_e4_sweet_spot.pdf", bbox_inches="tight")
plt.close()
print("✓ P06")


# ---
# P07 - E4 vs E1 at top_p=0.4  (grouped bars, 3 metrics)
# ---
fig, ax = plt.subplots(figsize=(9, 5))

metrics  = ["MIA AUC", "TPR @ 1% FPR", "Task weighted F1"]
e4_vals  = [e4_auc[sp], e4_tpr[sp], e4_f1[sp]]
e1_vals  = [e1_auc[sp], e1_tpr[sp], e1_f1[sp]]

xi   = np.array([0, 1.8, 3.6])
bw   = 0.55

b4 = ax.bar(xi - bw/2, e4_vals, bw, color=C["e4"], label="E4  informed",  alpha=0.88)
b1 = ax.bar(xi + bw/2, e1_vals, bw, color=C["e1"], label="E1  random",    alpha=0.88)

ax.bar_label(b4, labels=[f"{v:.3f}" for v in e4_vals],
             padding=5, fontsize=12, fontweight="bold", color=C["e4"])
ax.bar_label(b1, labels=[f"{v:.3f}" for v in e1_vals],
             padding=5, fontsize=12, fontweight="bold", color=C["e1"])

ax.set_xticks(xi); ax.set_xticklabels(metrics, fontsize=13)
ax.set_ylim(0, 1.18)
ax.set_ylabel("Value")
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.legend()

ax.axvspan(1.8 - 0.6, 1.8 + 0.6, alpha=0.07, color=C["e4"])

plt.tight_layout()
plt.savefig(f"{OUT}/P07_e1_bars.pdf", bbox_inches="tight")
plt.close()
print("✓ P07")


# ---
# P08 - All conditions AUC sweep
# ---
fig, ax = plt.subplots(figsize=(10, 5.5))

ax.axhline(BASE_AUC, color=C["baseline"], linewidth=1.5, linestyle="--",
           alpha=0.7, label=f"Baseline ({BASE_AUC:.3f})")
ax.plot(x, e1_auc, "D-", color=C["e1"], label="E1  Random")
ax.plot(x, e2_auc, "o-", color=C["e2"], label="E2  Single-layer")
ax.plot(x, e3_auc, "s-", color=C["e3"], label="E3  Four-layer")
ax.plot(x, e4_auc, "^-", color=C["e4"], label="E4  Global  ★")
ax.axvline(sp, color=C["e4"], linewidth=1.2, linestyle=":", alpha=0.5)

ax.set_xticks(x); ax.set_xticklabels(xlabs, rotation=40, ha="right")
ax.set_xlabel("top_p  (fraction of neurons suppressed)")
ax.set_ylabel("MIA AUC")
ax.set_ylim(0.68, 1.02)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.legend(loc="lower left", ncol=2)

plt.tight_layout()
plt.savefig(f"{OUT}/P08_all_auc.pdf", bbox_inches="tight")
plt.close()
print("✓ P08")


# ---
# P09 - All conditions F1 sweep
# ---
fig, ax = plt.subplots(figsize=(10, 5.5))

ax.axhline(BASE_F1, color=C["baseline"], linewidth=1.5, linestyle="--",
           alpha=0.7, label=f"Baseline ({BASE_F1:.3f})")
ax.plot(x, e1_f1, "D-", color=C["e1"], label="E1  Random")
ax.plot(x, e2_f1, "o-", color=C["e2"], label="E2  Single-layer")
ax.plot(x, e3_f1, "s-", color=C["e3"], label="E3  Four-layer")
ax.plot(x, e4_f1, "^-", color=C["e4"], label="E4  Global  ★")
ax.axvline(sp, color=C["e4"], linewidth=1.2, linestyle=":", alpha=0.5)

ax.set_xticks(x); ax.set_xticklabels(xlabs, rotation=40, ha="right")
ax.set_xlabel("top_p  (fraction of neurons suppressed)")
ax.set_ylabel("Task weighted F1")
ax.set_ylim(0.0, 1.05)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.legend(loc="lower left", ncol=2)

plt.tight_layout()
plt.savefig(f"{OUT}/P09_all_f1.pdf", bbox_inches="tight")
plt.close()
print("✓ P09")


print(f"\n 9 figures saved to {OUT}/")
