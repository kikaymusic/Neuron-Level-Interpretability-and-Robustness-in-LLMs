"""
generate_subsection_figs.py
Una figura AUC + F1 dual-axis por experimento + figura comparativa.
Nomenclatura:
  E1 = random selection (naive baseline)
  E2 = single-layer suppression (l*, Ibanez/Lumia)
  E3 = multi-layer suppression
  E4 = global neuron suppression (main contribution)
Output: figs_sections/
"""

import json, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = "figs_sections"
os.makedirs(OUT, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
with open("../results/sweep_results_malwspecsys.json") as f:
    mem = json.load(f)
with open("../results/task_utility_malwspecsys.json") as f:
    task = json.load(f)

results    = mem["results"]
base_auc   = mem["baseline"]["best_auc"]
base_f1    = task["baseline_f1"]
task_by_p  = {r["top_p"]: r for r in task["results"]}
top_p_vals = [r["top_p"] for r in results]

# keys in the JSON use old naming; map to new experiment labels
#   old e4_rnd  -> new E1 (random)
#   old e1_p3   -> new E2 (single-layer / l*)
#   old e2_p4a  -> new E3 (multi-layer)
#   old e3_p4b  -> new E4 (global)
def series(auc_key, f1_key):
    return (
        [r[auc_key] for r in results],
        [task_by_p[p][f1_key] for p in top_p_vals],
    )

e1_auc, e1_f1 = series("e4_rnd_best_auc_def",  "e4_rnd_weighted_f1")   # random
e2_auc, e2_f1 = series("e1_p3_best_auc_def",   "e1_p3_weighted_f1")    # single-layer
e3_auc, e3_f1 = series("e2_p4a_best_auc_def",  "e2_p4a_weighted_f1")   # multi-layer
e4_auc, e4_f1 = series("e3_p4b_best_auc_def",  "e3_p4b_weighted_f1")   # global

x        = np.arange(len(top_p_vals))
x_labels = [f"{p*100:.1f}%" for p in top_p_vals]
sp_idx   = top_p_vals.index(0.4)

COLORS = {
    "baseline": "#555555",
    "e1":       "#E07B39",   # random       -> warm orange
    "e2":       "#2F8EC3",   # single-layer -> blue
    "e3":       "#4BAE8A",   # multi-layer  -> green
    "e4":       "#9B6BC5",   # global       -> purple
}


# ── Helper: dual-axis AUC + F1 ─────────────────────────────────────────────
def draw_dual(auc_vals, f1_vals, color, title, fname, sp_label, sp_offset=0.9):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    l_auc, = ax1.plot(x, auc_vals, "o-", color=color, linewidth=2.2,
                      markersize=6, label="MIA AUC (left)")
    ax1.axhline(base_auc, color=COLORS["baseline"], linestyle="--",
                linewidth=1.2, alpha=0.7)
    ax1.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)

    l_f1, = ax2.plot(x, f1_vals, "s--", color=color, linewidth=2.0,
                     markersize=6, alpha=0.65, label="Task F1 (right)")
    ax2.axhline(base_f1, color=COLORS["baseline"], linestyle="-.",
                linewidth=1.0, alpha=0.55)

    ax1.axvspan(sp_idx - 0.35, sp_idx + 0.35, alpha=0.10, color=color)
    ax1.annotate(sp_label,
                 xy=(sp_idx, auc_vals[sp_idx]),
                 xytext=(sp_idx + sp_offset, auc_vals[sp_idx] + 0.04),
                 arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                 fontsize=11, color=color)

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=13)
    ax1.set_xlabel("Fraction of neurons suppressed (p)", fontsize=15)
    ax1.set_ylabel("MIA AUC", fontsize=15)
    ax2.set_ylabel("Task weighted F1", fontsize=15)
    ax1.tick_params(axis="y", labelsize=13)
    ax2.tick_params(axis="y", labelsize=13)
    ax1.set_ylim(0.48, 1.02)
    ax2.set_ylim(0.0, 1.25)
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    legend_handles = [
        l_auc, l_f1,
        plt.Line2D([0], [0], color=COLORS["baseline"], linestyle="--",
                   linewidth=1.2, label=f"Baseline AUC ({base_auc:.3f})"),
        plt.Line2D([0], [0], color=COLORS["baseline"], linestyle="-.",
                   linewidth=1.0, label=f"Baseline F1 ({base_f1:.3f})"),
    ]
    ax1.legend(handles=legend_handles,
               labels=["MIA AUC", "Task F1",
                       f"Baseline AUC ({base_auc:.3f})",
                       f"Baseline F1 ({base_f1:.3f})"],
               fontsize=13, loc="lower left", ncol=2, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{OUT}/{fname}", dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(f"{OUT}/{fname.replace('.png', '.pdf')}", bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════
draw_dual(e1_auc, e1_f1, COLORS["e1"],
          "E1: Random neuron selection - MIA AUC and task F1 vs suppression fraction",
          "E1_random_auc_f1.png",
          sp_label="p=0.4\n(reference)", sp_offset=0.9)

draw_dual(e2_auc, e2_f1, COLORS["e2"],
          "E2: Single-layer suppression (l*) - MIA AUC and task F1 vs suppression fraction",
          "E2_single_layer_auc_f1.png",
          sp_label="p=0.4\nL11 suppressed", sp_offset=0.9)

draw_dual(e3_auc, e3_f1, COLORS["e3"],
          "E3: Multi-layer suppression - MIA AUC and task F1 vs suppression fraction",
          "E3_multi_layer_auc_f1.png",
          sp_label="p=0.4\nAUC floor (0.844)", sp_offset=0.9)

draw_dual(e4_auc, e4_f1, COLORS["e4"],
          "E4: Global neuron suppression - MIA AUC and task F1 vs suppression fraction",
          "E4_global_auc_f1.png",
          sp_label="sweet spot\np=0.4\nTPR=0.8%", sp_offset=0.9)

# ══════════════════════════════════════════════════════════════════════════
# Comparative figure - 2 panels
# ══════════════════════════════════════════════════════════════════════════
fig, (ax_auc, ax_f1) = plt.subplots(1, 2, figsize=(14, 5))

exp_data = [
    ("E1: Random",              e1_auc, e1_f1, COLORS["e1"], "D--", 1.6),
    ("E2: Single-layer (l*)",   e2_auc, e2_f1, COLORS["e2"], "o-",  2.0),
    ("E3: Multi-layer",         e3_auc, e3_f1, COLORS["e3"], "s-",  2.0),
    ("E4: Global (ours)",       e4_auc, e4_f1, COLORS["e4"], "^-",  2.2),
]

for label, auc_v, f1_v, col, sty, lw in exp_data:
    ax_auc.plot(x, auc_v, sty, color=col, linewidth=lw, markersize=6, label=label)
    ax_f1.plot(x,  f1_v,  sty, color=col, linewidth=lw, markersize=6, label=label)

ax_auc.axhline(base_auc, color=COLORS["baseline"], linestyle="--",
               linewidth=1.2, label=f"Baseline ({base_auc:.3f})")
ax_auc.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
ax_auc.axvspan(sp_idx - 0.35, sp_idx + 0.35, alpha=0.08, color="gray")

ax_f1.axhline(base_f1, color=COLORS["baseline"], linestyle="--",
              linewidth=1.2, label=f"Baseline ({base_f1:.3f})")
ax_f1.axvspan(sp_idx - 0.35, sp_idx + 0.35, alpha=0.08, color="gray",
              label="p=0.4 (operating point)")

for ax in (ax_auc, ax_f1):
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_xlabel("Fraction of neurons suppressed (p)", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower left")

ax_auc.set_ylabel("MIA AUC", fontsize=10)
ax_auc.set_ylim(0.48, 1.02)
ax_f1.set_ylabel("Task weighted F1", fontsize=10)
ax_f1.set_ylim(0.0, 1.05)
plt.tight_layout()
plt.savefig(f"{OUT}/comparative_auc_f1.png", dpi=150, bbox_inches="tight", pad_inches=0.05)
plt.savefig(f"{OUT}/comparative_auc_f1.pdf", bbox_inches="tight", pad_inches=0.05)
plt.close()
print("Saved: comparative_auc_f1.png")

# ══════════════════════════════════════════════════════════════════════════
# Baseline: MIA AUC by transformer layer
# ══════════════════════════════════════════════════════════════════════════
# Values from sweep at p=0.001 (1 neuron suppressed = essentially undefended)
with open("sweep_20260428_205017/sweep_results_20260428_205017.json") as _f:
    _sweep = json.load(_f)
_r0 = _sweep["results"][0]
_apl = json.loads(_r0["e1_p3_auc_per_layer"])
layers   = list(range(12))
auc_base = [_apl[str(l)] for l in layers]
lstar    = int(max(range(12), key=lambda l: auc_base[l]))  # computed from data

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(layers, auc_base, "o-", color="#C0392B", linewidth=2.2, markersize=7,
        zorder=3)
ax.plot(lstar, auc_base[lstar], "o", color="#C0392B", markersize=14, zorder=4)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0,
           label="Random chance (0.5)")

ax.annotate(f"L{lstar} = {auc_base[lstar]:.3f}",
            xy=(lstar, auc_base[lstar]),
            xytext=(lstar - 2.2, auc_base[lstar] + 0.015),
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.3),
            fontsize=11, color="#C0392B", fontweight="bold")

ax.set_xticks(layers)
ax.set_xticklabels([f"L{l}" for l in layers], fontsize=10)
ax.set_xlabel("Transformer layer", fontsize=11)
ax.set_ylabel("MIA AUC", fontsize=11)
ax.set_ylim(0.48, 1.02)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.4)
ax.legend(fontsize=10, loc="upper left")

# ax.set_title("Membership signal by layer (baseline, no defence)", fontsize=12, pad=6)
plt.tight_layout()
plt.savefig(f"{OUT}/baseline_auc_by_layer.png", dpi=150,
            bbox_inches="tight", pad_inches=0.05)
plt.savefig(f"{OUT}/baseline_auc_by_layer.pdf",
            bbox_inches="tight", pad_inches=0.05)
plt.close()
print("Saved: baseline_auc_by_layer.png / .pdf")

print(f"\nDone. Figures in {OUT}/")
for f in sorted(os.listdir(OUT)):
    print(" ", f)
