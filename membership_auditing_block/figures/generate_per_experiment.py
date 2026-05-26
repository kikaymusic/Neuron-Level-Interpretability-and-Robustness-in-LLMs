"""
generate_per_experiment.py
Una figura por experimento. Cada figura tiene:
  - Heatmap (capas L0-L11 × top_p) mostrando AUC por capa
  - Gráfica complementaria específica para ese experimento

Nomenclatura actual (nueva):
  E1 = Random control
  E2 = Single-layer suppression (l*=L11)
  E3 = Four-layer suppression (L8–L11)
  E4 = Global informed suppression

Salida: figs_per_experiment/
  E1_random_control.pdf
  E2_single_layer.pdf
  E3_four_layers.pdf
  E4_global.pdf
"""

import json, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── Datos ──────────────────────────────────────────────────────────────────
with open('../results/sweep_results_malwspecsys.json') as f:
    sweep = json.load(f)

with open('../results/task_utility_malwspecsys.json') as f:
    task_j = json.load(f)

task_rows     = {r['top_p']: r for r in task_j['results']}
results       = sorted(sweep['results'], key=lambda r: r['top_p'])
TOP_P         = [r['top_p'] for r in results]
LAYERS        = list(range(12))
BASE_AUC      = sweep['baseline']['best_auc']
BASE_F1       = task_j['baseline_f1']
BASE_TPR      = next(r for r in results if r['top_p'] == 0.001)['e4_rnd_tpr_at_fpr1']

OUT = 'figs_per_experiment'
os.makedirs(OUT, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────
def parse_layer_auc(row, prefix):
    """Devuelve lista ordenada L0..L11 con el AUC del probe en esa capa."""
    d = json.loads(row[f'{prefix}_auc_per_layer'])
    return [d.get(str(l), np.nan) for l in LAYERS]

def build_matrix(prefix):
    """Matriz (top_p × capa) con AUC por capa para cada top_p."""
    return np.array([parse_layer_auc(r, prefix) for r in results])

def best_layer_per_tp(prefix):
    return [int(r[f'{prefix}_best_layer_def']) for r in results]

def serie(key):
    return [r[key] for r in results]

def f1_serie(key):
    return [task_rows[p][key] for p in TOP_P]

# Colores por experimento
COL = {'e1': '#2E7D32', 'e2': '#E65100', 'e3': '#6A1B9A', 'e4': '#616161'}

# Colormap del heatmap: rojo=AUC alto (vulnerable), verde=AUC bajo (defendido)
CMAP  = 'RdYlGn_r'
VMIN, VMAX = 0.45, 0.98

x_labs  = [str(p) for p in TOP_P]
l_labs  = [f'L{l}' for l in LAYERS]


def draw_heatmap(ax, matrix, prefix, title):
    """Dibuja el heatmap de AUC por capa con el best-layer marcado."""
    im = ax.imshow(matrix.T, aspect='auto', cmap=CMAP,
                   vmin=VMIN, vmax=VMAX, origin='lower')

    # Eje X: top_p
    ax.set_xticks(range(len(TOP_P)))
    ax.set_xticklabels(x_labs, rotation=40, ha='right', fontsize=9)
    ax.set_xlabel('top_p', fontsize=10)

    # Eje Y: capas
    ax.set_yticks(LAYERS)
    ax.set_yticklabels(l_labs, fontsize=9)
    ax.set_ylabel('Transformer layer', fontsize=10)

    ax.set_title(title, fontsize=11, fontweight='bold')

    # Marcar la mejor capa en cada top_p (punto blanco)
    for xi, bl in enumerate(best_layer_per_tp(prefix)):
        ax.plot(xi, bl, 'wo', markersize=5, markeredgecolor='k',
                markeredgewidth=0.6, zorder=3)

    # Colorbar
    sm = ScalarMappable(cmap=CMAP, norm=Normalize(vmin=VMIN, vmax=VMAX))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('MIA AUC', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    return im


# ══════════════════════════════════════════════════════════════════════════
# E2 — Single-layer suppression (l*=L11)
# Historia: L11 se suprime pero L10 absorbe la señal
# ══════════════════════════════════════════════════════════════════════════
mat_e1 = build_matrix('e1_p3')

fig, (ax_h, ax_r) = plt.subplots(1, 2, figsize=(13, 5))

draw_heatmap(ax_h, mat_e1, 'e1_p3',
             'E2 — Per-layer AUC\n(white dot = best layer for the attacker)')

# Panel derecho: L11 vs L10 como top_p sube
l11_auc = [parse_layer_auc(r, 'e1_p3')[11] for r in results]
l10_auc = [parse_layer_auc(r, 'e1_p3')[10] for r in results]
xi = np.arange(len(TOP_P))

ax_r.axhline(BASE_AUC, color='#1565C0', linestyle='--', linewidth=1.5,
             alpha=0.7, label=f'Baseline ({BASE_AUC:.3f})')
ax_r.plot(xi, l11_auc, 'o--', color=COL['e1'], linewidth=2.2, markersize=7,
          label='L11  (defended)')
ax_r.plot(xi, l10_auc, 's-',  color='#C62828', linewidth=2.2, markersize=7,
          label='L10  (undefended — absorbs signal)')
ax_r.set_xticks(xi)
ax_r.set_xticklabels(x_labs, rotation=40, ha='right', fontsize=9)
ax_r.set_xlabel('top_p', fontsize=10)
ax_r.set_ylabel('MIA AUC', fontsize=10)
ax_r.set_ylim(0.50, 1.02)
ax_r.set_title('E2 — Layer shift: L11 ↓  but  L10 compensates',
               fontsize=11, fontweight='bold')
ax_r.legend(fontsize=10)
ax_r.grid(True, alpha=0.3)
ax_r.spines['top'].set_visible(False)
ax_r.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{OUT}/E2_single_layer.pdf', bbox_inches='tight')
plt.close()
print('Saved: E2_single_layer.pdf')


# ══════════════════════════════════════════════════════════════════════════
# E3 — Four-layer suppression
# Historia: el suelo del atacante se desplaza a L7
# ══════════════════════════════════════════════════════════════════════════
mat_e2 = build_matrix('e2_p4a')

fig, (ax_h, ax_r) = plt.subplots(1, 2, figsize=(13, 5))

im = draw_heatmap(ax_h, mat_e2, 'e2_p4a',
                  'E3 — Per-layer AUC\n(L8–L11 defended; L7 becomes the floor)')

# Marcar visualmente las capas defendidas con líneas en el eje Y
for defended_l in [8, 9, 10, 11]:
    ax_h.axhline(defended_l - 0.5, color='white', linewidth=0.8, alpha=0.5)
    ax_h.axhline(defended_l + 0.5, color='white', linewidth=0.8, alpha=0.5)
ax_h.axhspan(7.5, 11.5, color='white', alpha=0.08, zorder=0)

# Panel derecho: E2 vs E3 best AUC + floor
e1_best = serie('e1_p3_best_auc_def')
e2_best = serie('e2_p4a_best_auc_def')
xi = np.arange(len(TOP_P))

ax_r.axhline(BASE_AUC, color='#1565C0', linestyle='--', linewidth=1.5,
             alpha=0.7, label=f'Baseline ({BASE_AUC:.3f})')
ax_r.axhline(0.8437, color=COL['e2'], linestyle=':', linewidth=2.0,
             alpha=0.8, label='L7 floor  (0.844)')
ax_r.plot(xi, e1_best, 'o-',  color=COL['e1'], linewidth=2.0, markersize=7,
          label='E2  (L11 only)')
ax_r.plot(xi, e2_best, 's-',  color=COL['e2'], linewidth=2.2, markersize=7,
          label='E3  (L8–L11)')
ax_r.set_xticks(xi)
ax_r.set_xticklabels(x_labs, rotation=40, ha='right', fontsize=9)
ax_r.set_xlabel('top_p', fontsize=10)
ax_r.set_ylabel('Best MIA AUC', fontsize=10)
ax_r.set_ylim(0.80, 1.02)
ax_r.set_title('E3 vs E2 — Attack retreats to L7\n(first undefended layer)',
               fontsize=11, fontweight='bold')
ax_r.legend(fontsize=10, loc='lower left')
ax_r.grid(True, alpha=0.3)
ax_r.spines['top'].set_visible(False)
ax_r.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{OUT}/E3_four_layers.pdf', bbox_inches='tight')
plt.close()
print('Saved: E3_four_layers.pdf')


# ══════════════════════════════════════════════════════════════════════════
# E4 — Global suppression
# Historia: cascade collapse + TPR@1%FPR → 0 al sweet spot
# ══════════════════════════════════════════════════════════════════════════
mat_e3 = build_matrix('e3_p4b')
e3_tpr = serie('e3_p4b_tpr_at_fpr1')
e3_f1  = f1_serie('e3_p4b_weighted_f1')

# Extender rango del colormap para capturar valores por debajo de azar (L11 invierte)
VMIN_E3 = 0.20

fig, (ax_h, ax_r) = plt.subplots(1, 2, figsize=(13, 5))

# Heatmap con rango extendido
im = ax_h.imshow(mat_e3.T, aspect='auto', cmap=CMAP,
                 vmin=VMIN_E3, vmax=VMAX, origin='lower')
ax_h.set_xticks(range(len(TOP_P)))
ax_h.set_xticklabels(x_labs, rotation=40, ha='right', fontsize=9)
ax_h.set_xlabel('top_p', fontsize=10)
ax_h.set_yticks(LAYERS)
ax_h.set_yticklabels(l_labs, fontsize=9)
ax_h.set_ylabel('Transformer layer', fontsize=10)
ax_h.set_title('E4 — Per-layer AUC\n(cascade collapse from L11 downward)',
               fontsize=11, fontweight='bold')
for xi_i, bl in enumerate(best_layer_per_tp('e3_p4b')):
    ax_h.plot(xi_i, bl, 'wo', markersize=5, markeredgecolor='k',
              markeredgewidth=0.6, zorder=3)
sm = ScalarMappable(cmap=CMAP, norm=Normalize(vmin=VMIN_E3, vmax=VMAX))
sm.set_array([])
cb = plt.colorbar(sm, ax=ax_h, fraction=0.046, pad=0.04)
cb.set_label('MIA AUC', fontsize=9)
cb.ax.tick_params(labelsize=8)

# Panel derecho: TPR@1%FPR — el resultado operacional
xi = np.arange(len(TOP_P))
sp = TOP_P.index(0.4)

ax_r2 = ax_r.twinx()
ax_r.plot(xi, e3_tpr, '^-', color=COL['e3'], linewidth=2.5, markersize=8,
          label='TPR@1%FPR  (left)')
ax_r.axhline(BASE_TPR, color='gray', linestyle='--', linewidth=1.5,
             alpha=0.7, label=f'Baseline TPR ({BASE_TPR:.3f})')
ax_r2.plot(xi, e3_f1, '^--', color='#9C27B0', linewidth=2.0, markersize=7,
           alpha=0.6, label='Task F1  (right)')
ax_r2.axhline(BASE_F1, color='#9C27B0', linestyle=':', linewidth=1.5,
              alpha=0.5, label=f'Baseline F1 ({BASE_F1:.3f})')

# Sweet spot
ax_r.axvline(sp, color=COL['e3'], linewidth=1.5, linestyle=':', alpha=0.7)
ax_r.text(sp + 0.12, max(e3_tpr) * 0.75,
          f'top_p=0.4\nTPR={e3_tpr[sp]:.3f}\nF1={e3_f1[sp]:.3f}',
          fontsize=9, color=COL['e3'], fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=COL['e3'], alpha=0.85))

ax_r.set_xticks(xi)
ax_r.set_xticklabels(x_labs, rotation=40, ha='right', fontsize=9)
ax_r.set_xlabel('top_p', fontsize=10)
ax_r.set_ylabel('TPR at 1% FPR', color=COL['e3'], fontsize=10)
ax_r2.set_ylabel('Task weighted F1', color='#9C27B0', fontsize=10)
ax_r.set_ylim(-0.005, 0.18)
ax_r2.set_ylim(0.0, 1.3)
ax_r2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax_r.set_title('E4 — Operational privacy risk\nvs task utility', fontsize=11, fontweight='bold')
ax_r.grid(True, alpha=0.3)
ax_r.spines['top'].set_visible(False)

lines1, labs1 = ax_r.get_legend_handles_labels()
lines2, labs2 = ax_r2.get_legend_handles_labels()
ax_r.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig(f'{OUT}/E4_global.pdf', bbox_inches='tight')
plt.close()
print('Saved: E4_global.pdf')


# ══════════════════════════════════════════════════════════════════════════
# E1 — Random control
# Historia: mismo k que E4 pero al azar — la selección importa
# ══════════════════════════════════════════════════════════════════════════
mat_e4 = build_matrix('e4_rnd')
e3_best = serie('e3_p4b_best_auc_def')
e4_best = serie('e4_rnd_best_auc_def')
e3_tpr  = serie('e3_p4b_tpr_at_fpr1')
e4_tpr  = serie('e4_rnd_tpr_at_fpr1')
e3_f1   = f1_serie('e3_p4b_weighted_f1')
e4_f1   = f1_serie('e4_rnd_weighted_f1')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Heatmap E3
ax = axes[0]
ax.imshow(mat_e3.T, aspect='auto', cmap=CMAP,
          vmin=VMIN_E3, vmax=VMAX, origin='lower')
ax.set_xticks(range(len(TOP_P)))
ax.set_xticklabels(x_labs, rotation=40, ha='right', fontsize=8)
ax.set_yticks(LAYERS); ax.set_yticklabels(l_labs, fontsize=8)
ax.set_xlabel('top_p', fontsize=9)
ax.set_ylabel('Layer', fontsize=9)
ax.set_title('E4  (informed)\nstructured cascade', fontsize=11, fontweight='bold')
for xi_i, bl in enumerate(best_layer_per_tp('e3_p4b')):
    ax.plot(xi_i, bl, 'wo', markersize=5, markeredgecolor='k', markeredgewidth=0.6, zorder=3)

# Heatmap E4
ax = axes[1]
im = ax.imshow(mat_e4.T, aspect='auto', cmap=CMAP,
               vmin=VMIN_E3, vmax=VMAX, origin='lower')
ax.set_xticks(range(len(TOP_P)))
ax.set_xticklabels(x_labs, rotation=40, ha='right', fontsize=8)
ax.set_yticks(LAYERS); ax.set_yticklabels(l_labs, fontsize=8)
ax.set_xlabel('top_p', fontsize=9)
ax.set_title('E1  (random)\ndiffuse degradation', fontsize=11, fontweight='bold')
for xi_i, bl in enumerate(best_layer_per_tp('e4_rnd')):
    ax.plot(xi_i, bl, 'wo', markersize=5, markeredgecolor='k', markeredgewidth=0.6, zorder=3)
sm = ScalarMappable(cmap=CMAP, norm=Normalize(vmin=VMIN_E3, vmax=VMAX))
sm.set_array([])
cb = plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
cb.set_label('MIA AUC', fontsize=8); cb.ax.tick_params(labelsize=8)

# Panel derecho: TPR@1%FPR E3 vs E4
ax = axes[2]
xi = np.arange(len(TOP_P))
sp = TOP_P.index(0.4)
ax.plot(xi, e3_tpr, '^-', color=COL['e3'], linewidth=2.2, markersize=7, label='E4  informed')
ax.plot(xi, e4_tpr, 'D-', color=COL['e4'], linewidth=2.0, markersize=7, label='E1  random')
ax.axhline(BASE_TPR, color='gray', linestyle='--', linewidth=1.5,
           alpha=0.7, label=f'Baseline ({BASE_TPR:.3f})')
ax.axvline(sp, color='gray', linewidth=1.0, linestyle=':', alpha=0.5)

# Anotación del gap
ax.annotate('', xy=(sp + 0.15, e4_tpr[sp]), xytext=(sp + 0.15, e3_tpr[sp]),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax.text(sp + 0.3, (e3_tpr[sp] + e4_tpr[sp]) / 2,
        '×13',
        fontsize=12, fontweight='bold', va='center')

ax.set_xticks(xi)
ax.set_xticklabels(x_labs, rotation=40, ha='right', fontsize=9)
ax.set_xlabel('top_p', fontsize=10)
ax.set_ylabel('TPR at 1% FPR', fontsize=10)
ax.set_ylim(-0.005, 0.25)
ax.set_title('E4 vs E1 — TPR@1%FPR\n(same k, different selection)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{OUT}/E1_random_control.pdf', bbox_inches='tight')
plt.close()
print('Saved: E1_random_control.pdf')

print(f'\nDone. 4 figures in {OUT}/')
