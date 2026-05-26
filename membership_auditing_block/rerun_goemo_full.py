"""
rerun_goemo_full.py
Full re-run of all four GoEmotions MIA experiments (E1-E4) with l*=L9,
using a consistent local protocol (train/test split on N=100).

Why l*=L9:
  The original sweep targeted L10 but the sweep's own per-layer evaluation
  shows L9 as the dominant layer post-suppression. L0-L9 lie before L10 in
  the encoder stack and are unaffected when L10 is suppressed. Re-running
  with l*=L9 (the layer that actually drove the sweep's best_auc_def) gives
  a coherent single-layer strategy.

Output:
  rerun_goemo_full_<timestamp>.json  — per-experiment results for all p
  rerun_goemo_full_<timestamp>.csv   — same, CSV format
"""

import csv, json, os
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

BASE   = "/Users/kikay/Documents/investigacion/Neuron-Level-Interpretability-and-Robustness-in-LLMs/data/goemotions"
BACKUP = "/Users/kikay/Desktop/backup_synapse"
TS     = datetime.now().strftime("%Y%m%d_%H%M%S")

# Set USE_FULL=True once extract_goemo_activations.py has been run
USE_FULL = True
ACTS_FILE = f"{BASE}/activations_full.json" if USE_FULL else f"{BASE}/activations.json"
MEM_FILE  = f"{BASE}/membership_full.pt"    if USE_FULL else f"{BASE}/membership.pt"

TOP_P_VALS   = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.65, 0.8]
K_TOP_LAYERS = 4
L, H         = 12, 768
TOTAL        = L * H
SEED         = 42

# ── 1. Load activations ────────────────────────────────────────────────────────
print("Loading activations...")
raw = []
with open(ACTS_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            raw.append(json.loads(line))

A = np.zeros((len(raw), L, H), dtype=np.float32)
for i, rec in enumerate(raw):
    feat = rec["features"][0]   # CLS token
    for lyr in feat["layers"]:
        A[i, lyr["index"], :] = lyr["values"]
N = len(raw)
print(f"Activations: {A.shape}")

# ── 2. Load membership labels ──────────────────────────────────────────────────
y_mem = np.array(torch.load(MEM_FILE, weights_only=False).tolist())
print(f"Members: {y_mem.sum()}  Non-members: {(1-y_mem).sum()}")

# ── 3. Train/test split ────────────────────────────────────────────────────────
tr_idx, te_idx = train_test_split(
    np.arange(N), test_size=0.2, random_state=SEED, stratify=y_mem
)
print(f"Train: {len(tr_idx)} (members={y_mem[tr_idx].sum()})  "
      f"Test: {len(te_idx)} (members={y_mem[te_idx].sum()})")

# ── 4. Compute baseline per-layer AUC from training data ──────────────────────
print("\nComputing per-layer baseline AUC on test set...")
base_auc_per_layer = {}
for lid in range(L):
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED,
                             class_weight="balanced")
    clf.fit(A[tr_idx, lid, :], y_mem[tr_idx])
    probs = clf.predict_proba(A[te_idx, lid, :])[:, 1]
    base_auc_per_layer[lid] = float(roc_auc_score(y_mem[te_idx], probs))

sorted_layers = sorted(base_auc_per_layer.items(), key=lambda x: x[1], reverse=True)
print("Per-layer baseline AUC (from training data):")
for lid, auc in sorted_layers:
    print(f"  L{lid:2d}: {auc:.4f}")

# l* = L11 (most vulnerable layer with N=2770)
L_STAR      = 11
TOP4_LAYERS = [lid for lid, _ in sorted_layers[:K_TOP_LAYERS]]
print(f"\nl* (override) = L{L_STAR}")
print(f"TOP4 (from data) = {['L'+str(l) for l in TOP4_LAYERS]}")
baseline_best_auc = max(base_auc_per_layer.values())
print(f"Baseline best AUC (test) = {baseline_best_auc:.4f}")

# ── 5. Neuron rankings from TRAIN activations ──────────────────────────────────
print("\nComputing neuron rankings...")

# Per-layer ranking for E2 and E3
layer_rank = {}
for lid in range(L):
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED,
                             class_weight="balanced")
    clf.fit(A[tr_idx, lid, :], y_mem[tr_idx])
    importance = np.abs(clf.coef_[0])
    layer_rank[lid] = np.argsort(importance)[::-1]

# Global ranking for E4
X_flat = A[tr_idx].reshape(len(tr_idx), -1)
clf_global = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED,
                                class_weight="balanced")
clf_global.fit(X_flat, y_mem[tr_idx])
global_importance = np.abs(clf_global.coef_[0])
global_rank = np.argsort(global_importance)[::-1]
print("Rankings ready.")

# ── Pre-train per-layer probes on BASELINE train activations (transferability) ─
# Probes are trained once on undefended activations. At evaluation they receive
# defended test activations — the attacker uses a stale probe, as in MalwSpecSys.
print("Pre-training baseline per-layer probes for transferability evaluation...")
baseline_probes = {}
for lid in range(L):
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED,
                             class_weight="balanced")
    clf.fit(A[tr_idx, lid, :], y_mem[tr_idx])
    baseline_probes[lid] = clf
print("Baseline probes ready.")

def tpr_at_fpr1(scores, y_true):
    n_neg = int((y_true == 0).sum())
    n_pos = int((y_true == 1).sum())
    if n_neg == 0 or n_pos == 0:
        return 0.0
    best_tpr = 0.0
    for thr in np.sort(scores)[::-1]:
        fp  = int(((scores >= thr) & (y_true == 0)).sum())
        tp  = int(((scores >= thr) & (y_true == 1)).sum())
        if fp / n_neg <= 0.01:
            best_tpr = tp / n_pos
    return best_tpr

# ── Helper: evaluate DEFENDED TEST activations with BASELINE-trained probes ───
def eval_defended(A_def):
    """Transferability protocol: baseline probes evaluated on defended test set."""
    A_def_te = A_def[te_idx]   # only test partition is defended for evaluation
    auc_pl = {}
    for lid in range(L):
        probs = baseline_probes[lid].predict_proba(A_def_te[:, lid, :])[:, 1]
        auc_pl[lid] = float(roc_auc_score(y_mem[te_idx], probs))

    best_layer = max(auc_pl, key=auc_pl.get)
    best_auc   = auc_pl[best_layer]
    scores     = baseline_probes[best_layer].predict_proba(
                     A_def_te[:, best_layer, :])[:, 1]
    tpr = tpr_at_fpr1(scores, y_mem[te_idx])
    return best_auc, best_layer, tpr, auc_pl

# ── 6. Sweep ───────────────────────────────────────────────────────────────────
print(f"\n{'p':>6}  {'k_g':>5}  "
      f"{'E1_auc':>7}  {'E2_auc':>7}  {'E3_auc':>7}  {'E4_auc':>7}  "
      f"{'E2_tpr':>7}  {'E4_tpr':>7}")
print("-" * 75)

rng     = np.random.default_rng(SEED)
results = []

for top_p in TOP_P_VALS:
    k_global    = max(1, int(top_p * TOTAL))
    k_per_layer = max(1, int(top_p * H))
    row = {"top_p": top_p, "k_global": k_global, "k_per_layer": k_per_layer}

    # E1: Random (k_global neurons)
    rnd_idx = rng.choice(TOTAL, size=k_global, replace=False)
    A_e1 = A.copy()
    for g in rnd_idx:
        A_e1[:, g // H, g % H] = 0.0
    e1_auc, e1_bl, e1_tpr, e1_pl = eval_defended(A_e1)
    row.update({"e1_best_auc": e1_auc, "e1_best_layer": e1_bl,
                "e1_tpr_fpr1": e1_tpr, "e1_auc_per_layer": e1_pl})

    # E2: Single-layer at l*=L9
    top_l9 = layer_rank[L_STAR][:k_per_layer].tolist()
    A_e2 = A.copy()
    A_e2[:, L_STAR, top_l9] = 0.0
    e2_auc, e2_bl, e2_tpr, e2_pl = eval_defended(A_e2)
    row.update({"e2_best_auc": e2_auc, "e2_best_layer": e2_bl,
                "e2_tpr_fpr1": e2_tpr, "e2_auc_per_layer": e2_pl})

    # E3: Multi-layer (top-4 by baseline AUC)
    A_e3 = A.copy()
    for lid in TOP4_LAYERS:
        top_lid = layer_rank[lid][:k_per_layer].tolist()
        A_e3[:, lid, top_lid] = 0.0
    e3_auc, e3_bl, e3_tpr, e3_pl = eval_defended(A_e3)
    row.update({"e3_best_auc": e3_auc, "e3_best_layer": e3_bl,
                "e3_tpr_fpr1": e3_tpr, "e3_auc_per_layer": e3_pl})

    # E4: Global suppression
    A_e4 = A.copy()
    for g in global_rank[:k_global]:
        A_e4[:, g // H, g % H] = 0.0
    e4_auc, e4_bl, e4_tpr, e4_pl = eval_defended(A_e4)
    row.update({"e4_best_auc": e4_auc, "e4_best_layer": e4_bl,
                "e4_tpr_fpr1": e4_tpr, "e4_auc_per_layer": e4_pl})

    results.append(row)
    print(f"{top_p:>6}  {k_global:>5}  "
          f"{e1_auc:>7.4f}  {e2_auc:>7.4f}  {e3_auc:>7.4f}  {e4_auc:>7.4f}  "
          f"{e2_tpr:>7.4f}  {e4_tpr:>7.4f}")

# ── 7. Save ────────────────────────────────────────────────────────────────────
out = {
    "run_ts": TS,
    "l_star": L_STAR,
    "top4_layers": TOP4_LAYERS,
    "baseline_best_auc": baseline_best_auc,
    "base_auc_per_layer": base_auc_per_layer,
    "n_train": len(tr_idx), "n_test": len(te_idx),
    "results": results,
}

json_path = f"{BACKUP}/rerun_goemo_full_{TS}.json"
with open(json_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved JSON: {json_path}")

# CSV
csv_path = f"{BACKUP}/rerun_goemo_full_{TS}.csv"
fieldnames = ["top_p", "k_global", "k_per_layer",
              "e1_best_auc", "e1_best_layer", "e1_tpr_fpr1",
              "e2_best_auc", "e2_best_layer", "e2_tpr_fpr1",
              "e3_best_auc", "e3_best_layer", "e3_tpr_fpr1",
              "e4_best_auc", "e4_best_layer", "e4_tpr_fpr1"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    w.writerows(results)
print(f"Saved CSV:  {csv_path}")

# Summary table
print(f"\n{'p':>6}  {'E1 AUC':>7}  {'E2 AUC':>7}  {'E3 AUC':>7}  {'E4 AUC':>7}  "
      f"{'E2 TPR':>7}  {'E4 TPR':>7}")
print(f"{'------':>6}  {'-------':>7}  {'-------':>7}  {'-------':>7}  {'-------':>7}  "
      f"{'-------':>7}  {'-------':>7}")
for r in results:
    print(f"{r['top_p']:>6}  {r['e1_best_auc']:>7.4f}  {r['e2_best_auc']:>7.4f}  "
          f"{r['e3_best_auc']:>7.4f}  {r['e4_best_auc']:>7.4f}  "
          f"{r['e2_tpr_fpr1']:>7.4f}  {r['e4_tpr_fpr1']:>7.4f}")
print(f"\nBaseline best AUC = {baseline_best_auc:.4f}  |  l* = L{L_STAR}  |  "
      f"TOP4 = {['L'+str(l) for l in TOP4_LAYERS]}")
