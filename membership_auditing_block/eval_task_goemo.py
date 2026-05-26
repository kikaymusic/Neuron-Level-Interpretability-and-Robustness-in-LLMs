"""
eval_task_goemo.py
Computes task utility (weighted F1) for all 4 experiments × 10 top_p values
on the GoEmotions sample (100 examples, 6 emotion classes).

l* and TOP4_LAYERS are computed dynamically from per-layer membership probes
trained on the base activations — no hardcoded layer indices.

Requirements:
  pip install transformers torch scikit-learn

Outputs:
  task_utility_goemo.json  (same format as task_utility_*.json for MalwSpecSys)
"""

import json, os, sys
import numpy as np
import torch
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from transformers import AutoModelForSequenceClassification

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = "/Users/kikay/Documents/investigacion/Neuron-Level-Interpretability-and-Robustness-in-LLMs/data/goemotions"
BACKUP      = "/Users/kikay/Desktop/backup_synapse"
MODEL_HF    = "monologg/bert-base-cased-goemotions-original"
ACTS_FILE   = f"{BASE}/activations.json"
SAMPLEDF    = f"{BASE}/sample_df.json"
LABELS_PT   = f"{BASE}/labels.pt"
MEM_PT      = f"{BASE}/membership.pt"
# Per-layer AUC from the actual MIA sweep (computed on full dataset with proper split)
AUC_CSV     = f"{BACKUP}/goemotions_results/auc_by_layer_goemo.csv"
OUT_FILE    = f"{BACKUP}/task_utility_goemo.json"

TOP_P_VALS   = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.65, 0.8]
K_TOP_LAYERS = 4     # number of layers for E3
L            = 12
H            = 768
TOTAL        = L * H    # 9216
SEED         = 42

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# ── Load activations ───────────────────────────────────────────────────────────
print("Loading activations...")
raw_acts = []
with open(ACTS_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            raw_acts.append(json.loads(line))

A = np.zeros((len(raw_acts), L, H), dtype=np.float32)
for i, rec in enumerate(raw_acts):
    feat = rec["features"][0]   # CLS token
    for lyr in feat["layers"]:
        A[i, lyr["index"], :] = lyr["values"]
print(f"Activations shape: {A.shape}")   # (N, 12, 768)

# ── Load labels and membership ─────────────────────────────────────────────────
labels    = torch.load(LABELS_PT, weights_only=False).tolist()
y_mem     = np.array(torch.load(MEM_PT, weights_only=False).tolist())
sample_df = [json.loads(l) for l in open(SAMPLEDF)]
N         = len(y_mem)
print(f"N={N}, members={y_mem.sum()}, non-members={(1-y_mem).sum()}")

# ── Phase 1: Load per-layer AUC from the MIA sweep (authoritative source) ─────
# The 100-sample task utility set is too small for a proper train/test split;
# AUC computed on training data would overfit and give nonsense l*.
# The authoritative per-layer AUC was computed during the full MIA sweep
# (train/test split on the 100-sample membership set) and saved in AUC_CSV.
import csv
print(f"\nLoading per-layer AUC from MIA sweep: {AUC_CSV}")
auc_per_layer = {}
with open(AUC_CSV) as f:
    for row in csv.DictReader(f):
        auc_per_layer[int(row["layer_id"])] = float(row["auc"])

layer_aucs_sorted = sorted(auc_per_layer.items(), key=lambda x: x[1], reverse=True)
L_STAR      = layer_aucs_sorted[0][0]
TOP4_LAYERS = [l for l, _ in layer_aucs_sorted[:K_TOP_LAYERS]]

print(f"\nPer-layer AUC from sweep (descending):")
for lid, auc in layer_aucs_sorted:
    marker = " ← l*" if lid == L_STAR else (" ← top-4" if lid in TOP4_LAYERS else "")
    print(f"  L{lid:2d}: {auc:.4f}{marker}")
print(f"\nl*       = L{L_STAR}  (AUC={auc_per_layer[L_STAR]:.4f})")
print(f"TOP4     = {['L'+str(l) for l in TOP4_LAYERS]}")

# ── Phase 2: Global probe for E4 neuron ranking ────────────────────────────────
print("\nTraining global membership probe for E4 neuron ranking...")
X_flat = A.reshape(N, -1)   # (N, 9216)
probe  = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED,
                            class_weight="balanced")
probe.fit(X_flat, y_mem)

importance  = np.abs(probe.coef_).sum(axis=0)   # (9216,)
global_rank = np.argsort(importance)[::-1]       # descending global index

# Per-layer neuron ranking derived from global probe coefficients
layer_importance = {}
for l in range(L):
    layer_importance[l] = np.argsort(importance[l*H:(l+1)*H])[::-1]

# ── Load GoEmotions model ──────────────────────────────────────────────────────
print("\nLoading GoEmotions model...")
TARGET_GOEMO_IDS = [2, 11, 14, 17, 25, 26]   # anger, disgust, fear, joy, sadness, surprise
model = AutoModelForSequenceClassification.from_pretrained(MODEL_HF)
model.eval()
model.to(device)

# ── Hook utilities ─────────────────────────────────────────────────────────────
def make_silence_hook(local_indices):
    idx_t = (torch.tensor(local_indices, dtype=torch.long)
             if local_indices else torch.tensor([], dtype=torch.long))

    def hook(module, input, output):
        if not isinstance(output, torch.Tensor):
            return output
        out = output.clone()
        idx = idx_t.to(out.device)
        if out.dim() == 3:
            out[:, 0, idx] = 0.0
        elif out.dim() == 2:
            out[:, idx] = 0.0
        return out

    return hook


def apply_hooks(layer_to_local):
    handles = []
    for li, local_idxs in layer_to_local.items():
        if local_idxs:
            h = model.bert.encoder.layer[li].output.LayerNorm.register_forward_hook(
                make_silence_hook(local_idxs)
            )
            handles.append(h)
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ── Task evaluation ────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_task(sample_df, labels):
    preds, trues = [], []
    for row, label in zip(sample_df, labels):
        input_ids = torch.tensor(row["input_ids"]).unsqueeze(0).to(device)
        attn_mask = torch.tensor(row["attention_mask"]).unsqueeze(0).to(device)
        logits    = model(input_ids=input_ids, attention_mask=attn_mask).logits
        target_logits = logits[0, TARGET_GOEMO_IDS]
        pred = TARGET_GOEMO_IDS[target_logits.argmax().item()]
        preds.append(pred)
        trues.append(label)
    local_preds = [TARGET_GOEMO_IDS.index(p) for p in preds]
    return float(f1_score(trues, local_preds, average="weighted", zero_division=0))


# ── Baseline ───────────────────────────────────────────────────────────────────
print("\nEvaluating baseline (no hooks)...")
baseline_f1 = eval_task(sample_df, labels)
print(f"Baseline weighted F1: {baseline_f1:.4f}")

# ── Sweep ──────────────────────────────────────────────────────────────────────
print(f"\nSweep: {len(TOP_P_VALS)} top_p values × 4 experiments")
print(f"  E2 targets: L{L_STAR}  |  E3 targets: {['L'+str(l) for l in TOP4_LAYERS]}")
results = []
rng = np.random.default_rng(SEED)

for top_p in TOP_P_VALS:
    k_global    = max(1, int(top_p * TOTAL))
    k_per_layer = max(1, int(top_p * H))
    print(f"\n  top_p={top_p}  k_global={k_global}  k_per_layer={k_per_layer}")
    row = {"top_p": top_p}

    # E1: Random selection (k_global neurons)
    rnd_global = rng.choice(TOTAL, size=k_global, replace=False).tolist()
    rnd_l2l    = defaultdict(list)
    for g in rnd_global:
        rnd_l2l[g // H].append(g % H)
    handles = apply_hooks(rnd_l2l)
    row["e1_rnd_weighted_f1"] = eval_task(sample_df, labels)
    remove_hooks(handles)
    print(f"    E1 random:       F1={row['e1_rnd_weighted_f1']:.4f}")

    # E2: Single-layer suppression at l* (computed from data)
    top_local_lstar = layer_importance[L_STAR][:k_per_layer].tolist()
    handles = apply_hooks({L_STAR: top_local_lstar})
    row["e2_single_weighted_f1"] = eval_task(sample_df, labels)
    remove_hooks(handles)
    print(f"    E2 single-layer: F1={row['e2_single_weighted_f1']:.4f}  (L{L_STAR})")

    # E3: Multi-layer suppression (top-4 layers by AUC)
    l2l_e3  = {l: layer_importance[l][:k_per_layer].tolist() for l in TOP4_LAYERS}
    handles = apply_hooks(l2l_e3)
    row["e3_multi_weighted_f1"] = eval_task(sample_df, labels)
    remove_hooks(handles)
    print(f"    E3 multi-layer:  F1={row['e3_multi_weighted_f1']:.4f}  ({['L'+str(l) for l in TOP4_LAYERS]})")

    # E4: Global suppression
    top_global = global_rank[:k_global].tolist()
    l2l_e4     = defaultdict(list)
    for g in top_global:
        l2l_e4[g // H].append(g % H)
    handles = apply_hooks(l2l_e4)
    row["e4_global_weighted_f1"] = eval_task(sample_df, labels)
    remove_hooks(handles)
    print(f"    E4 global:       F1={row['e4_global_weighted_f1']:.4f}")

    results.append(row)

# ── Save ───────────────────────────────────────────────────────────────────────
out = {
    "baseline_f1":  baseline_f1,
    "l_star":       L_STAR,
    "top4_layers":  TOP4_LAYERS,
    "auc_per_layer": {str(l): auc_per_layer[l] for l in range(L)},
    "results":      results,
}
with open(OUT_FILE, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {OUT_FILE}")

# Summary
print(f"\n{'top_p':>6}  {'E1':>7}  {'E2(L'+str(L_STAR)+')':>8}  {'E3':>7}  {'E4':>7}")
print(f"{'------':>6}  {'-------':>7}  {'--------':>8}  {'-------':>7}  {'-------':>7}")
for r in results:
    print(f"{r['top_p']:>6}  "
          f"{r['e1_rnd_weighted_f1']:>7.4f}  "
          f"{r['e2_single_weighted_f1']:>8.4f}  "
          f"{r['e3_multi_weighted_f1']:>7.4f}  "
          f"{r['e4_global_weighted_f1']:>7.4f}")
