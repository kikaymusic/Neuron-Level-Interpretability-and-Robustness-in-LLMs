"""
Task utility evaluation via full BigBird forward pass.

Methodology identical to original pipeline (synapseFINAL_run.py):
  - AutoModelForSequenceClassification.from_pretrained + load_state_dict
  - 50 examples, 10 per class, seed=42
  - Hooks recovered from activation file comparison (no probe retraining)
  - Weighted F1 (same as quick_baseline_f1 in original script)

Usage (on server, from project root):
    python task_utility_eval.py

Output:
    data/BigBird/task_utility_<timestamp>.json
"""

import ast
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification

# Paths
BASE_DIR   = "/Users/kikay/Desktop/backup_synapse"
CSV_PATH   = f"{BASE_DIR}/BigBird_tokens_PT.csv"
ACT_BASE   = f"{BASE_DIR}/activations_bigbird.json"
SWEEP_DIR  = f"{BASE_DIR}/sweep_20260428_205017"
MODEL_PATH = f"{BASE_DIR}/best_model_BigBird.pth"
MODEL_HF   = "google/bigbird-roberta-base"

# Config
SEED        = 42
NUM_LABELS  = 5
N_PER_CLASS = 10
TOP_P_LIST  = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.65, 0.8]
EXPERIMENTS = {
    "e1_p3":  "E1_P3",
    "e2_p4a": "E2_P4A",
    "e3_p4b": "E3_P4B",
    "e4_rnd": "E4_RND",
}

# Hook functions (exact copy from synapseFINAL_run.py)
def make_cls_silence_hook(indices):
    idxs = torch.tensor(indices, dtype=torch.long)

    def hook(module, inp, output):
        if isinstance(output, tuple):
            if len(output) == 0:
                return output
            hidden = output[0]
        else:
            hidden = output
        if not hasattr(hidden, "dim") or hidden.dim() != 3:
            return output
        if idxs.numel() == 0:
            return output
        new_hidden = hidden.clone()
        cls_token = new_hidden[:, 0, :]
        cls_token[:, idxs] = 0.0
        new_hidden[:, 0, :] = cls_token
        if isinstance(output, tuple):
            return (new_hidden,) + tuple(output[1:])
        else:
            return new_hidden

    return hook


def get_encoder_layers(model):
    if hasattr(model, "bert"):
        return model.bert.encoder.layer
    raise NotImplementedError("Unsupported architecture - expected model.bert")


# Recover which neurons were zeroed by comparing activation files
def extract_hook_map(base_path, def_path, n_lines=60):
    """
    Read n_lines from base and defended JSONL files, compare CLS activations.
    Returns {layer_id: [local_neuron_indices]} for layers where neurons were zeroed.

    Logic: a neuron at layer L position i was silenced if:
      - def[example, i] == 0.0 for ALL n_lines examples
      - base[example, i] != 0.0 for AT LEAST ONE example
    The hook sets values to exactly 0.0, so this is a clean comparison.
    """
    base_cls = {}  # {layer_id: list of np.array(768)}
    def_cls  = {}

    def read_cls(path, n):
        out = {}
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                obj = json.loads(line)
                for feat in obj["features"]:
                    if feat["token"] != "[CLS]":
                        continue
                    for layer in feat["layers"]:
                        out.setdefault(layer["index"], []).append(
                            np.array(layer["values"], dtype=np.float32))
                    break
        return out

    base_cls = read_cls(base_path, n_lines)
    def_cls  = read_cls(def_path,  n_lines)

    hook_map = {}
    for lid in sorted(base_cls.keys()):
        if lid not in def_cls:
            continue
        base_arr = np.stack(base_cls[lid])  # (n_lines, 768)
        def_arr  = np.stack(def_cls[lid])

        zeroed = np.where(
            np.all(def_arr == 0.0, axis=0) &
            np.any(base_arr != 0.0, axis=0)
        )[0].tolist()

        if zeroed:
            hook_map[lid] = zeroed

    return hook_map


# Forward pass with optional hooks
def run_forward(model, sample_df, labels, device, hook_map=None):
    """
    Run full model forward pass on sample_df.
    hook_map: {layer_id: [neuron_indices]} - applied before inference, removed after.
    Returns weighted F1.
    """
    layers  = get_encoder_layers(model)
    handles = []

    if hook_map:
        for lid, idxs in hook_map.items():
            handles.append(
                layers[int(lid)].register_forward_hook(make_cls_silence_hook(idxs))
            )

    preds = []
    model.eval()
    with torch.no_grad():
        for _, row in sample_df.iterrows():
            input_ids = torch.tensor(row["input_ids"]).unsqueeze(0).to(device)
            att_mask  = torch.tensor(row["attention_mask"]).unsqueeze(0).to(device)
            logits    = model(input_ids=input_ids, attention_mask=att_mask).logits
            preds.append(int(logits.argmax(dim=-1)))

    for h in handles:
        h.remove()

    return float(f1_score(labels, preds, average="weighted", zero_division=0))


# Main
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("  Task utility: full BigBird forward pass")
    print(f"  {N_PER_CLASS} examples per class × {NUM_LABELS} classes = {N_PER_CLASS * NUM_LABELS} total")
    print("=" * 60)

    # 1. Load CSV
    print("\n-- Step 1: Loading CSV --")
    print(f"  CSV: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found.")
        sys.exit(1)

    # Read only label column first to select the 50 rows, then parse only those
    df_meta = pd.read_csv(CSV_PATH, usecols=["label"])
    print(f"  Total rows: {len(df_meta)}")

    le = LabelEncoder()
    df_meta["label_enc"] = le.fit_transform(df_meta["label"].astype(str))
    print(f"  Classes: {dict(enumerate(le.classes_))}")
    print(f"  Class distribution: {df_meta['label_enc'].value_counts().sort_index().to_dict()}")

    # Select 50 indices (10 per class) before loading heavy columns
    selected_indices = []
    for cls_id in sorted(df_meta["label_enc"].unique()):
        idx = df_meta[df_meta["label_enc"] == cls_id].sample(
            n=min(N_PER_CLASS, (df_meta["label_enc"] == cls_id).sum()),
            random_state=SEED
        ).index.tolist()
        selected_indices.extend(idx)
    selected_indices = sorted(selected_indices)
    print(f"  Selected {len(selected_indices)} row indices.")

    # Load ONLY the 50 selected rows - skip everything else from disk
    print("  Loading only the 50 selected rows from CSV...")
    all_row_nums  = set(range(len(df_meta)))
    rows_to_skip  = sorted(all_row_nums - set(selected_indices))
    # skiprows counts from row 1 (row 0 = header is always kept)
    df = pd.read_csv(
        CSV_PATH,
        skiprows=[r + 1 for r in rows_to_skip],
        usecols=["label", "input_ids", "attention_mask"]
    ).reset_index(drop=True)

    def parse_ids(x):
        if isinstance(x, list):
            return x
        x = x.strip()
        if x.startswith("["):
            return ast.literal_eval(x)
        return list(map(int, x.split(",")))

    df["input_ids"]      = df["input_ids"].apply(parse_ids)
    df["attention_mask"] = df["attention_mask"].apply(parse_ids)
    df["label_enc"]      = le.transform(df["label"].astype(str))

    sample_len = sum(df["attention_mask"].iloc[0])
    print(f"  Real tokens in first selected row: {sample_len} / {len(df['attention_mask'].iloc[0])}")

    # 2. Already selected above - just rename and confirm
    print(f"\n-- Step 2: Confirming balanced selection --")
    sample_df = df
    labels    = sample_df["label_enc"].tolist()
    print(f"  {len(sample_df)} examples, distribution: {sample_df['label_enc'].value_counts().sort_index().to_dict()}")

    # 3. Load BigBird model
    print("\n-- Step 3: Loading BigBird model --")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")
    print(f"  Loading architecture from: {MODEL_HF}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_HF, num_labels=NUM_LABELS)
    print(f"  Loading weights from: {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("  ✓ Model ready.")

    # 4. Baseline
    print("\n-- Step 4: Baseline (no hooks) --")
    base_f1 = run_forward(model, sample_df, labels, device, hook_map=None)
    print(f"  Baseline weighted F1 = {base_f1:.4f}")

    # 5. Sweep
    print("\n-- Step 5: Sweep over conditions --")
    results = []
    for top_p in TOP_P_LIST:
        print(f"\n  top_p = {top_p}")
        row = {"top_p": top_p, "baseline_weighted_f1": base_f1}

        for key, tag in EXPERIMENTS.items():
            p_str    = str(top_p).replace(".", "p")
            def_path = f"{SWEEP_DIR}/A_def_{tag}_p{p_str}.json"

            if not os.path.exists(def_path):
                print(f"    [{key}] MISSING: {def_path}")
                row[f"{key}_weighted_f1"] = None
                row[f"{key}_n_neurons"]   = None
                continue

            hook_map  = extract_hook_map(ACT_BASE, def_path)
            n_neurons = sum(len(v) for v in hook_map.values())
            def_f1    = run_forward(model, sample_df, labels, device, hook_map=hook_map)

            print(f"    [{key}] {n_neurons} neurons silenced in layers "
                  f"{sorted(hook_map.keys())} → weighted_f1 = {def_f1:.4f}")

            row[f"{key}_weighted_f1"] = def_f1
            row[f"{key}_n_neurons"]   = n_neurons
            row[f"{key}_hook_layers"] = sorted(hook_map.keys())

        results.append(row)

    # 6. Save
    out_path = f"{BASE_DIR}/task_utility_{ts}.json"
    output = {
        "run_ts":      ts,
        "n_per_class": N_PER_CLASS,
        "n_total":     len(sample_df),
        "metric":      "weighted_f1",
        "method":      "full_forward_pass_with_recovered_hooks",
        "baseline_f1": base_f1,
        "label_mapping": dict(enumerate(le.classes_.tolist())),
        "top_p_values": TOP_P_LIST,
        "results":     results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Summary table
    print("\n" + "=" * 60)
    print(f"  Baseline weighted F1: {base_f1:.4f}")
    print(f"\n  {'top_p':>6}  {'E1':>7}  {'E2':>7}  {'E3':>7}  {'E4':>7}")
    print(f"  {'------':>6}  {'-------':>7}  {'-------':>7}  {'-------':>7}  {'-------':>7}")
    for r in results:
        print(f"  {r['top_p']:>6}  "
              f"{r.get('e1_p3_weighted_f1') or 0:>7.4f}  "
              f"{r.get('e2_p4a_weighted_f1') or 0:>7.4f}  "
              f"{r.get('e3_p4b_weighted_f1') or 0:>7.4f}  "
              f"{r.get('e4_rnd_weighted_f1') or 0:>7.4f}")
    print(f"\nSaved → {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
