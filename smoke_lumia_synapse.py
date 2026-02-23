# smoke_lumia_synapse.py
import os
import json
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from lumia_unimodal_tb_synapse import (
    lumia_layerwise_auc_from_activations,
    lumia_neuron_ranking_and_mask_from_layer,
)

BASE_PATH = "."
MODEL_DIR = "./data/BigBird"  # ajusta si toca
activations_path = os.path.join(MODEL_DIR, "activations.json")
labels_path = os.path.join(MODEL_DIR, "labels_numeric.txt")

out_dir = os.path.join(MODEL_DIR, "lumia_results")
os.makedirs(out_dir, exist_ok=True)

# 1) Load task labels (only for stratified TEMP membership labels)
y_task = np.loadtxt(labels_path, dtype=np.int64)
N = len(y_task)

# 2) TEMP membership labels (replace later with real train/test membership)
p_member = 0.5
seed_mem = 42
sss = StratifiedShuffleSplit(n_splits=1, train_size=p_member, random_state=seed_mem)
member_idx, _ = next(sss.split(np.zeros_like(y_task), y_task))
y_mem = np.zeros(N, dtype=np.int64)
y_mem[member_idx] = 1

print(f"TEMP y_mem: members={y_mem.sum()} nonmembers={(y_mem==0).sum()}")

# 3) Load activations.json
with open(activations_path, "r") as f:
    activations = json.load(f)

# 4) Build A[N,L,H] from your stored format (assumes activations[i][0] is flat [L*H])
# IMPORTANT: set these correctly (or infer if you have them in metadata)
num_layers = 12      # change if your model differs
hidden_size = 768    # change if your model differs

L = int(num_layers)
H = int(hidden_size)

A = np.zeros((N, L, H), dtype=np.float32)
for i in range(N):
    flat = np.array(activations[i][0], dtype=np.float32)  # [L*H]
    A[i] = flat.reshape(L, H)

print("Built A:", A.shape)

# 5) Phase 1: AUC by layer + l_star
df_auc, l_star = lumia_layerwise_auc_from_activations(
    activations=A,
    membership_labels=y_mem,
    seed=123,
    test_size=0.2,
    out_csv=os.path.join(out_dir, "auc_by_layer.csv"),
)

print("Phase1 OK. l_star =", l_star)
print(df_auc.head())

# 6) Phase 2: neuron ranking + mask_M in l_star
mask_obj = lumia_neuron_ranking_and_mask_from_layer(
    activations=A,
    membership_labels=y_mem,
    layer_id=l_star,
    top_p=0.01,
    seed=123,
    test_size=0.2,
    out_scores_csv=os.path.join(out_dir, f"neuron_scores_layer{l_star}.csv"),
    out_mask_json=os.path.join(out_dir, f"mask_M_layer{l_star}_top0p01.json"),
)

print("Phase2 OK. mask top_k =", mask_obj["top_k"])
