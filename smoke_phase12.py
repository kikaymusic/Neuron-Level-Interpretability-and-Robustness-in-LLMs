import numpy as np
from lumia_unimodal_tb_synapse import (
    lumia_layerwise_auc_from_activations,
    lumia_neuron_ranking_and_mask_from_layer,
)

# Fake activations: N samples, L layers, H hidden dims
N, L, H = 200, 6, 32
rng = np.random.default_rng(123)

A = rng.normal(size=(N, L, H)).astype(np.float32)

# Inject a membership signal into one layer/dimension so AUC > 0.5 somewhere
y_mem = rng.integers(0, 2, size=N).astype(int)
signal_layer = 3
signal_dim = 7
A[:, signal_layer, signal_dim] += (y_mem * 1.5).astype(np.float32)

df_auc, l_star = lumia_layerwise_auc_from_activations(
    activations=A,
    membership_labels=y_mem,
    seed=123,
    test_size=0.2,
    out_csv="./results/_smoke/auc_by_layer.csv",
)

print("AUC head:")
print(df_auc.head())
print("l_star:", l_star)

mask_obj = lumia_neuron_ranking_and_mask_from_layer(
    activations=A,
    membership_labels=y_mem,
    layer_id=l_star,
    top_p=0.1,  # 10% just for smoke
    seed=123,
    test_size=0.2,
    out_scores_csv="./results/_smoke/neuron_scores.csv",
    out_mask_json="./results/_smoke/mask_M.json",
)

print("mask_obj:", mask_obj)
