
# Minimal LUMIA operational API for SYNAPSE integration

import os
import json
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def lumia_layerwise_auc_from_activations(
    activations: np.ndarray,
    membership_labels: np.ndarray,
    seed: int = 123,
    test_size: float = 0.2,
    out_csv: Optional[str] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Phase 1 (layer-wise): train a linear membership probe per layer and compute AUC.
    activations: [N, L, H]
    membership_labels: [N] with 0/1
    """
    A = np.asarray(activations)
    y = np.asarray(membership_labels).astype(int)

    if A.ndim != 3:
        raise ValueError(f"Expected activations shape [N,L,H], got {A.shape}")

    N, L, _H = A.shape

    idx_all = np.arange(N)
    idx_train, idx_test = train_test_split(
        idx_all,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    y_train = y[idx_train]
    y_test = y[idx_test]

    rows = []
    for layer_id in range(L):
        X_train = A[idx_train, layer_id, :]
        X_test = A[idx_test, layer_id, :]

        clf = LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            random_state=seed
        )
        clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        rows.append({"layer_id": int(layer_id), "auc": float(auc)})

    df_auc = pd.DataFrame(rows).sort_values("auc", ascending=False)
    l_star = int(df_auc.iloc[0]["layer_id"])

    if out_csv is not None:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df_auc.to_csv(out_csv, index=False)

    return df_auc, l_star


def lumia_neuron_ranking_and_mask_from_layer(
    activations: np.ndarray,
    membership_labels: np.ndarray,
    layer_id: int,
    top_p: float = 0.01,
    seed: int = 123,
    test_size: float = 0.2,
    out_scores_csv: Optional[str] = None,
    out_mask_json: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Phase 2 (neuron-wise): train a membership probe on layer_id and rank neurons by |w|.
    Build mask_M as global indices: layer_id*H + neuron_id.
    activations: [N, L, H]
    """
    A = np.asarray(activations)
    y = np.asarray(membership_labels).astype(int)

    if A.ndim != 3:
        raise ValueError(f"Expected activations shape [N,L,H], got {A.shape}")

    N, L, H = A.shape
    if not (0 <= layer_id < L):
        raise ValueError(f"layer_id={layer_id} out of range [0, {L-1}]")

    idx_all = np.arange(N)
    idx_train, _idx_test = train_test_split(
        idx_all,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    X_train = A[idx_train, layer_id, :]
    y_train = y[idx_train]

    clf = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=seed
    )
    clf.fit(X_train, y_train)

    w = clf.coef_.reshape(-1)  # [H]
    scores = np.abs(w)

    k = max(1, int(round(H * float(top_p))))
    rank_idx = np.argsort(scores)[::-1]
    top_local = rank_idx[:k].astype(int).tolist()

    indices_global = [int(layer_id * H + j) for j in top_local]

    if out_scores_csv is not None:
        os.makedirs(os.path.dirname(out_scores_csv), exist_ok=True)
        df_scores = pd.DataFrame({
            "layer_id": [int(layer_id)] * H,
            "neuron_id": np.arange(H, dtype=int),
            "score": scores.astype(float)
        }).sort_values("score", ascending=False)
        df_scores.to_csv(out_scores_csv, index=False)

    mask_obj = {
        "layer_id": int(layer_id),
        "hidden_dim": int(H),
        "top_p": float(top_p),
        "top_k": int(k),
        "indices_global": indices_global
    }

    if out_mask_json is not None:
        os.makedirs(os.path.dirname(out_mask_json), exist_ok=True)
        with open(out_mask_json, "w", encoding="utf-8") as f:
            json.dump(mask_obj, f, indent=2)

    return mask_obj