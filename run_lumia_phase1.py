import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def load_activations_json(path: str) -> np.ndarray:
    """
    Expected: activations saved as nested lists with shape [N, L, H].
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    A = np.asarray(obj, dtype=np.float32)
    if A.ndim != 3:
        raise ValueError(f"Expected activations [N,L,H], got {A.shape}")
    return A


def build_membership_labels_dummy(n: int, seed: int = 123, member_frac: float = 0.5) -> np.ndarray:
    """
    Dummy membership: deterministic random split.
    member=1, non-member=0.
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    idx = rng.permutation(n)
    k = int(round(n * member_frac))
    y[idx[:k]] = 1
    return y


def layerwise_auc(A: np.ndarray, y_mem: np.ndarray, seed: int = 123, test_size: float = 0.2) -> pd.DataFrame:
    N, L, _H = A.shape
    idx_all = np.arange(N)

    idx_train, idx_test = train_test_split(
        idx_all,
        test_size=test_size,
        random_state=seed,
        stratify=y_mem
    )

    rows = []
    for layer_id in range(L):
        X_train = A[idx_train, layer_id, :]
        X_test = A[idx_test, layer_id, :]

        clf = LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed)
        clf.fit(X_train, y_mem[idx_train])

        proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_mem[idx_test], proba)
        rows.append({"layer_id": int(layer_id), "auc": float(auc)})

    df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    return df


def main():
    activ_path = "./data/BigBird/activations.json"
    out_path = "./data/BigBird/auc_by_layer.csv"

    A = load_activations_json(activ_path)
    N, L, H = A.shape
    print(f"activations: N={N}, L={L}, H={H}")

    # Dummy membership for now (replace later with real train vs test split)
    y_mem = build_membership_labels_dummy(N, seed=123, member_frac=0.5)

    df_auc = layerwise_auc(A, y_mem, seed=123, test_size=0.2)
    l_star = int(df_auc.iloc[0]["layer_id"])
    print("l_star =", l_star, "best_auc =", float(df_auc.iloc[0]["auc"]))

    df_auc.to_csv(out_path, index=False)
    print("saved", out_path)


if __name__ == "__main__":
    main()
