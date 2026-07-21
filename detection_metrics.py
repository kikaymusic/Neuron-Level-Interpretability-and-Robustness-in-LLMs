"""
Detection-oriented metrics for the malware task (reviewer R3.2).

Adds a *normal-vs-malicious* evaluation lens on top of the existing multiclass
malware model, WITHOUT retraining. The malicious score of an example is

    malicious_score = 1 - P(normal_class)

and the binary target is (true_label != normal_class). From the ROC of that score
we report:

    - EER            : equal-error rate (FPR == FNR operating point)
    - TPR@1%FPR      : detection rate at a fixed 1% false-alarm point
                       (the operationally most relevant metric in security
                       detection; same one used in the authors' other work)
    - FAR            : false-alarm rate = benign flagged as malicious
    - MAR            : miss rate        = malicious missed as normal
    - ROC-AUC

FAR/MAR are reported at two operating points:
    - "argmax": the model's own decision (predicted class == normal or not)
    - "eer"   : the balanced ROC threshold

Not applied to the per-family multiclass breakdown or to GoEmotions (those are
not detection problems). macro-F1 remains the primary metric elsewhere.

Pure post-processing: numpy + scikit-learn only, no GPU, no model. Feeds R3.2.
"""

from __future__ import annotations
import numpy as np


def _as_probs(scores: np.ndarray) -> np.ndarray:
    """Accept either probabilities (rows ~sum to 1) or raw logits (softmax them)."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D (N, C); got shape {scores.shape}")
    row_sums = scores.sum(axis=1)
    looks_like_probs = np.all(scores >= 0) and np.allclose(row_sums, 1.0, atol=1e-3)
    if looks_like_probs:
        return scores
    # softmax over classes (numerically stable)
    z = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def detection_metrics(scores, y_true, normal_idx: int = 3, fpr_target: float = 0.01) -> dict:
    """
    Parameters
    ----------
    scores : array (N, C)
        Per-example class probabilities (or logits; auto-softmaxed).
    y_true : array (N,)
        Integer true class labels in [0, C).
    normal_idx : int
        Index of the benign/"Normal" class (MalwSpecSys: 3).
    fpr_target : float
        False-alarm operating point for TPR@x%FPR (default 0.01 = 1%).

    Returns
    -------
    dict with: eer, eer_threshold, tpr_at_fpr, fpr_target, roc_auc,
               far_argmax, mar_argmax, far_eer, mar_eer,
               n, n_normal, n_malicious.
        Metrics that are undefined (e.g. only one class present) are NaN.
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    probs = _as_probs(scores)
    y_true = np.asarray(y_true).astype(int)
    n, C = probs.shape
    if not (0 <= normal_idx < C):
        raise ValueError(f"normal_idx={normal_idx} out of range for C={C} classes")
    if y_true.shape[0] != n:
        raise ValueError("scores and y_true length mismatch")

    malicious_score = 1.0 - probs[:, normal_idx]          # higher = more malicious
    y_bin = (y_true != normal_idx).astype(int)            # 1 = malicious, 0 = normal
    n_mal = int(y_bin.sum())
    n_norm = int((1 - y_bin).sum())

    out = {
        "n": int(n), "n_normal": n_norm, "n_malicious": n_mal,
        "fpr_target": float(fpr_target),
        "eer": np.nan, "eer_threshold": np.nan, "tpr_at_fpr": np.nan,
        "roc_auc": np.nan,
        "far_argmax": np.nan, "mar_argmax": np.nan,
        "far_eer": np.nan, "mar_eer": np.nan,
    }

    # FAR/MAR at the model's own (argmax) operating point — always well defined
    pred = probs.argmax(axis=1)
    pred_mal = (pred != normal_idx).astype(int)
    if n_norm > 0:
        out["far_argmax"] = float(((pred_mal == 1) & (y_bin == 0)).sum() / n_norm)
    if n_mal > 0:
        out["mar_argmax"] = float(((pred_mal == 0) & (y_bin == 1)).sum() / n_mal)

    # ROC-based metrics need both classes present
    if n_mal == 0 or n_norm == 0:
        return out

    fpr, tpr, thr = roc_curve(y_bin, malicious_score)
    fnr = 1.0 - tpr
    out["roc_auc"] = float(roc_auc_score(y_bin, malicious_score))

    # EER: operating point where FPR and FNR cross
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    out["eer"] = float((fpr[i] + fnr[i]) / 2.0)
    out["eer_threshold"] = float(thr[i])

    # TPR at a fixed low FPR (interpolated on the ROC curve)
    out["tpr_at_fpr"] = float(np.interp(fpr_target, fpr, tpr))

    # FAR/MAR at the EER threshold (decide malicious if score >= threshold)
    dec_mal = (malicious_score >= out["eer_threshold"]).astype(int)
    out["far_eer"] = float(((dec_mal == 1) & (y_bin == 0)).sum() / n_norm)
    out["mar_eer"] = float(((dec_mal == 0) & (y_bin == 1)).sum() / n_mal)
    return out


# ---------------------------------------------------------------------------
# Self-test (runs locally, no GPU): synthetic data with a KNOWN answer.
# ---------------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(0)
    C, normal_idx = 5, 3

    # 1) Perfect separator -> EER 0, AUC 1, TPR@1%FPR 1, zero errors.
    N = 400
    y = rng.integers(0, C, size=N)
    probs = np.full((N, C), 0.02)
    for k in range(N):
        # give almost all mass to the TRUE class -> argmax correct, score clean
        probs[k] = 0.02
        probs[k, y[k]] = 0.92
    probs /= probs.sum(1, keepdims=True)
    m = detection_metrics(probs, y, normal_idx=normal_idx)
    assert abs(m["roc_auc"] - 1.0) < 1e-6, m
    assert m["eer"] < 1e-6, m
    assert abs(m["tpr_at_fpr"] - 1.0) < 1e-6, m
    assert abs(m["far_argmax"]) < 1e-9 and abs(m["mar_argmax"]) < 1e-9, m
    print("[selftest] perfect separator OK:", {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})

    # 2) Random scores -> AUC ~0.5, EER ~0.5.
    N = 5000
    y = rng.integers(0, C, size=N)
    probs = rng.dirichlet(np.ones(C), size=N)
    m = detection_metrics(probs, y, normal_idx=normal_idx)
    assert 0.42 < m["roc_auc"] < 0.58, m
    assert 0.42 < m["eer"] < 0.58, m
    print("[selftest] random scores OK:  auc=%.3f eer=%.3f tpr@1%%=%.3f far=%.3f mar=%.3f"
          % (m["roc_auc"], m["eer"], m["tpr_at_fpr"], m["far_argmax"], m["mar_argmax"]))

    # 3) Logits accepted (auto-softmax) -> same as softmaxing first.
    logits = rng.normal(size=(200, C)) * 3
    y = rng.integers(0, C, size=200)
    a = detection_metrics(logits, y, normal_idx=normal_idx)
    e = np.exp(logits - logits.max(1, keepdims=True)); sm = e / e.sum(1, keepdims=True)
    b = detection_metrics(sm, y, normal_idx=normal_idx)
    assert abs(a["roc_auc"] - b["roc_auc"]) < 1e-9, (a, b)
    print("[selftest] logits==softmax OK")

    # 4) Degenerate (one class only) -> ROC metrics NaN, argmax metrics defined.
    y = np.full(50, normal_idx)
    probs = rng.dirichlet(np.ones(C), size=50)
    m = detection_metrics(probs, y, normal_idx=normal_idx)
    assert np.isnan(m["eer"]) and not np.isnan(m["far_argmax"]) and np.isnan(m["mar_argmax"]), m
    print("[selftest] degenerate one-class OK")
    print("ALL SELFTESTS PASSED")


if __name__ == "__main__":
    _selftest()
