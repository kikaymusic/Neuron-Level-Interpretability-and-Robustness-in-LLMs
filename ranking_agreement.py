"""
Neuron-ranking agreement — shared analysis for reviewers R3.4 / R7.6 / R4.3
([CLS] vs mean-pooled rankings) and #8 (probe vs conductance/IG).

Given two per-neuron importance vectors over the SAME neuron space (length L*hidden),
quantify how much they agree:

    - Spearman & Kendall rank correlation over all neurons
    - top-k overlap: |top_a ∩ top_b| as a Jaccard index and as a fraction of k

Used to answer "how much does the choice of representation / attribution method change
the identified salient neurons?". Pure numpy + scipy — unit-testable locally, no GPU.
"""

from __future__ import annotations
import numpy as np


def top_k_set(importance, k: int) -> set:
    imp = np.asarray(importance, dtype=float)
    return set(int(i) for i in np.argsort(imp)[-k:])


def compare_rankings(importance_a, importance_b, top_k_frac: float = 0.10) -> dict:
    """
    Parameters
    ----------
    importance_a, importance_b : array (num_neurons,)
        Per-neuron importance from the two methods (e.g. probe weights vs conductance,
        or [CLS] probe vs mean-pool probe). Higher = more important.
    top_k_frac : float
        Fraction of neurons that defines the "top-k" set (default 10%).

    Returns
    -------
    dict: n, k, spearman, kendall, topk_intersection, topk_overlap (∩/k),
          topk_jaccard (∩/∪).
    """
    from scipy.stats import spearmanr, kendalltau

    a = np.asarray(importance_a, dtype=float)
    b = np.asarray(importance_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"importance vectors must be 1D and equal length; got {a.shape} vs {b.shape}")
    n = a.size
    k = max(1, round(n * top_k_frac))

    top_a, top_b = top_k_set(a, k), top_k_set(b, k)
    inter = len(top_a & top_b)
    union = len(top_a | top_b)

    sp = spearmanr(a, b).correlation
    kt = kendalltau(a, b).correlation
    return {
        "n": int(n),
        "k": int(k),
        "top_k_frac": float(top_k_frac),
        "spearman": float(sp) if sp == sp else float("nan"),
        "kendall": float(kt) if kt == kt else float("nan"),
        "topk_intersection": int(inter),
        "topk_overlap": float(inter / k),
        "topk_jaccard": float(inter / union) if union else float("nan"),
    }


def probe_importance(probe) -> np.ndarray:
    """Per-neuron importance from a trained linear probe: sum of |weights| over classes.
    Matches the notebook's get_top_k_neurons_exact ranking criterion."""
    W = probe.linear.weight.detach().abs()          # [num_classes, num_neurons]
    return W.sum(dim=0).cpu().numpy()                # [num_neurons]


# ---------------------------------------------------------------------------
# Self-test (local, no GPU).
# ---------------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(0)
    n = 1000
    a = rng.normal(size=n)

    # identical -> perfect agreement
    r = compare_rankings(a, a.copy(), top_k_frac=0.1)
    assert abs(r["spearman"] - 1) < 1e-9 and r["topk_jaccard"] == 1.0, r
    print(f"[selftest] identical OK: spearman={r['spearman']:.3f} jaccard={r['topk_jaccard']:.3f}")

    # reversed -> anticorrelated, disjoint tops
    r = compare_rankings(a, -a, top_k_frac=0.1)
    assert abs(r["spearman"] + 1) < 1e-9 and r["topk_intersection"] == 0, r
    print(f"[selftest] reversed OK: spearman={r['spearman']:.3f} inter={r['topk_intersection']}")

    # unrelated -> ~0 correlation, small overlap
    b = rng.normal(size=n)
    r = compare_rankings(a, b, top_k_frac=0.1)
    assert abs(r["spearman"]) < 0.15 and r["topk_overlap"] < 0.3, r
    print(f"[selftest] unrelated OK: spearman={r['spearman']:.3f} overlap={r['topk_overlap']:.3f}")

    # partial: b = a with noise -> high but <1 agreement
    b = a + rng.normal(scale=0.5, size=n)
    r = compare_rankings(a, b, top_k_frac=0.1)
    assert 0.5 < r["spearman"] < 0.99 and 0.3 < r["topk_overlap"] < 1.0, r
    print(f"[selftest] noisy-copy OK: spearman={r['spearman']:.3f} overlap={r['topk_overlap']:.3f}")
    print("ALL SELFTESTS PASSED")


if __name__ == "__main__":
    _selftest()
