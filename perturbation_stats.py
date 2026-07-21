"""
Random-neuron control + significance for the probe ranking (reviewers R1.3 / R4.2,
and the statistical machinery for R4.4 / R7.3).

Idea: if the linear-probe ranking is meaningful (neurons it flags as important are
functionally used by the model, not merely correlated), then silencing the TOP-k
ranked neurons must degrade the task **more** than silencing k RANDOM neurons.
This module runs that comparison over many random draws (seeds) and quantifies it:

    - degradation of the top-k silencing (deterministic given the probe)
    - degradation of k random neurons: distribution over `n_seeds` draws (mean +/- std)
    - one-sided empirical p-value  = P(random draw degrades at least as much as top-k)
    - z-score of the top-k degradation w.r.t. the random distribution

A small empirical p (and a large positive z) is evidence that the ranking captures
functionally relevant units, addressing the correlation-vs-causation objection.

Pure numpy: the model-dependent part is injected as a callable `eval_metric(indices)`,
so all the statistics here are unit-testable locally without a GPU.
"""

from __future__ import annotations
import numpy as np


def random_index_sets(total_neurons: int, k: int, n_seeds: int, base_seed: int = 0):
    """`n_seeds` reproducible sets of k distinct neuron indices drawn from [0, total)."""
    if k > total_neurons:
        raise ValueError(f"k={k} > total_neurons={total_neurons}")
    sets = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        sets.append(sorted(int(x) for x in rng.choice(total_neurons, size=k, replace=False)))
    return sets


def top_vs_random(eval_metric,
                  baseline: float,
                  top_indices,
                  total_neurons: int,
                  n_seeds: int = 20,
                  base_seed: int = 0,
                  higher_is_better: bool = True) -> dict:
    """
    Parameters
    ----------
    eval_metric : callable(list[int]) -> float
        Evaluates the PRIMARY task metric (e.g. macro-F1) with the given neuron
        indices silenced. Model/data are captured by the closure.
    baseline : float
        The same metric with NO neuron silenced.
    top_indices : sequence[int]
        The top-k neurons from the probe ranking.
    total_neurons : int
        Size of the pool to draw random neurons from (L * hidden_size).
    n_seeds : int
        Number of random draws.
    higher_is_better : bool
        True for F1/accuracy (degradation = baseline - metric).

    Returns
    -------
    dict with k, baseline, top_metric, top_drop, random_mean_metric,
        random_mean_drop, random_std_drop, p_empirical, z_score, n_seeds,
        random_drops (list).
    """
    top_indices = [int(i) for i in top_indices]
    k = len(top_indices)
    sign = 1.0 if higher_is_better else -1.0

    top_metric = float(eval_metric(top_indices))
    rnd_metrics = np.array(
        [float(eval_metric(idx)) for idx in random_index_sets(total_neurons, k, n_seeds, base_seed)],
        dtype=float,
    )

    top_drop = sign * (baseline - top_metric)          # how much the top-k hurts
    rnd_drops = sign * (baseline - rnd_metrics)         # how much random draws hurt
    mu = float(rnd_drops.mean())
    sd = float(rnd_drops.std(ddof=1)) if n_seeds > 1 else float("nan")

    # one-sided empirical p: fraction of random draws that hurt >= top-k (Laplace-smoothed)
    p_emp = float((np.sum(rnd_drops >= top_drop) + 1) / (n_seeds + 1))
    z = float((top_drop - mu) / sd) if (sd == sd and sd > 0) else float("nan")

    return {
        "k": k,
        "baseline": float(baseline),
        "top_metric": top_metric,
        "top_drop": float(top_drop),
        "random_mean_metric": float(rnd_metrics.mean()),
        "random_mean_drop": mu,
        "random_std_drop": sd,
        "p_empirical": p_emp,
        "z_score": z,
        "n_seeds": int(n_seeds),
        "random_drops": [float(x) for x in rnd_drops],
    }


def aggregate_seeds(values, higher_is_better: bool = True) -> dict:
    """mean +/- std (and a 95% normal CI) for a list of per-seed metric values.
    Generic helper for R4.4/R7.3 (report mean +/- std over seeds for any experiment)."""
    v = np.asarray(list(values), dtype=float)
    n = v.size
    mean = float(v.mean()) if n else float("nan")
    std = float(v.std(ddof=1)) if n > 1 else float("nan")
    half = 1.96 * std / np.sqrt(n) if (n > 1 and std == std) else float("nan")
    return {"n": int(n), "mean": mean, "std": std,
            "ci95_low": (mean - half) if half == half else float("nan"),
            "ci95_high": (mean + half) if half == half else float("nan")}


# ---------------------------------------------------------------------------
# Self-test (local, no GPU): a synthetic model where the "top" neurons genuinely
# matter, so the statistics must flag the top-k as significantly more damaging.
# ---------------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(0)
    TOTAL = 500
    # Assign each neuron a hidden "importance"; the metric drops proportionally to
    # the summed importance of the silenced neurons. Top neurons have high importance.
    importance = np.zeros(TOTAL)
    importance[:50] = rng.uniform(0.8, 1.0, size=50)   # 50 genuinely important neurons
    importance[50:] = rng.uniform(0.0, 0.05, size=TOTAL - 50)
    BASELINE = 0.90

    def eval_metric(indices):
        drop = 0.02 * importance[np.asarray(indices, dtype=int)].sum()
        return max(0.0, BASELINE - drop)

    top_indices = list(range(20))  # top-20 (all important)
    r = top_vs_random(eval_metric, BASELINE, top_indices, TOTAL, n_seeds=50, base_seed=1)
    assert r["top_drop"] > r["random_mean_drop"], r
    assert r["p_empirical"] <= 1 / (50 + 1) + 1e-9, r      # no random draw beats top
    assert r["z_score"] > 3, r
    print("[selftest] top-vs-random OK: "
          f"top_drop={r['top_drop']:.4f}  random={r['random_mean_drop']:.4f}"
          f"+/-{r['random_std_drop']:.4f}  p={r['p_empirical']:.4f}  z={r['z_score']:.2f}")

    # Control: if "top" is actually random, it must NOT look significant.
    fake_top = sorted(int(x) for x in rng.choice(TOTAL, size=20, replace=False))
    r2 = top_vs_random(eval_metric, BASELINE, fake_top, TOTAL, n_seeds=50, base_seed=7)
    assert r2["p_empirical"] > 0.05, r2
    print(f"[selftest] random-as-top NOT significant OK: p={r2['p_empirical']:.3f}")

    agg = aggregate_seeds([0.80, 0.82, 0.79, 0.81, 0.83])
    assert 0.80 < agg["mean"] < 0.82 and agg["std"] > 0, agg
    print(f"[selftest] aggregate_seeds OK: {agg['mean']:.3f} +/- {agg['std']:.3f} "
          f"CI95=[{agg['ci95_low']:.3f}, {agg['ci95_high']:.3f}]")
    print("ALL SELFTESTS PASSED")


if __name__ == "__main__":
    _selftest()
