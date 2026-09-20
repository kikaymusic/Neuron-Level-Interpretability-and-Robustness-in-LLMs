"""
Intervention-based effectiveness comparison of neuron rankings (reviewer R1.1-b, round 2).

Round 1 answered "compare against existing methods" with ranking AGREEMENT (Spearman /
top-k overlap, see ranking_agreement.py). The reviewer replied that agreement is not an
effectiveness comparison, and asked to "conduct the same intervention experiments using
different neuron-ranking methods and compare their effectiveness under identical
experimental conditions".

This module runs exactly that: the SAME global-silencing sweep, driven by several neuron
orderings, holding everything else fixed (same eval subset, same fractions, same hooks,
same metric). The only thing that varies between arms is the criterion used to order the
neurons, so the degradation curves isolate the contribution of the ranking itself.

Arms compared (all over the same L*hidden neuron space):
  - probe                    : SYNAPSE's linear-probe weight ranking (operational method)
  - probeless                : Probeless, from NeuroX -- no probe is trained at all
                               (Antverg & Belinkov, ICLR 2022)
  - conductance              : integrated-gradients-based neuron importance
                               (Dhamdhere et al., ICLR 2019)
  - activation_times_gradient: the cheap standard neuron-importance proxy
  - random                   : the floor (reuses perturbation_stats.random_index_sets)

Ablation-based evaluation of a neuron ranking -- ablate in rank order, measure the drop,
compare against a random ordering -- is the standard protocol in this literature
(Sajjad, Durrani & Dalvi, TACL 2022).

Pure numpy: the model-dependent part is injected as a callable `eval_metric(indices)`,
so everything here is unit-testable locally without a GPU. Mirrors the design of
perturbation_stats.py.
"""

from __future__ import annotations
import numpy as np

import perturbation_stats


def ordering_from_importance(importance) -> list:
    """Neuron indices sorted by DECREASING importance.

    Use for methods that return a per-neuron score vector (probe, conductance,
    activation x gradient). Probeless already returns an ordering, so it is passed
    through as-is.
    """
    imp = np.asarray(importance, dtype=float)
    if imp.ndim != 1:
        raise ValueError(f"importance must be 1D; got shape {imp.shape}")
    return [int(i) for i in np.argsort(imp)[::-1]]


def sweep_ordering(eval_metric,
                   baseline: float,
                   ordering,
                   total_neurons: int,
                   percentages,
                   higher_is_better: bool = True) -> list:
    """Silence the top-k of `ordering` at each fraction and record the metric.

    Parameters
    ----------
    eval_metric : callable(list[int]) -> float
        Evaluates the task metric with the given neuron indices silenced. Model and
        data are captured by the closure -- the SAME closure must be used for every
        arm, which is what makes the conditions identical.
    baseline : float
        The same metric with no neuron silenced.
    ordering : sequence[int]
        Neuron indices in decreasing order of importance.
    total_neurons : int
        L * hidden_size. `ordering` must be a permutation of range(total_neurons).
    percentages : sequence[float]
        Fractions to sweep. Pass the notebook's SWEEP_PCTS -- do NOT invent new ones.

    Returns
    -------
    list[dict] with one row per fraction: percentage, k, baseline, metric, drop.
    """
    order = [int(i) for i in ordering]
    if len(order) != total_neurons:
        raise ValueError(f"ordering has {len(order)} entries, expected {total_neurons}")

    rows = []
    for pct in percentages:
        k = perturbation_stats.exact_k(total_neurons, pct) if hasattr(perturbation_stats, "exact_k") \
            else max(1, round(total_neurons * pct))
        m = eval_metric(order[:k])
        drop = (baseline - m) if higher_is_better else (m - baseline)
        rows.append({"percentage": float(pct), "k": int(k),
                     "baseline": float(baseline), "metric": float(m), "drop": float(drop)})
    return rows


def compare_orderings(eval_metric,
                      baseline: float,
                      orderings: dict,
                      total_neurons: int,
                      percentages,
                      n_random_seeds: int = 20,
                      base_seed: int = 0,
                      higher_is_better: bool = True) -> list:
    """Run `sweep_ordering` for every named ordering, plus a random floor.

    The random arm reuses perturbation_stats.random_index_sets so that this comparison
    and the Round-1 random-neuron control draw their random sets the same way.

    Returns
    -------
    list[dict]: rows of (method, percentage, k, baseline, metric, drop). The random arm
    additionally carries metric_std / drop_std over the draws.
    """
    rows = []
    for name, ordering in orderings.items():
        for r in sweep_ordering(eval_metric, baseline, ordering, total_neurons,
                                percentages, higher_is_better):
            rows.append({"method": name, **r})

    # n_random_seeds=0 skips the random arm. Use that when a previous run already
    # produced it under the same closure and sweep (random_control_malware.csv), so it
    # can be merged in afterwards instead of costing another n_seeds*len(pcts) evals.
    if n_random_seeds <= 0:
        return rows

    for pct in percentages:
        k = max(1, round(total_neurons * pct))
        vals = [eval_metric(idx) for idx in
                perturbation_stats.random_index_sets(total_neurons, k, n_random_seeds, base_seed)]
        vals = np.asarray(vals, dtype=float)
        drops = (baseline - vals) if higher_is_better else (vals - baseline)
        rows.append({"method": "random", "percentage": float(pct), "k": int(k),
                     "baseline": float(baseline), "metric": float(vals.mean()),
                     "drop": float(drops.mean()), "metric_std": float(vals.std()),
                     "drop_std": float(drops.std()), "n_seeds": int(n_random_seeds)})
    return rows


def compare_orderings_multiseed(make_eval,
                                orderings: dict,
                                total_neurons: int,
                                percentages,
                                n_seeds: int = 5,
                                higher_is_better: bool = True) -> list:
    """Repeat the whole comparison over `n_seeds` re-sampled evaluation sets.

    The rankings themselves are deterministic given the model, so the only source of
    variance is the evaluation sample -- which is exactly the multi-seed protocol already
    used elsewhere in the notebook (R4.4 / R7.3), where each seed re-samples the eval set
    rather than re-running a stochastic model.

    Parameters
    ----------
    make_eval : callable(seed) -> (eval_metric, baseline)
        Builds the evaluation closure and its no-silencing baseline for that seed. Every
        arm within a seed must share the returned closure; that is what keeps conditions
        identical across rankings.
    orderings : dict[str, sequence[int]]
        Same orderings reused across all seeds (they do not depend on the eval sample).

    Returns
    -------
    list[dict]: one row per (method, percentage) with metric_mean / metric_std /
    drop_mean / drop_std / n_seeds, plus the per-seed values in `metric_seeds`.
    """
    per_seed = []
    for s in range(n_seeds):
        eval_metric, baseline = make_eval(s)
        rows = []
        for name, ordering in orderings.items():
            for r in sweep_ordering(eval_metric, baseline, ordering, total_neurons,
                                    percentages, higher_is_better):
                rows.append({"method": name, **r})
        per_seed.append(rows)

    out = []
    for name in orderings:
        for pct in percentages:
            vals, drops, bases = [], [], []
            for rows in per_seed:
                r = next(r for r in rows
                         if r["method"] == name and r["percentage"] == float(pct))
                vals.append(r["metric"]); drops.append(r["drop"]); bases.append(r["baseline"])
            vals = np.asarray(vals, dtype=float); drops = np.asarray(drops, dtype=float)
            out.append({"method": name, "percentage": float(pct),
                        "k": max(1, round(total_neurons * pct)),
                        "baseline": float(np.mean(bases)),
                        "metric": float(vals.mean()), "metric_std": float(vals.std()),
                        "drop": float(drops.mean()), "drop_std": float(drops.std()),
                        "n_seeds": int(n_seeds),
                        "metric_seeds": ";".join(f"{v:.6f}" for v in vals)})
    return out


def aggregate_seed_files(paths) -> list:
    """Combine per-seed sweep files into mean +/- std rows.

    Designed for an INCREMENTAL protocol: each seed is written as its own CSV as soon as
    it finishes, so a run stopped early still leaves usable seeds, and the aggregate can
    be rebuilt at any time from whatever exists. The single-pass file from the first run
    is simply seed 0 (it used _make_eval_sample(SAMPLE_N, 0)), so it can be included here
    rather than discarded.

    Parameters
    ----------
    paths : sequence[str]
        Per-seed CSVs, each with columns method, percentage, k, baseline, metric, drop.

    Returns
    -------
    list[dict]: one row per (method, percentage), with metric/metric_std, drop/drop_std
    and n_seeds reflecting how many files actually contributed.
    """
    import pandas as pd

    frames = []
    for i, p in enumerate(paths):
        df = pd.read_csv(p)
        df = df[df["method"] != "random"]          # the random arm has its own protocol
        df["_seed_file"] = i
        frames.append(df)
    if not frames:
        return []
    allrows = pd.concat(frames, ignore_index=True)

    out = []
    for (method, pct), g in allrows.groupby(["method", "percentage"], sort=False):
        out.append({"method": method, "percentage": float(pct),
                    "k": int(g["k"].iloc[0]),
                    "baseline": float(g["baseline"].mean()),
                    "metric": float(g["metric"].mean()),
                    "metric_std": float(g["metric"].std(ddof=0)),
                    "drop": float(g["drop"].mean()),
                    "drop_std": float(g["drop"].std(ddof=0)),
                    "n_seeds": int(len(g))})
    return sorted(out, key=lambda r: (r["method"], r["percentage"]))


def effectiveness_summary(rows) -> list:
    """Collapse the per-fraction curves into one number per method.

    Two summaries, both standard ways to read an ablation curve:
      - auc_drop      : mean degradation across the swept fractions. Higher = the ranking
                        concentrates more damage overall.
      - frac_to_half  : smallest swept fraction at which the metric falls below half the
                        baseline; None if it never does. Lower = more effective ranking.

    Reporting a scalar matters here: the reviewer asked which ranking is MORE EFFECTIVE,
    and eighteen fractions per arm do not answer that on their own.
    """
    out = []
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    for method, rs in by_method.items():
        rs = sorted(rs, key=lambda r: r["percentage"])
        baseline = rs[0]["baseline"]
        half = baseline / 2.0
        frac = next((r["percentage"] for r in rs if r["metric"] < half), None)
        d = {"method": method,
             "auc_drop": float(np.mean([r["drop"] for r in rs])),
             "frac_to_half": frac,
             "n_points": len(rs)}
        # Multi-seed rows carry drop_std: propagate the spread of the summary statistic
        # (mean over fractions of independent per-fraction stds).
        if all("drop_std" in r for r in rs):
            d["auc_drop_std"] = float(np.sqrt(np.mean([r["drop_std"] ** 2 for r in rs])))
            d["n_seeds"] = rs[0].get("n_seeds")
        out.append(d)
    return sorted(out, key=lambda d: -d["auc_drop"])


# ---------------------------------------------------------------------------
# Self-test (local, no GPU). Simulates a model whose performance depends on a
# known set of "important" neurons, and checks that a ranking which finds them
# beats a ranking that does not, and that both are compared identically.
# ---------------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(0)
    total = 1000
    true_important = set(int(i) for i in rng.choice(total, 100, replace=False))
    baseline = 0.9

    def eval_metric(indices):
        hit = len(set(indices) & true_important) / len(true_important)
        return max(0.0, baseline * (1.0 - hit))

    good = [i for i in sorted(range(total), key=lambda i: i not in true_important)]
    bad = list(range(total))  # arbitrary, ignores the signal

    pcts = [0.05, 0.10, 0.25, 0.50]
    rows = compare_orderings(eval_metric, baseline,
                             {"good": good, "bad": bad}, total, pcts, n_random_seeds=5)
    summ = {d["method"]: d for d in effectiveness_summary(rows)}

    assert summ["good"]["auc_drop"] > summ["bad"]["auc_drop"], summ
    assert summ["good"]["auc_drop"] > summ["random"]["auc_drop"], summ
    assert summ["good"]["frac_to_half"] <= 0.10, summ
    assert len([r for r in rows if r["method"] == "good"]) == len(pcts)

    imp = np.zeros(total)
    for i in true_important:
        imp[i] = 1.0
    assert set(ordering_from_importance(imp)[:100]) == true_important

    print("ranking_effectiveness self-test OK")
    for d in effectiveness_summary(rows):
        print("   ", d)


if __name__ == "__main__":
    _selftest()
