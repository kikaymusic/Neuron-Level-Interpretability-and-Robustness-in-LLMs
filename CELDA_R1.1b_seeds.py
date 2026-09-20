# === Ranking-effectiveness comparison, INCREMENTAL SEEDS (R1.1-b, round 2) ===
# Adds seeds to the comparison one at a time, writing each seed to its own CSV as soon as
# it finishes. Stopping the run early therefore loses nothing: the aggregate is rebuilt
# from whatever seed files exist, and the caption reports however many were completed.
#
# The first run already produced seed 0 (it used _make_eval_sample(SAMPLE_N, 0)), so that
# file is reused rather than recomputed -- only SEEDS below are executed.
#
# The rankings are deterministic given the model, so the only source of variance is the
# evaluation sample, which is exactly what the manuscript's multi-seed protocol varies
# (R4.4 / R7.3). Orderings are computed once and reused; within each seed every arm shares
# one evaluation closure, which is what keeps conditions identical across rankings.
#
# PLACE THIS CELL IMMEDIATELY AFTER the attribution cell ([#8]): it reuses imp_ag and
# imp_cond, which that cell never writes to disk.
import ranking_effectiveness, ranking_agreement
import neurox.interpretation.probeless as probeless
from sklearn.metrics import f1_score
import pandas as pd, numpy as np, torch, os, glob, time

# Seed 0 comes from the single-pass run already on disk, so these are the ADDITIONAL
# seeds to compute. [1, 2] gives three seeds in total, which is enough to check whether
# the ordering of the methods is stable; raise to [1, 2, 3, 4] and re-run to add more
# (seeds already written to disk are skipped, so nothing is recomputed).
SEEDS = [1, 2]
RUN_RANKING_EFFECTIVENESS = True

_res = f"{BASE_PATH}/results"

if not RUN_RANKING_EFFECTIVENESS:
    print("[R1.1-b] skipped (RUN_RANKING_EFFECTIVENESS=False)")
else:
    model.load_state_dict(torch.load(weights_path, map_location=model.device), strict=False)
    model.eval()

    def _hooks_for(indices):
        hd = model.config.hidden_size; nl = model.config.num_hidden_layers
        layers = get_encoder_layers(model); handles = []
        for i in range(nl):
            idxs = [idx - i * hd for idx in indices if i * hd <= idx < (i + 1) * hd]
            if idxs:
                tgt = layers[i].output if hasattr(layers[i], "output") else layers[i]
                handles.append(tgt.register_forward_hook(make_cls_silence_hook(idxs)))
        return handles

    def _make_eval(seed):
        _sdf, _yt = _make_eval_sample(SAMPLE_N, seed)

        def _f1(indices):
            h = _hooks_for(indices)
            try:
                p = eval_probs(model, _sdf)
            finally:
                for hh in h: hh.remove()
            return f1_score(_yt, p.argmax(1), average="macro", zero_division=0)

        return _f1, _f1([])

    _total = model.config.hidden_size * model.config.num_hidden_layers

    # --- orderings: computed once, reused by every seed ---------------------------------
    _orderings = {"probe": ranking_effectiveness.ordering_from_importance(
        ranking_agreement.probe_importance(probe))}
    try:
        _orderings["probeless"] = probeless.get_neuron_ordering(X, y)
    except Exception as e:
        print(f"[R1.1-b][WARN] probeless failed ({type(e).__name__}: {e})")
    for _name, _var in [("activation_times_gradient", "imp_ag"), ("conductance", "imp_cond")]:
        if _var in globals():
            _orderings[_name] = ranking_effectiveness.ordering_from_importance(globals()[_var])
        else:
            print(f"[R1.1-b][WARN] {_var} missing; {_name} skipped (needs RUN_ATTRIBUTION=True)")
    print(f"[R1.1-b] total_neurons={_total} arms={list(_orderings)} n_pcts={len(SWEEP_PCTS)}")

    # --- one seed at a time, saved as it goes -------------------------------------------
    for _s in SEEDS:
        _path = f"{_res}/ranking_effectiveness_seed{_s}_malware.csv"
        if os.path.exists(_path):
            print(f"[R1.1-b] seed {_s} already on disk, skipping")
            continue
        _t = time.perf_counter()
        _f1, _base = _make_eval(_s)
        _rows = []
        for _name, _o in _orderings.items():
            for _r in ranking_effectiveness.sweep_ordering(
                    _f1, _base, _o, _total, SWEEP_PCTS, higher_is_better=True):
                _rows.append({"method": _name, **_r})
        pd.DataFrame(_rows).to_csv(_path, index=False)
        print(f"[R1.1-b] seed {_s} done in {time.perf_counter()-_t:.0f}s "
              f"(baseline={_base:.4f}) -> {_path}")

    # --- aggregate whatever exists, including seed 0 from the single-pass run -----------
    _files = []
    _seed0 = f"{_res}/ranking_effectiveness_malware.csv"
    if os.path.exists(_seed0):
        _files.append(_seed0)
    _files += sorted(glob.glob(f"{_res}/ranking_effectiveness_seed*_malware.csv"))

    if _files:
        _agg = ranking_effectiveness.aggregate_seed_files(_files)
        pd.DataFrame(_agg).to_csv(f"{_res}/ranking_effectiveness_agg_malware.csv", index=False)
        _summ = ranking_effectiveness.effectiveness_summary(_agg)
        pd.DataFrame(_summ).to_csv(
            f"{_res}/ranking_effectiveness_agg_summary_malware.csv", index=False)
        print(f"[R1.1-b] aggregated {len(_files)} seed file(s):")
        for _d in _summ:
            print("   ", _d)
    else:
        print("[R1.1-b][WARN] no seed files found to aggregate")
