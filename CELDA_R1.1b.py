# === Intervention-effectiveness comparison of neuron rankings (R1.1-b, round 2) ===
# Round 1 compared rankings by AGREEMENT. The reviewer asked for an EFFECTIVENESS
# comparison: the same intervention, driven by different rankings, under identical
# conditions. This cell runs the global-silencing sweep once per ranking, holding the
# evaluation closure, the sample, the fractions and the metric fixed, so the only thing
# that varies between arms is the neuron ordering.
#
# PLACE THIS CELL IMMEDIATELY AFTER the attribution cell ([#8]): it reuses the importance
# vectors that cell computes and never writes to disk.
#
# The random arm is NOT recomputed here: random_control_malware.csv from the previous
# round already holds it, produced by the same closure, the same SWEEP_PCTS and the same
# 20 draws. Recomputing it would cost 360 evaluations per model for numbers already owned.
#
# _f1 / _hooks_for are defined below verbatim as in the random-control cell, so this cell
# also works with RUN_RANDOM_CONTROL=False (that cell keeps its definitions inside its
# else-branch). The definitions are identical on purpose: identical conditions is the
# substance of the reviewer's request.
import ranking_effectiveness, ranking_agreement
import neurox.interpretation.probeless as probeless
from sklearn.metrics import f1_score
import pandas as pd, numpy as np, torch, os

RUN_RANKING_EFFECTIVENESS = True

if not RUN_RANKING_EFFECTIVENESS:
    print("[R1.1-b] skipped (RUN_RANKING_EFFECTIVENESS=False)")
else:
    # --- clean model, exactly as the multi-seed cell does before evaluating ---
    model.load_state_dict(torch.load(weights_path, map_location=model.device), strict=False)
    model.eval()

    _sdf, _yt = _make_eval_sample(SAMPLE_N, 0)

    def _hooks_for(indices):
        hd = model.config.hidden_size; nl = model.config.num_hidden_layers
        layers = get_encoder_layers(model); handles = []
        for i in range(nl):
            idxs = [idx - i * hd for idx in indices if i * hd <= idx < (i + 1) * hd]
            if idxs:
                tgt = layers[i].output if hasattr(layers[i], "output") else layers[i]
                handles.append(tgt.register_forward_hook(make_cls_silence_hook(idxs)))
        return handles

    def _f1(indices):
        h = _hooks_for(indices)
        try:
            p = eval_probs(model, _sdf)
        finally:
            for hh in h: hh.remove()
        return f1_score(_yt, p.argmax(1), average="macro", zero_division=0)

    _total = model.config.hidden_size * model.config.num_hidden_layers
    _base = _f1([])
    print(f"[R1.1-b] total_neurons={_total} baseline_macroF1={_base:.4f} "
          f"n_pcts={len(SWEEP_PCTS)}")

    _orderings = {}

    # 1) probe -- SYNAPSE's operational ranking
    _orderings["probe"] = ranking_effectiveness.ordering_from_importance(
        ranking_agreement.probe_importance(probe))

    # 2) Probeless (NeuroX) -- existing method, trains no probe at all. Returns an
    #    ORDERING already, so it is passed through without conversion.
    try:
        _orderings["probeless"] = probeless.get_neuron_ordering(X, y)
        print(f"[R1.1-b] probeless ordering: {len(_orderings['probeless'])} neurons")
    except Exception as e:
        print(f"[R1.1-b][WARN] probeless failed ({type(e).__name__}: {e})")

    # 3-4) the two attribution references computed by the [#8] cell
    for _name, _var in [("activation_times_gradient", "imp_ag"), ("conductance", "imp_cond")]:
        if _var in globals():
            _orderings[_name] = ranking_effectiveness.ordering_from_importance(globals()[_var])
        else:
            print(f"[R1.1-b][WARN] {_var} not available; {_name} arm skipped "
                  f"(needs RUN_ATTRIBUTION=True in this same execution)")

    # n_random_seeds=0 -> no random arm here; it is merged later from
    # random_control_malware.csv, which used this same closure and sweep.
    _rows = ranking_effectiveness.compare_orderings(
        _f1, _base, _orderings, _total, SWEEP_PCTS,
        n_random_seeds=0, base_seed=0, higher_is_better=True)

    os.makedirs(f"{BASE_PATH}/results", exist_ok=True)
    _out = f"{BASE_PATH}/results/ranking_effectiveness_malware.csv"
    pd.DataFrame(_rows).to_csv(_out, index=False)

    _summ = ranking_effectiveness.effectiveness_summary(_rows)
    pd.DataFrame(_summ).to_csv(
        f"{BASE_PATH}/results/ranking_effectiveness_summary_malware.csv", index=False)

    print(f"[R1.1-b] saved -> {_out}")
    print("[R1.1-b] summary (higher auc_drop = ranking concentrates more damage):")
    for _d in _summ:
        print("   ", _d)
