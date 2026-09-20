#!/usr/bin/env python3
"""
Build the trimmed notebook for the R1.1-b run (round 2).

The full SYNAPSErevision notebook takes ~9 min on BERT/DistilBERT but ~8-9.5 h on
BigBird/Longformer, because cells 33..93 run the whole original attack suite over long
sequences. None of that is needed to answer R1.1-b, and all of it has already been run
and reported.

This script keeps only what the comparison needs, WITHOUT editing any kept cell, so the
code that runs is byte-identical to the code that produced the published results:

    cells 0..32  setup, model, activations, probe, hook helpers, get_encoder_layers
    cell 94      eval_probs / _make_eval_sample (definitions precede the RUN_ gate)
    cell 102     attribution: imp_ag / imp_cond over the same neuron space
    + CELDA_R1.1b.py appended as the final cell

It also forces the cost knobs off for the experiments already completed in round 1, and
rewrites MODEL so one notebook is produced per model.

Usage:
    python3 make_nb_r1_1b.py BERT DistilBERT BigBird Longformer
"""
import json, re, sys

SRC = "SYNAPSErevision.ipynb"
# Set NEW_CELL=CELDA_R1.1b_seeds.py (or pass --seeds) for the multi-seed version.
NEW_CELL = "CELDA_R1.1b.py"
KEEP = list(range(0, 33)) + [94, 102]

# Round-1 experiments: already done, already reported. RUN_ATTRIBUTION stays True because
# the new cell consumes imp_ag / imp_cond, which are never persisted to disk.
KNOBS = {
    "RUN_DETECTION_METRICS": "False",
    "RUN_RANDOM_CONTROL": "False",
    "RUN_MEANPOOL_ABLATION": "False",
    "RUN_BITFLIP": "False",
    "RUN_ATTRIBUTION": "True",
}


def build(model_name: str, new_cell: str = NEW_CELL, prefix: str = "R11b_") -> str:
    nb = json.load(open(SRC))
    cells = nb["cells"]
    if max(KEEP) >= len(cells):
        raise SystemExit(f"{SRC} has {len(cells)} cells; expected at least {max(KEEP)+1}. "
                         "The notebook changed -- re-check the KEEP indices before running.")

    kept = [cells[i] for i in KEEP]

    for c in kept:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        s = re.sub(r'^MODEL\s*=\s*"[^"]+"', f'MODEL = "{model_name}"', s, count=1, flags=re.M)
        for knob, val in KNOBS.items():
            s = re.sub(rf'^{knob}\s*=\s*\w+', f'{knob} = {val}', s, count=1, flags=re.M)
        # Conductance (integrated gradients over every layer) is the memory peak of the
        # whole pipeline. On a unified-memory machine it is what gets OOM-killed for the
        # 4096-token models, so allow shrinking its budget per model.
        if ATTR_SAMPLES is not None:
            s = re.sub(r'^ATTRIBUTION_N_SAMPLES\s*=\s*\d+',
                       f'ATTRIBUTION_N_SAMPLES = {ATTR_SAMPLES}', s, count=1, flags=re.M)
        if ATTR_STEPS is not None:
            s = re.sub(r'^ATTRIBUTION_STEPS\s*=\s*\d+',
                       f'ATTRIBUTION_STEPS = {ATTR_STEPS}', s, count=1, flags=re.M)
        c["source"] = s.splitlines(keepends=True)
        c["outputs"] = []
        c["execution_count"] = None

    new_src = open(new_cell).read()
    # Which seeds this model still needs. BERT/DistilBERT reuse seed 0 from the
    # single-pass run and add 1,2; a model with no single-pass file needs 0,1,2 so that
    # every model ends up on the same three seeds. SEEDS lives in the appended cell, so
    # it is rewritten here rather than in the loop above.
    if SEED_LIST is not None:
        new_src = re.sub(r'^SEEDS\s*=\s*\[[^\]]*\]', f'SEEDS = {SEED_LIST}',
                         new_src, count=1, flags=re.M)

    kept.append({"cell_type": "code", "metadata": {}, "outputs": [],
                 "execution_count": None,
                 "source": new_src.splitlines(keepends=True)})

    nb["cells"] = kept
    out = f"{prefix}{model_name}.ipynb"
    json.dump(nb, open(out, "w"), indent=1)
    return out


ATTR_SAMPLES = None   # override ATTRIBUTION_N_SAMPLES (default 32) via --attr-samples N
ATTR_STEPS = None     # override ATTRIBUTION_STEPS      (default 20) via --attr-steps N
SEED_LIST = None      # override SEEDS in the multi-seed cell via --seeds-list 0,1,2


if __name__ == "__main__":
    args = sys.argv[1:]
    for flag, var in [("--attr-samples", "ATTR_SAMPLES"), ("--attr-steps", "ATTR_STEPS")]:
        if flag in args:
            i = args.index(flag)
            globals()[var] = int(args[i + 1])
            del args[i:i + 2]
            print(f"[{flag}] {globals()[var]}")
    if "--seeds-list" in args:
        i = args.index("--seeds-list")
        SEED_LIST = [int(x) for x in args[i + 1].split(",")]
        del args[i:i + 2]
        print(f"[--seeds-list] {SEED_LIST}")
    if "--seeds" in args:
        args.remove("--seeds")
        NEW_CELL = "CELDA_R1.1b_seeds.py"
        PREFIX = "R11bS_"
        print(f"[multi-seed] using {NEW_CELL}")
    else:
        PREFIX = "R11b_"
    models = args or ["BERT", "DistilBERT", "BigBird", "Longformer"]
    for m in models:
        print(f"{build(m, NEW_CELL, PREFIX)}  "
              f"({len(KEEP)+1} cells, from {len(json.load(open(SRC))['cells'])})")
