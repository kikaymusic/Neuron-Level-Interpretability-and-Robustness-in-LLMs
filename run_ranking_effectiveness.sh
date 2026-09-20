#!/bin/bash
# Round-2 revision: intervention-effectiveness comparison of neuron rankings.
#
# Repeats the global silencing sweep with the neuron ordering supplied by four ranking
# criteria (probe, Probeless, conductance, activation x gradient) under identical
# conditions, over five seeds, and aggregates the per-seed results.
#
# Run inside the project container, from the project root:
#
#   bash run_ranking_effectiveness.sh
#
# Seeds already present on disk are skipped, so the script can be re-run to add seeds.
# It never deletes activations or previous results.

set -u
cd "$(dirname "$0")"
exec > >(tee -a ranking_effectiveness_run.log) 2>&1

echo "===== $(date) ====="
pip install -q -r requirements-spark.txt

if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ABORTED: CUDA is not available"
    exit 1
fi

export HF_HUB_OFFLINE=1
export HF_HOME="$PWD/.hf_cache"
export MPLBACKEND=Agg

MODELS="DistilBERT BERT BigBird Longformer"
SEEDS="${SEEDS:-0,1,2,3,4}"

for M in $MODELS; do
    python3 make_nb_r1_1b.py --seeds --seeds-list "$SEEDS" "$M" || exit 1
done

for M in $MODELS; do
    echo "===== $M  $(date)"
    if papermill "R11bS_$M.ipynb" "R11bS_$M.out.ipynb" --log-output > "ranking_effectiveness_$M.log" 2>&1; then
        cat "data/$M/results/ranking_effectiveness_agg_summary_malware.csv" 2>/dev/null
    else
        echo "FAILED: $M"
        tail -25 "ranking_effectiveness_$M.log"
    fi
done
