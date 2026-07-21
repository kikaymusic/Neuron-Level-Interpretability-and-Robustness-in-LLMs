#!/bin/bash
# Calibración de tiempo: BigBird a config real (probe 1000, 5 semillas, eval 200).
# Pensado para correr DETACHED en la Spark.
cd /workspace
pip install -q -r requirements-spark.txt
export MPLBACKEND=Agg
export HF_HOME=/workspace/.hf_cache

# === Limpieza para una calibración limpia ===
# Borra OUTPUTS y CACHÉS que el notebook regenera. NO toca inputs (best_model_*.pth,
# *_tokens_PT.csv, ni los datos de GoEmotions: sample/labels/tsv).
rm -f  data/BigBird/activations*.json
rm -f  data/BigBird/reduced/BigBird_tokens_reduced.csv \
       data/BigBird/labels_numeric.txt \
       data/BigBird/labels_mapping.json
rm -rf data/BigBird/results
rm -f  data/goemotions/activations.json data/goemotions/activations_mean*.json
rm -rf data/goemotions/results
rm -f  SYNAPSErevision.out.ipynb
mkdir -p data/BigBird/results data/goemotions/results

START=$(date +%s)
if papermill SYNAPSErevision.ipynb SYNAPSErevision.out.ipynb --log-output; then
  MIN=$(( ($(date +%s) - START) / 60 ))
  echo "BigBird calibración TERMINADA en ${MIN} min"
  exit 0
else
  echo "BigBird calibración FALLÓ (mira run_bbcal.log)"
  exit 1
fi
