#!/bin/bash
# Uso: bash run_nb.sh <notebook.ipynb>   (para SYNAPSE_goemotions.ipynb o SYNAPSE_gpt2.ipynb)
NB="$1"
if [ -z "$NB" ]; then echo "Falta el notebook: bash run_nb.sh SYNAPSE_goemotions.ipynb"; exit 1; fi
cd /workspace
pip install -q -r requirements-spark.txt
export MPLBACKEND=Agg
export HF_HOME=/workspace/.hf_cache
OUT="${NB%.ipynb}.out.ipynb"
START=$(date +%s)
if papermill "$NB" "$OUT" --log-output; then
  MIN=$(( ($(date +%s) - START) / 60 ))
  echo "${NB} TERMINADO en ${MIN} min"
else
  echo "${NB} FALLÓ"; exit 1
fi
