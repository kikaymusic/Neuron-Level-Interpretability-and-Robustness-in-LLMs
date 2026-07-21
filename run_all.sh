#!/bin/bash
# Corre en secuencia lo que queda (una sola GPU, sin contención). BigBird ya está
# hecho -> NO se repite. Un solo lanzamiento, desatendido.
cd /workspace
pip install -q -r requirements-spark.txt
export MPLBACKEND=Agg
export HF_HOME=/workspace/.hf_cache

run_malware() {
  local M="$1"
  python - "$M" <<'PY'
import json, sys, re
M = sys.argv[1]; nb = json.load(open("SYNAPSErevision.ipynb"))
for c in nb["cells"]:
    if c["cell_type"] == "code":
        s = "".join(c["source"])
        if re.search(r'^\s*MODEL\s*=\s*"[^"]+"', s, re.M):
            c["source"] = re.sub(r'MODEL\s*=\s*"[^"]+"', f'MODEL = "{M}"', s, count=1).splitlines(keepends=True)
            break
json.dump(nb, open(f"SYNAPSE_{M}.ipynb", "w"), indent=1)
PY
  rm -f  data/$M/activations*.json data/$M/reduced/*.csv data/$M/labels_numeric.txt data/$M/labels_mapping.json
  rm -rf data/$M/results; mkdir -p data/$M/results
  local t=$(date +%s)
  if papermill SYNAPSE_$M.ipynb SYNAPSE_$M.out.ipynb --log-output; then
    echo "$M en $(( ($(date +%s)-t)/60 )) min"
  else
    echo "$M FALLÓ (sigo con el resto)"
  fi
}

run_nb() {
  local NB="$1"; local t=$(date +%s)
  if papermill "$NB" "${NB%.ipynb}.out.ipynb" --log-output; then
    echo "${NB} en $(( ($(date +%s)-t)/60 )) min"
  else
    echo "${NB} FALLÓ (sigo)"
  fi
}

for M in BERT DistilBERT Longformer; do run_malware "$M"; done
run_nb SYNAPSE_goemotions.ipynb
run_nb SYNAPSE_gpt2.ipynb

echo "TODO TERMINADO"
