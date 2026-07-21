#!/bin/bash
# Uso:  bash run_model.sh <MODEL>     (BERT | BigBird | DistilBERT | Longformer)
# Corre la suite completa de malware (+ GoEmotions) para UN modelo. Config completa
# (5 semillas, 200 ejemplos held-out, 20 draws).
M="$1"
if [ -z "$M" ]; then echo "Falta el modelo: bash run_model.sh BERT"; exit 1; fi
cd /workspace
pip install -q -r requirements-spark.txt
export MPLBACKEND=Agg
export HF_HOME=/workspace/.hf_cache

# 1) fijar MODEL en una copia del notebook
python - "$M" <<'PY'
import json, sys, re
M = sys.argv[1]
nb = json.load(open("SYNAPSErevision.ipynb"))
done = False
for c in nb["cells"]:
    if c["cell_type"] == "code":
        s = "".join(c["source"])
        if re.search(r'^\s*MODEL\s*=\s*"[^"]+"', s, re.M):
            s = re.sub(r'MODEL\s*=\s*"[^"]+"', f'MODEL = "{M}"', s, count=1)
            c["source"] = s.splitlines(keepends=True); done = True
            break
assert done, "no encontré la línea MODEL=..."
json.dump(nb, open(f"SYNAPSE_{M}.ipynb", "w"), indent=1)
print(f"[run] MODEL = {M}")
PY

# 2) limpiar cachés + resultados de ESTE modelo (regenera desde su CSV)
rm -f  data/$M/activations*.json data/$M/reduced/*.csv \
       data/$M/labels_numeric.txt data/$M/labels_mapping.json
rm -rf data/$M/results; mkdir -p data/$M/results

# 3) correr
START=$(date +%s)
if papermill SYNAPSE_$M.ipynb SYNAPSE_$M.out.ipynb --log-output; then
  MIN=$(( ($(date +%s) - START) / 60 ))
  echo "$M TERMINADO en ${MIN} min"
  exit 0
else
  echo "$M FALLÓ (mira el log)"
  exit 1
fi
