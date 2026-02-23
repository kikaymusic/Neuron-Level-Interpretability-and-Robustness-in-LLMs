#!/bin/bash

# Example for NGB dataset
datasets=("arxiv" "dm_mathematics" "github" "hackernews" "pile_cc" "pubmed_central" "wikipedia_(en)")
models=("125m")
ngrams=("ngram_7_0.2" "ngram_13_0.2" "ngram_13_0.8")
rounds=("1")

# Training and experimental loop for gpt-neo 125M

for dataset in "${datasets[@]}"; do
  for size in "${models[@]}"; do
    for ngram in "${ngrams[@]}"; do
      for round in "${rounds[@]}"; do
        python ./code/lumia_unimodal_ngb.py -d "$dataset" -m "EleutherAI/gpt-neo-$size" -s "$ngram" -g 1 -r "$round"
      done
    done
  done
done
