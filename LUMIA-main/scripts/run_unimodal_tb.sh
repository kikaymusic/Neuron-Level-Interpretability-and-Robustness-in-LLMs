#!/bin/bash

# Example for NGB dataset
datasets=("wikimia")
models=("125m")
rounds=("1")

# Training and experimental loop for gpt-neo 125M

for dataset in "${datasets[@]}"; do
  for size in "${models[@]}"; do
      for round in "${rounds[@]}"; do
        python ./code/lumia_unimodal_tb.py -d "$dataset" -m "EleutherAI/gpt-neo-$size" -s "" -g 1 -r "$round"
    done
  done
done
