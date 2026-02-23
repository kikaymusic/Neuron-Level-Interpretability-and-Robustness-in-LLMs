#!/bin/bash

# Define variables
rounds=("0")
sizes=("0.5b")
datasets=("science" "ao2" "chartqa" "magpie" "iconqa" "textcap" "math")

for round in "${rounds[@]}"; do
  for size in "${sizes[@]}"; do
    for dataset in "${datasets[@]}"; do
        python ./code/lumia_multimodal.py \
          -d "${dataset}" \
          -m "llava-hf/llava-onevision-qwen2-${size}-si-hf" \
          -g 0 \
          -r "${round}" \
          -s 500
    done
  done
done