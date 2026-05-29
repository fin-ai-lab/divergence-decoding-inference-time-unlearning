#!/bin/bash
# Finetune MUSE retain/forget models sequentially.
# Each model runs in a separate Python instance to avoid memory issues.
# Run from the repo root (Benchmarks/).

echo "Starting finetune process for all models..."

# Baseline model and output directory
BASELINE_MODEL_1_3B="princeton-nlp/Sheared-LLaMA-1.3B"
MODEL_DIR_1_3B="models/1.3b/"
mkdir -p "$MODEL_DIR_1_3B"

# Data files and the baseline each model finetunes from
declare -a data_files=("data/news/raw/retain1.txt" "data/news/raw/forget.txt" "data/news/scal/forget_2.txt" "data/news/scal/forget_3.txt" "data/news/scal/forget_4.txt" "data/news/sust/forget_2.txt" "data/news/sust/forget_3.txt" "data/news/sust/forget_4.txt")
declare -a baseline_models_1_3b=("$BASELINE_MODEL_1_3B" "$BASELINE_MODEL_1_3B" "$BASELINE_MODEL_1_3B" "$BASELINE_MODEL_1_3B" "$BASELINE_MODEL_1_3B" "${MODEL_DIR_1_3B}model_2" "${MODEL_DIR_1_3B}model_6" "${MODEL_DIR_1_3B}model_7")

for i in {1..2}; do
    echo "Training Model $i (1.3B)..."
    CUDA_VISIBLE_DEVICES=0 python -u scripts/train/finetune_single_model.py $i "${data_files[$((i-1))]}" "${baseline_models_1_3b[$((i-1))]}"
done
