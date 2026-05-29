# sweeps/tofu/eco.sh — ECO (Embedding-COrrupted prompts) on TOFU, optimal config.
#
# Trains a RoBERTa prompt classifier on forget vs retain, then evaluates the 8B
# target with embedding corruption at inference. Only the optimal point is run:
#   lr=1e-5, strength=50, dims=1, threshold=0.99
#
#   classifier -> models/tofu/eco/clf_lr1e-5
#   eval       -> saves/eval/tofu/eco/lr-1e-5_str-50
#
# 1 GPU / 8 CPU / 64 GB (defaults).  Trains a classifier -> SYNC_WEIGHTS=1.
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/tofu/eco.sh

SWEEP_NAME="tofu-eco"
SWEEP_VALUES=("lr-1e-5_str-50")

SYNC_WEIGHTS="1"

sweep_run_cmd() {
    cat <<'CMD'
set -e

TARGET="open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
FORGET="data/TOFU_downloaded/forget10.jsonl"
RETAIN="data/TOFU_downloaded/retain90.jsonl"
CLF_DIR="models/tofu/eco/clf_lr1e-5"

# Step 1: train classifier
if [ -f "${CLF_DIR}/config.json" ]; then
    echo "==> SKIP TOFU ECO classifier (exists)"
else
    echo "==> TOFU ECO classifier: lr=1e-5"
    python scripts/train/finetune_model_eco.py \
        --forget_data ${FORGET} \
        --retain_data ${RETAIN} \
        --output_dir ${CLF_DIR} \
        --learning_rate 1e-5 \
        --epochs 30 \
        --batch_size 16
fi

# Step 2: eval
echo "==> TOFU ECO eval: lr=1e-5 strength=50"
python src/eval.py experiment=eval/tofu/default \
    +model.model_handler=ECO \
    +model.model_eco_target=${TARGET} \
    +model.model_eco_classifier=${CLF_DIR} \
    +model.model_eco_strength=50 \
    +model.model_eco_dims=1 \
    +model.model_eco_threshold=0.99 \
    retain_logs_path=saves/eval/tofu/baselines/retrain/TOFU_EVAL.json \
    task_name=tofu/eco/lr-1e-5_str-50
CMD
}
