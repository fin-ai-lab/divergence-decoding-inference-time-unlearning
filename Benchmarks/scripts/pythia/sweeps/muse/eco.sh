# sweeps/muse/eco.sh — ECO (Embedding-COrrupted prompts) on MUSE News, optimal config.
#
# Trains a RoBERTa prompt classifier on forget vs retain, then evaluates the 7B
# target with embedding corruption at inference. Only the optimal point is run:
#   lr=2e-5, strength=50, dims=1, threshold=0.99
#
#   classifier -> models/muse/eco/clf_lr2e-5
#   eval       -> saves/eval/muse/eco/lr-2e-5_str-50
#
# 1 GPU / 8 CPU / 64 GB (defaults).  Trains a classifier -> SYNC_WEIGHTS=1.
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/muse/eco.sh

SWEEP_NAME="muse-eco"
SWEEP_VALUES=("lr-2e-5_str-50")

SYNC_WEIGHTS="1"

sweep_run_cmd() {
    cat <<'CMD'
set -e

TARGET="muse-bench/MUSE-news_target"
FORGET="data/news/raw/forget.txt"
RETAIN="data/news/raw/retain1.txt"
CLF_DIR="models/muse/eco/clf_lr2e-5"

# Step 1: train classifier
if [ -f "${CLF_DIR}/config.json" ]; then
    echo "==> SKIP MUSE ECO classifier (exists)"
else
    echo "==> MUSE ECO classifier: lr=2e-5"
    python scripts/train/finetune_model_eco.py \
        --forget_data ${FORGET} \
        --retain_data ${RETAIN} \
        --output_dir ${CLF_DIR} \
        --learning_rate 2e-5 \
        --epochs 30 \
        --batch_size 16
fi

# Step 2: eval
echo "==> MUSE ECO eval: lr=2e-5 strength=50"
python src/eval.py experiment=eval/muse/default.yaml \
    data_split=News \
    +model.model_handler=ECO \
    +model.model_eco_target=${TARGET} \
    +model.model_eco_classifier=${CLF_DIR} \
    +model.model_eco_strength=50 \
    +model.model_eco_dims=1 \
    +model.model_eco_threshold=0.99 \
    retain_logs_path=saves/eval/muse/baselines/retrain/MUSE_EVAL.json \
    task_name=muse/eco/lr-2e-5_str-50
CMD
}
