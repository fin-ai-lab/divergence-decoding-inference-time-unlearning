# sweeps/tofu/offset.sh — Offset Unlearning on TOFU, optimal config.
#
# Trains one side of a same-init offset pair (GA on forget + KL on retain) from
# the 1B full model, then evaluates via DD (alpha=1.0) with the trained offset as
# the retain model and the untrained 1B full as the forget model. Optimal point:
#   lr=7e-6
#
#   offset -> models/tofu/offset/lr-7e-6
#   eval   -> saves/eval/tofu/offset/lr-7e-6
#
# 1 GPU / 8 CPU / 64 GB (defaults).  Trains the offset model -> SYNC_WEIGHTS=1.
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/tofu/offset.sh

SWEEP_NAME="tofu-offset"
SWEEP_VALUES=("lr-7e-6")

SYNC_WEIGHTS="1"

sweep_run_cmd() {
    cat <<'CMD'
set -e

TARGET="open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
OFFSET_BASE="open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
FORGET="data/TOFU_downloaded/forget10.jsonl"
RETAIN="data/TOFU_downloaded/retain90.jsonl"
OUT_DIR="models/tofu/offset/lr-7e-6"

# Step 1: train offset model
if [ -f "${OUT_DIR}/config.json" ]; then
    echo "==> SKIP TOFU Offset train (exists)"
else
    echo "==> TOFU Offset train: lr=7e-6"
    python scripts/train/finetune_model_offset_unlearning.py \
        --target_model ${TARGET} \
        --offset_model ${OFFSET_BASE} \
        --forget_data ${FORGET} \
        --retain_data ${RETAIN} \
        --output_dir ${OUT_DIR} \
        --learning_rate 7e-6 \
        --epochs 5 \
        --batch_size 2 \
        --gradient_accumulation_steps 8
fi

# Step 2: eval via DD (alpha=1.0)
echo "==> TOFU Offset eval: lr=7e-6"
python src/eval.py experiment=eval/tofu/default \
    +model.model_handler=DD \
    +model.model_dd_big=${TARGET} \
    +model.model_dd_retain=${OUT_DIR} \
    +model.model_dd_forget=${OFFSET_BASE} \
    +model.model_dd_use_ngram=No \
    +model.model_dd_alpha=1.0 \
    +model.model_dd_log_alpha=No \
    retain_logs_path=saves/eval/tofu/baselines/retrain/TOFU_EVAL.json \
    task_name=tofu/offset/lr-7e-6
CMD
}
