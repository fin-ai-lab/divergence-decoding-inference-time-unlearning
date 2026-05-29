# sweeps/tofu/uld.sh — ULD (Unlearning from Logit Difference) on TOFU, optimal config.
#
# Builds a LoRA assistant (r=32, alpha=32) on the target with reversed objectives
# (CE on forget + uniform CE on retain), then evaluates the logit subtraction
# target_logits - beta * assistant_logits. Optimal point:
#   lr=2e-3, beta=0.75
#
#   assistant -> models/tofu/uld/lr-2e-3
#   eval      -> saves/eval/tofu/uld/lr-2e-3
#
# 1 GPU / 8 CPU / 64 GB (defaults).  Trains LoRA adapters -> SYNC_WEIGHTS=1.
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/tofu/uld.sh

SWEEP_NAME="tofu-uld"
SWEEP_VALUES=("lr-2e-3")

SYNC_WEIGHTS="1"

sweep_run_cmd() {
    cat <<'CMD'
set -e

TARGET="open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
FORGET="data/TOFU_downloaded/forget10.jsonl"
RETAIN="data/TOFU_downloaded/retain90.jsonl"
OUT_DIR="models/tofu/uld/lr-2e-3"

# Step 1: train LoRA assistant
if [ -d "${OUT_DIR}" ]; then
    echo "==> SKIP TOFU ULD train (exists)"
else
    echo "==> TOFU ULD train: lr=2e-3"
    python scripts/train/finetune_model_uld.py \
        --target_model ${TARGET} \
        --forget_data ${FORGET} \
        --retain_data ${RETAIN} \
        --output_dir ${OUT_DIR} \
        --learning_rate 2e-3 \
        --epochs 5 \
        --batch_size 8 \
        --gradient_accumulation_steps 4 \
        --retain_weight 6.5
fi

# Step 2: eval
echo "==> TOFU ULD eval: lr=2e-3 beta=0.75"
python src/eval.py experiment=eval/tofu/default \
    +model.model_handler=ULD \
    +model.model_uld_target=${TARGET} \
    +model.model_uld_assistant=${OUT_DIR} \
    +model.model_uld_beta=0.75 \
    retain_logs_path=saves/eval/tofu/baselines/retrain/TOFU_EVAL.json \
    task_name=tofu/uld/lr-2e-3
CMD
}
