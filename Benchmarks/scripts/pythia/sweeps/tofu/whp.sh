# sweeps/tofu/whp.sh — WHP (Who's Harry Potter) on TOFU, optimal config.
#
# Finetunes a reinforced model on the forget set, then evaluates the contrastive
# decode  baseline - alpha * ReLU(reinforced - baseline). Only the optimal point:
#   reinforced lr=1e-5, epochs=10 ; alpha=3.0
#
#   reinforced -> models/tofu/whp/lr-1e-5
#   eval       -> saves/eval/tofu/whp/lr-1e-5_alpha-3_0
#
# 1 GPU / 8 CPU / 64 GB (defaults).  Trains the reinforced model -> SYNC_WEIGHTS=1.
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/tofu/whp.sh

SWEEP_NAME="tofu-whp"
SWEEP_VALUES=("lr-1e-5_alpha-3_0")

SYNC_WEIGHTS="1"

sweep_run_cmd() {
    cat <<'CMD'
set -e

TARGET="open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
FORGET="data/TOFU_downloaded/forget10.jsonl"
OUT_DIR="models/tofu/whp/lr-1e-5"

# Step 1: finetune reinforced model
if [ -f "${OUT_DIR}/config.json" ]; then
    echo "==> SKIP TOFU WHP reinforced (exists)"
else
    echo "==> TOFU WHP reinforced finetune: lr=1e-5"
    python scripts/train/finetune_model_whp.py \
        --model_dir ${TARGET} \
        --forget_data ${FORGET} \
        --output_dir ${OUT_DIR} \
        --learning_rate 1e-5 \
        --epochs 10 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --max_len 2048
fi

# Step 2: eval
echo "==> TOFU WHP eval: alpha=3.0"
python src/eval.py experiment=eval/tofu/default \
    +model.model_handler=WHP \
    +model.model_whp_baseline=${TARGET} \
    +model.model_whp_reinforced=${OUT_DIR} \
    +model.model_whp_alpha=3.0 \
    retain_logs_path=saves/eval/tofu/baselines/retrain/TOFU_EVAL.json \
    task_name=tofu/whp/lr-1e-5_alpha-3_0
CMD
}
