# sweeps/muse/uld.sh — ULD (Unlearning from Logit Difference) on MUSE News.
#
# Builds a LoRA assistant (r=32, alpha=32) on the 7B target with reversed
# objectives, then evaluates target_logits - beta * assistant_logits (beta=0.5).
# Two optima -> one SLURM job each:
#   lr=2e-3 -> saves/eval/muse/uld/lr-2e-3
#   lr=5e-3 -> saves/eval/muse/uld/lr-5e-3
#
# 1 GPU / 8 CPU / 64 GB (defaults).  Trains LoRA adapters -> SYNC_WEIGHTS=1.
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/muse/uld.sh

SWEEP_NAME="muse-uld"
SWEEP_VALUES=("2e-3" "5e-3")

SYNC_WEIGHTS="1"

sweep_run_cmd() {
    local LR="$1"
    cat <<CMD
set -e

TARGET="muse-bench/MUSE-news_target"
FORGET="data/news/raw/forget.txt"
RETAIN="data/news/raw/retain1.txt"
OUT_DIR="models/muse/uld/lr-${LR}"

# Step 1: train LoRA assistant
if [ -d "\${OUT_DIR}" ]; then
    echo "==> SKIP MUSE ULD train lr=${LR} (exists)"
else
    echo "==> MUSE ULD train: lr=${LR}"
    python scripts/train/finetune_model_uld.py \\
        --target_model \${TARGET} \\
        --forget_data \${FORGET} \\
        --retain_data \${RETAIN} \\
        --output_dir \${OUT_DIR} \\
        --learning_rate ${LR} \\
        --epochs 5 \\
        --batch_size 8 \\
        --gradient_accumulation_steps 4 \\
        --retain_weight 6.5
fi

# Step 2: eval
echo "==> MUSE ULD eval: lr=${LR} beta=0.5"
python src/eval.py experiment=eval/muse/default.yaml \\
    data_split=News \\
    +model.model_handler=ULD \\
    +model.model_uld_target=\${TARGET} \\
    +model.model_uld_assistant=\${OUT_DIR} \\
    +model.model_uld_beta=0.5 \\
    retain_logs_path=saves/eval/muse/baselines/retrain/MUSE_EVAL.json \\
    task_name=muse/uld/lr-${LR}
CMD
}
