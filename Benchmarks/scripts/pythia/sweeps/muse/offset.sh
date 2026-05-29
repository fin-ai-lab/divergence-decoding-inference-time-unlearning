# sweeps/muse/offset.sh — Offset Unlearning on MUSE News, optimal configs.
#
# Trains a same-init offset model (GA on forget + KL on retain) from the 1.3B
# base, then evaluates via DD (alpha=1.0). Two per-metric optima -> one SLURM
# job each:
#   lr=1e-5 -> saves/eval/muse/offset/lr-1e-5
#   lr=5e-5 -> saves/eval/muse/offset/lr-5e-5
#
# 1 GPU / 8 CPU / 64 GB (defaults).  Trains the offset model -> SYNC_WEIGHTS=1.
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/muse/offset.sh

SWEEP_NAME="muse-offset"
SWEEP_VALUES=("1e-5" "5e-5")

SYNC_WEIGHTS="1"

sweep_run_cmd() {
    local LR="$1"
    cat <<CMD
set -e

TARGET="muse-bench/MUSE-news_target"
OFFSET_BASE="princeton-nlp/Sheared-LLaMA-1.3B"
FORGET="data/news/raw/forget.txt"
RETAIN="data/news/raw/retain1.txt"
OUT_DIR="models/muse/offset/lr-${LR}"

# Step 1: train offset model
if [ -f "\${OUT_DIR}/config.json" ]; then
    echo "==> SKIP MUSE Offset train lr=${LR} (exists)"
else
    echo "==> MUSE Offset train: lr=${LR}"
    python scripts/train/finetune_model_offset_unlearning.py \\
        --target_model \${TARGET} \\
        --offset_model \${OFFSET_BASE} \\
        --forget_data \${FORGET} \\
        --retain_data \${RETAIN} \\
        --output_dir \${OUT_DIR} \\
        --learning_rate ${LR} \\
        --epochs 5 \\
        --batch_size 2 \\
        --gradient_accumulation_steps 8
fi

# Step 2: eval via DD (alpha=1.0)
echo "==> MUSE Offset eval: lr=${LR}"
python src/eval.py experiment=eval/muse/default.yaml \\
    data_split=News \\
    +model.model_handler=DD \\
    +model.model_dd_big=\${TARGET} \\
    +model.model_dd_retain=\${OUT_DIR} \\
    +model.model_dd_forget=\${OFFSET_BASE} \\
    +model.model_dd_use_ngram=No \\
    +model.model_dd_alpha=1.0 \\
    +model.model_dd_log_alpha=No \\
    retain_logs_path=saves/eval/muse/baselines/retrain/MUSE_EVAL.json \\
    task_name=muse/offset/lr-${LR}
CMD
}
