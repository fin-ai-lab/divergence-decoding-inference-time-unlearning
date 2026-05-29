# sweeps/muse/guard.sh — GUARD on MUSE News, optimal configs.
#
# Trains an MLP prompt classifier (lr=1e-3) on penultimate-layer embeddings, then
# evaluates GUARD's constrained beam search at two SBERT delta optima -> one
# SLURM job each (both reuse the same classifier, trained idempotently):
#   delta=0.3 -> saves/eval/muse/guard/lr-1e-3_delta-0_3
#   delta=0.7 -> saves/eval/muse/guard/lr-1e-3_delta-0_7
#
#   classifier -> models/muse/guard/clf_lr1e-3
#
# 1 GPU / 8 CPU / 64 GB (defaults).  Trains a classifier -> SYNC_WEIGHTS=1.
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/muse/guard.sh

SWEEP_NAME="muse-guard"
SWEEP_VALUES=("0.3" "0.7")

SYNC_WEIGHTS="1"

sweep_run_cmd() {
    local DELTA="$1"
    local DELTA_STR="${DELTA//./_}"
    cat <<CMD
set -e

TARGET="muse-bench/MUSE-news_target"
FORGET="data/news/raw/forget.txt"
RETAIN="data/news/raw/retain1.txt"
CLF_DIR="models/muse/guard/clf_lr1e-3"

# Step 1: train MLP classifier (shared across deltas; idempotent)
if [ -f "\${CLF_DIR}/classifier.pt" ]; then
    echo "==> SKIP MUSE GUARD classifier (exists)"
else
    echo "==> MUSE GUARD classifier: lr=1e-3"
    python scripts/train/finetune_model_guard.py \\
        --model_dir \${TARGET} \\
        --forget_data \${FORGET} \\
        --retain_data \${RETAIN} \\
        --output_dir \${CLF_DIR} \\
        --learning_rate 1e-3 \\
        --epochs 50 \\
        --batch_size 32 \\
        --mlp_hidden 256 \\
        --embed_batch_size 8
fi

# Step 2: eval at this delta
echo "==> MUSE GUARD eval: lr=1e-3 delta=${DELTA}"
python src/eval.py experiment=eval/muse/default.yaml \\
    data_split=News \\
    +model.model_handler=GUARD \\
    +model.model_guard_target=\${TARGET} \\
    +model.model_guard_classifier=\${CLF_DIR} \\
    +model.model_guard_beam_width=7 \\
    +model.model_guard_beta=1.0 \\
    +model.model_guard_delta=${DELTA} \\
    +model.model_guard_threshold=0.5 \\
    retain_logs_path=saves/eval/muse/baselines/retrain/MUSE_EVAL.json \\
    task_name=muse/guard/lr-1e-3_delta-${DELTA_STR}
CMD
}
