# sweeps/muse/whp.sh — WHP (Who's Harry Potter) on MUSE News, optimal config.
#
# Finetunes a reinforced model on the forget text (lr=1e-5, epochs=10), then
# evaluates the contrastive decode baseline - alpha * ReLU(reinforced - baseline)
# at the optimal alpha=2.0.
#
#   reinforced -> models/muse/whp/news
#   eval       -> saves/eval/muse/whp/alpha-2_0
#
# 1 GPU / 8 CPU / 64 GB (defaults).  Trains the reinforced model -> SYNC_WEIGHTS=1.
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/muse/whp.sh

SWEEP_NAME="muse-whp"
SWEEP_VALUES=("alpha-2_0")

SYNC_WEIGHTS="1"

sweep_run_cmd() {
    cat <<'CMD'
set -e

TARGET="muse-bench/MUSE-news_target"
FORGET="data/news/raw/forget.txt"
OUT_DIR="models/muse/whp/news"

# Step 1: finetune reinforced model
if [ -f "${OUT_DIR}/config.json" ]; then
    echo "==> SKIP MUSE WHP reinforced (exists)"
else
    echo "==> MUSE WHP reinforced finetune: lr=1e-5 epochs=10"
    python scripts/train/finetune_model_whp.py \
        --model_dir ${TARGET} \
        --forget_data ${FORGET} \
        --output_dir ${OUT_DIR} \
        --learning_rate 1e-5 \
        --epochs 10 \
        --batch_size 4 \
        --max_len 2048
fi

# Step 2: eval
echo "==> MUSE WHP eval: alpha=2.0"
python src/eval.py experiment=eval/muse/default.yaml \
    data_split=News \
    +model.model_handler=WHP \
    +model.model_whp_baseline=${TARGET} \
    +model.model_whp_reinforced=${OUT_DIR} \
    +model.model_whp_alpha=2.0 \
    retain_logs_path=saves/eval/muse/baselines/retrain/MUSE_EVAL.json \
    task_name=muse/whp/alpha-2_0
CMD
}
