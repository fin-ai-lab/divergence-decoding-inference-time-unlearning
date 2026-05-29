# sweeps/muse/lunar.sh — LUNAR on MUSE News, optimal config.
#
# Edits MLP down_proj weights (layer 22, coeff 2.0) via activation-direction
# perturbation; the output is a plain model evaluated directly. LUNAR consumes
# the .json data variants. Only the optimal point:
#   lr=0.005 (5e-3), epochs=20
#
#   model -> models/muse/lunar/lr-0005
#   eval  -> saves/eval/muse/lunar/lr-0005
#
# 1 GPU / 8 CPU / 64 GB (defaults).  Trains an edited model -> SYNC_WEIGHTS=1.
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/muse/lunar.sh

SWEEP_NAME="muse-lunar"
SWEEP_VALUES=("lr-0005")

SYNC_WEIGHTS="1"

sweep_run_cmd() {
    cat <<'CMD'
set -e

TARGET="muse-bench/MUSE-news_target"
FORGET="data/news/raw/forget.json"
RETAIN="data/news/raw/retain1.json"
OUT_DIR="models/muse/lunar/lr-0005"

# Step 1: train (edit) model
if [ -f "${OUT_DIR}/config.json" ]; then
    echo "==> SKIP MUSE LUNAR train (exists)"
else
    echo "==> MUSE LUNAR train: lr=0.005 layers=22 coeff=2.0"
    python scripts/train/finetune_model_lunar.py \
        --model_dir ${TARGET} \
        --forget_data ${FORGET} \
        --retain_data ${RETAIN} \
        --output_dir ${OUT_DIR} \
        --learning_rate 0.005 \
        --epochs 20 \
        --layers 22 \
        --coeff 2.0 \
        --batch_size 64 \
        --act_batch_size 2
fi

# Step 2: eval the edited model directly
echo "==> MUSE LUNAR eval: lr=0.005"
python src/eval.py experiment=eval/muse/default.yaml \
    data_split=News \
    model.model_args.pretrained_model_name_or_path=${OUT_DIR} \
    retain_logs_path=saves/eval/muse/baselines/retrain/MUSE_EVAL.json \
    task_name=muse/lunar/lr-0005
CMD
}
