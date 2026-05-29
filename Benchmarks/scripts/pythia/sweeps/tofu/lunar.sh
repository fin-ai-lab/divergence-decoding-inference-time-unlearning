# sweeps/tofu/lunar.sh — LUNAR on TOFU, optimal config.
#
# Edits MLP down_proj weights (layer 22, coeff 2.0) via activation-direction
# perturbation; the output is a plain model evaluated directly. Optimal point:
#   lr=1e-3 (0.001), epochs=20
#
# NOTE: the original repo's dir label "lr-0001" comes from "${lr//./}" on
# lr=0.001 (dots stripped -> "0001"), i.e. learning rate 1e-3 — NOT 1e-4.
# An earlier migration used 1e-4 (10x too small), which produced a no-op edit
# (eval == target). Keep this at 1e-3 to reproduce the paper's LUNAR result.
#
#   model -> models/tofu/lunar/lr-0001
#   eval  -> saves/eval/tofu/lunar/lr-0001
#
# 1 GPU / 8 CPU / 64 GB (defaults).  Trains an edited model -> SYNC_WEIGHTS=1.
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/tofu/lunar.sh

SWEEP_NAME="tofu-lunar"
SWEEP_VALUES=("lr-0001")

SYNC_WEIGHTS="1"

sweep_run_cmd() {
    cat <<'CMD'
set -e

TARGET="open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
FORGET="data/TOFU_downloaded/forget10.jsonl"
RETAIN="data/TOFU_downloaded/retain90.jsonl"
OUT_DIR="models/tofu/lunar/lr-0001"

# Step 1: train (edit) model
if [ -f "${OUT_DIR}/config.json" ]; then
    echo "==> SKIP TOFU LUNAR train (exists)"
else
    echo "==> TOFU LUNAR train: lr=1e-3 layers=22 coeff=2.0"
    python scripts/train/finetune_model_lunar.py \
        --model_dir ${TARGET} \
        --forget_data ${FORGET} \
        --retain_data ${RETAIN} \
        --output_dir ${OUT_DIR} \
        --learning_rate 1e-3 \
        --epochs 20 \
        --layers 22 \
        --coeff 2.0 \
        --batch_size 64 \
        --act_batch_size 4
fi

# Step 2: eval the edited model directly
echo "==> TOFU LUNAR eval: lr=1e-3"
python src/eval.py experiment=eval/tofu/default \
    model.model_args.pretrained_model_name_or_path=${OUT_DIR} \
    retain_logs_path=saves/eval/tofu/baselines/retrain/TOFU_EVAL.json \
    task_name=tofu/lunar/lr-0001
CMD
}
