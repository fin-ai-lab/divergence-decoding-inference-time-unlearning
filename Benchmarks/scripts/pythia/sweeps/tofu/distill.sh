# sweeps/tofu/distill.sh — Distill the TOFU DD teacher into the 8B student.
#
# Trains the 8B student to match a frozen DD teacher (8B target + 1B retain/forget,
# teacher alpha=1.5) via KL on the forget10 split, then evaluates the epoch-10
# student checkpoint as a plain model.
#
# Optimal config (only run): lr=4e-5, epochs=10, temperature=1.5.
#
#   train  -> models/tofu/distill/lr-4e-05-temp-1.5/checkpoint-epoch-10
#   eval   -> saves/eval/tofu/distill/lr-4e-05-epoch-10-temp-1.5
#
# 8B student + frozen DD teacher need ~141 GB VRAM -> single H200.
# Trains a model -> SYNC_WEIGHTS=1.
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/tofu/distill.sh

SWEEP_NAME="tofu-distill"
SWEEP_VALUES=("lr-4e-05-temp-1.5")

SBATCH_EXTRA="--gres=gpu:h200:1"
SYNC_WEIGHTS="1"

sweep_run_cmd() {
    cat <<'CMD'
set -e

OUT_DIR="models/tofu/distill/lr-4e-05-temp-1.5"
STUDENT_CKPT="${OUT_DIR}/checkpoint-epoch-10"

# Step 1: distill (saves only the epoch-10 checkpoint)
if [ -f "${STUDENT_CKPT}/config.json" ]; then
    echo "==> SKIP TOFU distill (student checkpoint exists)"
else
    echo "==> TOFU distill: lr=4e-5 epochs=10 temp=1.5 (teacher alpha=1.5)"
    python scripts/distill/distill_model_tofu.py \
        --student_model open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
        --dd_big open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
        --dd_retain open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90 \
        --dd_forget open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        --dd_alpha 1.5 \
        --learning_rate 4e-5 \
        --num_epochs 10 \
        --temperature 1.5 \
        --save_epochs 10 \
        --output_dir "${OUT_DIR}"
fi

# Step 2: eval the distilled student as a plain model
echo "==> TOFU distill eval"
python src/eval.py experiment=eval/tofu/default \
    model=Llama-3.1-8B-Instruct \
    model.model_args.pretrained_model_name_or_path="${STUDENT_CKPT}" \
    retain_logs_path=saves/eval/tofu/baselines/retrain/TOFU_EVAL.json \
    task_name=tofu/distill/lr-4e-05-epoch-10-temp-1.5
CMD
}
