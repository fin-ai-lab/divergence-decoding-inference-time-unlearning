# sweeps/muse/distill.sh — Distill the MUSE DD teacher into the 7B student.
#
# Trains the 7B MUSE target to match a frozen DD teacher (7B target + 1.3b
# retain/forget verifiers) via KL on the forget text, then evaluates the
# epoch-5 student checkpoint as a plain model. Two per-metric optima:
#   temp=2.0, teacher alpha=0.9  (verbmem-optimal)
#   temp=1.5, teacher alpha=0.8  (knowmem-optimal)
# both at lr=1.25e-4, epochs=5.
#
#   train -> models/muse/distill/lr-1.25e-4-epoch-5-temp-<T>/checkpoint-epoch-5
#   eval  -> saves/eval/muse/distill/lr-1.25e-4-epoch-5-temp-<T>
#
# 7B student + frozen DD teacher need ~141 GB VRAM -> single H200.
# Trains models -> SYNC_WEIGHTS=1.
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/muse/distill.sh

SWEEP_NAME="muse-distill"
SWEEP_VALUES=("temp-2.0" "temp-1.5")

SBATCH_EXTRA="--gres=gpu:h200:1"
SYNC_WEIGHTS="1"

sweep_run_cmd() {
    local VAL="$1"
    local TEMP="${VAL#temp-}"
    local ALPHA
    case "${TEMP}" in
        2.0) ALPHA=0.9 ;;
        1.5) ALPHA=0.8 ;;
    esac
    cat <<CMD
set -e

OUT_DIR="models/muse/distill/lr-1.25e-4-epoch-5-temp-${TEMP}"
STUDENT_CKPT="\${OUT_DIR}/checkpoint-epoch-5"

# Step 1: distill (saves only the epoch-5 checkpoint)
if [ -f "\${STUDENT_CKPT}/config.json" ]; then
    echo "==> SKIP MUSE distill temp=${TEMP} (student checkpoint exists)"
else
    echo "==> MUSE distill: lr=1.25e-4 epochs=5 temp=${TEMP} (teacher alpha=${ALPHA})"
    python scripts/distill/distill_model_muse.py \\
        --student_model muse-bench/MUSE-news_target \\
        --dd_big muse-bench/MUSE-news_target \\
        --dd_retain models/muse/verifiers/1.3b/model_1 \\
        --dd_forget models/muse/verifiers/1.3b/model_2 \\
        --dd_alpha ${ALPHA} \\
        --data_path data/news/raw/forget.txt \\
        --learning_rate 1.25e-4 \\
        --num_epochs 5 \\
        --temperature ${TEMP} \\
        --save_epochs 5 \\
        --output_dir "\${OUT_DIR}"
fi

# Step 2: eval the distilled student as a plain model
echo "==> MUSE distill eval: temp=${TEMP}"
python src/eval.py experiment=eval/muse/default.yaml \\
    data_split=News \\
    model=Llama-2-7b-hf \\
    model.model_args.pretrained_model_name_or_path="\${STUDENT_CKPT}" \\
    retain_logs_path=saves/eval/muse/baselines/retrain/MUSE_EVAL.json \\
    task_name=muse/distill/lr-1.25e-4-epoch-5-temp-${TEMP}
CMD
}
