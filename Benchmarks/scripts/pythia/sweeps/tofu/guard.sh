# sweeps/tofu/guard.sh — GUARD on TOFU, optimal config.
#
# Trains an MLP prompt classifier on penultimate-layer embeddings, then evaluates
# GUARD's constrained beam search (trie + SBERT penalties). Only the optimal point:
#   lr=1e-3, delta=0.3, beam_width=7, beta=1.0, threshold=0.5
#
#   classifier -> models/tofu/guard/clf_lr1e-3
#   eval       -> saves/eval/tofu/guard/lr-1e-3_delta-0_3
#
# 1 GPU / 8 CPU / 64 GB (defaults).  Trains a classifier -> SYNC_WEIGHTS=1.
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/tofu/guard.sh

SWEEP_NAME="tofu-guard"
SWEEP_VALUES=("lr-1e-3_delta-0_3")

SYNC_WEIGHTS="1"

sweep_run_cmd() {
    cat <<'CMD'
set -e

TARGET="open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
FORGET="data/TOFU_downloaded/forget10.jsonl"
RETAIN="data/TOFU_downloaded/retain90.jsonl"
CLF_DIR="models/tofu/guard/clf_lr1e-3"

# Step 1: train MLP classifier
if [ -f "${CLF_DIR}/classifier.pt" ]; then
    echo "==> SKIP TOFU GUARD classifier (exists)"
else
    echo "==> TOFU GUARD classifier: lr=1e-3"
    python scripts/train/finetune_model_guard.py \
        --model_dir ${TARGET} \
        --forget_data ${FORGET} \
        --retain_data ${RETAIN} \
        --output_dir ${CLF_DIR} \
        --learning_rate 1e-3 \
        --epochs 50 \
        --batch_size 32 \
        --mlp_hidden 256 \
        --embed_batch_size 8
fi

# Step 2: eval
echo "==> TOFU GUARD eval: lr=1e-3 delta=0.3"
python src/eval.py experiment=eval/tofu/default \
    +model.model_handler=GUARD \
    +model.model_guard_target=${TARGET} \
    +model.model_guard_classifier=${CLF_DIR} \
    +model.model_guard_beam_width=7 \
    +model.model_guard_beta=1.0 \
    +model.model_guard_delta=0.3 \
    +model.model_guard_threshold=0.5 \
    retain_logs_path=saves/eval/tofu/baselines/retrain/TOFU_EVAL.json \
    task_name=tofu/guard/lr-1e-3_delta-0_3
CMD
}
