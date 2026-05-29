# sweeps/muse/verifiers.sh — Train the MUSE DD verifier (retain/forget) models.
#
# Finetunes Sheared-LLaMA base checkpoints into the per-size retain/forget pair
# used as the DD retain/forget models for MUSE. One SLURM job per (size, model_n):
#   model_1 -> retain (data/news/raw/retain1.txt)
#   model_2 -> forget (data/news/raw/forget.txt)
#
# finetune_single_model.py writes to models/<size>/model_<n> (derived from the
# baseline name); we relocate it to the documented verifier path afterwards.
#
#   models/muse/verifiers/1.3b/{model_1,model_2}
#   models/muse/verifiers/2.7b/{model_1,model_2}
#
# 1 GPU / 8 CPU / 64 GB (defaults).  Trains verifiers -> SYNC_WEIGHTS=1.
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/muse/verifiers.sh

SWEEP_NAME="muse-verifiers"
SWEEP_VALUES=("1.3b-1" "1.3b-2" "2.7b-1" "2.7b-2")

SYNC_WEIGHTS="1"

sweep_run_cmd() {
    local VAL="$1"
    local SIZE="${VAL%-*}"   # 1.3b | 2.7b
    local NUM="${VAL#*-}"    # 1 (retain) | 2 (forget)
    local BASELINE DATAFILE
    case "${SIZE}" in
        1.3b) BASELINE="princeton-nlp/Sheared-LLaMA-1.3B" ;;
        2.7b) BASELINE="princeton-nlp/Sheared-LLaMA-2.7B" ;;
    esac
    case "${NUM}" in
        1) DATAFILE="data/news/raw/retain1.txt" ;;
        2) DATAFILE="data/news/raw/forget.txt" ;;
    esac
    cat <<CMD
set -e

DEST="models/muse/verifiers/${SIZE}/model_${NUM}"

if [ -f "\${DEST}/config.json" ]; then
    echo "==> SKIP MUSE verifier ${SIZE}/model_${NUM} (exists)"
else
    echo "==> MUSE verifier train: ${SIZE} model_${NUM} (${DATAFILE})"
    python scripts/train/finetune_single_model.py ${NUM} ${DATAFILE} ${BASELINE}
    # Relocate from the script's default models/<size>/model_<n> to the
    # documented verifier path.
    mkdir -p "models/muse/verifiers/${SIZE}"
    rm -rf "\${DEST}"
    mv "models/${SIZE}/model_${NUM}" "\${DEST}"
fi
CMD
}
