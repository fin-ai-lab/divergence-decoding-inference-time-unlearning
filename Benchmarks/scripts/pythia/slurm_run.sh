#!/bin/bash
#SBATCH --account=pi-bll
#SBATCH --partition=bll
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=pgpu015,pgpu018
#SBATCH --output=slurm-%j.out
#
# slurm_run.sh — Generic SLURM batch script for the Divergence-Decoding
# benchmarks (MUSE + TOFU) on the Pythia cluster.
#
# Resource policy: 1 GPU + 8 CPUs + 64 GB = exactly 1/8 of a node, so eight
# single-GPU jobs pack onto each node and no GPU sits idle. Known-bad nodes
# pgpu015 / pgpu018 are excluded. Heavier jobs (8B full fine-tunes, distillation)
# override the GPU type to an H200 via SBATCH_EXTRA from their sweep file, e.g.
#   SBATCH_EXTRA="--gres=gpu:h200:1"
# but never exceed 1 GPU / 8 CPU / 64 GB so packing is preserved.
#
# The script is command-agnostic. Sweep files pass configuration via env vars:
#   RUN_CMD        — the full command to execute (e.g. python src/eval.py ...)
#   SYNC_WEIGHTS   — if "1", model weights (models/, saves/unlearn checkpoints)
#                    are persisted to ${CHECKPOINT_DIR} on bll01 for HF upload.
#
# Everything heavy lives on /hpc_temp (code, data, models, saves, venv, hf cache).
# NFS home holds only the code landing zone.

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

NFS_REPO="${HOME}/dd-benchmarks"
REPO_DIR="/hpc_temp/${USER}/dd-benchmarks"

# Source from NFS first (REPO_DIR on /hpc_temp/ may not exist yet). SLURM copies
# this script to a spool dir, so BASH_SOURCE-relative paths don't work here.
source "${NFS_REPO}/scripts/pythia/lib/common.sh"

BLL01_HOST="bll01"
DATA_DIR="/hpc_temp/${USER}/dd-data"
MODEL_DIR="/hpc_temp/${USER}/dd-models"
SAVES_DIR="/hpc_temp/${USER}/dd-saves"
VENV_DIR="/hpc_temp/${USER}/dd-venv"
HF_CACHE="/hpc_temp/${USER}/.cache/huggingface"

# RUN_CMD is transported base64-encoded (RUN_CMD_B64) so multi-line commands with
# control-flow/quotes survive ssh+export; fall back to a raw RUN_CMD if present.
if [ -n "${RUN_CMD_B64:-}" ]; then
    RUN_CMD=$(printf '%s' "${RUN_CMD_B64}" | base64 -d)
fi
: "${RUN_CMD:?ERROR: RUN_CMD/RUN_CMD_B64 not set. This script should be submitted by a sweep file.}"
SYNC_WEIGHTS="${SYNC_WEIGHTS:-0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/lab/dd-unlearning-checkpoints}"

# ── Job info ──────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════════════"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Node         : $(hostname)"
echo "  GPUs         : ${SLURM_GPUS_ON_NODE:-1}"
echo "  Start        : $(date)"
echo "  Repo         : ${REPO_DIR}"
echo "  Sync weights : ${SYNC_WEIGHTS}"
echo "  RUN_CMD      : ${RUN_CMD}"
echo "════════════════════════════════════════════════════════════════"

# ── Stage repo + data ─────────────────────────────────────────────────────────

mkdir -p "${REPO_DIR}" "${DATA_DIR}" "${MODEL_DIR}" "${SAVES_DIR}" "${HF_CACHE}"

echo "==> Syncing repo code from NFS to /hpc_temp/..."
rsync -a --delete \
    --exclude='/data/' --exclude='/models/' --exclude='/saves/' \
    --exclude='slurm-*.out' --exclude='__pycache__/' --exclude='*.pyc' \
    --exclude='.venv/' \
    "${NFS_REPO}/" "${REPO_DIR}/"

echo "==> Syncing data from ${BLL01_HOST}..."
rsync -az --compress-level=1 \
    "${BLL01_HOST}:${BLL01_REPO_DIR}/data/" \
    "${DATA_DIR}/"
echo "==> Data ready: $(du -sh "${DATA_DIR}" 2>/dev/null | cut -f1)"

# Symlink data, models, saves into the repo so relative paths resolve.
ln -sfn "${DATA_DIR}" "${REPO_DIR}/data"
ln -sfn "${MODEL_DIR}" "${REPO_DIR}/models"
ln -sfn "${SAVES_DIR}" "${REPO_DIR}/saves"

# Pull existing eval JSONs from bll01 so finished work can be skipped, and so
# training jobs can read the retrain baseline's *_EVAL.json (retain_logs_path).
echo "==> Syncing existing eval results from ${BLL01_HOST}..."
rsync -az \
    --include='*/' \
    --include='*.json' --include='*.csv' --include='*.log' --include='*.txt' \
    --exclude='*' \
    "${BLL01_HOST}:${BLL01_REPO_DIR}/saves/" \
    "${SAVES_DIR}/" 2>/dev/null || true

# Pull any already-trained model weights from the persistent checkpoint store so
# eval-only jobs can find verifier/baseline models trained by earlier jobs.
echo "==> Syncing existing model weights from ${BLL01_HOST}:${CHECKPOINT_DIR}..."
rsync -az \
    "${BLL01_HOST}:${CHECKPOINT_DIR}/models/" \
    "${MODEL_DIR}/" 2>/dev/null || true

# ── Setup uv + venv ──────────────────────────────────────────────────────────

export HF_HOME="${HF_CACHE}"
export HF_TOKEN="${HF_TOKEN:-}"
export TRITON_CACHE_DIR="/hpc_temp/${USER}/.triton"
export PATH="${HOME}/.local/bin:${PATH}"
export UV_CACHE_DIR="/hpc_temp/${USER}/.cache/uv"
export UV_LINK_MODE=copy

if [ -f "${HOME}/.cache/huggingface/token" ]; then
    cp "${HOME}/.cache/huggingface/token" "${HF_CACHE}/token"
fi

DEPS_HASH=$(md5sum "${REPO_DIR}/pyproject.toml" 2>/dev/null | cut -d' ' -f1)
DEPS_MARKER="${VENV_DIR}/.deps_hash"
VENV_LOCK="/hpc_temp/${USER}/dd-venv.lock"

# Lock so concurrent jobs don't race on venv creation. Wait up to 30 min: the
# first build compiles torch deps and can be slow.
exec 9>"${VENV_LOCK}"
echo "==> Waiting for venv lock..."
flock -w 1800 9 || error "Timed out waiting for venv lock"
echo "==> Lock acquired."

if [ ! -f "${VENV_DIR}/bin/python" ] || [ ! -f "${DEPS_MARKER}" ] || \
   [ "$(cat "${DEPS_MARKER}" 2>/dev/null)" != "${DEPS_HASH}" ]; then
    echo "==> Building venv at ${VENV_DIR}..."
    # --clear recreates the venv even if one already exists (e.g. a stale venv
    # from a previous dependency set). /hpc_temp is only visible on compute
    # nodes, so the venv can only be (re)built from inside a job like this.
    uv venv --clear --python 3.11 "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    cd "${REPO_DIR}"
    uv pip install ".[lm-eval]"
    # flash-attn is required by the Llama-3.2-1B/3B verifier configs
    # (attn_implementation: flash_attention_2). Install a prebuilt wheel matching
    # torch 2.4 / cu12 / cp311 so we never need an nvcc source build on the node.
    FLASH_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    python -c "import flash_attn" 2>/dev/null || uv pip install "${FLASH_WHL}"
    echo "${DEPS_HASH}" > "${DEPS_MARKER}"
else
    echo "==> Using existing venv (deps up to date)."
    source "${VENV_DIR}/bin/activate"
fi

flock -u 9
exec 9>&-
echo "==> Lock released."

echo "==> Python: $(python --version)"
echo "==> torch:  $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"

# ── Run ───────────────────────────────────────────────────────────────────────

cd "${REPO_DIR}"
echo ""
echo "==> Running command..."

RUN_SCRIPT="/tmp/dd-run-${SLURM_JOB_ID}.sh"
echo "${RUN_CMD}" > "${RUN_SCRIPT}"
chmod +x "${RUN_SCRIPT}"
set +e
bash "${RUN_SCRIPT}" 2>&1
RUN_EXIT=$?
set -e
rm -f "${RUN_SCRIPT}"
[ ${RUN_EXIT} -ne 0 ] && echo "==> Command FAILED with exit code ${RUN_EXIT}"

# ── Sync results back to bll01 ────────────────────────────────────────────────

# Always: eval JSON/logs (lightweight) back into the repo's saves/ for analysis.
# Retried + non-fatal (results also persist on /hpc_temp and can be re-pulled).
if [ -d "${REPO_DIR}/saves" ]; then
    echo ""
    echo "==> Syncing eval results back to ${BLL01_HOST} (JSON/logs only)..."
    set +e
    for attempt in 1 2 3 4 5; do
        rsync -az --timeout=120 -e "ssh -o ConnectTimeout=30" \
            --include='*/' \
            --include='*.json' --include='*.csv' --include='*.log' --include='*.txt' \
            --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pt' --exclude='*.pth' \
            "${SAVES_DIR}/" \
            "${BLL01_HOST}:${BLL01_REPO_DIR}/saves/" && { echo "==> Eval results synced."; break; }
        echo "==> eval-sync attempt ${attempt} failed; retrying..."; sleep $((attempt * 15))
    done
    set -e
fi

# Training jobs: persist model weights to the durable checkpoint store on bll01
# (/hpc_temp is wiped after 14 days; these become the HF upload artifacts).
# Best-effort + retried: the weights also live on /hpc_temp (persistent ~14d,
# shared across compute nodes), so a transient bll01 sshd hiccup (many jobs
# rsyncing at once) must NOT fail an otherwise-successful training job.
if [ "${SYNC_WEIGHTS}" = "1" ]; then
    echo ""
    echo "==> Persisting model weights to ${BLL01_HOST}:${CHECKPOINT_DIR} (best-effort)..."
    set +e
    ssh -o ConnectTimeout=30 "${BLL01_HOST}" "mkdir -p ${CHECKPOINT_DIR}/models ${CHECKPOINT_DIR}/saves"
    persist_ok=1
    for SRC_DST in "${MODEL_DIR}/:${CHECKPOINT_DIR}/models/" "${SAVES_DIR}/unlearn/:${CHECKPOINT_DIR}/saves/unlearn/"; do
        SRC="${SRC_DST%%:*}"; DST="${SRC_DST#*:}"
        [ -d "${SRC}" ] && [ -n "$(ls -A "${SRC}" 2>/dev/null)" ] || continue
        for attempt in 1 2 3 4 5; do
            rsync -az --timeout=120 --partial -e "ssh -o ConnectTimeout=30" "${SRC}" "${BLL01_HOST}:${DST}" && break
            echo "==> weight rsync attempt ${attempt} failed (${SRC}); retrying..."; sleep $((attempt * 20)); persist_ok=0
        done
    done
    set -e
    [ "${persist_ok}" = "1" ] && echo "==> Weights persisted." || \
        warn "Weight persist had retries/failures; weights remain on /hpc_temp (${MODEL_DIR}, ${SAVES_DIR}/unlearn) and can be re-synced later."
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Done: $(date) (run exit ${RUN_EXIT})"
echo "════════════════════════════════════════════════════════════════"
exit ${RUN_EXIT}
