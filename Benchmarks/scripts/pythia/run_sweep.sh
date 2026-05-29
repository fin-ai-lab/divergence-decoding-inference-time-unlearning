#!/bin/bash
# run_sweep.sh — Submit a job sweep to Pythia from bll01.
#
# Usage:
#   ./scripts/pythia/run_sweep.sh sweeps/tofu/dd_rank.sh
#   ./scripts/pythia/run_sweep.sh --partition standard_l40s sweeps/muse/eco.sh
#
# A sweep file defines:
#   SWEEP_NAME            job-name prefix (required)
#   SWEEP_VALUES=( ... )  list of sweep values (required)
#   sweep_run_cmd()       VAL -> shell command string (required)
# Optional, set inside the sweep file:
#   SBATCH_EXTRA          extra sbatch flags, e.g. "--gres=gpu:h200:1"
#   SYNC_WEIGHTS          "1" to persist trained weights to the checkpoint store
#
# What it does:
#   1. One-time setup (SSH keys, HF token, uv on pythia)
#   2. Rsyncs repo code to pythia (NFS home landing zone)
#   3. Submits one SLURM job per sweep value (1 GPU / 8 CPU / 64 GB each)

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
source "${PYTHIA_DIR}/lib/setup.sh"
source "${PYTHIA_DIR}/lib/sync.sh"

# ── Parse args ───────────────────────────────────────────────────────────────

PARTITION=""
DEPENDENCY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition) PARTITION="$2"; shift 2 ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        *) SWEEP_FILE="$1"; shift ;;
    esac
done

SWEEP_FILE="${SWEEP_FILE:?Usage: $0 [--partition P] <sweep_file>}"
[[ "${SWEEP_FILE}" = /* ]] || SWEEP_FILE="${PYTHIA_DIR}/${SWEEP_FILE}"
[ -f "${SWEEP_FILE}" ] || error "Sweep file not found: ${SWEEP_FILE}"

# Defaults a sweep file may override
SBATCH_EXTRA=""
SYNC_WEIGHTS="0"
source "${SWEEP_FILE}"

: "${SWEEP_NAME:?Sweep file must define SWEEP_NAME}"
[ ${#SWEEP_VALUES[@]} -gt 0 ] || error "Sweep file must define SWEEP_VALUES array"
declare -F sweep_run_cmd >/dev/null || error "Sweep file must define sweep_run_cmd()"

[ -n "${PARTITION}" ] && SBATCH_EXTRA="--partition=${PARTITION} ${SBATCH_EXTRA}"
[ -n "${DEPENDENCY}" ] && SBATCH_EXTRA="${SBATCH_EXTRA} --dependency=${DEPENDENCY}"

info "Sweep: ${SWEEP_NAME} (${#SWEEP_VALUES[@]} jobs)  sync_weights=${SYNC_WEIGHTS}  extra='${SBATCH_EXTRA}'"

# ── Setup & sync ─────────────────────────────────────────────────────────────

pythia_setup
pythia_sync

# ── Submit ───────────────────────────────────────────────────────────────────

JOB_IDS=()
for VAL in "${SWEEP_VALUES[@]}"; do
    RUN_CMD=$(sweep_run_cmd "${VAL}")
    info "Submitting (${VAL})..."

    # base64-encode RUN_CMD so multi-line commands with shell control-flow,
    # quotes, and $ survive the ssh + export transport intact.
    RUN_CMD_B64=$(printf '%s' "${RUN_CMD}" | base64 | tr -d '\n')
    SBATCH_OUTPUT=$(ssh "${PYTHIA_HOST}" \
        "cd ${PYTHIA_REPO} && export RUN_CMD_B64='${RUN_CMD_B64}' SYNC_WEIGHTS='${SYNC_WEIGHTS}' && sbatch \
            --export=ALL \
            --job-name=${SWEEP_NAME} \
            ${SBATCH_EXTRA:-} \
            scripts/pythia/slurm_run.sh")
    JOB_ID=$(echo "${SBATCH_OUTPUT}" | awk '{print $4}')
    [ -z "${JOB_ID}" ] && error "sbatch failed for ${VAL}: ${SBATCH_OUTPUT}"
    JOB_IDS+=("${JOB_ID}")
    info "Submitted job ${JOB_ID} (${VAL})"
done

echo ""
echo "  Sweep: ${SWEEP_NAME} — submitted ${#JOB_IDS[@]} jobs: ${JOB_IDS[*]}"
echo "  Monitor:    ssh pythia 'squeue -u \$USER'"
echo "  Cancel all: ssh pythia 'scancel ${JOB_IDS[*]}'"
echo ""
