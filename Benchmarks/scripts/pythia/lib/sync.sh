#!/bin/bash
# sync.sh — Rsync repo code to the Pythia NFS home (a tiny code landing zone).
#
# Heavy artifacts (data/models/saves/venv/hf-cache) are NEVER synced here; they
# live on /hpc_temp on the compute node. Only source code + configs land on NFS
# home so that `sbatch` can read slurm_run.sh and the compute node can copy the
# code to /hpc_temp.
#
# Requires: common.sh sourced first (provides LOCAL_REPO, PYTHIA_HOST, PYTHIA_REPO)

pythia_sync() {
    info "Syncing repo code to ${PYTHIA_HOST}:${PYTHIA_REPO} ..."

    rsync -az --delete \
        --exclude='/data/' \
        --exclude='/models/' \
        --exclude='/saves/' \
        --exclude='.git/' \
        --exclude='.venv/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='slurm-*.out' \
        --exclude='*.egg-info/' \
        --exclude='wandb/' \
        --exclude='outputs/' \
        --exclude='multirun/' \
        --exclude='.nfs*' \
        "${LOCAL_REPO}/" "${PYTHIA_HOST}:${PYTHIA_REPO}/"

    info "Repo synced."
}
