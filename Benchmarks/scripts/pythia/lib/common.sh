#!/bin/bash
# common.sh — Shared helpers and configuration for Pythia cluster scripts.
#
# Usage (from a script in scripts/pythia/):
#   source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
#
# Provides:
#   - info(), warn(), error() logging helpers
#   - PYTHIA_DIR, LOCAL_REPO, PYTHIA_HOST, PYTHIA_REPO
#   - .env loaded and validated

# ── Helpers ──────────────────────────────────────────────────────────────────

info()  { echo -e "\033[1;34m==>\033[0m $*"; }
warn()  { echo -e "\033[1;33m==>\033[0m $*"; }
error() { echo -e "\033[1;31m==>\033[0m $*" >&2; exit 1; }

# ── Paths ────────────────────────────────────────────────────────────────────

# PYTHIA_DIR = the scripts/pythia/ directory (parent of lib/)
PYTHIA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# LOCAL_REPO = the Benchmarks/ repo root (two levels up from scripts/pythia/)
LOCAL_REPO="$(cd "${PYTHIA_DIR}/../.." && pwd)"

# ── Load .env ────────────────────────────────────────────────────────────────

ENV_FILE="${PYTHIA_DIR}/.env"
[ -f "${ENV_FILE}" ] || error "${ENV_FILE} not found. Copy .env.example to .env and fill in your values."
source "${ENV_FILE}"

# ── Derived variables ────────────────────────────────────────────────────────

PYTHIA_HOST="${PYTHIA_HOST:-pythia}"
# Code landing zone on the (tiny) NFS home. Heavy artifacts never live here —
# data/models/saves/venv/hf-cache all live on /hpc_temp on the compute node.
PYTHIA_REPO="${PYTHIA_HOME}/dd-benchmarks"
