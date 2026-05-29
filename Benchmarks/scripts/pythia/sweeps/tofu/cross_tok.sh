# sweeps/tofu/cross_tok.sh — Cross-tokenizer Divergence-Decoding on TOFU (forget10).
#
# Cross-tokenizer DD steers a small auxiliary verifier from a DIFFERENT model
# family than the target (OLMo-2 / Gemma-3 / Qwen3) onto the TOFU target's
# vocabulary via the DD handler's cross-tokenizer mode
# (+model.model_dd_cross_tokenizer=Yes  +model.model_dd_verifier_tokenizer=<aux>).
#
# Aux bases (TOFU):
#   OLMo  = allenai/OLMo-2-0425-1B-Instruct   Gemma = google/gemma-3-1b-it
#   Qwen  = Qwen/Qwen3-1.7B
# Target = open-unlearning/tofu_Llama-3.1-8B-Instruct_full
#
# NO hyperparameter sweep — only the paper's OPTIMAL config per (model, variant):
#   OLMo  Linear lr=5e-5 alpha=3.0 | Rank lr=1e-5 topk=1000
#   Gemma Linear lr=3e-5 alpha=2.4 | Rank lr=1e-5 topk=1
#   Qwen  Linear lr=3e-5 alpha=2.9 | Rank lr=1e-5 topk=1000
#
# Verifier (retain90, forget10) pairs are pre-trained and live on the shared array
# at VERIFIER_BASE (compute nodes read /data/lab directly), so no retraining: each
# job just runs the cross-tok DD eval. Rank adds +model.topk_vocab=TOFU
# +model.model_dd_monte_carlo=Yes.
#
# Eval dir names match the analysis (find_optimal_cross_tok_configs in
# scripts/analysis/tofu_scores.py): <short>-lr<lr>-{alpha-<a>|topk-<k>}.
#
# 1 GPU / 8 CPU / 64 GB (slurm_run.sh defaults). Eval only -> SYNC_WEIGHTS=0.
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/tofu/cross_tok.sh

SWEEP_NAME="tofu-cross-tok"

SYNC_WEIGHTS="0"

# One job per (model, variant) so they pack 8/node and parallelise.
SWEEP_VALUES=(
    "OLMo-linear"
    "OLMo-rank"
    "Gemma-linear"
    "Gemma-rank"
    "Qwen-linear"
    "Qwen-rank"
)

sweep_run_cmd() {
    local VAL="$1"
    cat <<CMD
set -e

TARGET="open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
# Pre-trained verifier pairs on the shared array (read directly by compute nodes).
VERIFIER_BASE="models/tofu/cross_tok"

# Confirm the (retain90, forget10) verifier pair for <short>,<lr> is present.
ensure_verifiers() {
    local SHORT="\$1"; local LR="\$2"
    for split in retain90 forget10; do
        if [ ! -f "\${VERIFIER_BASE}/\${split}_\${SHORT}_lr\${LR}/config.json" ]; then
            echo "ERROR: missing verifier \${VERIFIER_BASE}/\${split}_\${SHORT}_lr\${LR}"; exit 1
        fi
    done
    echo "==> using pre-trained /data/lab verifiers for \${SHORT} lr=\${LR}"
}

# Linear cross-tok DD eval. Args: <aux_hf> <short> <lr> <alpha>
eval_linear() {
    local AUX="\$1"; local SHORT="\$2"; local LR="\$3"; local ALPHA="\$4"
    local CFG="\${SHORT}-lr\${LR}-alpha-\${ALPHA}"
    echo "==> TOFU cross-tok LINEAR \${SHORT} lr=\${LR} alpha=\${ALPHA} -> \${CFG}"
    python src/eval.py experiment=eval/tofu/default \\
        +model.model_handler=DD \\
        +model.model_dd_big=\${TARGET} \\
        +model.model_dd_retain=\${VERIFIER_BASE}/retain90_\${SHORT}_lr\${LR} \\
        +model.model_dd_forget=\${VERIFIER_BASE}/forget10_\${SHORT}_lr\${LR} \\
        +model.model_dd_use_ngram=No \\
        +model.model_dd_alpha=\${ALPHA} \\
        +model.model_dd_cross_tokenizer=Yes \\
        +model.model_dd_verifier_tokenizer=\${AUX} \\
        task_name=tofu/cross_tok/\${CFG}
}

# Rank cross-tok DD eval. Args: <aux_hf> <short> <lr> <topk>
eval_rank() {
    local AUX="\$1"; local SHORT="\$2"; local LR="\$3"; local TOPK="\$4"
    local CFG="\${SHORT}-lr\${LR}-topk-\${TOPK}"
    echo "==> TOFU cross-tok RANK \${SHORT} lr=\${LR} topk=\${TOPK} -> \${CFG}"
    python src/eval.py experiment=eval/tofu/default \\
        +model.model_handler=DD \\
        +model.model_dd_big=\${TARGET} \\
        +model.model_dd_retain=\${VERIFIER_BASE}/retain90_\${SHORT}_lr\${LR} \\
        +model.model_dd_forget=\${VERIFIER_BASE}/forget10_\${SHORT}_lr\${LR} \\
        +model.model_dd_use_ngram=No \\
        +model.model_dd_topk=\${TOPK} \\
        +model.model_dd_cross_tokenizer=Yes \\
        +model.model_dd_verifier_tokenizer=\${AUX} \\
        +model.topk_vocab=TOFU \\
        +model.model_dd_monte_carlo=Yes \\
        task_name=tofu/cross_tok/\${CFG}
}

case "${VAL}" in
    # ── OLMo (aux=allenai/OLMo-2-0425-1B-Instruct) ───────────────────────────
    OLMo-linear)
        AUX="allenai/OLMo-2-0425-1B-Instruct"; SHORT="OLMo-2-0425-1B-Instruct"
        ensure_verifiers "\${SHORT}" 5e-5
        eval_linear "\${AUX}" "\${SHORT}" 5e-5 3.0
        ;;
    OLMo-rank)
        AUX="allenai/OLMo-2-0425-1B-Instruct"; SHORT="OLMo-2-0425-1B-Instruct"
        ensure_verifiers "\${SHORT}" 1e-5
        eval_rank "\${AUX}" "\${SHORT}" 1e-5 1000
        ;;

    # ── Gemma (aux=google/gemma-3-1b-it, needs eager attention) ──────────────
    Gemma-linear)
        AUX="google/gemma-3-1b-it"; SHORT="gemma-3-1b-it"
        ensure_verifiers "\${SHORT}" 3e-5
        eval_linear "\${AUX}" "\${SHORT}" 3e-5 2.4
        ;;
    Gemma-rank)
        AUX="google/gemma-3-1b-it"; SHORT="gemma-3-1b-it"
        ensure_verifiers "\${SHORT}" 1e-5
        eval_rank "\${AUX}" "\${SHORT}" 1e-5 1
        ;;

    # ── Qwen (aux=Qwen/Qwen3-1.7B) ───────────────────────────────────────────
    Qwen-linear)
        AUX="Qwen/Qwen3-1.7B"; SHORT="Qwen3-1.7B"
        ensure_verifiers "\${SHORT}" 3e-5
        eval_linear "\${AUX}" "\${SHORT}" 3e-5 2.9
        ;;
    Qwen-rank)
        AUX="Qwen/Qwen3-1.7B"; SHORT="Qwen3-1.7B"
        ensure_verifiers "\${SHORT}" 1e-5
        eval_rank "\${AUX}" "\${SHORT}" 1e-5 1000
        ;;
esac

echo "==> TOFU cross-tok done for ${VAL}."
CMD
}
