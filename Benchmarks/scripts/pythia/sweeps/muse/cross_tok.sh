# sweeps/muse/cross_tok.sh — Cross-tokenizer Divergence-Decoding on MUSE News.
#
# Cross-tokenizer DD steers a small auxiliary verifier from a DIFFERENT model
# family than the target (OLMo-2 / Gemma-3 / Qwen3) onto the MUSE target's
# vocabulary via the DD handler's cross-tokenizer mode
# (+model.model_dd_cross_tokenizer=Yes  +model.model_dd_verifier_tokenizer=<aux>).
#
# Aux bases (MUSE):
#   OLMo  = allenai/OLMo-2-0425-1B        Gemma = google/gemma-3-1b-pt
#   Qwen  = Qwen/Qwen3-1.7B-Base
# Target = muse-bench/MUSE-news_target
# Data   = forget data/news/raw/forget.txt   retain data/news/raw/retain1.txt
#
# NO hyperparameter sweep — only the paper's OPTIMAL config per (model, variant).
# Where verbmem and knowmem peak at different points, both points are evaluated
# (one eval run emits all MUSE metrics; the paper reads the relevant one).
#
# Each job (a) finetunes the retain + forget verifier from the aux base at the
# optimal LR (skipped if already present), then (b) runs the cross-tok DD eval
# at the optimal alpha (Linear) or topk (Rank). Rank uses
# +model.topk_vocab=MUSE +model.model_dd_monte_carlo=Yes.
#
# Verifier weights -> models/muse/cross_tok/{model_1,model_2}_<short>_lr<lr>
# Eval results      -> saves/eval/muse/cross_tok/<model>-<variant>-<cfg>
#
# 1 GPU / 8 CPU / 64 GB (slurm_run.sh defaults). Trains verifiers -> SYNC_WEIGHTS=1.
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/muse/cross_tok.sh

SWEEP_NAME="muse-cross-tok"

SYNC_WEIGHTS="1"

# One job per (model, variant). Jobs that peak at different verbmem/knowmem
# points run both evals inside the same job.
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

TARGET="muse-bench/MUSE-news_target"
FORGET_DATA="data/news/raw/forget.txt"
RETAIN_DATA="data/news/raw/retain1.txt"

# Finetune a (retain, forget) verifier pair from the aux base at a given LR.
# Args: <aux_hf> <short> <lr> [extra finetune args...]
finetune_pair() {
    local AUX="\$1"; local SHORT="\$2"; local LR="\$3"; shift 3
    local RETAIN_DIR="models/muse/cross_tok/model_1_\${SHORT}_lr\${LR}"
    local FORGET_DIR="models/muse/cross_tok/model_2_\${SHORT}_lr\${LR}"
    if [ -f "\${RETAIN_DIR}/config.json" ]; then
        echo "==> SKIP retain verifier \${SHORT} lr=\${LR} (exists)"
    else
        echo "==> Finetune retain verifier: \${AUX} lr=\${LR} -> \${RETAIN_DIR}"
        python scripts/train/finetune_verifier.py \\
            --model \${AUX} --data \${RETAIN_DATA} --output \${RETAIN_DIR} \\
            --lr \${LR} --epochs 10 --batch_size 4 --grad_accum 8 --max_len 2048 "\$@"
    fi
    if [ -f "\${FORGET_DIR}/config.json" ]; then
        echo "==> SKIP forget verifier \${SHORT} lr=\${LR} (exists)"
    else
        echo "==> Finetune forget verifier: \${AUX} lr=\${LR} -> \${FORGET_DIR}"
        python scripts/train/finetune_verifier.py \\
            --model \${AUX} --data \${FORGET_DATA} --output \${FORGET_DIR} \\
            --lr \${LR} --epochs 10 --batch_size 4 --grad_accum 8 --max_len 2048 "\$@"
    fi
}

# Linear cross-tok DD eval. Args: <aux_hf> <short> <lr> <alpha> <task_cfg>
eval_linear() {
    local AUX="\$1"; local SHORT="\$2"; local LR="\$3"; local ALPHA="\$4"; local CFG="\$5"
    echo "==> MUSE cross-tok LINEAR \${SHORT} lr=\${LR} alpha=\${ALPHA}"
    python src/eval.py experiment=eval/muse/default.yaml \\
        data_split=News \\
        +model.model_handler=DD \\
        +model.model_dd_use_ngram=No \\
        +model.model_dd_big=\${TARGET} \\
        +model.model_dd_retain=models/muse/cross_tok/model_1_\${SHORT}_lr\${LR} \\
        +model.model_dd_forget=models/muse/cross_tok/model_2_\${SHORT}_lr\${LR} \\
        +model.model_dd_alpha=\${ALPHA} \\
        +model.model_dd_cross_tokenizer=Yes \\
        +model.model_dd_verifier_tokenizer=\${AUX} \\
        task_name=muse/cross_tok/\${CFG}
}

# Rank cross-tok DD eval. Args: <aux_hf> <short> <lr> <topk> <task_cfg>
eval_rank() {
    local AUX="\$1"; local SHORT="\$2"; local LR="\$3"; local TOPK="\$4"; local CFG="\$5"
    echo "==> MUSE cross-tok RANK \${SHORT} lr=\${LR} topk=\${TOPK}"
    python src/eval.py experiment=eval/muse/default.yaml \\
        data_split=News \\
        +model.model_handler=DD \\
        +model.model_dd_use_ngram=No \\
        +model.model_dd_big=\${TARGET} \\
        +model.model_dd_retain=models/muse/cross_tok/model_1_\${SHORT}_lr\${LR} \\
        +model.model_dd_forget=models/muse/cross_tok/model_2_\${SHORT}_lr\${LR} \\
        +model.model_dd_topk=\${TOPK} \\
        +model.model_dd_cross_tokenizer=Yes \\
        +model.model_dd_verifier_tokenizer=\${AUX} \\
        +model.topk_vocab=MUSE \\
        +model.model_dd_monte_carlo=Yes \\
        task_name=muse/cross_tok/\${CFG}
}

case "${VAL}" in
    # ── OLMo (aux=allenai/OLMo-2-0425-1B) ────────────────────────────────────
    OLMo-linear)
        AUX="allenai/OLMo-2-0425-1B"; SHORT="OLMo-2-0425-1B"
        finetune_pair "\${AUX}" "\${SHORT}" 8e-5
        set +e
        eval_linear "\${AUX}" "\${SHORT}" 8e-5 1.4 "OLMo-linear-lr8e-5-alpha-1.4"
        eval_linear "\${AUX}" "\${SHORT}" 8e-5 0.8 "OLMo-linear-lr8e-5-alpha-0.8"
        ;;
    OLMo-rank)
        AUX="allenai/OLMo-2-0425-1B"; SHORT="OLMo-2-0425-1B"
        finetune_pair "\${AUX}" "\${SHORT}" 5e-5
        set +e
        eval_rank "\${AUX}" "\${SHORT}" 5e-5 1000 "OLMo-rank-lr5e-5-topk-1000"
        ;;

    # ── Gemma (aux=google/gemma-3-1b-pt, needs eager attention) ──────────────
    Gemma-linear)
        AUX="google/gemma-3-1b-pt"; SHORT="gemma-3-1b-pt"
        finetune_pair "\${AUX}" "\${SHORT}" 3e-5 --attn_impl eager
        set +e
        eval_linear "\${AUX}" "\${SHORT}" 3e-5 0.7 "Gemma-linear-lr3e-5-alpha-0.7"
        ;;
    Gemma-rank)
        AUX="google/gemma-3-1b-pt"; SHORT="gemma-3-1b-pt"
        finetune_pair "\${AUX}" "\${SHORT}" 5e-5 --attn_impl eager
        set +e
        eval_rank "\${AUX}" "\${SHORT}" 5e-5 100 "Gemma-rank-lr5e-5-topk-100"
        eval_rank "\${AUX}" "\${SHORT}" 5e-5 500 "Gemma-rank-lr5e-5-topk-500"
        ;;

    # ── Qwen (aux=Qwen/Qwen3-1.7B-Base) ──────────────────────────────────────
    Qwen-linear)
        AUX="Qwen/Qwen3-1.7B-Base"; SHORT="Qwen3-1.7B-Base"
        # verbmem peaks at lr=8e-5 alpha=0.6, knowmem at lr=3e-5 alpha=0.9
        finetune_pair "\${AUX}" "\${SHORT}" 8e-5
        finetune_pair "\${AUX}" "\${SHORT}" 3e-5
        set +e
        eval_linear "\${AUX}" "\${SHORT}" 8e-5 0.6 "Qwen-linear-lr8e-5-alpha-0.6"
        eval_linear "\${AUX}" "\${SHORT}" 3e-5 0.9 "Qwen-linear-lr3e-5-alpha-0.9"
        ;;
    Qwen-rank)
        AUX="Qwen/Qwen3-1.7B-Base"; SHORT="Qwen3-1.7B-Base"
        finetune_pair "\${AUX}" "\${SHORT}" 5e-5
        set +e
        eval_rank "\${AUX}" "\${SHORT}" 5e-5 1000 "Qwen-rank-lr5e-5-topk-1000"
        ;;
esac

echo "==> MUSE cross-tok done for ${VAL}."
CMD
}
