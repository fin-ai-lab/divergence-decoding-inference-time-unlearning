# sweeps/tofu/dd_rank.sh — TOFU Rank (top-k) Divergence-Decoding sweep.
#
# Eval-only. One SLURM job per (aux size, top-k) grid point.
#
#   aux  ∈ {1B, 3B}   (Llama-3.2 retain90 / full pair against the 8B target)
#   topk ∈ {1,5,20,50,100,200,500,1000}
#
# Rank-DD adds +model.topk_vocab=TOFU and +model.model_dd_monte_carlo=Yes.
# task_name -> saves/eval/tofu/dd_rank/<1B|3B>-topk-<k>
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/tofu/dd_rank.sh

SWEEP_NAME="tofu-dd-rank"

# 1 GPU / 8 CPU / 64 GB (slurm_run.sh defaults) — do NOT set SBATCH_EXTRA.
# Eval-only — do NOT set SYNC_WEIGHTS.

_TOPKS=(1 5 20 50 100 200 500 1000)

SWEEP_VALUES=()
for _aux in 1B 3B; do
    for _k in "${_TOPKS[@]}"; do
        SWEEP_VALUES+=("${_aux}-${_k}")
    done
done

sweep_run_cmd() {
    local VAL="$1"
    local AUX="${VAL%%-*}"   # 1B | 3B
    local TOPK="${VAL#*-}"   # top-k value
    local SIZE
    case "${AUX}" in
        1B) SIZE="3.2-1B" ;;
        3B) SIZE="3.2-3B" ;;
    esac
    cat <<CMD
python src/eval.py experiment=eval/tofu/default \\
  +model.model_handler=DD \\
  +model.model_dd_big=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \\
  +model.model_dd_retain=open-unlearning/tofu_Llama-${SIZE}-Instruct_retain90 \\
  +model.model_dd_forget=open-unlearning/tofu_Llama-${SIZE}-Instruct_full \\
  +model.model_dd_topk=${TOPK} \\
  +model.model_dd_use_ngram=No \\
  +model.topk_vocab=TOFU \\
  +model.model_dd_monte_carlo=Yes \\
  task_name=tofu/dd_rank/${AUX}-topk-${TOPK}
CMD
}
