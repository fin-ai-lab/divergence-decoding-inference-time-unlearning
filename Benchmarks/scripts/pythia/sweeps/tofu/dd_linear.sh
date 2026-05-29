# sweeps/tofu/dd_linear.sh — TOFU Linear Divergence-Decoding alpha sweep (Fig.3 / Table 6).
#
# Eval-only. One SLURM job per (aux size, alpha) grid point so they pack 8/node
# and parallelise across the cluster. No training, no weight sync.
#
#   aux  ∈ {1B, 3B}   (Llama-3.2 retain90 / full pair against the 8B target)
#   alpha grid (20 values, see below)
#
# task_name -> saves/eval/tofu/dd_linear/<1B|3B>-alpha-<a>
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/tofu/dd_linear.sh

SWEEP_NAME="tofu-dd-linear"

# 1 GPU / 8 CPU / 64 GB (slurm_run.sh defaults) — do NOT set SBATCH_EXTRA.
# Eval-only — do NOT set SYNC_WEIGHTS.

_ALPHAS=(0.5 1.0 1.1 1.2 1.3 1.4 1.5 2.0 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 4.0)

SWEEP_VALUES=()
for _aux in 1B 3B; do
    for _a in "${_ALPHAS[@]}"; do
        SWEEP_VALUES+=("${_aux}-${_a}")
    done
done

sweep_run_cmd() {
    local VAL="$1"
    local AUX="${VAL%%-*}"      # 1B | 3B
    local ALPHA="${VAL#*-}"     # alpha value
    local SIZE                  # Llama-3.2 model-size token
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
  +model.model_dd_use_ngram=No \\
  +model.model_dd_alpha=${ALPHA} \\
  task_name=tofu/dd_linear/${AUX}-alpha-${ALPHA}
CMD
}
