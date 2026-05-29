# sweeps/muse/dd_linear.sh — MUSE News Linear Divergence-Decoding alpha sweep.
#
# Eval-only. One SLURM job per (verifier size, alpha) grid point.
#   size  ∈ {1.3b, 2.7b}   (verifier retain=model_1 / forget=model_2 vs 7B target)
#   alpha grid (14 values, see below)
#
# Requires sweeps/muse/verifiers.sh to have populated
#   models/muse/verifiers/<size>/{model_1,model_2}
#
# task_name -> saves/eval/muse/dd_linear/<size>-alpha-<a>
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/muse/dd_linear.sh

SWEEP_NAME="muse-dd-linear"

# 1 GPU / 8 CPU / 64 GB (slurm_run.sh defaults) — do NOT set SBATCH_EXTRA.
# Eval-only — do NOT set SYNC_WEIGHTS.

_ALPHAS=(0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 2.0 2.5 3.0)

SWEEP_VALUES=()
for _size in 1.3b 2.7b; do
    for _a in "${_ALPHAS[@]}"; do
        SWEEP_VALUES+=("${_size}-${_a}")
    done
done

sweep_run_cmd() {
    local VAL="$1"
    local SIZE="${VAL%-*}"    # 1.3b | 2.7b
    local ALPHA="${VAL##*-}"  # alpha value
    cat <<CMD
python src/eval.py experiment=eval/muse/default.yaml \\
  data_split=News \\
  +model.model_handler=DD \\
  +model.model_dd_use_ngram=No \\
  +model.model_dd_big=muse-bench/MUSE-news_target \\
  +model.model_dd_retain=models/muse/verifiers/${SIZE}/model_1 \\
  +model.model_dd_forget=models/muse/verifiers/${SIZE}/model_2 \\
  +model.model_dd_alpha=${ALPHA} \\
  task_name=muse/dd_linear/${SIZE}-alpha-${ALPHA}
CMD
}
