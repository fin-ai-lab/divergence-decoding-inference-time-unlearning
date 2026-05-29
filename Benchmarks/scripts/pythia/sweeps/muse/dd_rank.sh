# sweeps/muse/dd_rank.sh — MUSE News Rank (top-k) Divergence-Decoding sweep.
#
# Eval-only. One SLURM job per (verifier size, top-k) grid point.
#   size ∈ {1.3b, 2.7b}   (verifier retain=model_1 / forget=model_2 vs 7B target)
#   topk ∈ {1,5,20,50,200,1000}
#
# Rank-DD adds +model.topk_vocab=MUSE and +model.model_dd_monte_carlo=Yes.
# Requires sweeps/muse/verifiers.sh first.
#
# task_name -> saves/eval/muse/dd_rank/<size>-topk-<k>
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/muse/dd_rank.sh

SWEEP_NAME="muse-dd-rank"

# 1 GPU / 8 CPU / 64 GB (slurm_run.sh defaults) — do NOT set SBATCH_EXTRA.
# Eval-only — do NOT set SYNC_WEIGHTS.

_TOPKS=(1 5 20 50 200 1000)

SWEEP_VALUES=()
for _size in 1.3b 2.7b; do
    for _k in "${_TOPKS[@]}"; do
        SWEEP_VALUES+=("${_size}-${_k}")
    done
done

sweep_run_cmd() {
    local VAL="$1"
    local SIZE="${VAL%-*}"    # 1.3b | 2.7b
    local TOPK="${VAL##*-}"   # top-k value
    cat <<CMD
python src/eval.py experiment=eval/muse/default.yaml \\
  data_split=News \\
  +model.model_handler=DD \\
  +model.model_dd_use_ngram=No \\
  +model.model_dd_big=muse-bench/MUSE-news_target \\
  +model.model_dd_retain=models/muse/verifiers/${SIZE}/model_1 \\
  +model.model_dd_forget=models/muse/verifiers/${SIZE}/model_2 \\
  +model.model_dd_topk=${TOPK} \\
  +model.topk_vocab=MUSE \\
  +model.model_dd_monte_carlo=Yes \\
  task_name=muse/dd_rank/${SIZE}-topk-${TOPK}
CMD
}
