# sweeps/muse/dd_trigram.sh — MUSE News Trigram Divergence-Decoding sweep.
#
# Eval-only. The retain/forget "verifiers" are trigram stupid-backoff models fit
# directly from the raw forget/retain text (no neural models needed). One SLURM
# job per grid point:
#   alpha ∈ {5,10,15,20,25,30}  -> saves/eval/muse/dd_trigram/alpha-<a>
#   topk  ∈ {1,2,3,5,10}        -> saves/eval/muse/dd_trigram/topk-<k>
#
# Uses +model.model_dd_use_ngram=Yes with list-valued retain/forget text paths.
# topk runs also add +model.topk_vocab=MUSE +model.model_dd_monte_carlo=Yes.
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/muse/dd_trigram.sh

SWEEP_NAME="muse-dd-trigram"

# 1 GPU / 8 CPU / 64 GB (slurm_run.sh defaults) — do NOT set SBATCH_EXTRA.
# Eval-only — do NOT set SYNC_WEIGHTS.

SWEEP_VALUES=()
for _a in 5 10 15 20 25 30; do SWEEP_VALUES+=("alpha-${_a}"); done
for _k in 1 2 3 5 10;       do SWEEP_VALUES+=("topk-${_k}");  done

sweep_run_cmd() {
    local VAL="$1"
    local KIND="${VAL%-*}"   # alpha | topk
    local NUM="${VAL#*-}"    # value
    if [ "${KIND}" = "alpha" ]; then
        cat <<CMD
python src/eval.py experiment=eval/muse/default.yaml \\
  data_split=News \\
  +model.model_handler=DD \\
  +model.model_dd_use_ngram=Yes \\
  +model.model_dd_big=muse-bench/MUSE-news_target \\
  +model.model_dd_retain=[data/news/raw/retain1.txt] \\
  +model.model_dd_forget=[data/news/raw/forget.txt] \\
  +model.model_dd_alpha=${NUM} \\
  task_name=muse/dd_trigram/alpha-${NUM}
CMD
    else
        cat <<CMD
python src/eval.py experiment=eval/muse/default.yaml \\
  data_split=News \\
  +model.model_handler=DD \\
  +model.model_dd_use_ngram=Yes \\
  +model.model_dd_big=muse-bench/MUSE-news_target \\
  +model.model_dd_retain=[data/news/raw/retain1.txt] \\
  +model.model_dd_forget=[data/news/raw/forget.txt] \\
  +model.model_dd_topk=${NUM} \\
  +model.topk_vocab=MUSE \\
  +model.model_dd_monte_carlo=Yes \\
  task_name=muse/dd_trigram/topk-${NUM}
CMD
    fi
}
