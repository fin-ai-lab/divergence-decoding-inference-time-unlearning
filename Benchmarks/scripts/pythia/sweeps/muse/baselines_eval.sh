# sweeps/muse/baselines_eval.sh — MUSE reference-model evals (target + retrain).
#
# Eval-only. Two jobs:
#   target  -> muse-bench/MUSE-news_target
#   retrain -> muse-bench/MUSE-news_retrain
#
# Both write MUSE_EVAL.json + MUSE_SUMMARY.json. The retrain run's
# saves/eval/muse/baselines/retrain/MUSE_EVAL.json is the retain_logs source
# consumed by the MUSE gradient baselines. Run this sweep FIRST.
#
# task_name -> saves/eval/muse/baselines/{target,retrain}
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/muse/baselines_eval.sh

SWEEP_NAME="muse-baselines"
SWEEP_VALUES=("target" "retrain")

# 1 GPU / 8 CPU / 64 GB (slurm_run.sh defaults) — do NOT set SBATCH_EXTRA.
# Eval-only — do NOT set SYNC_WEIGHTS.

sweep_run_cmd() {
    local KIND="$1"
    local MODEL_PATH
    case "${KIND}" in
        target)  MODEL_PATH="muse-bench/MUSE-news_target" ;;
        retrain) MODEL_PATH="muse-bench/MUSE-news_retrain" ;;
    esac
    cat <<CMD
python src/eval.py experiment=eval/muse/default.yaml \\
  data_split=News \\
  model=Llama-2-7b-hf \\
  model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \\
  task_name=muse/baselines/${KIND}
CMD
}
