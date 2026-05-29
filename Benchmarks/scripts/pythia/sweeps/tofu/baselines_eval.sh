# sweeps/tofu/baselines_eval.sh — TOFU reference-model evals (target + retrain).
#
# Eval-only. Two jobs:
#   target  -> open-unlearning/tofu_Llama-3.1-8B-Instruct_full
#   retrain -> open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90
#
# Both write TOFU_EVAL.json + TOFU_SUMMARY.json. The retrain run's
# saves/eval/tofu/baselines/retrain/TOFU_EVAL.json is the retain_logs source
# consumed by every gradient-baseline train/eval. Run this sweep FIRST.
#
# task_name -> saves/eval/tofu/baselines/{target,retrain}
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/tofu/baselines_eval.sh

SWEEP_NAME="tofu-baselines"
SWEEP_VALUES=("target" "retrain")

# 1 GPU / 8 CPU / 64 GB (slurm_run.sh defaults) — do NOT set SBATCH_EXTRA.
# Eval-only — do NOT set SYNC_WEIGHTS.

sweep_run_cmd() {
    local KIND="$1"
    local MODEL_PATH
    case "${KIND}" in
        target)  MODEL_PATH="open-unlearning/tofu_Llama-3.1-8B-Instruct_full" ;;
        retrain) MODEL_PATH="open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90" ;;
    esac
    cat <<CMD
python src/eval.py experiment=eval/tofu/default \\
  model=Llama-3.1-8B-Instruct \\
  model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \\
  task_name=tofu/baselines/${KIND}
CMD
}
