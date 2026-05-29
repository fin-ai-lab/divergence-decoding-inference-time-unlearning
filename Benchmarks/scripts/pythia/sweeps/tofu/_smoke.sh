# Smoke test for the single cross-tokenizer-capable environment (transformers 4.57.6).
# (1) check_crosstok.py confirms transformers can instantiate OLMo-2 / gemma-3 /
#     Qwen3 (the cross-tokenizer auxiliaries).
# (2) Re-runs TOFU Linear DD (alpha=1.5, 1B); should still match the original repo
#     (validated to ~1e-4 on the prior env).
SWEEP_NAME="tofu-smoke"
SWEEP_VALUES=("1.5")
sweep_run_cmd() {
    local A="$1"
    cat <<CMD
set -e
echo "==> cross-tok architecture load check"
python scripts/pythia/check_crosstok.py
echo "==> TOFU Linear DD alpha=${A} (1B) eval"
python src/eval.py experiment=eval/tofu/default \
  +model.model_handler=DD \
  +model.model_dd_big=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
  +model.model_dd_retain=open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90 \
  +model.model_dd_forget=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  +model.model_dd_use_ngram=No \
  +model.model_dd_alpha=${A} \
  task_name=tofu_linear/alpha-${A}-3.2-1B
CMD
}
