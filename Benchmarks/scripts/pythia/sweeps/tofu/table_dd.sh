# High-priority: the OPTIMAL TOFU DD eval points needed for the main TOFU table
# (Table 5) and appendix (Table 6) — Linear alpha=1.5 / Rank k=20 for the 1B aux,
# plus the 3B optima. These are the only DD points the table needs; the full
# alpha/topk grid (dd_linear.sh / dd_rank.sh, Fig 3) is deprioritized separately.
# Writes to the same task_name paths as the grids, so they're idempotent.
SWEEP_NAME="tofu-table-dd"
SWEEP_VALUES=("linear-1B-1.5" "rank-1B-20" "linear-3B-1.2" "rank-3B-20")

sweep_run_cmd() {
    local VAL="$1"
    local VARIANT="${VAL%%-*}"; local REST="${VAL#*-}"
    local AUX="${REST%%-*}"; local P="${REST#*-}"
    local SIZE; case "$AUX" in 1B) SIZE="3.2-1B";; 3B) SIZE="3.2-3B";; esac
    local COMMON="experiment=eval/tofu/default +model.model_handler=DD \
+model.model_dd_big=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
+model.model_dd_retain=open-unlearning/tofu_Llama-${SIZE}-Instruct_retain90 \
+model.model_dd_forget=open-unlearning/tofu_Llama-${SIZE}-Instruct_full \
+model.model_dd_use_ngram=No"
    if [ "$VARIANT" = "linear" ]; then
        cat <<CMD
python src/eval.py ${COMMON} +model.model_dd_alpha=${P} task_name=tofu/dd_linear/${AUX}-alpha-${P}
CMD
    else
        cat <<CMD
python src/eval.py ${COMMON} +model.model_dd_topk=${P} +model.topk_vocab=TOFU +model.model_dd_monte_carlo=Yes task_name=tofu/dd_rank/${AUX}-topk-${P}
CMD
    fi
}
