# sweeps/tofu/gradient.sh — TOFU gradient-based unlearning baselines.
#
# EXACT reproduction of the original procedure (scripts/tofu_unlearn.py): the model
# is trained ONE EPOCH AT A TIME, each step resuming from the previous epoch's
# checkpoint with trainer.args.num_train_epochs={current_epoch}. This reproduces the
# original's incremental LR-scheduler trajectory exactly — a single num_train_epochs=N
# run would decay the LR over N epochs instead and yield a different checkpoint.
# We stop at each method's OPTIMAL epoch (no 10-epoch sweep, no DPO) and eval that
# one checkpoint. forget10, bs=16, grad_accum=1, 2 GPUs -> 13 steps/epoch.
#
#   method        trainer     experiment                  lr      epoch  ckpt
#   grad_ascent   GradAscent  unlearn/tofu/default.yaml   2e-6    3      39
#   graddiff      GradDiff    unlearn/tofu/default.yaml   2e-6    3      39
#   npo           NPO         unlearn/tofu/default.yaml   4e-6    2      26
#   rmu           RMU         unlearn/tofu/default.yaml   8e-7    4      52
#   simnpo        SimNPO      unlearn/tofu/default.yaml   3e-6    3      39
#   undial        UNDIAL      unlearn/tofu/default.yaml   4e-6    10     130
#
#   train -> saves/unlearn/tofu/gradient/<method>_<lr>/checkpoint-<n>
#   eval  -> saves/eval/tofu/gradient/<method>_<lr>/checkpoint-<n>
#
# One SLURM job per method (fanned out). 2 H100 / 16 CPU / 128 GB each.
# Usage: ./scripts/pythia/run_sweep.sh sweeps/tofu/gradient.sh

SWEEP_NAME="tofu-gradient"
SWEEP_VALUES=("grad_ascent" "graddiff" "npo" "rmu" "simnpo" "undial")

SBATCH_EXTRA="--gres=gpu:h100:2 --cpus-per-task=16 --mem=128G"
SYNC_WEIGHTS="1"

sweep_run_cmd() {
    local METHOD="$1"
    local TRAINER EXPERIMENT LR EPOCH
    case "${METHOD}" in
        grad_ascent) TRAINER=GradAscent; EXPERIMENT=unlearn/tofu/default.yaml; LR=2e-6;  EPOCH=3  ;;
        graddiff)    TRAINER=GradDiff;   EXPERIMENT=unlearn/tofu/default.yaml; LR=2e-6;  EPOCH=3  ;;
        npo)         TRAINER=NPO;        EXPERIMENT=unlearn/tofu/default.yaml; LR=4e-6;  EPOCH=2  ;;
        rmu)         TRAINER=RMU;        EXPERIMENT=unlearn/tofu/default.yaml; LR=8e-7;  EPOCH=4  ;;
        simnpo)      TRAINER=SimNPO;     EXPERIMENT=unlearn/tofu/default.yaml; LR=3e-6;  EPOCH=3  ;;
        undial)      TRAINER=UNDIAL;     EXPERIMENT=unlearn/tofu/default.yaml; LR=4e-6;  EPOCH=10 ;;
    esac
    cat <<CMD
set -e
TASK="tofu/gradient/${METHOD}_${LR}"
CKDIR="saves/unlearn/\${TASK}"
RETAIN_LOGS="saves/eval/tofu/baselines/retrain/TOFU_EVAL.json"
PORT=\$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
FINAL_CK=\$(( ${EPOCH} * 13 ))

# Step 1: incremental epoch-by-epoch training (matches scripts/tofu_unlearn.py).
for EP in \$(seq 1 ${EPOCH}); do
    CK=\$(( EP * 13 )); PREV=\$(( (EP - 1) * 13 ))
    if [ -f "\${CKDIR}/checkpoint-\${CK}/model.safetensors.index.json" ] || [ -f "\${CKDIR}/checkpoint-\${CK}/model.safetensors" ]; then
        echo "==> SKIP ${METHOD} epoch \${EP} (checkpoint-\${CK} exists)"; continue
    fi
    RESUME=""
    [ \${EP} -gt 1 ] && RESUME="trainer.args.resume_from_checkpoint=\${CKDIR}/checkpoint-\${PREV}"
    echo "==> TOFU ${METHOD} train epoch \${EP}/${EPOCH} (lr=${LR}) -> checkpoint-\${CK}"
    accelerate launch \
        --config_file configs/accelerate/default_config.yaml \
        --main_process_port \${PORT} \
        src/train.py --config-name=unlearn.yaml \
        experiment=${EXPERIMENT} \
        trainer=${TRAINER} \
        task_name=\${TASK} \
        model=Llama-3.1-8B-Instruct \
        forget_split=forget10 \
        retain_split=retain90 \
        model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
        retain_logs_path=\${RETAIN_LOGS} \
        trainer.args.per_device_train_batch_size=16 \
        trainer.args.gradient_accumulation_steps=1 \
        trainer.args.num_train_epochs=\${EP} \
        trainer.args.learning_rate=${LR} \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true \
        trainer.args.save_strategy=epoch \
        \${RESUME}
done

# Step 2: eval the optimal checkpoint only.
echo "==> TOFU ${METHOD} eval: checkpoint-\${FINAL_CK}"
python src/eval.py experiment=eval/tofu/default \
    model=Llama-3.1-8B-Instruct \
    forget_split=forget10 \
    holdout_split=holdout10 \
    model.model_args.pretrained_model_name_or_path=\${CKDIR}/checkpoint-\${FINAL_CK} \
    retain_logs_path=\${RETAIN_LOGS} \
    paths.output_dir=saves/eval/\${TASK}/checkpoint-\${FINAL_CK} \
    task_name=\${TASK}/checkpoint-\${FINAL_CK}
CMD
}
