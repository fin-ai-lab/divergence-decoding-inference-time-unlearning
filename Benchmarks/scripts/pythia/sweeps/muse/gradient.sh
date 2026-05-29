# sweeps/muse/gradient.sh — MUSE News gradient-based unlearning baselines.
#
# Full fine-tunes of the 7B target (accelerate + DeepSpeed ZeRO-3 offload, 2 GPUs)
# with per-epoch checkpoints, then eval. MUSE forget with bs=4 x grad_accum=4 on
# 2 GPUs -> 12 optimizer steps/epoch.
#
# One SLURM job per method (SWEEP_VALUES = method keys + case statement):
#   graddiff  GradDiff  default trainer cfg (lr 1e-5, 10 ep) -> eval final ckpt
#   npo       NPO       default trainer cfg (lr 1e-5, 10 ep) -> eval final ckpt
#   simnpo    SimNPO    default trainer cfg (lr 1e-5, 10 ep) -> eval final ckpt
#   undial    UNDIAL    lr=1e-5, 10 ep                       -> eval checkpoint-12
#
#   train -> saves/unlearn/muse/gradient/<method>/checkpoint-<n>
#   eval  -> saves/eval/muse/gradient/<method>
#   retain_logs -> saves/eval/muse/baselines/retrain/MUSE_EVAL.json
#
# 2 GPUs (accelerate + deepspeed on 7B).  Trains models -> SYNC_WEIGHTS=1.
#
# Usage: ./scripts/pythia/run_sweep.sh sweeps/muse/gradient.sh

SWEEP_NAME="muse-gradient"
SWEEP_VALUES=("graddiff" "npo" "simnpo" "undial")

SBATCH_EXTRA="--gres=gpu:h100:2 --cpus-per-task=16 --mem=128G"
SYNC_WEIGHTS="1"

sweep_run_cmd() {
    local METHOD="$1"
    local TRAINER LR EPOCH EVAL_CKPT
    case "${METHOD}" in
        # MUSE used single-run training (not TOFU's incremental resume) and took
        # the FINAL epoch of each run, so the LR schedule matches num_train_epochs=EPOCH.
        # UNDIAL's optimal is epoch 1 (checkpoint-12), so it trains 1 epoch (schedule
        # over 1 epoch) and evals the final checkpoint — NOT epoch 1 of a 10-epoch run.
        graddiff) TRAINER=GradDiff; LR=1e-5; EPOCH=10; EVAL_CKPT=last ;;
        npo)      TRAINER=NPO;      LR=1e-5; EPOCH=10; EVAL_CKPT=last ;;
        simnpo)   TRAINER=SimNPO;   LR=1e-5; EPOCH=10; EVAL_CKPT=last ;;
        undial)   TRAINER=UNDIAL;   LR=1e-5; EPOCH=1;  EVAL_CKPT=last ;;
    esac

    cat <<CMD
set -e

TASK="muse/gradient/${METHOD}"
TASK_DIR="saves/unlearn/\${TASK}"
RETAIN_LOGS="saves/eval/muse/baselines/retrain/MUSE_EVAL.json"
PORT=\$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')

# Step 1: train (per-epoch checkpoints)
echo "==> MUSE ${METHOD} train: lr=${LR} epochs=${EPOCH}"
accelerate launch \\
    --config_file configs/accelerate/default_config.yaml \\
    --main_process_port \${PORT} \\
    src/train.py --config-name=unlearn.yaml \\
    experiment=unlearn/muse/default.yaml \\
    trainer=${TRAINER} \\
    model=Llama-2-7b-hf \\
    data_split=News \\
    forget_split=forget \\
    retain_split=retain1 \\
    task_name=\${TASK} \\
    retain_logs_path=\${RETAIN_LOGS} \\
    trainer.args.per_device_train_batch_size=4 \\
    trainer.args.gradient_accumulation_steps=4 \\
    trainer.args.num_train_epochs=${EPOCH} \\
    trainer.args.learning_rate=${LR} \\
    trainer.args.ddp_find_unused_parameters=true \\
    trainer.args.gradient_checkpointing=true \\
    trainer.args.save_strategy=epoch

# Step 2: pick the checkpoint to eval
EVAL_CKPT="${EVAL_CKPT}"
if [ "\${EVAL_CKPT}" = "last" ]; then
    CKPT_N=\$(ls -d "\${TASK_DIR}"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
else
    CKPT_N="\${EVAL_CKPT}"
fi
CKPT_DIR="\${TASK_DIR}/checkpoint-\${CKPT_N}"
echo "==> MUSE ${METHOD} eval: checkpoint-\${CKPT_N}"

python src/eval.py experiment=eval/muse/default.yaml \\
    data_split=News \\
    model=Llama-2-7b-hf \\
    model.model_args.pretrained_model_name_or_path=\${CKPT_DIR} \\
    retain_logs_path=\${RETAIN_LOGS} \\
    task_name=muse/gradient/${METHOD}
CMD
}
