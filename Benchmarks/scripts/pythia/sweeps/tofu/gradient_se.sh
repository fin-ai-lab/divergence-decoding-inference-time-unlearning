# sweeps/tofu/gradient_se.sh — DEADLINE fast-path: SimNPO + UNDIAL only, to obtain
# the per-index TOFU_EVAL.json needed for bootstrap SEs (the original repo's EVAL.json
# for these two were deleted). Uses a single continuous run (num_train_epochs=N) rather
# than the exact incremental schedule, since the bootstrap SE half-width is essentially
# schedule-insensitive. The point estimates in the table use the original (copied) values.
SWEEP_NAME="tofu-gradient-se"
SWEEP_VALUES=("simnpo" "undial")
SBATCH_EXTRA="--gres=gpu:h100:2 --cpus-per-task=16 --mem=128G"
SYNC_WEIGHTS="1"

sweep_run_cmd() {
    local METHOD="$1"
    local TRAINER LR EPOCH CK
    case "${METHOD}" in
        simnpo) TRAINER=SimNPO; LR=3e-6; EPOCH=3;  CK=39  ;;
        undial) TRAINER=UNDIAL; LR=4e-6; EPOCH=10; CK=130 ;;
    esac
    cat <<CMD
set -e
TASK="tofu/gradient/${METHOD}_${LR}"
CKDIR="saves/unlearn/\${TASK}"
RETAIN_LOGS="saves/eval/tofu/baselines/retrain/TOFU_EVAL.json"
PORT=\$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
if [ ! -f "\${CKDIR}/checkpoint-${CK}/model.safetensors.index.json" ] && [ ! -f "\${CKDIR}/checkpoint-${CK}/model.safetensors" ]; then
    echo "==> ${METHOD} single-run train: lr=${LR} epochs=${EPOCH}"
    accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port \${PORT} \
        src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default.yaml trainer=${TRAINER} task_name=\${TASK} \
        model=Llama-3.1-8B-Instruct forget_split=forget10 retain_split=retain90 \
        model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
        retain_logs_path=\${RETAIN_LOGS} \
        trainer.args.per_device_train_batch_size=16 trainer.args.gradient_accumulation_steps=1 \
        trainer.args.num_train_epochs=${EPOCH} trainer.args.learning_rate=${LR} \
        trainer.args.ddp_find_unused_parameters=true trainer.args.gradient_checkpointing=true \
        trainer.args.save_strategy=epoch
fi
echo "==> ${METHOD} eval checkpoint-${CK}"
# NOTE: do NOT set paths.output_dir here — overriding it suppresses the
# fine-grained TOFU_EVAL.json write (only SUMMARY is produced), which we need
# for bootstrap SEs. task_name alone routes output to
# saves/eval/tofu/gradient/<method>_<lr>/checkpoint-<CK> AND writes EVAL.json.
python src/eval.py experiment=eval/tofu/default forget_split=forget10 holdout_split=holdout10 \
    model=Llama-3.1-8B-Instruct \
    model.model_args.pretrained_model_name_or_path=\${CKDIR}/checkpoint-${CK} \
    retain_logs_path=\${RETAIN_LOGS} \
    task_name=\${TASK}/checkpoint-${CK}
CMD
}
