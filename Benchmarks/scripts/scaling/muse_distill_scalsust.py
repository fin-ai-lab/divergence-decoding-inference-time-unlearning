#!/usr/bin/env python3
"""
Orchestrator for MUSE scaling & sustainability distillation experiments
(paper Figure 11).

Finetunes MUSE 1.3B models and runs, for each scaling step (forget-set size)
and sustainability step (sequential forget requests):
  1. Finetune 1.3B verifier models (1-8) sequentially on GPU 0
  2. DD teacher eval for each config
  3. Distill DD -> student for each config
  4. Eval distilled students

Fig 11 methods (Linear DD, Rank DD, GradDiff, NPO, SimNPO, and "Optimal")
share these DD teacher/student evals; this driver produces the DD/distill
arm. All phases are resumable: existing outputs are skipped.

Output layout (per SAVES_LAYOUT.md)
-----------------------------------
  Scaling steps        -> saves/eval/muse/scaling/{dd,distill}_<config>/
  Sustainability steps -> saves/eval/muse/sustainability/{dd,distill}_<config>/
  base_step1 (shared Step 1) lives under saves/eval/muse/scaling/.

Run from the repo root (Benchmarks/):
    python scripts/scaling/muse_distill_scalsust.py
"""

import subprocess
import sys
import os
import json
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

PRETRAINED = "princeton-nlp/Sheared-LLaMA-1.3B"
DD_BIG = "muse-bench/MUSE-News_target"
DD_ALPHA = 0.85
DISTILL_LR = 1e-4
DISTILL_TEMP = 1.5
DISTILL_EPOCHS = 5
DISTILL_BATCH = 4
DISTILL_GRAD_ACCUM = 8
DISTILL_DATA = "data/news/raw/forget.txt"

MODEL_DIR = "models/1.3b"
DISTILL_MODEL_DIR = "models/MUSE_Distill_ScalSust"

# Eval task_name -> saves/eval/<task_name>/ ; scaling and sustainability split
# per SAVES_LAYOUT.md. base_step1 is the shared Step 1 and lives under scaling/.
SCALING_TASK_DIR = "muse/scaling"
SUST_TASK_DIR = "muse/sustainability"

# Model definitions: (model_num, data_file, baseline). Listed in dependency
# order — model_6/7/8 chain off model_2/6/7, which precede them here.
FINETUNE_MODELS = [
    (1, "data/news/raw/retain1.txt", PRETRAINED),
    (2, "data/news/raw/forget.txt", PRETRAINED),
    (3, "data/news/scal/forget_2.txt", PRETRAINED),
    (4, "data/news/scal/forget_3.txt", PRETRAINED),
    (5, "data/news/scal/forget_4.txt", PRETRAINED),
    (6, "data/news/sust/forget_2.txt", f"{MODEL_DIR}/model_2"),
    (7, "data/news/sust/forget_3.txt", f"{MODEL_DIR}/model_6"),
    (8, "data/news/sust/forget_4.txt", f"{MODEL_DIR}/model_7"),
]

# Experiment configs: (name, forget_model_num)
# base_step1 is the shared Step 1 of both scaling and sustainability.
EXPERIMENT_CONFIGS = [
    ("base_step1", 2),
    ("scal_step2", 3),
    ("scal_step3", 4),
    ("scal_step4", 5),
    ("sust_step2", 6),
    ("sust_step3", 7),
    ("sust_step4", 8),
]

# Sustainability distillation chains: each step starts from the previous step's checkpoint.
# Scaling steps and base_step1 start fresh from DD_BIG.
SUST_STUDENT_CHAIN = {
    "sust_step2": "base_step1",
    "sust_step3": "sust_step2",
    "sust_step4": "sust_step3",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def task_dir_for(config_name):
    """Return the eval task subdir (scaling vs sustainability) for a config."""
    return SUST_TASK_DIR if config_name.startswith("sust_") else SCALING_TASK_DIR


def dd_task_name(config_name):
    return f"{task_dir_for(config_name)}/dd_{config_name}"


def distill_task_name(config_name):
    return f"{task_dir_for(config_name)}/distill_{config_name}"


def run_command(cmd, env=None):
    """Run a shell command, return True on success."""
    if env:
        full_env = os.environ.copy()
        full_env.update(env)
    else:
        full_env = None

    sys.stdout.flush()
    result = subprocess.run(cmd, shell=True, env=full_env, stdout=sys.stdout, stderr=subprocess.STDOUT)
    sys.stdout.flush()
    if result.returncode != 0:
        print(f"Command failed (exit {result.returncode}): {cmd}")
        return False
    return True


def finetune_completed(model_num):
    """Check if a finetuned model already exists."""
    config_file = Path(MODEL_DIR) / f"model_{model_num}" / "config.json"
    if config_file.exists():
        print(f"Finetune already completed: model_{model_num}")
        return True
    return False


def dd_eval_completed(config_name):
    """Check if DD eval results exist."""
    summary = Path("saves/eval") / dd_task_name(config_name) / "MUSE_SUMMARY.json"
    if summary.exists():
        try:
            data = json.loads(summary.read_text())
            if len(data.keys()) >= 3:
                print(f"DD eval already completed: dd_{config_name}")
                return True
        except (json.JSONDecodeError, IOError):
            pass
    return False


def distill_completed(config_name):
    """Check if distillation checkpoint exists."""
    config_file = Path(DISTILL_MODEL_DIR) / config_name / "checkpoint-epoch-5" / "config.json"
    if config_file.exists():
        print(f"Distillation already completed: {config_name}")
        return True
    return False


def distill_eval_completed(config_name):
    """Check if distilled model eval results exist."""
    summary = Path("saves/eval") / distill_task_name(config_name) / "MUSE_SUMMARY.json"
    if summary.exists():
        try:
            data = json.loads(summary.read_text())
            if len(data.keys()) >= 3:
                print(f"Distill eval already completed: distill_{config_name}")
                return True
        except (json.JSONDecodeError, IOError):
            pass
    return False


# ── Phase functions ──────────────────────────────────────────────────────────

def run_finetune(model_num, data_file, baseline, gpu_id):
    """Finetune a single 1.3B model on the given GPU."""
    if finetune_completed(model_num):
        return True

    print(f"==> Finetuning model_{model_num} on GPU {gpu_id}: {data_file} (baseline: {baseline})")
    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    cmd = f"python -u scripts/train/finetune_single_model.py {model_num} {data_file} {baseline}"
    return run_command(cmd, env)


def run_dd_eval(config_name, forget_model_num, gpu_id):
    """Run DD teacher eval on the given GPU."""
    if dd_eval_completed(config_name):
        return True

    forget_path = f"{MODEL_DIR}/model_{forget_model_num}/"
    retain_path = f"{MODEL_DIR}/model_1/"
    task_name = dd_task_name(config_name)

    print(f"==> DD eval {config_name} on GPU {gpu_id} (forget=model_{forget_model_num})")
    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    cmd = f"""python src/eval.py \
    experiment=eval/muse/default.yaml \
    data_split=News \
    +model.model_handler=DD \
    +model.model_dd_use_ngram=No \
    +model.model_dd_big={DD_BIG} \
    +model.model_dd_retain={retain_path} \
    +model.model_dd_forget={forget_path} \
    +model.model_dd_alpha={DD_ALPHA} \
    task_name={task_name}"""

    return run_command(cmd, env)


def run_distillation(config_name, forget_model_num, gpu_id):
    """Run DD distillation on the given GPU."""
    if distill_completed(config_name):
        return True

    forget_path = f"{MODEL_DIR}/model_{forget_model_num}"
    retain_path = f"{MODEL_DIR}/model_1"
    output_dir = f"{DISTILL_MODEL_DIR}/{config_name}"

    # Sustainability steps chain from the previous step's checkpoint
    prev_step = SUST_STUDENT_CHAIN.get(config_name)
    if prev_step:
        student_path = f"{DISTILL_MODEL_DIR}/{prev_step}/checkpoint-epoch-{DISTILL_EPOCHS}"
    else:
        student_path = DD_BIG

    print(f"==> Distilling {config_name} on GPU {gpu_id} (forget=model_{forget_model_num}, student={student_path})")
    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    cmd = f"""python scripts/distill/distill_model_muse.py \
    --learning_rate {DISTILL_LR} \
    --num_epochs {DISTILL_EPOCHS} \
    --per_device_batch_size {DISTILL_BATCH} \
    --gradient_accumulation_steps {DISTILL_GRAD_ACCUM} \
    --dd_alpha {DD_ALPHA} \
    --dd_big {DD_BIG} \
    --dd_retain {retain_path} \
    --dd_forget {forget_path} \
    --student_model {student_path} \
    --data_path {DISTILL_DATA} \
    --temperature {DISTILL_TEMP} \
    --output_dir {output_dir} \
    --save_epochs {DISTILL_EPOCHS}"""

    return run_command(cmd, env)


def run_distill_eval(config_name, gpu_id):
    """Run evaluation on a distilled model checkpoint."""
    if distill_eval_completed(config_name):
        return True

    model_path = f"{DISTILL_MODEL_DIR}/{config_name}/checkpoint-epoch-{DISTILL_EPOCHS}"
    task_name = distill_task_name(config_name)

    print(f"==> Eval distilled {config_name} on GPU {gpu_id}")
    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    cmd = f"""python src/eval.py \
    experiment=eval/muse/default.yaml \
    data_split=News \
    model.model_args.pretrained_model_name_or_path={model_path} \
    task_name={task_name}"""

    return run_command(cmd, env)


def run_phase(phase_name, configs, run_fn):
    """Run a phase over configs sequentially on GPU 0.

    Args:
        phase_name: display name for logging
        configs: list of (config_name, forget_model_num) tuples
        run_fn: function(config_name, forget_model_num, gpu_id) -> bool
    """
    print(f"\n{'='*60}")
    print(f"Phase: {phase_name}")
    print(f"{'='*60}")

    for config_name, forget_model_num in configs:
        run_fn(config_name, forget_model_num, 0)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MUSE Distillation: Scaling & Sustainability")
    print("=" * 60)
    print(f"DD alpha: {DD_ALPHA}")
    print(f"Distill: lr={DISTILL_LR}, temp={DISTILL_TEMP}, epochs={DISTILL_EPOCHS}")
    print(f"Batch: {DISTILL_BATCH} x {DISTILL_GRAD_ACCUM} = {DISTILL_BATCH * DISTILL_GRAD_ACCUM}")
    print("=" * 60)

    # ── Phase 1: Finetune models 1-8 ────────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 1: Finetuning 1.3B verifier models")
    print(f"{'='*60}")

    for model_num, data_file, baseline in FINETUNE_MODELS:
        run_finetune(model_num, data_file, baseline, 0)

    # ── Phase 2: DD teacher evals ────────────────────────────────────────
    run_phase(
        "DD Teacher Evals",
        EXPERIMENT_CONFIGS,
        run_dd_eval,
    )

    # ── Phase 3: Distillation ────────────────────────────────────────────
    run_phase(
        "Distillation (DD -> Student)",
        EXPERIMENT_CONFIGS,
        run_distillation,
    )

    # ── Phase 4: Eval distilled models ───────────────────────────────────
    # Wrap run_distill_eval to match the 3-arg signature expected by run_phase
    distill_eval_configs = [(name, None) for name, _ in EXPERIMENT_CONFIGS]
    run_phase(
        "Distilled Model Evals",
        distill_eval_configs,
        lambda name, _forget, gpu_id: run_distill_eval(name, gpu_id),
    )

    print(f"\n{'='*60}")
    print("All phases complete!")
    print(f"Scaling eval results:        saves/eval/{SCALING_TASK_DIR}/{{dd,distill}}_*/MUSE_SUMMARY.json")
    print(f"Sustainability eval results: saves/eval/{SUST_TASK_DIR}/{{dd,distill}}_*/MUSE_SUMMARY.json")
    print(f"Run: python scripts/analysis/analyze_distill_scalsust.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
