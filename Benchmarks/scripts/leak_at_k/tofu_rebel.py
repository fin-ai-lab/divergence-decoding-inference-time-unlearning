#!/usr/bin/env python3
"""
Run REBEL and Leak@K adversarial attacks on the TOFU unlearning methods
(paper Figure 17, Leak@K).

Methods attacked (optimal config per method, no sweeps):
  - Target / Retrain baselines (8B, vLLM)
  - 7 gradient unlearning methods (DPO, GradAscent, GradDiff, NPO, RMU,
    SimNPO, UNDIAL) — retrained on the fly at their optimal lr/checkpoint
  - Linear DD (alpha=1.5) and Rank DD (topk=20, monte-carlo)
  - Distill DD (retrained student model)
  - Inference-time baselines: Offset, ULD, WHP (retrained on the fly)

All model weights and results are written under saves/ and models/,
which on Pythia are symlinked to /hpc_temp/ (unlimited scratch).

Leak@K results are written to: saves/eval/tofu/leak_at_k/<method>/

vLLM environment
----------------
REBEL needs vllm (torch>=2.6) which conflicts with the training stack
(torch==2.4.1), so it runs from a separate Python venv. Create one with uv
and point REBEL_PYTHON at it, e.g.:

    uv venv /path/to/rebel-venv
    uv pip install --python /path/to/rebel-venv vllm
    export REBEL_PYTHON=/path/to/rebel-venv/bin/python

If REBEL_PYTHON is unset, the current interpreter is used (assumes vllm is
already importable).

Environment variables
---------------------
  REBEL_PYTHON              python with vllm installed (default: sys.executable)
  VLLM_TP                   tensor-parallel size for the vLLM hacker/judge,
                            read by REBEL/root/config.py (default: 2)
  GPU_MEMORY_UTILIZATION    vLLM mem fraction read by REBEL/root/config.py for
                            pure-vLLM targets (default: 0.45)
  GPU_MEMORY_UTILIZATION_DD vLLM mem fraction used by this driver when a DD
                            target shares the GPU with the hacker/judge (default: 0.45)
  REBEL_BATCH_SIZE          attack batch size (default: 1024)

Usage
-----
    # Full run (all methods, both Leak@K and REBEL evolutionary):
    python scripts/leak_at_k/tofu_rebel.py

    # Only specific methods:
    python scripts/leak_at_k/tofu_rebel.py --methods Target,SimNPO,Linear_DD

    # Only Leak@K (skip full REBEL evolutionary) — Figure 17:
    python scripts/leak_at_k/tofu_rebel.py --leak-only

    # DD runner mode (called internally by the orchestrator):
    python scripts/leak_at_k/tofu_rebel.py --dd-run --name Linear_DD --mode leak \
        --dd-style alpha --dd-param 1.5
"""

import argparse
import gc
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# Repo root is two levels up: scripts/leak_at_k/tofu_rebel.py -> Benchmarks/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Constants ─────────────────────────────────────────────────────────────────
REBEL_DIR = PROJECT_ROOT / "src" / "REBEL"
REBEL_DATA = REBEL_DIR / "data" / "tofu_forget10.jsonl"
# saves/ is symlinked to /hpc_temp/$USER/unlearning-saves on Pythia
RESULTS_BASE = PROJECT_ROOT / "saves" / "eval" / "tofu" / "leak_at_k"

TOKENIZER_8B = "meta-llama/Llama-3.1-8B-Instruct"
TOKENIZER_1B = "meta-llama/Llama-3.2-1B-Instruct"

# REBEL needs vllm (torch>=2.6) which conflicts with training (torch==2.4.1).
# Run it from a separate venv and export REBEL_PYTHON (see module docstring).
# Falls back to current Python if not set (e.g. local dev with vllm installed).
REBEL_PYTHON = os.environ.get("REBEL_PYTHON", sys.executable)

LEAK_NUM_ATTACKS = 1000
LEAK_K_VALUES = [10, 100, 500, 1000]
REBEL_MUTATIONS = "1500,80,50,40,40"
REBEL_TOP_K = "20,12,8,5,3"

# DD runs share GPU with the vLLM hacker/judge, so keep mem util lower.
# Pure-vLLM targets use GPU_MEMORY_UTILIZATION (read by REBEL/root/config.py).
DD_GPU_MEM_UTIL = float(os.environ.get("GPU_MEMORY_UTILIZATION_DD", "0.45"))
REBEL_BATCH_SIZE = int(os.environ.get("REBEL_BATCH_SIZE", "1024"))

MODEL_8B = "Llama-3.1-8B-Instruct"
FORGET_SPLIT = "forget10"
RETAIN_SPLIT = "retain90"

TRAINER_EXPERIMENTS = {
    "DPO": "unlearn/tofu/idk.yaml",
    "GradAscent": "unlearn/tofu/default.yaml",
    "GradDiff": "unlearn/tofu/default.yaml",
    "NPO": "unlearn/tofu/default.yaml",
    "RMU": "unlearn/tofu/default.yaml",
    "SimNPO": "unlearn/tofu/default.yaml",
    "UNDIAL": "unlearn/tofu/default.yaml",
}

# DD base models (all on HuggingFace)
DD_BIG = "open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
DD_RETAIN = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
DD_FORGET = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"

# ── Hardcoded optimal configs (from tofu_scores.py on 2026-03-26) ─────────
# Avoids importing tofu_scores.py which has module-level file I/O

OPTIMAL_UNLEARN = {
    "DPO":        {"lr": "4e-6",  "epoch": 2,  "checkpoint": 26},
    "GradAscent": {"lr": "2e-6",  "epoch": 3,  "checkpoint": 39},
    "GradDiff":   {"lr": "2e-6",  "epoch": 3,  "checkpoint": 39},
    "NPO":        {"lr": "4e-6",  "epoch": 2,  "checkpoint": 26},
    "RMU":        {"lr": "8e-7",  "epoch": 4,  "checkpoint": 52},
    "SimNPO":     {"lr": "2e-6",  "epoch": 7,  "checkpoint": 91},
    "UNDIAL":     {"lr": "4e-6",  "epoch": 10, "checkpoint": 130},
}

OPTIMAL_DD = {
    "Linear_DD": {"style": "alpha", "param": 1.5, "extra_args": []},
    "Rank_DD":   {"style": "topk",  "param": 20,  "extra_args": ["--dd-monte-carlo"]},
}

OPTIMAL_DISTILL = {"lr": 4e-05, "epoch": 10, "temperature": 1.5}

# Offset Unlearning: trains 1B offset model, eval uses DD with alpha=1.0
OFFSET_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
OPTIMAL_OFFSET = {"lr": "5e-6", "epochs": 5, "batch_size": 2, "grad_accum": 8}

# ULD: trains truncated LoRA assistant, eval uses target - beta * assistant
ULD_TARGET = "open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
OPTIMAL_ULD = {"lr": "1e-3", "epochs": 5, "batch_size": 8, "grad_accum": 4,
               "retain_weight": 6.5, "beta": 0.75}

# WHP: finetunes reinforced model on forget set, eval uses baseline - alpha * ReLU(reinforced - baseline)
WHP_BASELINE = "open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
OPTIMAL_WHP = {"lr": "1e-5", "alpha": 3.0, "epochs": 10, "batch_size": 4, "grad_accum": 4}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run_cmd(cmd, env=None, cwd=None):
    """Run a shell command, return True on success."""
    if env:
        full_env = os.environ.copy()
        full_env.update(env)
    else:
        full_env = None
    print(f"\n>>> {cmd[:200]}...\n", flush=True)
    result = subprocess.run(cmd, shell=True, env=full_env, cwd=cwd)
    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode}): {cmd[:200]}")
        return False
    return True


def cleanup_model_files(path):
    """Delete model weight files from a directory to free disk on /hpc_temp/."""
    extensions = [".safetensors", ".bin", ".pt", ".pth"]
    path = Path(path)
    if not path.exists():
        return
    for f in path.iterdir():
        if any(f.name.endswith(ext) for ext in extensions):
            f.unlink()
            print(f"  Deleted: {f}")
    for name in ["config.json", "generation_config.json", "tokenizer.json",
                  "tokenizer_config.json", "special_tokens_map.json",
                  "trainer_state.json", "training_args.bin", "model.safetensors.index.json"]:
        p = path / name
        if p.exists():
            p.unlink()


def parse_leak_asr(results_dir, k=None):
    """Parse Leak@ whole_generation_tofu.json -> ASR (fraction of leaked examples).

    Parameters
    ----------
    results_dir : path-like
        Directory containing whole_generation_tofu.json.
    k : int or None
        If given, only consider the first k attack prompts per example (Leak@K).
        None means use all attacks.
    """
    path = Path(results_dir) / "whole_generation_tofu.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    total = len(data)
    if total == 0:
        return 0.0
    leaked = 0
    for idx, attacks in data.items():
        for i, entry in enumerate(attacks):
            if k is not None and i >= k:
                break
            ev = entry[-1] if isinstance(entry, (list, tuple)) else entry
            if isinstance(ev, dict) and ev.get("leaked", False):
                leaked += 1
                break
    return leaked / total


def parse_rebel_asr(results_dir):
    """Parse REBEL results -> ASR (fraction of examples with leak reports)."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return None
    # Count total examples from data file
    total = sum(1 for _ in open(REBEL_DATA))
    if total == 0:
        return 0.0
    # Count leaked indices from leak_report files (including baseline_only/)
    leaked_indices = set()
    for p in results_dir.rglob("leak_report_idx*.json"):
        name = p.stem
        parts = name.split("idx")
        if len(parts) >= 2:
            idx_str = parts[1].split("_")[0].split(".")[0]
            try:
                leaked_indices.add(int(idx_str))
            except ValueError:
                pass
    return len(leaked_indices) / total


# ── Retraining ────────────────────────────────────────────────────────────────
# Weights go to saves/unlearn/ which is symlinked to /hpc_temp/ on Pythia.

def retrain_unlearning_method(method, lr, num_epochs, master_port):
    """Retrain a single unlearning method. Returns (task_name, checkpoint)."""
    experiment = TRAINER_EXPERIMENTS[method]
    task_name = f"tofu_rebel/tofu_{MODEL_8B}_{FORGET_SPLIT}_{method}_{lr}"
    model_path = f"open-unlearning/tofu_{MODEL_8B}_full"
    checkpoint = num_epochs * 13

    cp_dir = PROJECT_ROOT / "saves" / "unlearn" / task_name / f"checkpoint-{checkpoint}"
    if (cp_dir / "config.json").exists():
        print(f"Checkpoint already exists: {cp_dir}")
        return task_name, checkpoint

    print(f"\nRetraining {method} lr={lr} for {num_epochs} epochs...")
    accel_config = PROJECT_ROOT / "configs" / "accelerate" / "default_config.yaml"
    cmd = (
        f"accelerate launch --config_file {accel_config} "
        f"--main_process_port {master_port} "
        f"src/train.py --config-name=unlearn.yaml "
        f"experiment={experiment} "
        f"trainer={method} "
        f"task_name={task_name} "
        f"model={MODEL_8B} "
        f"forget_split={FORGET_SPLIT} "
        f"retain_split={RETAIN_SPLIT} "
        f"model.model_args.pretrained_model_name_or_path={model_path} "
        f"retain_logs_path=saves/eval/tofu_{MODEL_8B}_{RETAIN_SPLIT}/TOFU_EVAL.json "
        f"trainer.args.per_device_train_batch_size=16 "
        f"trainer.args.gradient_accumulation_steps=1 "
        f"trainer.args.num_train_epochs={num_epochs} "
        f"trainer.args.learning_rate={lr} "
        f"trainer.args.ddp_find_unused_parameters=true "
        f"trainer.args.gradient_checkpointing=true "
        f"trainer.args.save_strategy=epoch"
    )
    env = {"CUDA_VISIBLE_DEVICES": "0,1"}
    if not run_cmd(cmd, env=env, cwd=str(PROJECT_ROOT)):
        raise RuntimeError(f"Training failed for {method}")

    # Delete all checkpoints except the optimal one
    unlearn_dir = PROJECT_ROOT / "saves" / "unlearn" / task_name
    for d in unlearn_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-") and d.name != f"checkpoint-{checkpoint}":
            cleanup_model_files(d)
    # Delete top-level model files
    cleanup_model_files(unlearn_dir)

    return task_name, checkpoint


def retrain_distill(lr, num_epochs, temperature, master_port):
    """Retrain distill DD model. Returns path to checkpoint.
    Writes to models/ which is symlinked to /hpc_temp/ on Pythia."""
    output_dir = PROJECT_ROOT / "models" / "TOFU_Distill_Rebel" / f"lr_{lr}-temp-{temperature}"
    checkpoint_dir = output_dir / f"checkpoint-epoch-{num_epochs}"

    if (checkpoint_dir / "config.json").exists():
        print(f"Distill checkpoint already exists: {checkpoint_dir}")
        return str(checkpoint_dir)

    print(f"\nRetraining Distill DD: lr={lr}, temp={temperature}, epochs={num_epochs}...")
    cmd = (
        f"python scripts/distill/distill_model_tofu.py "
        f"--learning_rate {lr} "
        f"--num_epochs {num_epochs} "
        f"--per_device_batch_size 4 "
        f"--gradient_accumulation_steps 8 "
        f"--dd_alpha 1.5 "
        f"--dd_big {DD_BIG} "
        f"--dd_retain {DD_RETAIN} "
        f"--dd_forget {DD_FORGET} "
        f"--temperature {temperature} "
        f"--output_dir {output_dir} "
        f"--save_epochs {num_epochs}"
    )
    env = {"CUDA_VISIBLE_DEVICES": "0,1"}
    if not run_cmd(cmd, env=env, cwd=str(PROJECT_ROOT)):
        raise RuntimeError("Distillation training failed")

    return str(checkpoint_dir)


def retrain_offset(lr, epochs, batch_size, grad_accum):
    """Retrain Offset Unlearning model. Returns path to trained offset model."""
    output_dir = PROJECT_ROOT / "models" / "offset" / f"tofu_lr{lr}"

    if (output_dir / "config.json").exists():
        print(f"Offset model already exists: {output_dir}")
        return str(output_dir)

    print(f"\nRetraining Offset: lr={lr}, epochs={epochs}...")
    cmd = (
        f"python scripts/train/finetune_model_offset_unlearning.py "
        f"--target_model {DD_BIG} "
        f"--offset_model {OFFSET_MODEL} "
        f"--forget_data data/TOFU_downloaded/forget10.jsonl "
        f"--retain_data data/TOFU_downloaded/retain90.jsonl "
        f"--output_dir {output_dir} "
        f"--learning_rate {lr} "
        f"--epochs {epochs} "
        f"--batch_size {batch_size} "
        f"--gradient_accumulation_steps {grad_accum}"
    )
    env = {"CUDA_VISIBLE_DEVICES": "0"}
    if not run_cmd(cmd, env=env, cwd=str(PROJECT_ROOT)):
        raise RuntimeError("Offset training failed")
    return str(output_dir)


def retrain_uld(lr, epochs, batch_size, grad_accum, retain_weight):
    """Retrain ULD assistant model. Returns path to trained assistant."""
    output_dir = PROJECT_ROOT / "models" / "uld" / f"tofu_lr{lr}"

    if (output_dir / "config.json").exists():
        print(f"ULD assistant already exists: {output_dir}")
        return str(output_dir)

    print(f"\nRetraining ULD assistant: lr={lr}, epochs={epochs}...")
    cmd = (
        f"python scripts/train/finetune_model_uld.py "
        f"--target_model {ULD_TARGET} "
        f"--forget_data data/TOFU_downloaded/forget10.jsonl "
        f"--retain_data data/TOFU_downloaded/retain90.jsonl "
        f"--output_dir {output_dir} "
        f"--learning_rate {lr} "
        f"--epochs {epochs} "
        f"--batch_size {batch_size} "
        f"--gradient_accumulation_steps {grad_accum} "
        f"--retain_weight {retain_weight}"
    )
    env = {"CUDA_VISIBLE_DEVICES": "0"}
    if not run_cmd(cmd, env=env, cwd=str(PROJECT_ROOT)):
        raise RuntimeError("ULD training failed")
    return str(output_dir)


def retrain_whp(lr, epochs, batch_size, grad_accum):
    """Retrain WHP reinforced model. Returns path to trained model."""
    lr_str = lr.replace("+", "").replace("-0", "-")
    output_dir = PROJECT_ROOT / "models" / "whp" / f"tofu_lr{lr_str}"

    if (output_dir / "config.json").exists():
        print(f"WHP reinforced model already exists: {output_dir}")
        return str(output_dir)

    print(f"\nRetraining WHP reinforced model: lr={lr}, epochs={epochs}...")
    cmd = (
        f"python scripts/train/finetune_model_whp.py "
        f"--model_dir {WHP_BASELINE} "
        f"--forget_data data/TOFU_downloaded/forget10.jsonl "
        f"--output_dir {output_dir} "
        f"--learning_rate {lr} "
        f"--epochs {epochs} "
        f"--batch_size {batch_size} "
        f"--gradient_accumulation_steps {grad_accum} "
        f"--max_len 2048"
    )
    env = {"CUDA_VISIBLE_DEVICES": "0"}
    if not run_cmd(cmd, env=env, cwd=str(PROJECT_ROOT)):
        raise RuntimeError("WHP training failed")
    return str(output_dir)


# ── REBEL attack runners ─────────────────────────────────────────────────────

def run_vllm_attack(name, model_id, tokenizer_id, mode):
    """Run Leak@ or REBEL via subprocess (vLLM target).
    Results go to saves/eval/tofu/leak_at_k/ -> /hpc_temp/ via symlink."""
    results_dir = str(RESULTS_BASE / name / mode)
    os.makedirs(results_dir, exist_ok=True)

    # Skip if results already exist
    if mode == "leak" and (Path(results_dir) / "whole_generation_tofu.json").exists():
        print(f"Leak@ results already exist for {name}, skipping")
        return True
    if mode == "rebel" and any(Path(results_dir).glob("leak_report_idx*.json")):
        print(f"REBEL results already exist for {name}, skipping")
        return True

    cmd = [
        REBEL_PYTHON, "-m", "root.main", mode,
        "--data-path", str(REBEL_DATA),
        "--results-dir", results_dir,
        "--model-id", model_id,
        "--tokenizer-id", tokenizer_id,
        "--data-kind", "tofu",
    ]
    if mode == "leak":
        cmd += ["--num-attacks", str(LEAK_NUM_ATTACKS)]
    else:
        cmd += ["--mutations-list", REBEL_MUTATIONS, "--top-k-list", REBEL_TOP_K]

    print(f"\n{'='*60}")
    print(f"  {mode.upper()} on {name}")
    print(f"  Model: {model_id}")
    print(f"  Results: {results_dir}")
    print(f"{'='*60}\n")

    env = {"REBEL_BATCH_SIZE": str(REBEL_BATCH_SIZE)}
    result = subprocess.run(cmd, cwd=str(REBEL_DIR), env={**os.environ, **env})
    if result.returncode != 0:
        print(f"WARNING: {mode} on {name} failed (exit {result.returncode})")
        return False
    return True


def run_dd_attack_subprocess(name, dd_style, dd_param, dd_extra_args, mode):
    """Launch DD attack as a subprocess (calls this script in --dd-run mode)."""
    results_dir = str(RESULTS_BASE / name / mode)
    os.makedirs(results_dir, exist_ok=True)

    if mode == "leak" and (Path(results_dir) / "whole_generation_tofu.json").exists():
        print(f"Leak@ results already exist for {name}, skipping")
        return True
    if mode == "rebel" and any(Path(results_dir).glob("leak_report_idx*.json")):
        print(f"REBEL results already exist for {name}, skipping")
        return True

    # DD runner needs vllm (via REBEL imports) so must use REBEL_PYTHON.
    # src.model.dd works fine under the REBEL venv (torch + transformers present).
    cmd = [
        REBEL_PYTHON, str(Path(__file__)),
        "--dd-run",
        "--name", name,
        "--mode", mode,
        "--dd-style", dd_style,
        "--dd-param", str(dd_param),
    ] + dd_extra_args

    print(f"\n{'='*60}")
    print(f"  {mode.upper()} on {name} (DD mode)")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"WARNING: {mode} DD attack on {name} failed (exit {result.returncode})")
        return False
    return True


def run_uld_attack_subprocess(name, uld_config, mode):
    """Launch ULD attack as a subprocess (calls this script in --uld-run mode)."""
    results_dir = str(RESULTS_BASE / name / mode)
    os.makedirs(results_dir, exist_ok=True)

    if mode == "leak" and (Path(results_dir) / "whole_generation_tofu.json").exists():
        print(f"Leak@ results already exist for {name}, skipping")
        return True
    if mode == "rebel" and any(Path(results_dir).glob("leak_report_idx*.json")):
        print(f"REBEL results already exist for {name}, skipping")
        return True

    cmd = [
        REBEL_PYTHON, str(Path(__file__)),
        "--uld-run",
        "--name", name,
        "--mode", mode,
        "--uld-target", uld_config["model_uld_target"],
        "--uld-assistant", uld_config["model_uld_assistant"],
        "--uld-beta", str(uld_config["model_uld_beta"]),
    ]

    print(f"\n{'='*60}")
    print(f"  {mode.upper()} on {name} (ULD mode)")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"WARNING: {mode} ULD attack on {name} failed (exit {result.returncode})")
        return False
    return True


def run_whp_attack_subprocess(name, whp_config, mode):
    """Launch WHP attack as a subprocess (calls this script in --whp-run mode)."""
    results_dir = str(RESULTS_BASE / name / mode)
    os.makedirs(results_dir, exist_ok=True)

    if mode == "leak" and (Path(results_dir) / "whole_generation_tofu.json").exists():
        print(f"Leak@ results already exist for {name}, skipping")
        return True
    if mode == "rebel" and any(Path(results_dir).glob("leak_report_idx*.json")):
        print(f"REBEL results already exist for {name}, skipping")
        return True

    cmd = [
        REBEL_PYTHON, str(Path(__file__)),
        "--whp-run",
        "--name", name,
        "--mode", mode,
        "--whp-baseline", whp_config["model_whp_baseline"],
        "--whp-reinforced", whp_config["model_whp_reinforced"],
        "--whp-alpha", str(whp_config["model_whp_alpha"]),
    ]

    print(f"\n{'='*60}")
    print(f"  {mode.upper()} on {name} (WHP mode)")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"WARNING: {mode} WHP attack on {name} failed (exit {result.returncode})")
        return False
    return True


def _run_dd_attack_inprocess(name, mode, dd_config):
    """Actually run a DD attack in-process. Called in --dd-run mode."""
    # REBEL imports need REBEL dir on sys.path
    sys.path.insert(0, str(REBEL_DIR))
    # src/model/__init__.py uses `from model.X import ...` (not `from src.model.X`),
    # which only resolves when src/ is on sys.path (or project is pip-installed).
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    os.chdir(str(REBEL_DIR))

    import root.config as rebel_config
    from root.approaches.evolutionary import EvolutionaryAttack
    from root.approaches.naive import StaticAttack
    from root.models.dd_target import DDTargetLLM
    from root.models.hacker import HackerLLM
    from root.models.judge import JudgeLLM
    from root.models.judge_hacker_simgleton import JudgeHackerSingleton
    from root.utils.data import load_sampels
    from root.utils.logger import AttackLogger

    results_dir = str(RESULTS_BASE / name / mode)
    os.makedirs(results_dir, exist_ok=True)

    rebel_config.apply_cli_config(
        data_path=str(REBEL_DATA),
        results_dir=results_dir,
        data_kind="tofu",
        model_id=name,
        tokenizer_id=name,
    )
    if mode == "leak":
        rebel_config.NUM_ATTACKS = LEAK_NUM_ATTACKS
    else:
        rebel_config.TOP_K_LIST = [20, 12, 8, 5, 3]
        rebel_config.MUTATIONS_LIST = [1500, 80, 50, 40, 40]
    rebel_config.BATCH_SIZE = REBEL_BATCH_SIZE // 2

    data = load_sampels(str(REBEL_DATA))
    print(f"Data loaded: {len(data)} examples")

    # DD target loads HF models on cuda:0; hacker/judge vLLM with TP=2 spans
    # both GPUs but DD_GPU_MEM_UTIL leaves room for DD's HF models (~20GB)
    rebel_config.GPU_MEM_UTIL = DD_GPU_MEM_UTIL
    target = DDTargetLLM(dd_config, device="cuda:0", batch_size=64)

    singleton = JudgeHackerSingleton(
        hacker_class=HackerLLM,
        judge_class=JudgeLLM,
        dtype=rebel_config.DTYPE,
        tensor_parallel_size=rebel_config.TP,
        gpu_mem_util=rebel_config.GPU_MEM_UTIL,
    )
    hacker = singleton.get_hacker()
    judge = singleton.get_judge()

    try:
        start = time.time()
        if mode == "leak":
            attack = StaticAttack(num_attacks=LEAK_NUM_ATTACKS)
            logger = AttackLogger(results_dir=Path(results_dir))
            attack.run(target, hacker, judge, data, logger)
        else:
            attack = EvolutionaryAttack(
                top_k_list=[20, 12, 8, 5, 3],
                mutations_list=[1500, 80, 50, 40, 40],
            )
            for idx in data:
                t0 = time.time()
                attack.run(
                    target, hacker, judge, data[idx],
                    use_trackers=False, do_stats=False,
                    stop_at_first=True, idx=idx,
                )
                print(f"Attack {idx} finished in {time.time() - t0:.1f}s")
                gc.collect()
        print(f"{mode} on {name} finished in {time.time() - start:.1f}s")
    finally:
        target.unload()
        hacker.unload()
        judge.unload()
        gc.collect()


def _run_uld_attack_inprocess(name, mode, uld_config):
    """Actually run a ULD attack in-process. Called in --uld-run mode."""
    sys.path.insert(0, str(REBEL_DIR))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    os.chdir(str(REBEL_DIR))

    import root.config as rebel_config
    from root.approaches.evolutionary import EvolutionaryAttack
    from root.approaches.naive import StaticAttack
    from root.models.uld_target import ULDTargetLLM
    from root.models.hacker import HackerLLM
    from root.models.judge import JudgeLLM
    from root.models.judge_hacker_simgleton import JudgeHackerSingleton
    from root.utils.data import load_sampels
    from root.utils.logger import AttackLogger

    results_dir = str(RESULTS_BASE / name / mode)
    os.makedirs(results_dir, exist_ok=True)

    rebel_config.apply_cli_config(
        data_path=str(REBEL_DATA),
        results_dir=results_dir,
        data_kind="tofu",
        model_id=name,
        tokenizer_id=name,
    )
    if mode == "leak":
        rebel_config.NUM_ATTACKS = LEAK_NUM_ATTACKS
    else:
        rebel_config.TOP_K_LIST = [20, 12, 8, 5, 3]
        rebel_config.MUTATIONS_LIST = [1500, 80, 50, 40, 40]
    rebel_config.BATCH_SIZE = REBEL_BATCH_SIZE // 2

    data = load_sampels(str(REBEL_DATA))
    print(f"Data loaded: {len(data)} examples")

    rebel_config.GPU_MEM_UTIL = DD_GPU_MEM_UTIL
    target = ULDTargetLLM(uld_config, device="cuda:0", batch_size=64)

    singleton = JudgeHackerSingleton(
        hacker_class=HackerLLM,
        judge_class=JudgeLLM,
        dtype=rebel_config.DTYPE,
        tensor_parallel_size=rebel_config.TP,
        gpu_mem_util=rebel_config.GPU_MEM_UTIL,
    )
    hacker = singleton.get_hacker()
    judge = singleton.get_judge()

    try:
        start = time.time()
        if mode == "leak":
            attack = StaticAttack(num_attacks=LEAK_NUM_ATTACKS)
            logger = AttackLogger(results_dir=Path(results_dir))
            attack.run(target, hacker, judge, data, logger)
        else:
            attack = EvolutionaryAttack(
                top_k_list=[20, 12, 8, 5, 3],
                mutations_list=[1500, 80, 50, 40, 40],
            )
            for idx in data:
                t0 = time.time()
                attack.run(
                    target, hacker, judge, data[idx],
                    use_trackers=False, do_stats=False,
                    stop_at_first=True, idx=idx,
                )
                print(f"Attack {idx} finished in {time.time() - t0:.1f}s")
                gc.collect()
        print(f"{mode} on {name} finished in {time.time() - start:.1f}s")
    finally:
        target.unload()
        hacker.unload()
        judge.unload()
        gc.collect()


def _run_whp_attack_inprocess(name, mode, whp_config):
    """Actually run a WHP attack in-process. Called in --whp-run mode."""
    sys.path.insert(0, str(REBEL_DIR))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    os.chdir(str(REBEL_DIR))

    import root.config as rebel_config
    from root.approaches.evolutionary import EvolutionaryAttack
    from root.approaches.naive import StaticAttack
    from root.models.whp_target import WHPTargetLLM
    from root.models.hacker import HackerLLM
    from root.models.judge import JudgeLLM
    from root.models.judge_hacker_simgleton import JudgeHackerSingleton
    from root.utils.data import load_sampels
    from root.utils.logger import AttackLogger

    results_dir = str(RESULTS_BASE / name / mode)
    os.makedirs(results_dir, exist_ok=True)

    rebel_config.apply_cli_config(
        data_path=str(REBEL_DATA),
        results_dir=results_dir,
        data_kind="tofu",
        model_id=name,
        tokenizer_id=name,
    )
    if mode == "leak":
        rebel_config.NUM_ATTACKS = LEAK_NUM_ATTACKS
    else:
        rebel_config.TOP_K_LIST = [20, 12, 8, 5, 3]
        rebel_config.MUTATIONS_LIST = [1500, 80, 50, 40, 40]
    rebel_config.BATCH_SIZE = REBEL_BATCH_SIZE // 2

    data = load_sampels(str(REBEL_DATA))
    print(f"Data loaded: {len(data)} examples")

    rebel_config.GPU_MEM_UTIL = DD_GPU_MEM_UTIL
    target = WHPTargetLLM(whp_config, device="cuda:0", batch_size=32)

    singleton = JudgeHackerSingleton(
        hacker_class=HackerLLM,
        judge_class=JudgeLLM,
        dtype=rebel_config.DTYPE,
        tensor_parallel_size=rebel_config.TP,
        gpu_mem_util=rebel_config.GPU_MEM_UTIL,
    )
    hacker = singleton.get_hacker()
    judge = singleton.get_judge()

    try:
        start = time.time()
        if mode == "leak":
            attack = StaticAttack(num_attacks=LEAK_NUM_ATTACKS)
            logger = AttackLogger(results_dir=Path(results_dir))
            attack.run(target, hacker, judge, data, logger)
        else:
            attack = EvolutionaryAttack(
                top_k_list=[20, 12, 8, 5, 3],
                mutations_list=[1500, 80, 50, 40, 40],
            )
            for idx in data:
                t0 = time.time()
                attack.run(
                    target, hacker, judge, data[idx],
                    use_trackers=False, do_stats=False,
                    stop_at_first=True, idx=idx,
                )
                print(f"Attack {idx} finished in {time.time() - t0:.1f}s")
                gc.collect()
        print(f"{mode} on {name} finished in {time.time() - start:.1f}s")
    finally:
        target.unload()
        hacker.unload()
        judge.unload()
        gc.collect()


# ── Orchestrator ──────────────────────────────────────────────────────────────

def build_model_list(methods_filter=None):
    """Build ordered list of models to attack (matches tofu_scores.py table)."""
    models = []

    # ── HuggingFace baselines (no retraining) ──
    for name, model_id, tok_id in [
        ("Target",      "open-unlearning/tofu_Llama-3.1-8B-Instruct_full",      TOKENIZER_8B),
        ("Retrain",     "open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90",  TOKENIZER_8B),
    ]:
        models.append({"name": name, "type": "vllm", "model_id": model_id, "tokenizer_id": tok_id})

    # ── 7 gradient unlearning methods (retrain optimal config) ──
    for method, cfg in OPTIMAL_UNLEARN.items():
        models.append({"name": method, "type": "unlearn", "method": method, **cfg})

    # ── DD methods (inference-time, no saved weights) ──
    for name, cfg in OPTIMAL_DD.items():
        models.append({"name": name, "type": "dd", **cfg})

    # ── Distill DD (retrain student) ──
    models.append({"name": "Distill_DD", "type": "distill", **OPTIMAL_DISTILL})

    # ── Offset Unlearning (retrain offset model, eval as DD with alpha=1) ──
    models.append({"name": "Offset", "type": "offset", **OPTIMAL_OFFSET})

    # ── ULD (retrain assistant, eval as target - beta * assistant) ──
    models.append({"name": "ULD", "type": "uld", **OPTIMAL_ULD})

    # ── WHP (retrain reinforced model, eval as baseline - alpha * ReLU(reinforced - baseline)) ──
    models.append({"name": "WHP", "type": "whp", **OPTIMAL_WHP})

    if methods_filter:
        allowed = set(methods_filter)
        models = [m for m in models if m["name"] in allowed]

    return models


def run_orchestrator(args):
    """Main orchestrator loop."""
    print("=" * 60)
    print("  TOFU Leak@K / REBEL Benchmark")
    print("=" * 60)

    methods_filter = args.methods.split(",") if args.methods else None
    models = build_model_list(methods_filter)

    print(f"\nModels to attack ({len(models)}):")
    for m in models:
        print(f"  - {m['name']} ({m['type']})")

    master_port = get_free_port()
    modes = []
    if not args.rebel_only:
        modes.append("leak")
    if not args.leak_only:
        modes.append("rebel")

    results = {}

    for model_info in models:
        name = model_info["name"]
        mtype = model_info["type"]
        print(f"\n{'#'*60}")
        print(f"# Processing: {name} ({mtype})")
        print(f"{'#'*60}")

        model_id = None
        tokenizer_id = None

        # ── Prepare model (retrain if needed) ──
        if mtype == "vllm":
            model_id = model_info["model_id"]
            tokenizer_id = model_info["tokenizer_id"]

        elif mtype == "unlearn":
            method = model_info["method"]
            lr = model_info["lr"]
            epoch = model_info["epoch"]
            task_name, cp = retrain_unlearning_method(method, lr, epoch, master_port)
            model_id = str(PROJECT_ROOT / "saves" / "unlearn" / task_name / f"checkpoint-{cp}")
            tokenizer_id = TOKENIZER_8B

        elif mtype == "distill":
            lr = model_info["lr"]
            epoch = model_info["epoch"]
            temp = model_info["temperature"]
            model_id = retrain_distill(lr, epoch, temp, master_port)
            tokenizer_id = TOKENIZER_1B

        elif mtype == "offset":
            lr = model_info["lr"]
            offset_model_path = retrain_offset(
                lr, model_info["epochs"], model_info["batch_size"], model_info["grad_accum"])
            # Offset uses DD at inference: big=target, retain=trained, forget=reference
            model_id = offset_model_path  # for cleanup later

        elif mtype == "uld":
            lr = model_info["lr"]
            uld_assistant_path = retrain_uld(
                lr, model_info["epochs"], model_info["batch_size"],
                model_info["grad_accum"], model_info["retain_weight"])

        elif mtype == "whp":
            lr = model_info["lr"]
            whp_reinforced_path = retrain_whp(
                lr, model_info["epochs"], model_info["batch_size"],
                model_info["grad_accum"])

        # ── Run attacks ──
        for mode in modes:
            if mtype == "dd":
                run_dd_attack_subprocess(
                    name,
                    model_info["style"],
                    model_info["param"],
                    model_info.get("extra_args", []),
                    mode,
                )
            elif mtype == "offset":
                # Offset eval uses DD: big=target, retain=trained offset, forget=reference offset
                run_dd_attack_subprocess(
                    name,
                    "alpha",       # DD style
                    1.0,           # alpha=1.0
                    ["--dd-offset-retain", offset_model_path,
                     "--dd-offset-forget", OFFSET_MODEL],
                    mode,
                )
            elif mtype == "uld":
                uld_config = {
                    "model_uld_target": ULD_TARGET,
                    "model_uld_assistant": uld_assistant_path,
                    "model_uld_beta": model_info["beta"],
                }
                run_uld_attack_subprocess(name, uld_config, mode)
            elif mtype == "whp":
                whp_config = {
                    "model_whp_baseline": WHP_BASELINE,
                    "model_whp_reinforced": whp_reinforced_path,
                    "model_whp_alpha": model_info["alpha"],
                }
                run_whp_attack_subprocess(name, whp_config, mode)
            else:
                run_vllm_attack(name, model_id, tokenizer_id, mode)

        # ── Cleanup weights (free /hpc_temp/ space for next model) ──
        if mtype == "unlearn" and model_id:
            print(f"Cleaning up weights: {model_id}")
            cleanup_model_files(model_id)
        elif mtype == "distill" and model_id:
            print(f"Cleaning up weights: {model_id}")
            cleanup_model_files(model_id)

        # ── Parse results ──
        result = {"name": name}
        leak_dir = RESULTS_BASE / name / "leak"
        rebel_dir = RESULTS_BASE / name / "rebel"
        if leak_dir.exists():
            result["leak_asr"] = parse_leak_asr(leak_dir)
            for k in LEAK_K_VALUES:
                result[f"leak@{k}"] = parse_leak_asr(leak_dir, k=k)
        if rebel_dir.exists():
            result["rebel_asr"] = parse_rebel_asr(rebel_dir)
        results[name] = result
        print(f"  Results so far: {result}")

    # ── Save summary (to saves/ -> /hpc_temp/, synced back by slurm_run.sh) ──
    summary_path = RESULTS_BASE / "results_summary.json"
    os.makedirs(RESULTS_BASE, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results summary: {summary_path}")
    print(f"{'='*60}")

    # ── Leak@K table ──
    k_headers = "".join(f" {'Leak@'+str(k):>10}" for k in LEAK_K_VALUES)
    print(f"\n{'Method':<15}{k_headers} {'REBEL ASR':>10}")
    print("-" * (15 + 10 * len(LEAK_K_VALUES) + 11))
    for name, r in results.items():
        cols = ""
        for k in LEAK_K_VALUES:
            v = r.get(f"leak@{k}")
            cols += f" {f'{v:.1%}':>10}" if isinstance(v, (int, float)) else f" {'N/A':>10}"
        rebel = f"{r.get('rebel_asr', 0):.1%}" if isinstance(r.get("rebel_asr"), (int, float)) else "N/A"
        print(f"{name:<15}{cols} {rebel:>10}")


# ── DD runner (subprocess entry point) ────────────────────────────────────────

def run_dd_mode(args):
    """Run a single DD attack in-process (called as subprocess by orchestrator)."""
    # Offset uses custom retain/forget models; standard DD uses the defaults
    if args.dd_offset_retain:
        dd_config = {
            "model_dd_big": DD_BIG,
            "model_dd_retain": args.dd_offset_retain,
            "model_dd_forget": args.dd_offset_forget,
            "model_dd_use_ngram": "No",
            "model_dd_alpha": float(args.dd_param),
        }
    else:
        dd_config = {
            "model_dd_big": DD_BIG,
            "model_dd_retain": DD_RETAIN,
            "model_dd_forget": DD_FORGET,
            "model_dd_use_ngram": "No",
        }
        if args.dd_style == "alpha":
            dd_config["model_dd_alpha"] = float(args.dd_param)
        elif args.dd_style == "topk":
            dd_config["model_dd_topk"] = int(float(args.dd_param))
            dd_config["topk_vocab"] = "TOFU"
            if args.dd_monte_carlo:
                dd_config["model_dd_monte_carlo"] = "Yes"

    _run_dd_attack_inprocess(args.name, args.mode, dd_config)


def run_uld_mode(args):
    """Run a single ULD attack in-process (called as subprocess by orchestrator)."""
    uld_config = {
        "model_uld_target": args.uld_target,
        "model_uld_assistant": args.uld_assistant,
        "model_uld_beta": float(args.uld_beta),
    }
    _run_uld_attack_inprocess(args.name, args.mode, uld_config)


def run_whp_mode(args):
    """Run a single WHP attack in-process (called as subprocess by orchestrator)."""
    whp_config = {
        "model_whp_baseline": args.whp_baseline,
        "model_whp_reinforced": args.whp_reinforced,
        "model_whp_alpha": float(args.whp_alpha),
    }
    _run_whp_attack_inprocess(args.name, args.mode, whp_config)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TOFU Leak@K / REBEL benchmark runner")

    # Orchestrator args
    parser.add_argument("--methods", default=None,
                        help="Comma-separated list of methods to run (default: all)")
    parser.add_argument("--leak-only", action="store_true",
                        help="Only run Leak@K attacks (skip REBEL evolutionary)")
    parser.add_argument("--rebel-only", action="store_true",
                        help="Only run REBEL evolutionary (skip Leak@K)")

    # DD runner args (used when called as subprocess)
    parser.add_argument("--dd-run", action="store_true",
                        help="DD runner mode (internal use)")
    parser.add_argument("--name", default=None)
    parser.add_argument("--mode", default=None, choices=["leak", "rebel"])
    parser.add_argument("--dd-style", default=None, choices=["alpha", "topk"])
    parser.add_argument("--dd-param", default=None)
    parser.add_argument("--dd-monte-carlo", action="store_true")
    parser.add_argument("--dd-offset-retain", default=None,
                        help="Offset: path to trained offset model (used as dd_retain)")
    parser.add_argument("--dd-offset-forget", default=None,
                        help="Offset: path to reference offset model (used as dd_forget)")

    # ULD runner args (used when called as subprocess)
    parser.add_argument("--uld-run", action="store_true",
                        help="ULD runner mode (internal use)")
    parser.add_argument("--uld-target", default=None)
    parser.add_argument("--uld-assistant", default=None)
    parser.add_argument("--uld-beta", default=None)

    # WHP runner args (used when called as subprocess)
    parser.add_argument("--whp-run", action="store_true",
                        help="WHP runner mode (internal use)")
    parser.add_argument("--whp-baseline", default=None)
    parser.add_argument("--whp-reinforced", default=None)
    parser.add_argument("--whp-alpha", default=None)

    args = parser.parse_args()

    if args.dd_run:
        run_dd_mode(args)
    elif args.uld_run:
        run_uld_mode(args)
    elif args.whp_run:
        run_whp_mode(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
