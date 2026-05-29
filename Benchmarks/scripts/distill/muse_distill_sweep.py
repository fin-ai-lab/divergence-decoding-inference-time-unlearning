#!/usr/bin/env python3
"""
Sweep over learning rates and temperatures for DD distillation on MUSE.

Uses alpha=0.85 (average of 0.8 for knowmem and 0.9 for verbmem).
Evaluates at epoch 5.
Picks optimal config by distance to retrain baseline.
"""

import subprocess
import os
import json
from pathlib import Path
from tabulate import tabulate


def run_command(cmd, env=None):
    """Run a shell command with optional environment variables"""
    if env:
        full_env = os.environ.copy()
        full_env.update(env)
    else:
        full_env = None

    print(f"\n{'='*60}")
    print(f"Running: {cmd[:100]}...")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, shell=True, env=full_env)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        return False
    return True


def get_model_output_dir():
    """Get the model output directory for training checkpoints"""
    return "models/MUSE_Distill"


def get_eval_output_dir():
    """Get the eval output directory for evaluation results"""
    return "saves/eval/muse_distill"


def training_completed(lr, epoch, temp):
    """Check if training checkpoint exists"""
    model_dir = get_model_output_dir()
    checkpoint_dir = Path(model_dir) / f"lr_{lr}-temp-{temp}" / f"checkpoint-epoch-{epoch}"
    if (checkpoint_dir / "config.json").exists():
        print(f"Training already completed: {checkpoint_dir}")
        return True
    return False


def get_task_name(lr, epoch, temp):
    """Generate task name for a config"""
    return f"lr-{lr}-epoch-{epoch}-temp-{temp}"


def evaluation_completed(lr, epoch, temp):
    """Check if evaluation is already completed"""
    task_name = get_task_name(lr, epoch, temp)
    eval_file = Path(f"saves/eval/muse_distill/{task_name}/MUSE_SUMMARY.json")
    if eval_file.exists():
        try:
            with open(eval_file, 'r') as f:
                data = json.load(f)
            if len(data.keys()) >= 3:
                print(f"Evaluation already completed: {eval_file}")
                return True
        except (json.JSONDecodeError, IOError):
            pass
    return False


def get_eval_results(lr, epoch, temp):
    """Get evaluation results from JSON file"""
    task_name = get_task_name(lr, epoch, temp)
    eval_file = Path(f"saves/eval/muse_distill/{task_name}/MUSE_SUMMARY.json")
    try:
        with open(eval_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def run_distillation(lr, num_epochs, output_dir, args, save_epochs=None, temperature=1.0):
    """Run distillation training"""
    cmd = f"""python scripts/distill/distill_model_muse.py \
        --learning_rate {lr} \
        --num_epochs {num_epochs} \
        --per_device_batch_size {args['batch_size']} \
        --gradient_accumulation_steps {args['grad_accum']} \
        --dd_alpha {args['dd_alpha']} \
        --dd_big {args['dd_big']} \
        --dd_retain {args['dd_retain']} \
        --dd_forget {args['dd_forget']} \
        --data_path {args['data_path']} \
        --temperature {temperature} \
        --output_dir {output_dir}"""

    if save_epochs:
        cmd += f" --save_epochs {','.join(map(str, save_epochs))}"

    env = {'CUDA_VISIBLE_DEVICES': args.get('gpu', '0')}
    return run_command(cmd, env)


def run_evaluation(model_path, output_dir, task_name, args):
    """Run MUSE evaluation on a distilled checkpoint"""
    cmd = f"""python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split={args['data_split']} \
        model.model_args.pretrained_model_name_or_path={model_path} \
        paths.output_dir={output_dir} \
        task_name=muse_distill/{task_name}"""

    env = {'CUDA_VISIBLE_DEVICES': args.get('gpu', '0')}
    return run_command(cmd, env)


def load_retrain_baseline():
    """Load retrain baseline scores for distance calculation"""
    try:
        info = json.load(open("saves/eval/muse_main/muse_retrain/MUSE_SUMMARY.json"))
        return {
            'forget_verbmem_ROUGE': info['forget_verbmem_ROUGE'] * 100,
            'forget_knowmem_ROUGE': info['forget_knowmem_ROUGE'] * 100,
            'retain_knowmem_ROUGE': info['retain_knowmem_ROUGE'] * 100
        }
    except FileNotFoundError:
        print("Warning: Could not load retrain baseline. Distance calculations will be skipped.")
        return None


def calculate_distance_verbmem(scores, retrain_scores):
    """Calculate euclidean distance from retrain using verbmem and retain metrics"""
    forget_verbmem_diff = scores['forget_verbmem_ROUGE'] - retrain_scores['forget_verbmem_ROUGE']
    retain_diff = scores['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE']
    return (forget_verbmem_diff**2 + retain_diff**2)**0.5


def calculate_distance_knowmem(scores, retrain_scores):
    """Calculate euclidean distance from retrain using knowmem and retain metrics"""
    forget_knowmem_diff = scores['forget_knowmem_ROUGE'] - retrain_scores['forget_knowmem_ROUGE']
    retain_diff = scores['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE']
    return (forget_knowmem_diff**2 + retain_diff**2)**0.5


def calculate_distance_avg(scores, retrain_scores):
    """Calculate average of verbmem and knowmem distances"""
    dist_verbmem = calculate_distance_verbmem(scores, retrain_scores)
    dist_knowmem = calculate_distance_knowmem(scores, retrain_scores)
    return (dist_verbmem + dist_knowmem) / 2


def run_sweep(args, learning_rates, epoch_checkpoints, temperatures):
    """Run distillation sweep"""
    model_output_dir = get_model_output_dir()
    max_epochs = max(epoch_checkpoints)
    results = []

    print(f"\n{'#'*80}")
    print(f"# Running sweep (alpha={args['dd_alpha']})")
    print(f"{'#'*80}")

    for temp in temperatures:
        for lr in learning_rates:
            print(f"\n{'#'*80}")
            print(f"# Temperature: {temp}, Learning Rate: {lr}")
            print(f"{'#'*80}")

            # Check if we need to train
            needs_training = False
            for epoch in epoch_checkpoints:
                if not training_completed(lr, epoch, temp):
                    needs_training = True
                    break

            # Run training if needed
            if needs_training:
                print(f"\nRunning distillation with temp={temp}, lr={lr} for {max_epochs} epochs...")
                lr_temp_output_dir = Path(model_output_dir) / f"lr_{lr}-temp-{temp}"
                success = run_distillation(lr, max_epochs, str(lr_temp_output_dir), args,
                                          save_epochs=epoch_checkpoints, temperature=temp)
                if not success:
                    print(f"Distillation failed for temp={temp}, lr={lr}")
                    continue
            else:
                print(f"\nAll checkpoints exist for temp={temp}, lr={lr}, skipping training...")

            # Evaluate each epoch checkpoint
            for epoch in epoch_checkpoints:
                checkpoint_path = Path(model_output_dir) / f"lr_{lr}-temp-{temp}" / f"checkpoint-epoch-{epoch}"
                task_name = get_task_name(lr, epoch, temp)

                if not checkpoint_path.exists():
                    print(f"Checkpoint not found: {checkpoint_path}")
                    continue

                # Run evaluation if needed
                if not evaluation_completed(lr, epoch, temp):
                    print(f"\nEvaluating lr={lr}, epoch={epoch}, temp={temp}...")
                    eval_output_dir = Path(get_eval_output_dir()) / task_name
                    success = run_evaluation(str(checkpoint_path), str(eval_output_dir), task_name, args)
                    if not success:
                        print(f"Evaluation failed for lr={lr}, epoch={epoch}, temp={temp}")
                        continue

                # Get results
                eval_results = get_eval_results(lr, epoch, temp)
                if eval_results:
                    results.append({
                        'lr': lr,
                        'epoch': epoch,
                        'temperature': temp,
                        'forget_verbmem_ROUGE': eval_results.get('forget_verbmem_ROUGE', 'N/A'),
                        'forget_knowmem_ROUGE': eval_results.get('forget_knowmem_ROUGE', 'N/A'),
                        'retain_knowmem_ROUGE': eval_results.get('retain_knowmem_ROUGE', 'N/A'),
                    })

    return results


def print_results_table(results, retrain_scores=None):
    """Print results table"""
    print(f"\n{'='*120}")
    print(f"DISTILLATION RESULTS SUMMARY")
    print(f"{'='*120}")

    if not results:
        print("No results collected.")
        return

    headers = ['Temp', 'LR', 'Epoch', 'Forget Verbmem', 'Forget Knowmem', 'Retain Knowmem', 'Dist Verbmem', 'Dist Knowmem', 'Dist Avg']
    table_data = []

    for r in sorted(results, key=lambda x: (x['temperature'], x['lr'], x['epoch'])):
        dist_verbmem = 'N/A'
        dist_knowmem = 'N/A'
        dist_avg = 'N/A'

        if retrain_scores and isinstance(r['forget_verbmem_ROUGE'], float):
            scores = {
                'forget_verbmem_ROUGE': r['forget_verbmem_ROUGE'] * 100,
                'forget_knowmem_ROUGE': r['forget_knowmem_ROUGE'] * 100 if isinstance(r['forget_knowmem_ROUGE'], float) else 0,
                'retain_knowmem_ROUGE': r['retain_knowmem_ROUGE'] * 100 if isinstance(r['retain_knowmem_ROUGE'], float) else 0,
            }
            dist_verbmem = calculate_distance_verbmem(scores, retrain_scores)
            dist_knowmem = calculate_distance_knowmem(scores, retrain_scores)
            dist_avg = (dist_verbmem + dist_knowmem) / 2

        row = [
            r['temperature'],
            f"{r['lr']:.0e}",
            r['epoch'],
            f"{r['forget_verbmem_ROUGE']*100:.2f}%" if isinstance(r['forget_verbmem_ROUGE'], float) else r['forget_verbmem_ROUGE'],
            f"{r['forget_knowmem_ROUGE']*100:.2f}%" if isinstance(r['forget_knowmem_ROUGE'], float) else r['forget_knowmem_ROUGE'],
            f"{r['retain_knowmem_ROUGE']*100:.2f}%" if isinstance(r['retain_knowmem_ROUGE'], float) else r['retain_knowmem_ROUGE'],
            f"{dist_verbmem:.2f}" if isinstance(dist_verbmem, float) else dist_verbmem,
            f"{dist_knowmem:.2f}" if isinstance(dist_knowmem, float) else dist_knowmem,
            f"{dist_avg:.2f}" if isinstance(dist_avg, float) else dist_avg,
        ]
        table_data.append(row)

    print(tabulate(table_data, headers=headers, tablefmt='grid'))

    # Find best config by average distance
    if retrain_scores:
        valid_results = []
        for r in results:
            if isinstance(r['forget_verbmem_ROUGE'], float):
                scores = {
                    'forget_verbmem_ROUGE': r['forget_verbmem_ROUGE'] * 100,
                    'forget_knowmem_ROUGE': r['forget_knowmem_ROUGE'] * 100 if isinstance(r['forget_knowmem_ROUGE'], float) else 0,
                    'retain_knowmem_ROUGE': r['retain_knowmem_ROUGE'] * 100 if isinstance(r['retain_knowmem_ROUGE'], float) else 0,
                }
                dist_avg = calculate_distance_avg(scores, retrain_scores)
                valid_results.append((r, dist_avg))

        if valid_results:
            best = min(valid_results, key=lambda x: x[1])
            print(f"\nBest config by average distance to retrain:")
            print(f"  Temperature: {best[0]['temperature']}, LR: {best[0]['lr']:.0e}, Epoch: {best[0]['epoch']}")
            print(f"  Avg Distance: {best[1]:.2f}")
            print(f"  Forget Verbmem: {best[0]['forget_verbmem_ROUGE']*100:.2f}%")
            print(f"  Forget Knowmem: {best[0]['forget_knowmem_ROUGE']*100:.2f}%" if isinstance(best[0]['forget_knowmem_ROUGE'], float) else "")
            print(f"  Retain Knowmem: {best[0]['retain_knowmem_ROUGE']*100:.2f}%" if isinstance(best[0]['retain_knowmem_ROUGE'], float) else "")


def main():
    eval_output_dir = get_eval_output_dir()

    # Base configuration
    args = {
        # Training config
        'dd_big': 'muse-bench/MUSE-News_target',
        'dd_retain': 'models/1.3b/model_1',
        'dd_forget': 'models/1.3b/model_2',
        'dd_alpha': 0.85,  # Average of 0.8 (knowmem) and 0.9 (verbmem)
        'data_path': 'data/news/raw/forget.txt',
        'batch_size': 16,
        'grad_accum': 2,  # effective batch size = 32
        'gpu': '0',
        # Eval config
        'data_split': 'News',
    }

    # Sweep configuration
    learning_rates = [2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4, 1.25e-4, 1.5e-4]
    epoch_checkpoints = [5]  # Fixed to 5 epochs
    temperatures = [0.25, 0.5, 1.0, 1.5, 2.0]

    print("=" * 80)
    print("MUSE DD Distillation Sweep")
    print("=" * 80)
    print(f"Alpha: {args['dd_alpha']} (average of verbmem=0.9 and knowmem=0.8)")
    print(f"Learning rates: {learning_rates}")
    print(f"Epoch checkpoints: {epoch_checkpoints}")
    print(f"Temperatures: {temperatures}")
    print(f"DD Big: {args['dd_big']}")
    print(f"DD Retain: {args['dd_retain']}")
    print(f"DD Forget: {args['dd_forget']}")
    print("=" * 80)

    # Load retrain baseline for distance calculations
    retrain_scores = load_retrain_baseline()

    results = run_sweep(args, learning_rates, epoch_checkpoints, temperatures)

    # Print results
    print_results_table(results, retrain_scores)

    # Save results to JSON
    results_file = Path(eval_output_dir) / "sweep_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    if retrain_scores:
        print(f"\nRetrain baseline:")
        print(f"  Forget Verbmem: {retrain_scores['forget_verbmem_ROUGE']:.2f}%")
        print(f"  Forget Knowmem: {retrain_scores['forget_knowmem_ROUGE']:.2f}%")
        print(f"  Retain Knowmem: {retrain_scores['retain_knowmem_ROUGE']:.2f}%")


if __name__ == "__main__":
    main()
