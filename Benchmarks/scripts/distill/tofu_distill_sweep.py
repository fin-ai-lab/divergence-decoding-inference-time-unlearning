#!/usr/bin/env python3
"""
Sweep over learning rates and epochs for DD distillation, then evaluate and print results.
"""

import subprocess
import os
import sys
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
    return "models/TOFU_Distill"


def get_eval_output_dir():
    """Get the eval output directory for evaluation results"""
    return "saves/eval/tofu_distill"


def training_completed(lr, epoch, temp):
    """Check if training checkpoint exists"""
    model_dir = get_model_output_dir()
    checkpoint_dir = Path(model_dir) / f"lr_{lr}-temp-{temp}" / f"checkpoint-epoch-{epoch}"
    # Check for model files
    if (checkpoint_dir / "config.json").exists():
        print(f"Training already completed: {checkpoint_dir}")
        return True
    return False


def evaluation_completed(lr, epoch, temp):
    """Check if evaluation is already completed"""
    eval_dir = get_eval_output_dir()
    eval_file = Path(eval_dir) / f"lr-{lr}-epoch-{epoch}-temp-{temp}" / "TOFU_SUMMARY.json"
    if eval_file.exists():
        try:
            with open(eval_file, 'r') as f:
                data = json.load(f)
            if len(data.keys()) >= 5:  # Has meaningful results
                print(f"Evaluation already completed: {eval_file}")
                return True
        except (json.JSONDecodeError, IOError):
            pass
    return False


def get_eval_results(lr, epoch, temp):
    """Get evaluation results from JSON file"""
    eval_dir = get_eval_output_dir()
    eval_file = Path(eval_dir) / f"lr-{lr}-epoch-{epoch}-temp-{temp}" / "TOFU_SUMMARY.json"
    try:
        with open(eval_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def run_distillation(lr, num_epochs, output_dir, args, save_epochs=None, temperature=1.0):
    """Run distillation training"""
    cmd = f"""python scripts/distill/distill_model_tofu.py \
        --learning_rate {lr} \
        --num_epochs {num_epochs} \
        --per_device_batch_size {args['batch_size']} \
        --gradient_accumulation_steps {args['grad_accum']} \
        --dd_alpha {args['dd_alpha']} \
        --dd_big {args['dd_big']} \
        --dd_retain {args['dd_retain']} \
        --dd_forget {args['dd_forget']} \
        --temperature {temperature} \
        --output_dir {output_dir}"""

    if save_epochs:
        cmd += f" --save_epochs {','.join(map(str, save_epochs))}"

    env = {'CUDA_VISIBLE_DEVICES': args.get('gpu', '0')}
    return run_command(cmd, env)


def run_evaluation(model_path, output_dir, args):
    """Run TOFU evaluation on a checkpoint"""
    cmd = f"""python src/eval.py \
        experiment=eval/tofu/default.yaml \
        forget_split={args['forget_split']} \
        holdout_split={args['holdout_split']} \
        model={args['model_config']} \
        task_name=distill_eval_{model_path} \
        model.model_args.pretrained_model_name_or_path={model_path} \
        paths.output_dir={output_dir} \
        retain_logs_path=saves/eval/tofu_{args['model_config']}_{args['retain_split']}/TOFU_EVAL.json"""

    env = {'CUDA_VISIBLE_DEVICES': args.get('gpu', '0')}
    return run_command(cmd, env)


def main():
    # Configuration
    model_output_dir = get_model_output_dir()
    eval_output_dir = get_eval_output_dir()

    # DD Model configuration
    args = {
        'dd_big': 'open-unlearning/tofu_Llama-3.1-8B-Instruct_full',
        'dd_retain': 'open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90',
        'dd_forget': 'open-unlearning/tofu_Llama-3.2-1B-Instruct_full',
        'dd_alpha': 1.5,
        'batch_size': 32,
        'grad_accum': 1,  # effective batch size = 32
        'gpu': '0',
        # Eval config
        'model_config': 'Llama-3.1-8B-Instruct',
        'forget_split': 'forget10',
        'holdout_split': 'holdout10',
        'retain_split': 'retain90',
    }

    # Sweep configuration
    learning_rates = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5]
    epoch_checkpoints = [5, 10]  # Which epochs to save and evaluate
    temperatures = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]  # Distillation temperatures to sweep
    max_epochs = max(epoch_checkpoints)

    # Results storage
    results = []

    print("=" * 80)
    print("DD Distillation Sweep")
    print("=" * 80)
    print(f"Learning rates: {learning_rates}")
    print(f"Epoch checkpoints: {epoch_checkpoints}")
    print(f"Temperatures: {temperatures}")
    print(f"Model output directory: {model_output_dir}")
    print(f"Eval output directory: {eval_output_dir}")
    print(f"DD Alpha: {args['dd_alpha']}")
    print("=" * 80)

    # Run sweep
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
                checkpoint_eval_dir = Path(eval_output_dir) / f"lr-{lr}-epoch-{epoch}-temp-{temp}"

                if not checkpoint_path.exists():
                    print(f"Checkpoint not found: {checkpoint_path}")
                    continue

                # Run evaluation if needed
                if not evaluation_completed(lr, epoch, temp):
                    print(f"\nEvaluating temp={temp}, lr={lr}, epoch={epoch}...")
                    success = run_evaluation(str(checkpoint_path), str(checkpoint_eval_dir), args)
                    if not success:
                        print(f"Evaluation failed for temp={temp}, lr={lr}, epoch={epoch}")
                        continue

                # Get results
                eval_results = get_eval_results(lr, epoch, temp)
                if eval_results:
                    results.append({
                        'lr': lr,
                        'epoch': epoch,
                        'temperature': temp,
                        'forget_quality': eval_results.get('forget_quality', 'N/A'),
                        'model_utility': eval_results.get('model_utility', 'N/A'),
                        'forget_truth_ratio': eval_results.get('forget_truth_ratio', 'N/A'),
                        'forget_rouge': eval_results.get('forget_rouge', 'N/A'),
                        'forget_probability': eval_results.get('forget_probability', 'N/A'),
                    })

    # Print summary table
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)

    if results:
        headers = ['Temp', 'LR', 'Epoch', 'Forget Quality', 'Model Utility', 'Truth Ratio', 'ROUGE', 'Probability']
        table_data = []
        for r in sorted(results, key=lambda x: (x['temperature'], x['lr'], x['epoch'])):
            row = [
                r['temperature'],
                f"{r['lr']:.0e}",
                r['epoch'],
                f"{r['forget_quality']:.4f}" if isinstance(r['forget_quality'], float) else r['forget_quality'],
                f"{r['model_utility']:.4f}" if isinstance(r['model_utility'], float) else r['model_utility'],
                f"{r['forget_truth_ratio']:.4f}" if isinstance(r['forget_truth_ratio'], float) else r['forget_truth_ratio'],
                f"{r['forget_rouge']:.4f}" if isinstance(r['forget_rouge'], float) else r['forget_rouge'],
                f"{r['forget_probability']:.4f}" if isinstance(r['forget_probability'], float) else r['forget_probability'],
            ]
            table_data.append(row)

        print(tabulate(table_data, headers=headers, tablefmt='grid'))

        # Find best config by forget_quality (if available)
        valid_results = [r for r in results if isinstance(r.get('forget_quality'), (int, float))]
        if valid_results:
            best = max(valid_results, key=lambda x: x['forget_quality'])
            print(f"\nBest config by Forget Quality:")
            print(f"  Temperature: {best['temperature']}, LR: {best['lr']:.0e}, Epoch: {best['epoch']}")
            print(f"  Forget Quality: {best['forget_quality']:.4f}")
            print(f"  Model Utility: {best['model_utility']:.4f}" if isinstance(best['model_utility'], float) else "")
    else:
        print("No results collected. Check if training and evaluation completed successfully.")

    # Save results to JSON
    results_file = Path(eval_output_dir) / "sweep_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Print comprehensive results using tofu_scores.py
    print("\n" + "=" * 80)
    print("Running tofu_scores.py for aggregate scoring...")
    print("=" * 80)
    try:
        sys.path.insert(0, 'scripts/analysis')
        from tofu_scores import print_distill_sweep_results
        print_distill_sweep_results()
    except ImportError as e:
        print(f"Could not import tofu_scores: {e}")
        print("Run 'python scripts/analysis/tofu_scores.py' separately to see aggregate scores.")


if __name__ == "__main__":
    main()
