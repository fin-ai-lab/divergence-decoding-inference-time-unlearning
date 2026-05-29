"""
TOFU Evaluation Score Computation with Bootstrap Confidence Intervals

This module computes aggregate scores for TOFU evaluation results with
bootstrap confidence intervals. It includes:

1. Caching: Bootstrap results are cached to avoid re-running expensive
   computations. The cache is stored in bootstrap_cache.json and is
   keyed by (eval_dir, n_samples, alpha, seed).

2. Bootstrap Integration: Uses TOFUBootstrapper from bootstrap_tofu.py
   to compute 99% confidence intervals via resampling.

3. Table Generation: Generates LaTeX and plain text tables with ± notation
   for confidence intervals:
   - LaTeX tables use $\\pm$ notation
   - Plain text tables use ± notation

To clear the cache, call: clear_bootstrap_cache()
"""

import json
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import hashlib
from bootstrap_tofu import TOFUBootstrapper
from muse_scores import palette, markers

privacy_keys = ["mia_min_k_plus_plus", "mia_min_k", "mia_loss", "mia_zlib"]
memorization_keys = ["extraction_strength", "exact_memorization", "forget_Q_A_PARA_Prob", "forget_truth_ratio"]
utility_keys = ["model_utility", "forget_Q_A_gibberish"]

# Bootstrap settings
BOOTSTRAP_N_SAMPLES = 1000  # Number of bootstrap iterations
BOOTSTRAP_ALPHA = 0.01      # 99% confidence intervals
BOOTSTRAP_SEED = 42         # For reproducibility

# Cache settings
BOOTSTRAP_CACHE_FILE = "bootstrap_cache.json"

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

retrain_priv_scores = {}
with open("saves/eval/tofu/baselines/retrain/TOFU_SUMMARY.json", "r") as f:
    data = json.load(f)
    for key in privacy_keys:
        retrain_priv_scores[key] = data[key]

with open("saves/eval/tofu/baselines/target/TOFU_SUMMARY.json", "r") as f:
    data = json.load(f)
    target_util = 2 / ((1 / data["model_utility"]) + (1 / data["forget_Q_A_gibberish"]))


# ── Result-path helpers (see SAVES_LAYOUT.md) ────────────────────────────────
# Baselines live under tofu/baselines/{target,retrain}. The neural-DD sweeps are
# keyed by the auxiliary-model label 1B/3B (the sweep task_name uses 1B/3B even
# though the underlying Llama-3.2 checkpoints are 3.2-1B / 3.2-3B). Gradient
# baselines eval to tofu/gradient/<method>_<lr>/checkpoint-<n>.

# Map the analysis's internal "3.2-1B" / "3.2-3B" size token to the sweep's
# 1B / 3B auxiliary-model label used in the result paths.
DD_AUX_LABEL = {"3.2-1B": "1B", "3.2-3B": "3B"}

# Gradient-method label -> lowercase method key used in the sweep task_name.
GRADIENT_METHOD_DIR = {
    "DPO": "dpo",
    "GradAscent": "grad_ascent",
    "GradDiff": "graddiff",
    "NPO": "npo",
    "RMU": "rmu",
    "SimNPO": "simnpo",
    "UNDIAL": "undial",
}


def dd_linear_dir(model_size, alpha):
    """saves/eval dir for a Linear-DD (alpha) sweep point."""
    return f"saves/eval/tofu/dd_linear/{DD_AUX_LABEL[model_size]}-alpha-{alpha}"


def dd_rank_dir(model_size, topk):
    """saves/eval dir for a Rank-DD (top-k) sweep point."""
    return f"saves/eval/tofu/dd_rank/{DD_AUX_LABEL[model_size]}-topk-{topk}"


def gradient_dir(method_key, lr, checkpoint):
    """saves/eval dir for a gradient-baseline checkpoint eval."""
    return f"saves/eval/tofu/gradient/{method_key}_{lr}/checkpoint-{checkpoint}"


def view_pretty(df):
    numeric_columns = df.select_dtypes(include=['number']).columns
    df[numeric_columns] = df[numeric_columns].round(2)
    print(df.to_string(index=False))

def generate_cache_key(eval_dir, n_samples, alpha, seed):
    """
    Generate a unique cache key for a bootstrap computation.

    Args:
        eval_dir: Directory containing evaluation data
        n_samples: Number of bootstrap samples
        alpha: Significance level
        seed: Random seed

    Returns:
        Hash string to use as cache key
    """
    # Normalize the path to handle relative vs absolute paths
    normalized_dir = os.path.normpath(os.path.abspath(eval_dir))
    key_string = f"{normalized_dir}_{n_samples}_{alpha}_{seed}"
    return hashlib.md5(key_string.encode()).hexdigest()

def load_bootstrap_cache():
    """Load the bootstrap cache from disk."""
    if os.path.exists(BOOTSTRAP_CACHE_FILE):
        try:
            with open(BOOTSTRAP_CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load cache file: {e}")
            return {}
    return {}

def save_bootstrap_cache(cache):
    """Save the bootstrap cache to disk."""
    try:
        with open(BOOTSTRAP_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except IOError as e:
        print(f"Warning: Could not save cache file: {e}")

def get_cached_bootstrap_results(eval_dir, n_samples, alpha, seed):
    """
    Retrieve cached bootstrap results if available.

    Returns:
        Cached results dict or None if not found
    """
    cache = load_bootstrap_cache()
    cache_key = generate_cache_key(eval_dir, n_samples, alpha, seed)

    if cache_key in cache:
        print(f"  Using cached bootstrap results for {eval_dir}")
        return cache[cache_key]
    return None

def save_cached_bootstrap_results(eval_dir, n_samples, alpha, seed, results):
    """
    Save bootstrap results to cache.

    Args:
        eval_dir: Directory containing evaluation data
        n_samples: Number of bootstrap samples
        alpha: Significance level
        seed: Random seed
        results: Bootstrap results to cache
    """
    cache = load_bootstrap_cache()
    cache_key = generate_cache_key(eval_dir, n_samples, alpha, seed)
    cache[cache_key] = results
    save_bootstrap_cache(cache)

def clear_bootstrap_cache():
    """Clear all cached bootstrap results."""
    if os.path.exists(BOOTSTRAP_CACHE_FILE):
        os.remove(BOOTSTRAP_CACHE_FILE)
        print(f"Bootstrap cache cleared: {BOOTSTRAP_CACHE_FILE}")
    else:
        print("No cache file found.")

#currently goes to the TOFU_SUMMARY.json file, but we can switch it to TOFU_EVAL.json
def calculate_info(directory):
    with open(directory, "r") as f:
        data = json.load(f)

    memorization_score = len(memorization_keys) / sum(1 / (1-data[key]) for key in memorization_keys)
    try:
        utility_score = (len(utility_keys) / sum(1 / data[key] for key in utility_keys)) / target_util
    except ZeroDivisionError:
        utility_score = 0.0
    agg_wo_privacy = 2 / ((1 / (memorization_score)) + (1 / (utility_score))) if utility_score > 0 else 0.0

    if "mia_min_k_plus_plus" in data:
        privacy_score = len(privacy_keys) / sum(1 / (1 - math.fabs(data[key] - retrain_priv_scores[key])) for key in privacy_keys)
        agg = 3 / ((1 / (memorization_score)) + (1 / (privacy_score)) + (1 / (utility_score))) if utility_score > 0 else 0.0
    else:
        agg = float('nan')
        privacy_score = float('nan')

    results = {
        "agg": agg,
        "agg_wo_privacy": agg_wo_privacy,
        "memorization_score": memorization_score,
        "privacy_score": privacy_score,
        "utility_score": utility_score
    }

    return results


# Default to point estimates (no bootstrap SEs): the per-index TOFU_EVAL.json files are
# gitignored (too large to commit), so analysis runs from TOFU_SUMMARY.json alone. Set
# SKIP_BOOTSTRAP = False (with EVAL.json present) to compute the 99% bootstrap CIs.
SKIP_BOOTSTRAP = True


def calculate_info_with_bootstrap(eval_dir, use_bootstrap=True,
                                   n_samples=BOOTSTRAP_N_SAMPLES,
                                   alpha=BOOTSTRAP_ALPHA,
                                   seed=BOOTSTRAP_SEED):
    """
    Calculate aggregate scores with bootstrap confidence intervals.

    Args:
        eval_dir: Directory containing TOFU_EVAL.json (for bootstrap) or TOFU_SUMMARY.json
        use_bootstrap: Whether to compute bootstrap CIs (requires TOFU_EVAL.json)
        n_samples: Number of bootstrap iterations
        alpha: Significance level for CIs (0.01 = 99% CI)
        seed: Random seed for reproducibility

    Returns:
        Dict with scores and optionally CI bounds for the 5 key metrics:
        - agg, agg_wo_privacy, memorization_score, privacy_score, utility_score
    """
    if SKIP_BOOTSTRAP:
        use_bootstrap = False
    # First get point estimates from SUMMARY
    summary_path = os.path.join(eval_dir, "TOFU_SUMMARY.json")
    point_estimates = calculate_info(summary_path)

    if not use_bootstrap:
        return point_estimates

    # Try to compute bootstrap CIs
    eval_json_path = os.path.join(eval_dir, "TOFU_EVAL.json")

    if not os.path.exists(eval_json_path):
        print(f"Warning: {eval_json_path} not found, skipping bootstrap")
        return point_estimates

    try:
        # Check cache first
        cached_results = get_cached_bootstrap_results(eval_dir, n_samples, alpha, seed)
        if cached_results is not None:
            return cached_results

        # Initialize bootstrapper
        bootstrapper = TOFUBootstrapper(
            eval_json_path=eval_json_path,
            config_yaml_path="tofu_config.yaml",
            retrain_summary_path="saves/eval/tofu/baselines/retrain/TOFU_SUMMARY.json",
            target_summary_path="saves/eval/tofu/baselines/target/TOFU_SUMMARY.json"
        )

        # Run bootstrap
        bootstrap_results = bootstrapper.bootstrap(
            n_samples=n_samples,
            alpha=alpha,
            seed=seed
        )

        # Extract the 5 key metrics with CIs
        key_metrics = ["agg", "agg_wo_privacy", "memorization_score",
                       "privacy_score", "utility_score"]

        results_with_ci = {}
        for metric in key_metrics:
            if metric in bootstrap_results:
                bs = bootstrap_results[metric]
                results_with_ci[metric] = bs["mean"]
                results_with_ci[f"{metric}_ci_lower"] = bs["ci_lower"]
                results_with_ci[f"{metric}_ci_upper"] = bs["ci_upper"]
                results_with_ci[f"{metric}_ci_half_width"] = bs["ci_half_width"]
            else:
                # Fallback to point estimate
                results_with_ci[metric] = point_estimates.get(metric, float('nan'))
                results_with_ci[f"{metric}_ci_lower"] = float('nan')
                results_with_ci[f"{metric}_ci_upper"] = float('nan')
                results_with_ci[f"{metric}_ci_half_width"] = float('nan')

        # Save to cache
        save_cached_bootstrap_results(eval_dir, n_samples, alpha, seed, results_with_ci)

        return results_with_ci

    except Exception as e:
        print(f"Warning: Bootstrap failed for {eval_dir}: {e}")
        return point_estimates

def _sweep_best(param_values, path_fn):
    """Find the param value that maximises agg and agg_wo_privacy.

    Args:
        param_values: Iterable of parameter values to sweep.
        path_fn: Callable that maps a param value to a TOFU_SUMMARY.json path.

    Returns:
        dict with best_param_agg, best_param_agg_wo_priv, best_agg, best_agg_wo_priv.
    """
    best_agg = -1
    best_agg_wo_priv = -1
    best_param_agg = None
    best_param_agg_wo_priv = None

    for val in param_values:
        try:
            info = calculate_info(path_fn(val))
            if not math.isnan(info["agg"]) and info["agg"] > best_agg:
                best_agg = info["agg"]
                best_param_agg = val
            if info["agg_wo_privacy"] > best_agg_wo_priv:
                best_agg_wo_priv = info["agg_wo_privacy"]
                best_param_agg_wo_priv = val
        except (FileNotFoundError, KeyError):
            pass

    return {
        "best_param_agg": best_param_agg,
        "best_param_agg_wo_priv": best_param_agg_wo_priv,
        "best_agg": best_agg,
        "best_agg_wo_priv": best_agg_wo_priv,
    }


TABLE_INFO_KEYS = [
    "agg", "memorization_score", "privacy_score", "utility_score",
    "agg_ci_half_width", "memorization_score_ci_half_width",
    "privacy_score_ci_half_width", "utility_score_ci_half_width",
]


def _add_table_entry(table_data, method, config, eval_dir, use_bootstrap=True, n_samples=BOOTSTRAP_N_SAMPLES):
    """Compute scores for eval_dir and append a row to table_data."""
    print(f"  Processing {method} ({config})...")
    info = calculate_info_with_bootstrap(eval_dir, use_bootstrap=use_bootstrap, n_samples=n_samples)
    table_data.append({
        "Method": method,
        "Config": config,
        **{k: info.get(k, float('nan')) for k in TABLE_INFO_KEYS},
    })


def find_optimal_dd_configs():
    """Find optimal DD configurations for both linear and rank methods - 1B and 3B only"""
    optimal_configs = {}

    linear_alphas = [0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 4.0]
    topk_values = [1, 5, 20, 50, 100, 200, 500, 1000]

    for model_size in ["3.2-1B", "3.2-3B"]:
        result = _sweep_best(
            linear_alphas,
            lambda a, ms=model_size: f"{dd_linear_dir(ms, a)}/TOFU_SUMMARY.json",
        )
        optimal_configs[f"Linear DD {model_size}"] = result

        result = _sweep_best(
            topk_values,
            lambda k, ms=model_size: f"{dd_rank_dir(ms, k)}/TOFU_SUMMARY.json",
        )
        optimal_configs[f"Rank DD {model_size}"] = result

    return optimal_configs


def find_optimal_cross_tok_configs():
    """Find optimal cross-tokenizer DD configurations for TOFU."""
    # label -> (short_name, [lr_values])
    cross_tok_models = {
        "OLMo": ("OLMo-2-0425-1B-Instruct", ["1e-5", "3e-5", "5e-5"]),
        "Gemma": ("gemma-3-1b-it", ["1e-5", "3e-5", "5e-5"]),
        "Qwen": ("Qwen3-1.7B", ["1e-5", "3e-5", "5e-5"]),
    }
    cross_tok_alphas = [round(x * 0.1, 1) for x in range(0, 31)]
    cross_tok_topks = [1, 5, 20, 100, 200, 500, 1000]

    optimal = {}
    for label, (short, lrs) in cross_tok_models.items():
        best_alpha = {"best_param_agg": None}
        best_topk = {"best_param_agg": None}
        for lr in lrs:
            result_alpha = _sweep_best(
                cross_tok_alphas,
                lambda a, s=short, l=lr: f"saves/eval/tofu/cross_tok/{s}-lr{l}-alpha-{a}/TOFU_SUMMARY.json",
            )
            result_topk = _sweep_best(
                cross_tok_topks,
                lambda k, s=short, l=lr: f"saves/eval/tofu/cross_tok/{s}-lr{l}-topk-{k}/TOFU_SUMMARY.json",
            )
            # Keep the best across LRs
            if result_alpha["best_param_agg"] is not None:
                if best_alpha["best_param_agg"] is None or result_alpha.get("best_agg", 0) > best_alpha.get("best_agg", 0):
                    best_alpha = {**result_alpha, "lr": lr}
            if result_topk["best_param_agg"] is not None:
                if best_topk["best_param_agg"] is None or result_topk.get("best_agg", 0) > best_topk.get("best_agg", 0):
                    best_topk = {**result_topk, "lr": lr}
        optimal[label] = {"alpha": best_alpha, "topk": best_topk}
    return optimal


def find_optimal_offset_configs():
    """Find optimal Offset Unlearning LR configuration"""
    lrs = ["1e-6", "5e-6", "7e-6", "8e-6", "1e-5", "5e-5"]
    return _sweep_best(
        lrs,
        lambda lr: f"saves/eval/tofu/offset/lr-{lr}/TOFU_SUMMARY.json",
    )


def find_optimal_uld_configs():
    """Find optimal ULD LR configuration"""
    lrs = ["5e-5", "5e-4", "1e-3", "2e-3", "3e-3", "5e-3"]
    return _sweep_best(
        lrs,
        lambda lr: f"saves/eval/tofu/uld/lr-{lr}/TOFU_SUMMARY.json",
    )


def find_optimal_whp_configs():
    """Find optimal WHP (lr, alpha) configuration"""
    lrs = ["1e-5", "3e-5", "5e-5", "5e-5"]
    alphas = ["0_5", "1_0", "1_5", "2_0", "3_0"]
    combos = [(lr, a) for lr in lrs for a in alphas]
    return _sweep_best(
        combos,
        lambda c: f"saves/eval/tofu/whp/lr-{c[0]}_alpha-{c[1]}/TOFU_SUMMARY.json",
    )


def find_optimal_guard_configs():
    """Find optimal GUARD (lr, delta) configuration"""
    lrs = ["1e-3", "5e-4", "1e-4"]
    deltas = ["0_3", "0_5", "0_7"]
    combos = [(lr, d) for lr in lrs for d in deltas]
    return _sweep_best(
        combos,
        lambda c: f"saves/eval/tofu/guard/lr-{c[0]}_delta-{c[1]}/TOFU_SUMMARY.json",
    )


def find_optimal_eco_configs():
    """Find optimal ECO (classifier lr, strength) configuration"""
    lrs = ["1e-5", "2e-5", "5e-5"]
    strengths = [50, 100, 200]
    combos = [(lr, s) for lr in lrs for s in strengths]
    return _sweep_best(
        combos,
        lambda c: f"saves/eval/tofu/eco/lr-{c[0]}_str-{c[1]}/TOFU_SUMMARY.json",
    )


def find_optimal_lunar_configs():
    """Find optimal LUNAR lr configuration"""
    lrs = ["0001", "0005", "001", "005", "01"]
    return _sweep_best(
        lrs,
        lambda lr: f"saves/eval/tofu/lunar/lr-{lr}/TOFU_SUMMARY.json",
    )


def find_optimal_distill_configs():
    """Find optimal configurations for DD distillation with learning rate, epoch, and temperature search"""
    learning_rates = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5]
    epochs = [5, 10]
    temperatures = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]

    best_agg = -1
    best_agg_wo_priv = -1
    best_config_agg = None
    best_config_agg_wo_priv = None

    all_results = []

    for temp in temperatures:
        for lr in learning_rates:
            for epoch in epochs:
                try:
                    summary_path = f"saves/eval/tofu/distill/lr-{lr}-epoch-{epoch}-temp-{temp}/TOFU_SUMMARY.json"
                    info = calculate_info(summary_path)

                    result = {
                        "temperature": temp,
                        "lr": lr,
                        "epoch": epoch,
                        "agg": info["agg"],
                        "agg_wo_privacy": info["agg_wo_privacy"],
                        "memorization_score": info["memorization_score"],
                        "privacy_score": info["privacy_score"],
                        "utility_score": info["utility_score"]
                    }
                    all_results.append(result)

                    # Check for best aggregate score (with privacy)
                    if not math.isnan(info["agg"]) and info["agg"] > best_agg:
                        best_agg = info["agg"]
                        best_config_agg = {"lr": lr, "epoch": epoch, "temperature": temp}

                    # Check for best aggregate score without privacy
                    if info["agg_wo_privacy"] > best_agg_wo_priv:
                        best_agg_wo_priv = info["agg_wo_privacy"]
                        best_config_agg_wo_priv = {"lr": lr, "epoch": epoch, "temperature": temp}

                except (FileNotFoundError, KeyError):
                    pass

    return {
        "best_config_agg": best_config_agg,
        "best_config_agg_wo_priv": best_config_agg_wo_priv,
        "best_agg": best_agg,
        "best_agg_wo_priv": best_agg_wo_priv,
        "all_results": all_results
    }


def find_optimal_unlearning_configs():
    """Find optimal configurations for unlearning methods with learning rate search"""
    unlearn_methods = {"DPO": "DPO", "GradAscent": "GradAscent", "GradDiff/GA-GDR": "GradDiff", "NPO": "NPO", "RMU": "RMU", "SimNPO": "SimNPO", "UNDIAL": "UNDIAL"}
    checkpoints = [13, 26, 39, 52, 65, 78, 91, 104, 117, 130]
    learning_rates = ["2e-6", "1e-6", "3e-6", "1.5e-6", "4e-6", "8e-7"]
    optimal_configs = {}

    for method, dirname in unlearn_methods.items():
        best_agg = -1
        best_agg_wo_priv = -1
        best_config_agg = None
        best_config_agg_wo_priv = None
        best_agg_wo_priv_score = -1

        # Search over learning rates and checkpoints
        for lr in learning_rates:
            for checkpoint in checkpoints:
                try:
                    method_dir = gradient_dir(GRADIENT_METHOD_DIR[dirname], lr, checkpoint)
                    info = calculate_info(f"{method_dir}/TOFU_SUMMARY.json")

                    # Check for best aggregate score (with privacy)
                    if not math.isnan(info["agg"]) and info["agg"] > best_agg:
                        best_agg = info["agg"]
                        best_config_agg = {
                            "checkpoint": checkpoint,
                            "learning_rate": lr,
                            "epoch": int(checkpoint / 13)
                        }
                    
                    # Check for best aggregate score without privacy
                    if info["agg_wo_privacy"] > best_agg_wo_priv:
                        best_agg_wo_priv = info["agg_wo_privacy"]
                        best_agg_wo_priv_score = info["agg_wo_privacy"]
                        best_config_agg_wo_priv = {
                            "checkpoint": checkpoint,
                            "learning_rate": lr,
                            "epoch": int(checkpoint / 13)
                        }

                except (FileNotFoundError, KeyError):
                    pass

        optimal_configs[method] = {
            "best_config_agg": best_config_agg,
            "best_config_agg_wo_priv": best_config_agg_wo_priv,
            "best_agg": best_agg,
            "best_agg_wo_priv": best_agg_wo_priv_score,
        }
    
    return optimal_configs

def generate_tofu_tables(use_bootstrap=True, n_samples=BOOTSTRAP_N_SAMPLES, latex=False):
    """
    Generate the TOFU tables with bootstrap confidence intervals.

    Args:
        use_bootstrap: Whether to compute and display bootstrap CIs
        n_samples: Number of bootstrap iterations
    """
    print("Finding optimal configurations...")
    dd_configs = find_optimal_dd_configs()
    cross_tok_configs = find_optimal_cross_tok_configs()
    unlearn_configs = find_optimal_unlearning_configs()
    distill_configs = find_optimal_distill_configs()
    offset_config = find_optimal_offset_configs()
    uld_config = find_optimal_uld_configs()
    whp_config = find_optimal_whp_configs()
    guard_config = find_optimal_guard_configs()
    eco_config = find_optimal_eco_configs()
    lunar_config = find_optimal_lunar_configs()

    # Table: With Privacy (includes Linear DD, Rank DD, and unlearning methods)
    table_data = []

    print("\nComputing scores" + (" with bootstrap CIs..." if use_bootstrap else "..."))

    # Add Target and Retrain baselines (8B)
    print("  Processing Target...")
    target_info = calculate_info_with_bootstrap(
        "saves/eval/tofu/baselines/target",
        use_bootstrap=use_bootstrap,
        n_samples=n_samples
    )

    print("  Processing Retrain...")
    retrain_info = calculate_info_with_bootstrap(
        "saves/eval/tofu/baselines/retrain",
        use_bootstrap=use_bootstrap,
        n_samples=n_samples
    )

    table_data.append({
        "Method": "Target",
        "Config": "Full",
        **{k: target_info.get(k, float('nan')) for k in
           ["agg", "memorization_score", "privacy_score", "utility_score",
            "agg_ci_half_width", "memorization_score_ci_half_width",
            "privacy_score_ci_half_width", "utility_score_ci_half_width"]}
    })

    table_data.append({
        "Method": "Retrain",
        "Config": "Retain90",
        **{k: retrain_info.get(k, float('nan')) for k in
           ["agg", "memorization_score", "privacy_score", "utility_score",
            "agg_ci_half_width", "memorization_score_ci_half_width",
            "privacy_score_ci_half_width", "utility_score_ci_half_width"]}
    })

    # Add Linear DD methods
    for model_size in ["3.2-1B"]:
        config_key = f"Linear DD {model_size}"
        if config_key in dd_configs and dd_configs[config_key]["best_param_agg"] is not None:
            alpha = dd_configs[config_key]["best_param_agg"]
            eval_dir = dd_linear_dir(model_size, alpha)
            try:
                print(f"  Processing Linear DD {model_size} α={alpha}...")
                info = calculate_info_with_bootstrap(
                    eval_dir,
                    use_bootstrap=use_bootstrap,
                    n_samples=n_samples
                )
                table_data.append({
                    "Method": f"Linear DD",
                    "Config": f"$\\alpha$={alpha}",
                    **{k: info.get(k, float('nan')) for k in
                       ["agg", "memorization_score", "privacy_score", "utility_score",
                        "agg_ci_half_width", "memorization_score_ci_half_width",
                        "privacy_score_ci_half_width", "utility_score_ci_half_width"]}
                })
            except FileNotFoundError:
                pass

    # Add Rank DD methods
    for model_size in ["3.2-1B"]:
        config_key = f"Rank DD {model_size}"
        if config_key in dd_configs and dd_configs[config_key]["best_param_agg"] is not None:
            topk = dd_configs[config_key]["best_param_agg"]
            eval_dir = dd_rank_dir(model_size, topk)
            try:
                print(f"  Processing Rank DD {model_size} k={topk}...")
                info = calculate_info_with_bootstrap(
                    eval_dir,
                    use_bootstrap=use_bootstrap,
                    n_samples=n_samples
                )
                table_data.append({
                    "Method": f"Rank DD",
                    "Config": f"k={topk}",
                    **{k: info.get(k, float('nan')) for k in
                       ["agg", "memorization_score", "privacy_score", "utility_score",
                        "agg_ci_half_width", "memorization_score_ci_half_width",
                        "privacy_score_ci_half_width", "utility_score_ci_half_width"]}
                })
            except FileNotFoundError:
                pass

    # Add cross-tokenizer DD methods
    cross_tok_shorts = {"OLMo": "OLMo-2-0425-1B-Instruct", "Gemma": "gemma-3-1b-it", "Qwen": "Qwen3-1.7B"}
    for label, results in cross_tok_configs.items():
        # Pick whichever of alpha/topk has the best agg score
        for sweep_type in ["alpha", "topk"]:
            r = results[sweep_type]
            if r["best_param_agg"] is not None:
                short = cross_tok_shorts[label]
                val = r["best_param_agg"]
                lr = r.get("lr", "1e-5")
                eval_dir = f"saves/eval/tofu/cross_tok/{short}-lr{lr}-{sweep_type}-{val}"
                prefix = "α" if sweep_type == "alpha" else "k"
                variant = "Linear" if sweep_type == "alpha" else "Rank"
                _add_table_entry(
                    table_data,
                    f"{label} {variant} CT-DD",
                    f"lr={lr}, ${prefix}$={val}",
                    eval_dir,
                    use_bootstrap=use_bootstrap,
                    n_samples=n_samples,
                )

    # Add unlearning methods
    method_dirnames = {"DPO": "DPO", "GradAscent": "GradAscent", "GradDiff/GA-GDR": "GradDiff", "NPO": "NPO", "RMU": "RMU", "SimNPO": "SimNPO", "UNDIAL": "UNDIAL"}
    for method, dirname in method_dirnames.items():
        if method in unlearn_configs and unlearn_configs[method]["best_config_agg"] is not None:
            config = unlearn_configs[method]["best_config_agg"]
            checkpoint = config["checkpoint"]
            lr = config["learning_rate"]
            epoch = config["epoch"]

            eval_dir = gradient_dir(GRADIENT_METHOD_DIR[dirname], lr, checkpoint)
            try:
                print(f"  Processing {method}...")
                info = calculate_info_with_bootstrap(
                    eval_dir,
                    use_bootstrap=use_bootstrap,
                    n_samples=n_samples
                )
                table_data.append({
                    "Method": method,
                    "Config": f"lr={lr}, e={epoch}",
                    **{k: info.get(k, float('nan')) for k in
                       ["agg", "memorization_score", "privacy_score", "utility_score",
                        "agg_ci_half_width", "memorization_score_ci_half_width",
                        "privacy_score_ci_half_width", "utility_score_ci_half_width"]}
                })
            except (FileNotFoundError, KeyError, Exception) as e:
                print(f"    Skipped {method}: {e}")
                pass

    # Add DD Distillation if available
    if distill_configs["best_config_agg"] is not None:
        config = distill_configs["best_config_agg"]
        lr = config["lr"]
        epoch = config["epoch"]
        temp = config.get("temperature", 1)
        eval_dir = f"saves/eval/tofu/distill/lr-{lr}-epoch-{epoch}-temp-{temp}"
        try:
            print(f"  Processing Distill DD...")
            info = calculate_info_with_bootstrap(
                eval_dir,
                use_bootstrap=use_bootstrap,
                n_samples=n_samples
            )
            table_data.append({
                "Method": "Distill DD",
                "Config": f"lr={lr:.0e}, T={temp}",
                **{k: info.get(k, float('nan')) for k in
                   ["agg", "memorization_score", "privacy_score", "utility_score",
                    "agg_ci_half_width", "memorization_score_ci_half_width",
                    "privacy_score_ci_half_width", "utility_score_ci_half_width"]}
            })
        except (FileNotFoundError, KeyError, Exception) as e:
            print(f"    Skipped Distill DD: {e}")
            pass

    # Add Offset Unlearning method
    if offset_config["best_param_agg"] is not None:
        lr = offset_config["best_param_agg"]
        eval_dir = f"saves/eval/tofu/offset/lr-{lr}"
        try:
            _add_table_entry(table_data, "$\\delta$-Unlearning", f"lr={lr}", eval_dir,
                             use_bootstrap=use_bootstrap, n_samples=n_samples)
        except (FileNotFoundError, Exception) as e:
            print(f"    Skipped Offset: {e}")

    # Add ULD method
    if uld_config["best_param_agg"] is not None:
        lr = uld_config["best_param_agg"]
        eval_dir = f"saves/eval/tofu/uld/lr-{lr}"
        try:
            _add_table_entry(table_data, "ULD", f"lr={lr}", eval_dir,
                             use_bootstrap=use_bootstrap, n_samples=n_samples)
        except (FileNotFoundError, Exception) as e:
            print(f"    Skipped ULD: {e}")

    # Add WHP method
    if whp_config["best_param_agg"] is not None:
        lr, alpha = whp_config["best_param_agg"]
        eval_dir = f"saves/eval/tofu/whp/lr-{lr}_alpha-{alpha}"
        alpha_display = alpha.replace("_", ".")
        try:
            _add_table_entry(table_data, "WHP", f"lr={lr}, $\\alpha$={alpha_display}", eval_dir,
                             use_bootstrap=use_bootstrap, n_samples=n_samples)
        except (FileNotFoundError, Exception) as e:
            print(f"    Skipped WHP: {e}")

    # Add GUARD method
    if guard_config["best_param_agg"] is not None:
        lr, delta = guard_config["best_param_agg"]
        eval_dir = f"saves/eval/tofu/guard/lr-{lr}_delta-{delta}"
        delta_display = delta.replace("_", ".")
        try:
            _add_table_entry(table_data, "GUARD", f"lr={lr}, $\\delta$={delta_display}", eval_dir,
                             use_bootstrap=use_bootstrap, n_samples=n_samples)
        except (FileNotFoundError, Exception) as e:
            print(f"    Skipped GUARD: {e}")

    # Add ECO method
    if eco_config["best_param_agg"] is not None:
        lr, strength = eco_config["best_param_agg"]
        eval_dir = f"saves/eval/tofu/eco/lr-{lr}_str-{strength}"
        try:
            _add_table_entry(table_data, "ECO", f"lr={lr}, str={strength}", eval_dir,
                             use_bootstrap=use_bootstrap, n_samples=n_samples)
        except (FileNotFoundError, Exception) as e:
            print(f"    Skipped ECO: {e}")

    # Add LUNAR method
    if lunar_config["best_param_agg"] is not None:
        lr = lunar_config["best_param_agg"]
        eval_dir = f"saves/eval/tofu/lunar/lr-{lr}"
        try:
            _add_table_entry(table_data, "LUNAR", f"lr=0.{lr}", eval_dir,
                             use_bootstrap=use_bootstrap, n_samples=n_samples)
        except (FileNotFoundError, Exception) as e:
            print(f"    Skipped LUNAR: {e}")

    df = pd.DataFrame(table_data)

    # Enforce display order
    method_order = [
        "Target", "Retrain", "Linear DD", "Rank DD", "Distill DD",
        "DPO", "GradAscent", "GradDiff/GA-GDR", "NPO", "RMU", "SimNPO", "UNDIAL", "LUNAR",
        "$\\delta$-Unlearning", "ULD", "WHP", "GUARD", "ECO",
    ]
    order_map = {m: i for i, m in enumerate(method_order)}
    df["_order"] = df["Method"].map(order_map).fillna(len(method_order))
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    # Find top 2 values for each metric column (for bolding), excluding baselines
    metric_cols = ['agg', 'memorization_score', 'privacy_score', 'utility_score']
    top2_indices = {}
    # Exclude Target and Retrain from top-2 consideration
    df_methods = df[~df['Method'].isin(['Target', 'Retrain'])]
    for col in metric_cols:
        # Get non-NaN values sorted descending (excluding baselines)
        valid_vals = df_methods[col].dropna().sort_values(ascending=False)
        if len(valid_vals) >= 2:
            top2_vals = valid_vals.head(2).tolist()
            # Find all indices that have these top 2 values
            top2_indices[col] = set(df_methods[df_methods[col].isin(top2_vals)].index.tolist())
        elif len(valid_vals) == 1:
            top2_indices[col] = set(df_methods[df_methods[col] == valid_vals.iloc[0]].index.tolist())
        else:
            top2_indices[col] = set()

    # Generate LaTeX table (booktabs). Bolds the top-2 value per metric (baselines
    # excluded), with a midrule after the Retrain baseline. When bootstrap CIs are
    # available they are appended with $\pm$ notation.
    if latex:
        print("\n" + "="*80)
        print("TABLE: TOFU Results (LaTeX)" + (" with Bootstrap CIs" if use_bootstrap else ""))
        print("="*80)

        def format_latex(val, ci_hw, bold=False):
            if pd.isna(val):
                return "N/A"
            if use_bootstrap and not pd.isna(ci_hw) and ci_hw > 0:
                content = f"{val:.2f} $\\pm$ {ci_hw:.2f}"
            else:
                content = f"{val:.2f}"
            return f"\\textbf{{{content}}}" if bold else content

        print("\\begin{tabular}{llcccc}")
        print("\\toprule")
        print("Method & Config & Agg. $\\uparrow$ & Mem. $\\uparrow$ & Priv. $\\uparrow$ & Utility $\\uparrow$ \\\\")
        print("\\midrule")

        for i, row in df.iterrows():
            agg_str = format_latex(row['agg'], row.get('agg_ci_half_width', float('nan')), bold=(i in top2_indices['agg']))
            mem_str = format_latex(row['memorization_score'], row.get('memorization_score_ci_half_width', float('nan')), bold=(i in top2_indices['memorization_score']))
            priv_str = format_latex(row['privacy_score'], row.get('privacy_score_ci_half_width', float('nan')), bold=(i in top2_indices['privacy_score']))
            util_str = format_latex(row['utility_score'], row.get('utility_score_ci_half_width', float('nan')), bold=(i in top2_indices['utility_score']))

            print(f"{row['Method']} & {row['Config']} & {agg_str} & {mem_str} & {priv_str} & {util_str} \\\\")
            if row['Method'] == "Retrain":
                print("\\midrule")

        print("\\bottomrule")
        print("\\end{tabular}")
        return

    # Generate Markdown table
    print("\n" + "="*80)
    print("TABLE: TOFU Results" + (" (SEs hidden for brevity)" if use_bootstrap else ""))
    print("="*80)

    def format_md(val, bold=False):
        if pd.isna(val):
            return "N/A"
        content = f"{val:.2f}"
        return f"**{content}**" if bold else content

    print(f"| Method | Config | Agg ↑ | Mem ↑ | Priv ↑ | Util ↑ |")
    print(f"| --- | --- | --- | --- | --- | --- |")

    for row_idx, (i, row) in enumerate(df.iterrows()):
        method = row['Method']
        config = row['Config']
        bold_row = row_idx < 5

        agg_str = format_md(row['agg'], bold=bold_row)
        mem_str = format_md(row['memorization_score'], bold=bold_row)
        priv_str = format_md(row['privacy_score'], bold=bold_row)
        util_str = format_md(row['utility_score'], bold=bold_row)

        if bold_row:
            print(f"| **{method}** | **{config}** | {agg_str} | {mem_str} | {priv_str} | {util_str} |")
        else:
            print(f"| {method} | {config} | {agg_str} | {mem_str} | {priv_str} | {util_str} |")


def appendix_table(use_bootstrap=True, n_samples=BOOTSTRAP_N_SAMPLES):
    """
    Generate one comprehensive appendix table showing all hyperparameter configurations and scores
    for both 1B and 3B models across Linear DD and Rank DD methods.
    Columns: Model, Method, Parameter, Agg, Mem, Priv, Utility

    Args:
        use_bootstrap: Whether to compute and display bootstrap CIs
        n_samples: Number of bootstrap iterations
    """

    print("\n" + "="*80)
    print("APPENDIX TABLE: All DD Hyperparameters (1B and 3B)" + (" with Bootstrap 99% CIs" if use_bootstrap else ""))
    print("="*80)

    all_data = []

    # Linear DD - All Alpha Values for 1B and 3B
    alpha_values = [0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 4.0]

    for model_size in ["3.2-1B", "3.2-3B"]:
        for alpha in alpha_values:
            eval_dir = dd_linear_dir(model_size, alpha)
            directory = f"{eval_dir}/TOFU_SUMMARY.json"
            try:
                info = calculate_info_with_bootstrap(eval_dir, use_bootstrap=use_bootstrap, n_samples=n_samples)
                all_data.append({
                    "Model": model_size.split('-')[1],
                    "Method": "Linear",
                    "Parameter": f"$\\alpha$={alpha}",
                    "Agg": info["agg"],
                    "Agg_ci": info.get("agg_ci_half_width"),
                    "Agg_wo_Priv": info["agg_wo_privacy"],
                    "Agg_wo_Priv_ci": info.get("agg_wo_privacy_ci_half_width"),
                    "Mem": info["memorization_score"],
                    "Mem_ci": info.get("memorization_score_ci_half_width"),
                    "Priv": info["privacy_score"],
                    "Priv_ci": info.get("privacy_score_ci_half_width"),
                    "Utility": info["utility_score"],
                    "Utility_ci": info.get("utility_score_ci_half_width")
                })
            except FileNotFoundError:
                pass

    # Rank DD - All TopK Values for 1B and 3B
    topk_values = [1, 5, 20, 50, 100, 200, 500, 1000]

    for model_size in ["3.2-1B", "3.2-3B"]:
        for topk in topk_values:
            eval_dir = dd_rank_dir(model_size, topk)
            directory = f"{eval_dir}/TOFU_SUMMARY.json"
            try:
                info = calculate_info_with_bootstrap(eval_dir, use_bootstrap=use_bootstrap, n_samples=n_samples)
                all_data.append({
                    "Model": model_size.split('-')[1],
                    "Method": "Rank",
                    "Parameter": f"k={topk}",
                    "Agg": info["agg"],
                    "Agg_ci": info.get("agg_ci_half_width"),
                    "Agg_wo_Priv": info["agg_wo_privacy"],
                    "Agg_wo_Priv_ci": info.get("agg_wo_privacy_ci_half_width"),
                    "Mem": info["memorization_score"],
                    "Mem_ci": info.get("memorization_score_ci_half_width"),
                    "Priv": info["privacy_score"],
                    "Priv_ci": info.get("privacy_score_ci_half_width"),
                    "Utility": info["utility_score"],
                    "Utility_ci": info.get("utility_score_ci_half_width")
                })
            except FileNotFoundError:
                pass

    df = pd.DataFrame(all_data)

    # Helper function to format value with CI
    def format_with_ci(val, ci_hw):
        if pd.isna(val):
            return "N/A"
        if use_bootstrap and not pd.isna(ci_hw) and ci_hw > 0:
            return f"{val:.2f} $\\pm$ {ci_hw:.2f}"
        else:
            return f"{val:.2f}"

    if not df.empty:
        print("\n\\begin{tabular}{llllccccc}")
        print("\\toprule")
        print("Model & Method & Parameter & Agg. $\\uparrow$ & Agg w/o Priv. $\\uparrow$ & Mem. $\\uparrow$ & Priv. $\\uparrow$ & Utility $\\uparrow$ \\\\")
        print("\\midrule")

        current_model = None
        current_method = None

        for _, row in df.iterrows():
            # Add midrule when model changes
            if current_model != row['Model']:
                if current_model is not None:
                    print("\\midrule")
                current_model = row['Model']
                current_method = None

            # Add midrule when method changes within same model
            if current_method != row['Method']:
                if current_method is not None:
                    print("\\midrule")
                current_method = row['Method']

            agg_val = format_with_ci(row['Agg'], row['Agg_ci'])
            agg_wo_priv_val = format_with_ci(row['Agg_wo_Priv'], row['Agg_wo_Priv_ci'])
            mem_val = format_with_ci(row['Mem'], row['Mem_ci'])
            priv_val = format_with_ci(row['Priv'], row['Priv_ci'])
            util_val = format_with_ci(row['Utility'], row['Utility_ci'])

            print(f"{row['Model']} & {row['Method']} & {row['Parameter']} & {agg_val} & {agg_wo_priv_val} & "
                  f"{mem_val} & {priv_val} & {util_val} \\\\")

        print("\\bottomrule")
        print("\\end{tabular}")
    else:
        print("No data found")

def tofu_model_scaling_plot():
    """
    Create a line plot showing how model scaling affects aggregate score for TOFU.
    Similar to MUSE model_scaling_plot but only considering aggregate score.
    Shows two lines: one for aggregate score (with privacy) and one for aggregate score without privacy.
    Hardcode trigram/ngram models to 0 since they don't work.
    Includes bootstrap confidence intervals as error bands.
    """
    print("Finding optimal configurations for TOFU model scaling plot...")
    dd_configs = find_optimal_dd_configs()

    # Model sizes and their x-axis positions
    model_sizes = ["Trigram", "3.2-1B", "3.2-3B"]
    model_size_x = {"Trigram": 0, "3.2-1B": 1, "3.2-3B": 3}

    # Prepare data for plotting
    plot_data = []

    # Dictionary to store CI info: {mean_score: (lower_error, upper_error)}
    error_bars = {}

    for model_size in model_sizes:
        x_pos = model_size_x[model_size]

        if model_size == "Trigram":
            # Hardcode trigram to 0 for both metrics
            error_bars[0.0] = (0.0, 0.0)
            for i in range(3): #have to trick sns
                plot_data.append({
                    'model_size': model_size,
                    'x_pos': x_pos,
                    'score': 0.0,
                    'method': 'Aggregate Score'
                })
                plot_data.append({
                    'model_size': model_size,
                    'x_pos': x_pos,
                    'score': 0.0,
                    'method': 'Agg. w/o Priv.'
                })
        else:
            # Find best methods for this model size and get their CIs
            linear_config_key = f"Linear DD {model_size}"
            rank_config_key = f"Rank DD {model_size}"

            best_agg = 0.0
            best_agg_wo_privacy = 0.0
            best_agg_ci_lower = 0.0
            best_agg_ci_upper = 0.0
            best_agg_wo_priv_ci_lower = 0.0
            best_agg_wo_priv_ci_upper = 0.0
            best_method = None
            best_method_wo_privacy = None

            # Check Linear DD
            if linear_config_key in dd_configs and dd_configs[linear_config_key]["best_param_agg"] is not None:
                alpha = dd_configs[linear_config_key]["best_param_agg"]
                eval_dir = dd_linear_dir(model_size, alpha)

                # Get bootstrap results with CIs
                info = calculate_info_with_bootstrap(eval_dir, use_bootstrap=True)

                linear_agg = info.get("agg", 0.0)
                linear_agg_wo_privacy = info.get("agg_wo_privacy", 0.0)
                linear_agg_ci_lower = info.get("agg_ci_lower", linear_agg)
                linear_agg_ci_upper = info.get("agg_ci_upper", linear_agg)
                linear_agg_wo_priv_ci_lower = info.get("agg_wo_privacy_ci_lower", linear_agg_wo_privacy)
                linear_agg_wo_priv_ci_upper = info.get("agg_wo_privacy_ci_upper", linear_agg_wo_privacy)

                if linear_agg > best_agg:
                    best_agg = linear_agg
                    best_agg_ci_lower = linear_agg_ci_lower
                    best_agg_ci_upper = linear_agg_ci_upper
                    best_method = "Linear DD"
                if linear_agg_wo_privacy > best_agg_wo_privacy:
                    best_agg_wo_privacy = linear_agg_wo_privacy
                    best_agg_wo_priv_ci_lower = linear_agg_wo_priv_ci_lower
                    best_agg_wo_priv_ci_upper = linear_agg_wo_priv_ci_upper
                    best_method_wo_privacy = "Linear DD"
                print(f"{model_size} Linear DD: agg={linear_agg:.4f}, agg_wo_privacy={linear_agg_wo_privacy:.4f}")

            # Check Rank DD
            if rank_config_key in dd_configs and dd_configs[rank_config_key]["best_param_agg"] is not None:
                topk = dd_configs[rank_config_key]["best_param_agg"]
                eval_dir = dd_rank_dir(model_size, topk)

                # Get bootstrap results with CIs
                info = calculate_info_with_bootstrap(eval_dir, use_bootstrap=True)

                rank_agg = info.get("agg", 0.0)
                rank_agg_wo_privacy = info.get("agg_wo_privacy", 0.0)
                rank_agg_ci_lower = info.get("agg_ci_lower", rank_agg)
                rank_agg_ci_upper = info.get("agg_ci_upper", rank_agg)
                rank_agg_wo_priv_ci_lower = info.get("agg_wo_privacy_ci_lower", rank_agg_wo_privacy)
                rank_agg_wo_priv_ci_upper = info.get("agg_wo_privacy_ci_upper", rank_agg_wo_privacy)

                if rank_agg > best_agg:
                    best_agg = rank_agg
                    best_agg_ci_lower = rank_agg_ci_lower
                    best_agg_ci_upper = rank_agg_ci_upper
                    best_method = "Rank DD"
                if rank_agg_wo_privacy > best_agg_wo_privacy:
                    best_agg_wo_privacy = rank_agg_wo_privacy
                    best_agg_wo_priv_ci_lower = rank_agg_wo_priv_ci_lower
                    best_agg_wo_priv_ci_upper = rank_agg_wo_priv_ci_upper
                    best_method_wo_privacy = "Rank DD"
                print(f"{model_size} Rank DD: agg={rank_agg:.4f}, agg_wo_privacy={rank_agg_wo_privacy:.4f}")

            # Store error bars: mean -> (ci_lower, ci_upper) as absolute bounds
            error_bars[round(best_agg, 5)] = (best_agg_ci_lower, best_agg_ci_upper)
            error_bars[round(best_agg_wo_privacy, 5)] = (best_agg_wo_priv_ci_lower, best_agg_wo_priv_ci_upper)

            for i in range(3): #have to trick sns
                plot_data.append({
                    'model_size': model_size,
                    'x_pos': x_pos,
                    'score': best_agg,
                    'method': 'Aggregate Score'
                })
                plot_data.append({
                    'model_size': model_size,
                    'x_pos': x_pos,
                    'score': best_agg_wo_privacy,
                    'method': 'Agg. w/o Priv.'
                })
            print(f"{model_size} Best: {best_method} with agg={best_agg:.4f}")
            print(f"{model_size} Best (w/o privacy): {best_method_wo_privacy} with agg_wo_privacy={best_agg_wo_privacy:.4f}")

    # Convert to DataFrame
    df = pd.DataFrame(plot_data)
    print(df)
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))

    # Plot Aggregate Score - seaborn green color (#2ca02c)
    # Plot Agg. w/o Priv. - seaborn purple color (#9467bd)
    hues = {
        'Aggregate Score': '#2ca02c',
        'Agg. w/o Priv.': '#9467bd'
    }

    # Error bar function to retrieve CIs based on mean score
    def get_error_bars(x):
        mean_val = round(x.mean(), 5)
        return error_bars.get(mean_val)
    
    print(error_bars)

    sns.lineplot(data=df, x='x_pos', y='score', hue='method', palette=hues,
                marker='s', markersize=8, linewidth=2, ax=ax, errorbar=get_error_bars, err_style="band")

    # Customize the plot
    ax.set_xlabel("Model Size Ratio ({p|q} / P)")
    ax.set_ylabel("Aggregate Score")
    ax.set_title("TOFU Model Scaling", fontsize=14)

    # Add invisible top axis to match layout
    ax_top = ax.twiny()
    ax_top.set_xticks([])
    ax_top.set_xlabel("")

    # Set x-axis labels
    ax.set_xticks([0, 1, 3])
    ax.set_xticklabels(['~0%', '12.5%', '37.5%'])
    ax.set_ylim(0, 1.0)

    # Position legend
    ax.legend(loc='lower right')
    plt.tight_layout()

    # Save
    plt.savefig("results/tofu_model_scaling_plot.png", dpi=600, bbox_inches="tight")
    plt.savefig("results/tofu_model_scaling_plot.pdf", dpi=600, bbox_inches="tight")

    print(f"\nTOFU model scaling plot saved as tofu_model_scaling_plot.png and tofu_model_scaling_plot.pdf")

def tofu_privacy_plot():
    """
    Create a plot showing privacy score vs alpha (bottom axis) and privacy score vs topk (top twinx axis).
    Shows inverse U-shaped curves where privacy score ranges from 0 to 1.
    Uses only 3.2-1B model size for clarity.
    Includes bootstrap confidence intervals as error bands.
    """

    alpha_data = []
    topk_data = []

    # Dictionary to store CI info: {mean_privacy_score: (lower_error, upper_error)}
    error_bars = {}

    # Load target baseline
    target_info = calculate_info_with_bootstrap("saves/eval/tofu/baselines/target", use_bootstrap=True)
    target_privacy = target_info["privacy_score"]
    target_privacy_ci_lower = target_info.get("privacy_score_ci_lower", target_privacy)
    target_privacy_ci_upper = target_info.get("privacy_score_ci_upper", target_privacy)
    print(f"Target privacy score: {target_privacy:.4f}")

    # Store error bars and add duplicate entries for target (alpha=0)
    error_bars[round(target_privacy, 5)] = (target_privacy_ci_lower, target_privacy_ci_upper)
    for i in range(3):  # trick sns with duplicate entries
        alpha_data.append({
            'alpha': 0.0,
            'privacy_score': target_privacy
        })

    # Load retrain baseline for reference
    retrain_info = calculate_info("saves/eval/tofu/baselines/retrain/TOFU_SUMMARY.json")
    retrain_privacy = retrain_info["privacy_score"]
    print(f"Retrain privacy score: {retrain_privacy:.4f}")

    # Model size to process
    model_size = "3.2-1B"

    # Alpha values to check
    alpha_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    print(f"Loading privacy data for {model_size}...")

    # Load alpha-based models
    for alpha in alpha_values:
        eval_dir = dd_linear_dir(model_size, alpha)
        try:
            info = calculate_info_with_bootstrap(eval_dir, use_bootstrap=True)
            if not math.isnan(info["privacy_score"]):
                privacy_score = info["privacy_score"]
                privacy_ci_lower = info.get("privacy_score_ci_lower", privacy_score)
                privacy_ci_upper = info.get("privacy_score_ci_upper", privacy_score)

                # Store error bars as (ci_lower, ci_upper) absolute bounds
                error_bars[round(privacy_score, 5)] = (privacy_ci_lower, privacy_ci_upper)

                # Add duplicate entries to trick seaborn
                for i in range(3):
                    alpha_data.append({
                        'alpha': alpha,
                        'privacy_score': privacy_score
                    })
                print(f"  Alpha {alpha}: privacy = {privacy_score:.4f}")
        except FileNotFoundError:
            print(f"Warning: Could not find {eval_dir}")
            continue

    # Load topk-based models
    topk_values = [1, 5, 20, 50, 200, 1000]

    for topk in topk_values:
        eval_dir = dd_rank_dir(model_size, topk)
        try:
            info = calculate_info_with_bootstrap(eval_dir, use_bootstrap=True)
            if not math.isnan(info["privacy_score"]):
                privacy_score = info["privacy_score"]
                privacy_ci_lower = info.get("privacy_score_ci_lower", privacy_score)
                privacy_ci_upper = info.get("privacy_score_ci_upper", privacy_score)

                # Store error bars
                error_bars[round(privacy_score, 5)] = (privacy_ci_lower, privacy_ci_upper)

                # Add duplicate entries to trick seaborn
                for i in range(3):
                    topk_data.append({
                        'topk': topk,
                        'privacy_score': privacy_score
                    })
                print(f"  TopK {topk}: privacy = {privacy_score:.4f}")
        except FileNotFoundError:
            print(f"Warning: Could not find {eval_dir}")
            continue

    if not alpha_data and not topk_data:
        print("No privacy data found. Please check that the files exist and contain privacy scores.")
        return

    # Convert to DataFrames
    df_alpha = pd.DataFrame(alpha_data)
    df_topk = pd.DataFrame(topk_data)

    # Error bar function to retrieve CIs based on mean score
    def get_error_bars(x):
        mean_val = round(x.mean(), 5)
        return error_bars.get(mean_val)

    print(error_bars)

    # Create the plot with twinx
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 3))

    # Use consistent colors from the palette
    linear_color = '#17becf'
    rank_color =  '#d62728'

    # Bottom plot: Alpha vs Privacy Score (Linear DD)
    if not df_alpha.empty:
        sns.lineplot(data=df_alpha, x='alpha', y='privacy_score',
                marker='o', markersize=8, linewidth=2,
                color=linear_color,
                label="Linear DD",
                errorbar=get_error_bars, err_style="band", ax=ax1)

    ax1.set_xlabel("Alpha (α)")
    ax1.set_ylabel("Privacy Score")

    # Create twin axis for topk
    ax2 = ax1.twiny()

    # Top plot: TopK vs Privacy Score (Rank DD)
    if not df_topk.empty:
        sns.lineplot(data=df_topk, x='topk', y='privacy_score',
                marker='X', markersize=8, linewidth=2, linestyle='--',
                color=rank_color,
                label="Rank DD",
                errorbar=get_error_bars, err_style="band", ax=ax2)

    ax2.set_xlabel("Top-k")
    ax2.set_xscale('log')

    # Add retrain baseline
    ax1.axhline(retrain_privacy, color='black', linestyle='--', alpha=0.7, label='Retrain')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Order: Linear DD, Rank DD, then retrain
    ordered_lines = []
    ordered_labels = []
    
    # Add Linear DD
    if "Linear DD" in labels1:
        idx = labels1.index("Linear DD")
        ordered_lines.append(lines1[idx])
        ordered_labels.append("Linear DD")
    
    # Add Rank DD
    if "Rank DD" in labels2:
        idx = labels2.index("Rank DD")
        ordered_lines.append(lines2[idx])
        ordered_labels.append("Rank DD")
    
    # Add retrain if present
    if 'Retrain' in labels1:
        idx = labels1.index('Retrain')
        ordered_lines.append(lines1[idx])
        ordered_labels.append('Retrain')
    
    ax1.legend(ordered_lines, ordered_labels, loc='lower right')
    ax2.legend_.remove() if ax2.legend_ else None

    plt.tight_layout()

    # Save the plot
    plt.savefig("results/tofu_privacy_plot.png", dpi=600, bbox_inches="tight")
    plt.savefig("results/tofu_privacy_plot.pdf", dpi=600, bbox_inches="tight")
    
    print(f"\nTOFU privacy plot saved as tofu_privacy_plot.png and tofu_privacy_plot.pdf")
    print(f"Processed {len(df_alpha)} alpha data points and {len(df_topk)} topk data points")

def tofu_agg_plot(with_privacy=True):
    """
    Create a plot showing aggregate score vs alpha (bottom axis) and aggregate score vs topk (top twinx axis).
    Uses only 3.2-1B model size for clarity.
    Similar styling to the privacy plot.
    Includes bootstrap confidence intervals as error bands.

    Args:
        with_privacy: If True, plot aggregate score with privacy. If False, plot without privacy.
    """

    # Determine which metric to use
    metric_key = 'agg' if with_privacy else 'agg_wo_privacy'
    metric_name = 'with privacy' if with_privacy else 'w/o privacy'
    y_label = 'Aggregate Score' if with_privacy else 'Agg. (w/o Priv.)'
    filename_suffix = '' if with_privacy else '_no_priv'
    ci_key_lower = 'agg_ci_lower' if with_privacy else 'agg_wo_privacy_ci_lower'
    ci_key_upper = 'agg_ci_upper' if with_privacy else 'agg_wo_privacy_ci_upper'

    alpha_data = []
    topk_data = []

    # Dictionary to store CI info: {mean_agg_score: (ci_lower, ci_upper)}
    error_bars = {}

    # Load target baseline
    target_info = calculate_info_with_bootstrap("saves/eval/tofu/baselines/target", use_bootstrap=True)
    target_agg = target_info[metric_key]
    target_agg_ci_lower = target_info.get(ci_key_lower, target_agg)
    target_agg_ci_upper = target_info.get(ci_key_upper, target_agg)
    print(f"Target aggregate score ({metric_name}): {target_agg:.4f}")

    # Store error bars and add duplicate entries for target (alpha=0)
    error_bars[round(target_agg, 5)] = (target_agg_ci_lower, target_agg_ci_upper)
    for i in range(3):  # trick sns with duplicate entries
        alpha_data.append({
            'alpha': 0.0,
            metric_key: target_agg
        })

    # Load retrain baseline for reference
    retrain_info = calculate_info("saves/eval/tofu/baselines/retrain/TOFU_SUMMARY.json")
    retrain_agg = retrain_info[metric_key]
    print(f"Retrain aggregate score ({metric_name}): {retrain_agg:.4f}")

    # Model size to process
    model_size = "3.2-1B"

    # Alpha values to check
    alpha_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    print(f"Loading aggregate score ({metric_name}) data for {model_size}...")

    # Load alpha-based models
    for alpha in alpha_values:
        eval_dir = dd_linear_dir(model_size, alpha)
        try:
            info = calculate_info_with_bootstrap(eval_dir, use_bootstrap=True)
            # Only check for NaN if with_privacy (agg_wo_privacy is never NaN)
            if with_privacy and not math.isnan(info[metric_key]):
                agg_score = info[metric_key]
                agg_ci_lower = info.get(ci_key_lower, agg_score)
                agg_ci_upper = info.get(ci_key_upper, agg_score)

                # Store error bars
                error_bars[round(agg_score, 5)] = (agg_ci_lower, agg_ci_upper)

                # Add duplicate entries to trick seaborn
                for i in range(3):
                    alpha_data.append({
                        'alpha': alpha,
                        metric_key: agg_score
                    })
                print(f"  Alpha {alpha}: {metric_key} = {agg_score:.4f}")
            elif not with_privacy:
                agg_score = info[metric_key]
                agg_ci_lower = info.get(ci_key_lower, agg_score)
                agg_ci_upper = info.get(ci_key_upper, agg_score)

                # Store error bars
                error_bars[round(agg_score, 5)] = (agg_ci_lower, agg_ci_upper)

                # Add duplicate entries to trick seaborn
                for i in range(3):
                    alpha_data.append({
                        'alpha': alpha,
                        metric_key: agg_score
                    })
                print(f"  Alpha {alpha}: {metric_key} = {agg_score:.4f}")
        except FileNotFoundError:
            print(f"Warning: Could not find {eval_dir}")
            continue

    # Load topk-based models
    topk_values = [1, 5, 20, 50, 200, 1000]

    for topk in topk_values:
        eval_dir = dd_rank_dir(model_size, topk)
        try:
            info = calculate_info_with_bootstrap(eval_dir, use_bootstrap=True)
            # Only check for NaN if with_privacy (agg_wo_privacy is never NaN)
            if with_privacy and not math.isnan(info[metric_key]):
                agg_score = info[metric_key]
                agg_ci_lower = info.get(ci_key_lower, agg_score)
                agg_ci_upper = info.get(ci_key_upper, agg_score)

                # Store error bars
                error_bars[round(agg_score, 5)] = (agg_ci_lower, agg_ci_upper)

                # Add duplicate entries to trick seaborn
                for i in range(3):
                    topk_data.append({
                        'topk': topk,
                        metric_key: agg_score
                    })
                print(f"  TopK {topk}: {metric_key} = {agg_score:.4f}")
            elif not with_privacy:
                agg_score = info[metric_key]
                agg_ci_lower = info.get(ci_key_lower, agg_score)
                agg_ci_upper = info.get(ci_key_upper, agg_score)

                # Store error bars
                error_bars[round(agg_score, 5)] = (agg_ci_lower, agg_ci_upper)

                # Add duplicate entries to trick seaborn
                for i in range(3):
                    topk_data.append({
                        'topk': topk,
                        metric_key: agg_score
                    })
                print(f"  TopK {topk}: {metric_key} = {agg_score:.4f}")
        except FileNotFoundError:
            print(f"Warning: Could not find {eval_dir}")
            continue

    if not alpha_data and not topk_data:
        print(f"No aggregate score ({metric_name}) data found. Please check that the files exist and contain aggregate scores.")
        return

    # Convert to DataFrames
    df_alpha = pd.DataFrame(alpha_data)
    df_topk = pd.DataFrame(topk_data)

    # Error bar function to retrieve CIs based on mean score
    def get_error_bars(x):
        mean_val = round(x.mean(), 5)
        return error_bars.get(mean_val)

    print(error_bars)

    # Create the plot with twinx
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 3))

    # Use consistent colors from the palette (same as privacy plot)
    linear_color = '#17becf'
    rank_color = '#d62728'

    # Bottom plot: Alpha vs Aggregate Score (Linear DD)
    if not df_alpha.empty:
        sns.lineplot(data=df_alpha, x='alpha', y=metric_key,
                marker='o', markersize=8, linewidth=2,
                color=linear_color,
                label="Linear DD",
                errorbar=get_error_bars, err_style="band", ax=ax1)

    ax1.set_xlabel("Alpha (α)")
    ax1.set_ylabel(y_label)

    # Create twin axis for topk
    ax2 = ax1.twiny()

    # Top plot: TopK vs Aggregate Score (Rank DD)
    if not df_topk.empty:
        sns.lineplot(data=df_topk, x='topk', y=metric_key,
                marker='X', markersize=8, linewidth=2, linestyle='--',
                color=rank_color,
                label="Rank DD",
                errorbar=get_error_bars, err_style="band", ax=ax2)

    ax2.set_xlabel("Top-k")
    ax2.set_xscale('log')

    # Add retrain baseline
    ax1.axhline(retrain_agg, color='black', linestyle='--', alpha=0.7, label='Retrain')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # Order: Linear DD, Rank DD, then retrain
    ordered_lines = []
    ordered_labels = []

    # Add Linear DD
    if "Linear DD" in labels1:
        idx = labels1.index("Linear DD")
        ordered_lines.append(lines1[idx])
        ordered_labels.append("Linear DD")

    # Add Rank DD
    if "Rank DD" in labels2:
        idx = labels2.index("Rank DD")
        ordered_lines.append(lines2[idx])
        ordered_labels.append("Rank DD")

    # Add retrain if present
    if 'Retrain' in labels1:
        idx = labels1.index('Retrain')
        ordered_lines.append(lines1[idx])
        ordered_labels.append('Retrain')

    ax1.legend(ordered_lines, ordered_labels, loc='lower right')
    ax2.legend_.remove() if ax2.legend_ else None

    plt.tight_layout()

    # Save the plot
    plt.savefig(f"results/tofu_agg{filename_suffix}_plot.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"results/tofu_agg{filename_suffix}_plot.pdf", dpi=600, bbox_inches="tight")

    print(f"\nTOFU aggregate score ({metric_name}) plot saved as tofu_agg{filename_suffix}_plot.png and tofu_agg{filename_suffix}_plot.pdf")
    print(f"Processed {len(df_alpha)} alpha data points and {len(df_topk)} topk data points")

def print_distill_sweep_results():
    """
    Print a comprehensive table of all DD distillation sweep results.
    Shows all temperature, learning rate, and epoch combinations with their scores.
    Organized by temperature groups.
    """
    print("\n" + "=" * 115)
    print("DD DISTILLATION SWEEP RESULTS")
    print("=" * 115)

    distill_configs = find_optimal_distill_configs()
    all_results = distill_configs.get("all_results", [])

    if not all_results:
        print("No distillation results found. Run tofu_distill_sweep.py first.")
        return

    # Sort by agg
    all_results_sorted = sorted(all_results, key=lambda x: (x.get("agg", 0)), reverse=True)

    # Print table header
    print(f"{'Temp':<8} {'LR':<12} {'Epoch':<8} {'Agg':<10} {'Agg w/o P':<12} {'Mem':<10} {'Priv':<10} {'Util':<10}")
    print("-" * 115)

    for r in all_results_sorted:
        temp = r.get('temperature', 1)

        lr_str = f"{r['lr']:.0e}"
        agg_str = f"{r['agg']:.4f}" if not math.isnan(r['agg']) else "N/A"
        agg_wo_p_str = f"{r['agg_wo_privacy']:.4f}" if r['agg_wo_privacy'] > 0 else "N/A"
        mem_str = f"{r['memorization_score']:.4f}" if r['memorization_score'] > 0 else "N/A"
        priv_str = f"{r['privacy_score']:.4f}" if not math.isnan(r['privacy_score']) else "N/A"
        util_str = f"{r['utility_score']:.4f}" if r['utility_score'] > 0 else "N/A"

        print(f"{temp:<8} {lr_str:<12} {r['epoch']:<8} {agg_str:<10} {agg_wo_p_str:<12} {mem_str:<10} {priv_str:<10} {util_str:<10}")

    print("-" * 115)

    # Print best configurations
    if distill_configs["best_config_agg"] is not None:
        cfg = distill_configs["best_config_agg"]
        temp = cfg.get('temperature', 1)
        print(f"\nBest config (with privacy):    temp={temp}, lr={cfg['lr']:.0e}, epoch={cfg['epoch']} -> Agg={distill_configs['best_agg']:.4f}")

    if distill_configs["best_config_agg_wo_priv"] is not None:
        cfg = distill_configs["best_config_agg_wo_priv"]
        temp = cfg.get('temperature', 1)
        print(f"Best config (w/o privacy):     temp={temp}, lr={cfg['lr']:.0e}, epoch={cfg['epoch']} -> Agg w/o P={distill_configs['best_agg_wo_priv']:.4f}")

    print("=" * 115)


def plot_distill_sweep_heatmap():
    """
    Plot a heatmap of DD distillation sweep results (epoch=10 only).
    Rows: temperature (0.5, 1.0, 1.5)
    Columns: learning rate (1e-5 through 5e-5)
    Values: aggregate score
    Cells with agg > 0.67 are highlighted as SOTA.
    """
    distill_configs = find_optimal_distill_configs()
    all_results = distill_configs.get("all_results", [])

    if not all_results:
        print("No distillation results found. Run tofu_distill_sweep.py first.")
        return

    # Filter to epoch=10 only
    results_epoch10 = [r for r in all_results if r['epoch'] == 10]

    # Define the grid
    temperatures = [0.5, 1.0, 1.5, 2.0]
    learning_rates = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5]
    lr_labels = ['1e-5', '2e-5', '3e-5', '4e-5', '5e-5', '6e-5']

    # Build the data matrix
    data = np.full((len(temperatures), len(learning_rates)), np.nan)
    for r in results_epoch10:
        temp = r['temperature']
        lr = r['lr']
        if temp in temperatures and lr in learning_rates:
            row_idx = temperatures.index(temp)
            col_idx = learning_rates.index(lr)
            data[row_idx, col_idx] = r['agg']

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 3))

    # Three-color scheme: <0.63 red, 0.63-0.67 yellow, >0.67 green (SOTA)
    from matplotlib.colors import ListedColormap
    three_cmap = ListedColormap(['#d9534f', '#f0ad4e', '#5cb85c'])  # red, yellow, green

    # Create categorical data: 0 for <0.63, 1 for 0.63-0.67, 2 for >0.67
    cat_data = np.where(data > 0.67, 2, np.where(data >= 0.63, 1, 0))
    cat_data = np.where(np.isnan(data), np.nan, cat_data)

    # Plot the heatmap with three colors
    heatmap = sns.heatmap(
        cat_data,
        ax=ax,
        annot=data,  # Show actual scores as annotations
        fmt='.3f',
        cmap=three_cmap,
        xticklabels=lr_labels,
        yticklabels=[str(t) for t in temperatures],
        cbar=False,
        vmin=0,
        vmax=2,
        linewidths=1,
        linecolor='white'
    )

    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Temperature')
    ax.tick_params(length=0)  # Remove tick marks

    plt.tight_layout()

    # Save
    plt.savefig("results/tofu_distill_heatmap.png", dpi=300, bbox_inches="tight")
    plt.savefig("results/tofu_distill_heatmap.pdf", dpi=300, bbox_inches="tight")
    print("\nHeatmap saved as tofu_distill_heatmap.png and tofu_distill_heatmap.pdf")


REBEL_CACHE_FILE = "rebel_cache.json"

def _load_rebel_cache():
    if os.path.exists(REBEL_CACHE_FILE):
        try:
            with open(REBEL_CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def _save_rebel_cache(cache):
    with open(REBEL_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def _get_rebel_memorization(method):
    """Get memorization score for a REBEL method by finding its eval dir."""
    eval_dir_map = {
        "Target": "saves/eval/tofu/baselines/target",
        "Retrain": "saves/eval/tofu/baselines/retrain",
    }
    if method in eval_dir_map:
        try:
            info = calculate_info(eval_dir_map[method] + "/TOFU_SUMMARY.json")
            return info["memorization_score"]
        except (FileNotFoundError, KeyError):
            return None

    unlearn_dirnames = {"DPO": "DPO", "GradAscent": "GradAscent", "GradDiff/GA-GDR": "GradDiff", "GradDiff": "GradDiff", "NPO": "NPO", "RMU": "RMU", "SimNPO": "SimNPO", "UNDIAL": "UNDIAL"}
    # Map directory names back to config keys (for REBEL which uses dir names)
    dir_to_config = {"GradDiff": "GradDiff/GA-GDR"}
    config_key = dir_to_config.get(method, method)
    if method in unlearn_dirnames:
        configs = find_optimal_unlearning_configs()
        if config_key in configs and configs[config_key]["best_config_agg"] is not None:
            c = configs[config_key]["best_config_agg"]
            dirname = unlearn_dirnames[method]
            path = f"{gradient_dir(GRADIENT_METHOD_DIR[dirname], c['learning_rate'], c['checkpoint'])}/TOFU_SUMMARY.json"
            try:
                return calculate_info(path)["memorization_score"]
            except (FileNotFoundError, KeyError):
                pass
        return None

    if method in ["Linear_DD", "Rank_DD"]:
        dd_configs = find_optimal_dd_configs()
        dd_type = method.replace("_DD", " DD")
        config_key = f"{dd_type} 3.2-1B"
        if config_key in dd_configs and dd_configs[config_key]["best_param_agg"] is not None:
            param = dd_configs[config_key]["best_param_agg"]
            if "Linear" in method:
                path = f"{dd_linear_dir('3.2-1B', param)}/TOFU_SUMMARY.json"
            else:
                path = f"{dd_rank_dir('3.2-1B', param)}/TOFU_SUMMARY.json"
            try:
                return calculate_info(path)["memorization_score"]
            except (FileNotFoundError, KeyError):
                pass
        return None

    if method == "Distill_DD":
        configs = find_optimal_distill_configs()
        if configs["best_config_agg"] is not None:
            c = configs["best_config_agg"]
            path = f"saves/eval/tofu/distill/lr-{c['lr']}-epoch-{c['epoch']}-temp-{c.get('temperature', 1)}/TOFU_SUMMARY.json"
            try:
                return calculate_info(path)["memorization_score"]
            except (FileNotFoundError, KeyError):
                pass
        return None

    if method in ("$\\delta$-Unlearning", "Offset"):
        config = find_optimal_offset_configs()
        if config["best_param_agg"] is not None:
            lr = config["best_param_agg"]
            path = f"saves/eval/tofu/offset/lr-{lr}/TOFU_SUMMARY.json"
            try:
                return calculate_info(path)["memorization_score"]
            except (FileNotFoundError, KeyError):
                pass
        return None

    if method in ("ULD",):
        config = find_optimal_uld_configs()
        if config["best_param_agg"] is not None:
            lr = config["best_param_agg"]
            path = f"saves/eval/tofu/uld/lr-{lr}/TOFU_SUMMARY.json"
            try:
                return calculate_info(path)["memorization_score"]
            except (FileNotFoundError, KeyError):
                pass
        return None

    if method in ("ECO",):
        config = find_optimal_eco_configs()
        if config["best_param_agg"] is not None:
            lr, strength = config["best_param_agg"]
            path = f"saves/eval/tofu/eco/lr-{lr}_str-{strength}/TOFU_SUMMARY.json"
            try:
                return calculate_info(path)["memorization_score"]
            except (FileNotFoundError, KeyError):
                pass
        return None

    return None

def generate_rebel_table():
    """Generate Leak@K table from REBEL adversarial attack results."""
    from pathlib import Path

    REBEL_RESULTS = Path("saves/eval/tofu/leak_at_k")
    K_VALUES = [1, 10, 100, 1000]

    # Method names matching the main table, in display order
    METHODS = [
        "Target", "Retrain",
        "Linear_DD", "Rank_DD", "Distill_DD",
        "DPO", "GradAscent", "GradDiff", "NPO", "RMU", "SimNPO", "UNDIAL", "LUNAR",
        "Offset", "ULD", "WHP", "GUARD", "ECO",
    ]

    def parse_leak_at_k(results_dir, ks):
        path = Path(results_dir) / "whole_generation_tofu.json"
        if not path.exists():
            return {k: None for k in ks}
        with open(path) as f:
            data = json.load(f)
        total = len(data)
        if total == 0:
            return {k: 0.0 for k in ks}
        max_k = max(ks)
        first_leak = {}
        for idx, attacks in data.items():
            for i, entry in enumerate(attacks):
                if i >= max_k:
                    break
                ev = entry[-1] if isinstance(entry, (list, tuple)) else entry
                if isinstance(ev, dict) and ev.get("leaked", False):
                    first_leak[idx] = i
                    break
        return {k: sum(1 for fl in first_leak.values() if fl < k) / total for k in ks}

    # Try loading from cache (invalidate if methods changed)
    cache = _load_rebel_cache()
    cached_methods = cache.get("methods", [])
    if cache.get("table_data") and cached_methods == METHODS:
        print("\n[Loaded REBEL table from cache. Delete rebel_cache.json to recompute.]")
        table_data = cache["table_data"]
    else:
        # Collect results
        table_data = []
        for method in METHODS:
            print(f"  Processing REBEL {method}...")
            leak_dir = REBEL_RESULTS / method / "leak"
            scores = parse_leak_at_k(leak_dir, K_VALUES)
            if all(v is None for v in scores.values()):
                continue
            utility = _get_rebel_memorization(method)
            table_data.append({
                "Method": method,
                "Utility": utility,
                **{f"Leak@{k}": scores[k] for k in K_VALUES},
            })

        if not table_data:
            print("\nNo REBEL Leak@ results found in saves/eval/tofu/leak_at_k/")
            return

        # Save to cache
        _save_rebel_cache({"table_data": table_data, "methods": METHODS})
        print("  [Cached REBEL table to rebel_cache.json]")

    # Display name mapping
    display_names = {
        "Linear_DD": "Linear DD", "Rank_DD": "Rank DD", "Distill_DD": "Distill DD",
        "GradDiff": "GradDiff/GA-GDR", "Offset": "$\\delta$-Unlearning",
    }

    # Simplified markdown table (Method, Leak@10, Leak@1000 only)
    bold_methods = {"Target", "Retrain", "Linear_DD", "Rank_DD", "Distill_DD"}
    print("\n" + "=" * 80)
    print("TABLE: REBEL Leak@")
    print("=" * 80)
    print(f"| Method | Leak@10 \u2193 | Leak@1000 \u2193 |")
    print(f"| --- | --- | --- |")
    for row in table_data:
        name = display_names.get(row["Method"], row["Method"])
        l10 = f"{row['Leak@10'] * 100:.1f}" if row.get("Leak@10") is not None else "---"
        l1000 = f"{row['Leak@1000'] * 100:.1f}" if row.get("Leak@1000") is not None else "---"
        if row["Method"] in bold_methods:
            print(f"| **{name}** | **{l10}** | **{l1000}** |")
        else:
            print(f"| {name} | {l10} | {l1000} |")

    # Line plot: Leak@K vs K
    # Extend palette/markers for methods not in muse_scores
    extra_palette = {
        "DPO": "#e377c2", "GradAscent": "#ff9896", "GradDiff/GA-GDR": "#bcbd22",
        "NPO": "#9467bd", "RMU": "#c5b0d5", "SimNPO": "#2ca02c",
        "UNDIAL": "#dbdb8d", "Distill DD": "#8c564b", "WHP": "#c49c94",
        "GUARD": "#e377c2",
        "ECO": "#dbdb8d",
        "LUNAR": "#f7b6d2",
    }
    extra_markers = {
        "DPO": "o", "GradAscent": "o", "GradDiff/GA-GDR": "o",
        "NPO": "o", "RMU": "o", "SimNPO": "o",
        "UNDIAL": "o", "Distill DD": "X", "WHP": "D", "GUARD": "D",
        "ECO": "P", "LUNAR": "P",
    }
    plot_palette = {**palette, **extra_palette}
    plot_markers = {**markers, **extra_markers}

    fig, ax = plt.subplots(figsize=(8, 5))
    for row in table_data:
        name = display_names.get(row["Method"], row["Method"])
        ks = [k for k in K_VALUES if row.get(f"Leak@{k}") is not None]
        vals = [row[f"Leak@{k}"] * 100 for k in ks]
        if not ks:
            continue
        ax.plot(ks, vals,
                marker=plot_markers.get(name, "o"),
                color=plot_palette.get(name, "#333333"),
                label=name, linewidth=2, markersize=7)

    ax.set_xscale("log")
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("n (number of attack attempts)")
    ax.set_ylabel("Leak@n")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig("results/rebel_leak_plot.png", dpi=600, bbox_inches="tight")
    plt.savefig("results/rebel_leak_plot.pdf", dpi=600, bbox_inches="tight")
    print("\nSaved rebel_leak_plot.png and rebel_leak_plot.pdf")
    plt.close()

    # Simplified plot: highlight DD methods, gray out the rest
    highlight_methods = {"Target", "Retrain", "Linear DD", "Rank DD", "Distill DD"}
    fig, ax = plt.subplots(figsize=(6, 5))
    other_added = False
    for row in table_data:
        name = display_names.get(row["Method"], row["Method"])
        ks = [k for k in K_VALUES if row.get(f"Leak@{k}") is not None]
        vals = [row[f"Leak@{k}"] * 100 for k in ks]
        if not ks:
            continue
        if name in highlight_methods:
            ax.plot(ks, vals,
                    marker=plot_markers.get(name, "o"),
                    color=plot_palette.get(name, "#333333"),
                    label=name, linewidth=2, markersize=7)
        else:
            ax.plot(ks, vals,
                    marker="o", color="#999999",
                    label="Baselines" if not other_added else None,
                    linewidth=1.5, markersize=5, alpha=0.7)
            other_added = True

    ax.set_xscale("log")
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Number of Attack Attempts")
    ax.set_ylabel("Leak@n")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig("results/rebel_leak_plot_simple.png", dpi=600, bbox_inches="tight")
    plt.savefig("results/rebel_leak_plot_simple.pdf", dpi=600, bbox_inches="tight")
    print("Saved rebel_leak_plot_simple.png and rebel_leak_plot_simple.pdf")
    plt.close()


def render_tofu_table_png(use_bootstrap=True, n_samples=BOOTSTRAP_N_SAMPLES):
    """Render the TOFU results table as a PNG with rows highlighted by category."""
    from matplotlib.colors import to_rgba

    # Reuse generate_tofu_tables logic to build the dataframe
    print("Finding optimal configurations...")
    dd_configs = find_optimal_dd_configs()
    cross_tok_configs = find_optimal_cross_tok_configs()
    unlearn_configs = find_optimal_unlearning_configs()
    distill_configs = find_optimal_distill_configs()
    offset_config = find_optimal_offset_configs()
    uld_config = find_optimal_uld_configs()
    whp_config = find_optimal_whp_configs()
    guard_config = find_optimal_guard_configs()
    eco_config = find_optimal_eco_configs()
    lunar_config = find_optimal_lunar_configs()

    table_data = []
    print("\nComputing scores" + (" with bootstrap CIs..." if use_bootstrap else "..."))

    # Target and Retrain
    print("  Processing Target...")
    target_info = calculate_info_with_bootstrap("saves/eval/tofu/baselines/target", use_bootstrap=use_bootstrap, n_samples=n_samples)
    print("  Processing Retrain...")
    retrain_info = calculate_info_with_bootstrap("saves/eval/tofu/baselines/retrain", use_bootstrap=use_bootstrap, n_samples=n_samples)

    for name, info, config in [("Target", target_info, "Full"), ("Retrain", retrain_info, "Retain90")]:
        table_data.append({
            "Method": name, "Config": config,
            **{k: info.get(k, float('nan')) for k in
               ["agg", "memorization_score", "privacy_score", "utility_score"]}
        })

    # Linear DD
    for model_size in ["3.2-1B"]:
        config_key = f"Linear DD {model_size}"
        if config_key in dd_configs and dd_configs[config_key]["best_param_agg"] is not None:
            alpha = dd_configs[config_key]["best_param_agg"]
            eval_dir = dd_linear_dir(model_size, alpha)
            try:
                info = calculate_info_with_bootstrap(eval_dir, use_bootstrap=use_bootstrap, n_samples=n_samples)
                table_data.append({"Method": "Linear DD", "Config": f"α={alpha}",
                    **{k: info.get(k, float('nan')) for k in ["agg", "memorization_score", "privacy_score", "utility_score"]}})
            except FileNotFoundError:
                pass

    # Rank DD
    for model_size in ["3.2-1B"]:
        config_key = f"Rank DD {model_size}"
        if config_key in dd_configs and dd_configs[config_key]["best_param_agg"] is not None:
            topk = dd_configs[config_key]["best_param_agg"]
            eval_dir = dd_rank_dir(model_size, topk)
            try:
                info = calculate_info_with_bootstrap(eval_dir, use_bootstrap=use_bootstrap, n_samples=n_samples)
                table_data.append({"Method": "Rank DD", "Config": f"k={topk}",
                    **{k: info.get(k, float('nan')) for k in ["agg", "memorization_score", "privacy_score", "utility_score"]}})
            except FileNotFoundError:
                pass

    # Distill DD
    if distill_configs["best_config_agg"] is not None:
        config = distill_configs["best_config_agg"]
        lr, epoch, temp = config["lr"], config["epoch"], config.get("temperature", 1)
        eval_dir = f"saves/eval/tofu/distill/lr-{lr}-epoch-{epoch}-temp-{temp}"
        try:
            info = calculate_info_with_bootstrap(eval_dir, use_bootstrap=use_bootstrap, n_samples=n_samples)
            table_data.append({"Method": "Distill DD", "Config": f"lr={lr:.0e}, T={temp}",
                **{k: info.get(k, float('nan')) for k in ["agg", "memorization_score", "privacy_score", "utility_score"]}})
        except Exception:
            pass

    # Unlearning methods
    method_dirnames = {"DPO": "DPO", "GradAscent": "GradAscent", "GradDiff/GA-GDR": "GradDiff",
                       "NPO": "NPO", "RMU": "RMU", "SimNPO": "SimNPO", "UNDIAL": "UNDIAL"}
    for method, dirname in method_dirnames.items():
        if method in unlearn_configs and unlearn_configs[method]["best_config_agg"] is not None:
            cfg = unlearn_configs[method]["best_config_agg"]
            eval_dir = gradient_dir(GRADIENT_METHOD_DIR[dirname], cfg['learning_rate'], cfg['checkpoint'])
            try:
                info = calculate_info_with_bootstrap(eval_dir, use_bootstrap=use_bootstrap, n_samples=n_samples)
                table_data.append({"Method": method, "Config": f"lr={cfg['learning_rate']}, e={cfg['epoch']}",
                    **{k: info.get(k, float('nan')) for k in ["agg", "memorization_score", "privacy_score", "utility_score"]}})
            except Exception:
                pass

    # Offset / ULD / WHP / GUARD / ECO / LUNAR
    simple_methods = [
        (offset_config, "$\\delta$-Unlearning", lambda c: (f"saves/eval/tofu/offset/lr-{c}", f"lr={c}")),
        (uld_config, "ULD", lambda c: (f"saves/eval/tofu/uld/lr-{c}", f"lr={c}")),
        (lunar_config, "LUNAR", lambda c: (f"saves/eval/tofu/lunar/lr-{c}", f"lr=0.{c}")),
    ]
    for cfg, name, path_fn in simple_methods:
        if cfg["best_param_agg"] is not None:
            eval_dir, config_str = path_fn(cfg["best_param_agg"])
            try:
                _add_table_entry(table_data, name, config_str, eval_dir, use_bootstrap=use_bootstrap, n_samples=n_samples)
            except Exception:
                pass

    if whp_config["best_param_agg"] is not None:
        lr, alpha = whp_config["best_param_agg"]
        try:
            _add_table_entry(table_data, "WHP", f"lr={lr}, α={alpha.replace('_','.')}", f"saves/eval/tofu/whp/lr-{lr}_alpha-{alpha}",
                             use_bootstrap=use_bootstrap, n_samples=n_samples)
        except Exception:
            pass

    if guard_config["best_param_agg"] is not None:
        lr, delta = guard_config["best_param_agg"]
        try:
            _add_table_entry(table_data, "GUARD", f"lr={lr}, δ={delta.replace('_','.')}", f"saves/eval/tofu/guard/lr-{lr}_delta-{delta}",
                             use_bootstrap=use_bootstrap, n_samples=n_samples)
        except Exception:
            pass

    if eco_config["best_param_agg"] is not None:
        lr, strength = eco_config["best_param_agg"]
        try:
            _add_table_entry(table_data, "ECO", f"lr={lr}, str={strength}", f"saves/eval/tofu/eco/lr-{lr}_str-{strength}",
                             use_bootstrap=use_bootstrap, n_samples=n_samples)
        except Exception:
            pass

    # Cross-tokenizer DD
    cross_tok_shorts = {"OLMo": "OLMo-2-0425-1B-Instruct", "Gemma": "gemma-3-1b-it", "Qwen": "Qwen3-1.7B"}
    for label, results in cross_tok_configs.items():
        for sweep_type in ["alpha", "topk"]:
            r = results[sweep_type]
            if r["best_param_agg"] is not None:
                short = cross_tok_shorts[label]
                val = r["best_param_agg"]
                lr = r.get("lr", "1e-5")
                eval_dir = f"saves/eval/tofu/cross_tok/{short}-lr{lr}-{sweep_type}-{val}"
                prefix = "α" if sweep_type == "alpha" else "k"
                variant = "Linear" if sweep_type == "alpha" else "Rank"
                _add_table_entry(table_data, f"{label} {variant} CT-DD", f"lr={lr}, {prefix}={val}",
                                 eval_dir, use_bootstrap=use_bootstrap, n_samples=n_samples)

    df = pd.DataFrame(table_data)

    # Enforce display order
    method_order = [
        "Target", "Retrain", "Linear DD", "Rank DD", "Distill DD",
        "DPO", "GradAscent", "GradDiff/GA-GDR", "NPO", "RMU", "SimNPO", "UNDIAL", "LUNAR",
        "$\\delta$-Unlearning", "ULD", "WHP", "GUARD", "ECO",
    ]
    # Append any cross-tok methods at the end
    for m in df["Method"].unique():
        if m not in method_order:
            method_order.append(m)
    order_map = {m: i for i, m in enumerate(method_order)}
    df["_order"] = df["Method"].map(order_map).fillna(len(method_order))
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    # Category definitions (matching muse_scores.py)
    dd_methods = {"Target", "Retrain", "Linear DD", "Rank DD", "Distill DD"}
    gradient_methods = {"DPO", "GradAscent", "GradDiff/GA-GDR", "NPO", "RMU", "SimNPO", "UNDIAL", "LUNAR"}
    inference_methods = {"$\\delta$-Unlearning", "WHP", "ULD", "GUARD", "ECO"}
    cross_tok_methods = {f"{label} {variant} CT-DD"
                         for label in ["OLMo", "Gemma", "Qwen"]
                         for variant in ["Linear", "Rank"]}

    # Colors for each category
    cat_colors = {
        "DD Methods": to_rgba("#1f77b4", 0.15),
        "Other Gradient-Based": to_rgba("#999999", 0.15),
        "Other Inference-Time": to_rgba("#e377c2", 0.15),
        "DD Cross-Tokenizer": to_rgba("#2ca02c", 0.15),
    }

    def get_category(method):
        if method in dd_methods:
            return "DD Methods"
        elif method in gradient_methods:
            return "Other Gradient-Based"
        elif method in inference_methods:
            return "Other Inference-Time"
        elif method in cross_tok_methods:
            return "DD Cross-Tokenizer"
        return "DD Methods"

    # Build display table
    metric_cols = ['agg', 'memorization_score', 'privacy_score', 'utility_score']
    display_cols = ["Method", "Config", "Agg ↑", "Mem ↑", "Priv ↑", "Util ↑"]
    cell_text = []
    row_colors = []
    for _, row in df.iterrows():
        method = row["Method"]
        # Clean up LaTeX for display
        display_method = method.replace("$\\delta$-", "δ-")
        display_config = row["Config"].replace("$\\alpha$", "α").replace("$\\delta$", "δ")
        cell_text.append([
            display_method, display_config,
            f"{row['agg']:.2f}" if not pd.isna(row['agg']) else "N/A",
            f"{row['memorization_score']:.2f}" if not pd.isna(row['memorization_score']) else "N/A",
            f"{row['privacy_score']:.2f}" if not pd.isna(row['privacy_score']) else "N/A",
            f"{row['utility_score']:.2f}" if not pd.isna(row['utility_score']) else "N/A",
        ])
        row_colors.append(cat_colors[get_category(method)])

    # Find top 2 per metric for bolding
    df_methods = df[~df['Method'].isin(['Target', 'Retrain'])]
    top2_indices = {}
    for col in metric_cols:
        valid_vals = df_methods[col].dropna().sort_values(ascending=False)
        if len(valid_vals) >= 2:
            top2_vals = valid_vals.head(2).tolist()
            top2_indices[col] = set(df_methods[df_methods[col].isin(top2_vals)].index.tolist())
        elif len(valid_vals) == 1:
            top2_indices[col] = set(df_methods[df_methods[col] == valid_vals.iloc[0]].index.tolist())
        else:
            top2_indices[col] = set()

    # Render with matplotlib
    n_rows = len(cell_text)
    row_h = 0.38
    fig_height = row_h * (n_rows + 1)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    table = ax.table(cellText=cell_text, colLabels=display_cols, loc='bottom', cellLoc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(list(range(len(display_cols))))

    # Style header
    for j in range(len(display_cols)):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    # Style data rows
    for i in range(n_rows):
        for j in range(len(display_cols)):
            cell = table[i + 1, j]
            cell.set_facecolor(row_colors[i])
            cell.set_edgecolor('#cccccc')
            # Bold top-2 metric cells
            if j >= 2:
                col_key = metric_cols[j - 2]
                if df.index[i] in top2_indices.get(col_key, set()):
                    cell.set_text_props(fontweight='bold')
            # Bold method name for DD methods
            if j == 0 and df.iloc[i]["Method"] in dd_methods:
                cell.set_text_props(fontweight='bold')

    fig.text(0.5, -0.01, "SEs omitted to reduce clutter.", ha='center', va='top',
             fontsize=12, fontstyle='italic', color='#555555')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig("results/tofu_table.png", dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor='white')
    print("Saved tofu_table.png")
    plt.close()


# Main execution
def _front_page_agg_ci_half_width(eval_dir):
    """Best-effort 99% bootstrap CI half-width for the `agg` score of one
    method's best-`agg` eval directory.

    Reuses the same bootstrap machinery as the main table
    (``calculate_info_with_bootstrap`` with the module-level 99%
    ``BOOTSTRAP_ALPHA``). Degrades gracefully: if ``eval_dir`` is ``None``,
    the directory / its ``TOFU_SUMMARY.json`` is missing, or no per-index
    ``TOFU_EVAL.json`` is present (so no bootstrap can run), it returns
    ``nan`` instead of raising — the caller then plots the point estimate
    with no error bar.
    """
    if not eval_dir:
        return float("nan")
    if not os.path.exists(os.path.join(eval_dir, "TOFU_SUMMARY.json")):
        return float("nan")
    try:
        info = calculate_info_with_bootstrap(eval_dir, use_bootstrap=True)
    except Exception:
        return float("nan")
    return info.get("agg_ci_half_width", float("nan"))


def collect_front_page_scores():
    """Collect best `agg` and `agg_wo_privacy` scores for every method.

    For each method we report its best-achievable score under each metric
    (i.e. the optimal config may differ between the two columns), mirroring
    how the tables/MUSE scatter pick per-metric optima.

    Each row also carries ``agg_ci_half_width``: the 99% bootstrap CI
    half-width for the `agg` score of that method's best-`agg` config,
    computed from its per-index ``TOFU_EVAL.json`` via the same machinery as
    the main table. When ``TOFU_EVAL.json`` is absent (results still being
    generated) the half-width is ``nan`` and downstream plots simply omit the
    error bar for that method.

    Returns a list of dicts: {name, agg, agg_wo_privacy, agg_ci_half_width}.
    """
    dd_configs = find_optimal_dd_configs()
    distill_configs = find_optimal_distill_configs()
    unlearn_configs = find_optimal_unlearning_configs()
    offset = find_optimal_offset_configs()
    uld = find_optimal_uld_configs()
    whp = find_optimal_whp_configs()
    guard = find_optimal_guard_configs()
    eco = find_optimal_eco_configs()
    lunar = find_optimal_lunar_configs()

    rows = []

    # --- Baselines (single config each) ---
    # (display name, TOFU_SUMMARY.json path, eval dir for bootstrap CIs)
    for name, path, ci_dir in [
        ("Target", "saves/eval/tofu/baselines/target/TOFU_SUMMARY.json",
         "saves/eval/tofu/baselines/target"),
        ("Retrain", "saves/eval/tofu/baselines/retrain/TOFU_SUMMARY.json",
         "saves/eval/tofu/baselines/retrain"),
    ]:
        info = calculate_info(path)
        rows.append({
            "name": name,
            "agg": info["agg"],
            "agg_wo_privacy": info["agg_wo_privacy"],
            "agg_ci_half_width": _front_page_agg_ci_half_width(ci_dir),
        })

    # --- Highlighted DD methods ---
    for name, key, dir_fn in [
        ("Linear DD", "Linear DD 3.2-1B", dd_linear_dir),
        ("Rank DD", "Rank DD 3.2-1B", dd_rank_dir),
    ]:
        c = dd_configs[key]
        ci_dir = dir_fn("3.2-1B", c["best_param_agg"]) if c["best_param_agg"] is not None else None
        rows.append({
            "name": name,
            "agg": c["best_agg"],
            "agg_wo_privacy": c["best_agg_wo_priv"],
            "agg_ci_half_width": _front_page_agg_ci_half_width(ci_dir),
        })

    distill_ci_dir = None
    if distill_configs["best_config_agg"] is not None:
        dc = distill_configs["best_config_agg"]
        distill_ci_dir = (f"saves/eval/tofu/distill/lr-{dc['lr']}-epoch-"
                          f"{dc['epoch']}-temp-{dc.get('temperature', 1)}")
    rows.append({
        "name": "Distill DD",
        "agg": distill_configs["best_agg"],
        "agg_wo_privacy": distill_configs["best_agg_wo_priv"],
        "agg_ci_half_width": _front_page_agg_ci_half_width(distill_ci_dir),
    })

    # --- Gradient-based unlearning methods (gray) ---
    for method in ["DPO", "GradAscent", "GradDiff/GA-GDR", "NPO", "RMU", "SimNPO", "UNDIAL"]:
        cfg = unlearn_configs.get(method)
        if not cfg or cfg["best_config_agg"] is None:
            continue
        bc = cfg["best_config_agg"]
        ci_dir = gradient_dir(
            GRADIENT_METHOD_DIR[{"GradDiff/GA-GDR": "GradDiff"}.get(method, method)],
            bc["learning_rate"], bc["checkpoint"],
        )
        rows.append({
            "name": method,
            "agg": cfg["best_agg"],
            "agg_wo_privacy": cfg["best_agg_wo_priv"],
            "agg_ci_half_width": _front_page_agg_ci_half_width(ci_dir),
        })

    # --- Inference-time / other methods (gray), all via _sweep_best ---
    # eval-dir builder maps each method's best-`agg` param back to its saves dir
    # so the 99% CI is computed from the same config that produced `best_agg`.
    def offset_dir(p):
        return f"saves/eval/tofu/offset/lr-{p}"

    def uld_dir(p):
        return f"saves/eval/tofu/uld/lr-{p}"

    def whp_dir(p):
        return f"saves/eval/tofu/whp/lr-{p[0]}_alpha-{p[1]}"

    def guard_dir(p):
        return f"saves/eval/tofu/guard/lr-{p[0]}_delta-{p[1]}"

    def eco_dir(p):
        return f"saves/eval/tofu/eco/lr-{p[0]}_str-{p[1]}"

    def lunar_dir(p):
        return f"saves/eval/tofu/lunar/lr-{p}"

    for name, cfg, dir_builder in [
        ("$\\delta$-Unlearning", offset, offset_dir),
        ("ULD", uld, uld_dir),
        ("WHP", whp, whp_dir),
        ("GUARD", guard, guard_dir),
        ("ECO", eco, eco_dir),
        ("LUNAR", lunar, lunar_dir),
    ]:
        if cfg.get("best_param_agg") is None and cfg.get("best_param_agg_wo_priv") is None:
            continue
        ci_dir = dir_builder(cfg["best_param_agg"]) if cfg.get("best_param_agg") is not None else None
        rows.append({
            "name": name,
            "agg": cfg["best_agg"] if cfg["best_agg"] >= 0 else float("nan"),
            "agg_wo_privacy": cfg["best_agg_wo_priv"] if cfg["best_agg_wo_priv"] >= 0 else float("nan"),
            "agg_ci_half_width": _front_page_agg_ci_half_width(ci_dir),
        })

    return rows


def tofu_front_page_scatter():
    """Original front-page strip plot of the TOFU Aggregate Score, one marker
    per method (MUSE styling), with a top legend instead of on-plot labels.

    The five highlighted methods (Target, Retrain, Linear/Rank/Distill DD) are
    drawn in full colour; all remaining methods collapse into a gray
    "Baselines" cloud. Unlike the original, the baseline cloud uses *zero*
    jitter, so every baseline marker sits on the single x=0 column.
    """
    rows = collect_front_page_scores()

    highlight = ["Target", "Retrain", "Linear DD", "Rank DD", "Distill DD"]
    gradient = {"DPO", "GradAscent", "GradDiff/GA-GDR", "NPO", "RMU", "SimNPO", "UNDIAL", "LUNAR"}
    inference = {"$\\delta$-Unlearning", "ULD", "WHP", "GUARD", "ECO"}

    def plot_name(name):
        if name in gradient or name in inference:
            return "Baselines"
        return name

    baseline_color = "#999999"

    # Single x column (Aggregate Score)
    xa = 0.0

    fig, ax = plt.subplots(1, 1, figsize=(5, 4.2))
    s = 260

    # --- Gray baseline cloud (drawn first, behind), zero jitter ---
    baseline_vals = [r["agg"] for r in rows
                     if plot_name(r["name"]) == "Baselines"
                     and not (isinstance(r["agg"], float) and math.isnan(r["agg"]))]
    ax.scatter([xa] * len(baseline_vals), baseline_vals, marker="o", s=s * 0.8,
               facecolor=baseline_color, edgecolor="black", linewidth=0.5,
               alpha=0.55, zorder=2, label="Baselines")

    # --- Highlighted methods (drawn on top, full colour) ---
    # No jitter or offsets: every marker sits on the single x=0 line.
    for name in highlight:
        row = next((r for r in rows if r["name"] == name), None)
        if row is None:
            continue
        val = row["agg"]
        if isinstance(val, float) and math.isnan(val):
            continue
        ax.scatter(xa, val,
                   marker=markers[name], s=s,
                   facecolor=palette[name], edgecolor="black", linewidth=1.2,
                   alpha=1.0, zorder=3, label=name)

    ax.set_xticks([])
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylabel("Aggregate Score")
    ax.margins(y=0.08)

    # Two-row legend on top. Interleaved so matplotlib's column-major fill yields:
    #   row 1: Target, Retrain, Baselines
    #   row 2: Linear DD, Rank DD, Distill DD
    legend_order = ["Target", "Linear DD", "Retrain",
                    "Rank DD", "Baselines", "Distill DD"]
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ordered = [(label_to_handle[l], l) for l in legend_order if l in label_to_handle]
    if ordered:
        fig.legend(
            [h for h, _ in ordered], [l for _, l in ordered],
            loc="upper center", ncol=3, frameon=False,
            handletextpad=0.4, columnspacing=0.8,
            bbox_to_anchor=(0.5, 1.11),
        )

    plt.tight_layout()
    plt.savefig("results/tofu_front_page_scatter.png", dpi=600, bbox_inches="tight")
    plt.savefig("results/tofu_front_page_scatter.pdf", dpi=600, bbox_inches="tight")
    print("\nSaved tofu_front_page_scatter.png and tofu_front_page_scatter.pdf")


def tofu_front_page_lineplot():
    """Front-page plot of the TOFU Aggregate Score where each method *family*
    is its own x-axis group:

      * Retrain          -> a dashed horizontal reference line across the plot
      * Divergence Decoding / Gradient-Based / Inference-Time
                          -> an x-axis group holding that family's individual
                             methods as points, coloured + marked per the
                             shared muse_scores style.

    Each visible point carries a 99% bootstrap confidence interval (vertical
    error bar) computed from that method's per-index ``TOFU_EVAL.json`` via the
    same machinery as the main table (``collect_front_page_scores`` -> the
    module-level 99% ``BOOTSTRAP_ALPHA``). Methods whose ``TOFU_EVAL.json`` is
    absent (results still generating) are plotted as the point estimate with no
    error bar rather than crashing.

    Only methods scoring above the y-floor are drawn (and legended); no jitter
    (points within a group are spread on a fixed, evenly-spaced ladder).
    """
    rows = collect_front_page_scores()
    by_name = {r["name"]: r["agg"] for r in rows}
    ci_by_name = {r["name"]: r.get("agg_ci_half_width", float("nan")) for r in rows}

    def vals_of(names):
        out = [(n, by_name.get(n, float("nan"))) for n in names]
        return [(n, v) for n, v in out
                if not (isinstance(v, float) and math.isnan(v))]

    # The muse_scores palette/marker key differs from the display name here.
    style_key = {"GradDiff/GA-GDR": "GradDiff"}

    # (x-axis label, member methods, family color, per-method styling?)
    groups = [
        ("Ours",
         ["Linear DD", "Rank DD", "Distill DD"], None, True),
        ("Gradient-Based",
         ["DPO", "GradAscent", "GradDiff/GA-GDR", "NPO", "RMU", "SimNPO", "UNDIAL", "LUNAR"],
         "#999999", False),
        ("Inference-Time",
         ["$\\delta$-Unlearning", "ULD", "WHP", "GUARD", "ECO"], "#666666", False),
    ]

    retrain = by_name.get("Retrain", float("nan"))

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2))
    point_spread = 0.22       # half-width of the point ladder within a group
    y_floor = 0.5             # y-axis floor; methods below this are cut off

    # --- Retrain dashed reference line across the full width ---
    if not (isinstance(retrain, float) and math.isnan(retrain)):
        ax.axhline(retrain, ls="--", color="black", lw=2, zorder=1)
        ax.text(len(groups) - 0.55, retrain + 0.008, "Retrain Benchmark",
                ha="right", va="bottom", fontsize=11, style="italic",
                color="black")

    for i, (label, methods, color, per_method) in enumerate(groups):
        members = vals_of(methods)
        if not members:
            continue
        names = [n for n, _ in members]
        ys = [v for _, v in members]

        # Evenly spaced point ladder within the group (no randomness).
        if len(ys) == 1:
            xs = [i]
        else:
            xs = i + np.linspace(-point_spread, point_spread, len(ys))

        # Plot only the methods that clear the y-floor, each with the colour +
        # marker defined in muse_scores, and add it to the legend.
        for x, name, y in zip(xs, names, ys):
            if y < y_floor:
                continue
            key = style_key.get(name, name)
            # 99% CI half-width; nan when no per-index TOFU_EVAL.json exists.
            hw = ci_by_name.get(name, float("nan"))
            if isinstance(hw, float) and (math.isnan(hw) or hw <= 0):
                # Graceful degradation: point estimate only, no error bar.
                ax.scatter(x, y, marker=markers.get(key, "o"), s=220,
                           facecolor=palette.get(key, "#888888"), edgecolor="black",
                           linewidth=0.8, alpha=1.0, zorder=3, label=key)
            else:
                # 99% confidence interval as a vertical error bar, with the
                # styled marker drawn on top of it.
                ax.errorbar(x, y, yerr=hw, fmt="none", ecolor=palette.get(key, "#888888"),
                            elinewidth=1.5, capsize=3, capthick=1.0, zorder=2)
                ax.scatter(x, y, marker=markers.get(key, "o"), s=220,
                           facecolor=palette.get(key, "#888888"), edgecolor="black",
                           linewidth=0.8, alpha=1.0, zorder=3, label=key)

    # Legend (colour + symbol) for every visible method: 6 rows x 2 columns,
    # with small markers and tight padding.
    leg = ax.legend(loc="lower left", ncol=2, frameon=False,
                    fontsize=11, handletextpad=0.1, columnspacing=0.3,
                    handlelength=0.6, borderpad=0.1, scatterpoints=1,
                    borderaxespad=0.6, markerscale=0.45)

    # Box the whole bottom-left corner: the left and bottom axes spines form
    # two sides, so we only draw the top and right sides (two lines) to close it.
    # Tight pad so the box stays clear of the Gradient-Based markers.
    fig.canvas.draw()
    bb = leg.get_window_extent().transformed(ax.transAxes.inverted())
    right, top = bb.x1 - 0.012, bb.y1
    box_kw = dict(color="black", lw=1.0, transform=ax.transAxes,
                  clip_on=False, zorder=5)
    ax.plot([0, right], [top, top], **box_kw)
    ax.plot([right, right], [0, top], **box_kw)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=11)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0.54, 0.88)
    ax.set_yticks([0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85])
    ax.set_ylabel("TOFU Aggregate Score")

    plt.tight_layout()
    plt.savefig("results/tofu_front_page_lineplot.png", dpi=600, bbox_inches="tight")
    plt.savefig("results/tofu_front_page_lineplot.pdf", dpi=600, bbox_inches="tight")
    print("\nSaved results/tofu_front_page_lineplot.png and results/tofu_front_page_lineplot.pdf")


def tofu_front_page_plot():
    """Front-page strip plot of the TOFU Aggregate Score, one marker per method
    (MUSE styling), with direct on-plot labels instead of a legend.

    The five highlighted methods (Target, Retrain, Linear/Rank/Distill DD) are
    drawn in full colour; all remaining methods collapse into gray
    Gradient (v) and Inference-Time (^) categories.
    """
    rows = collect_front_page_scores()

    highlight = ["Target", "Retrain", "Linear DD", "Rank DD", "Distill DD"]
    gradient = {"DPO", "GradAscent", "GradDiff/GA-GDR", "NPO", "RMU", "SimNPO", "UNDIAL", "LUNAR"}
    inference = {"$\\delta$-Unlearning", "ULD", "WHP", "GUARD", "ECO"}

    def plot_name(name):
        if name in gradient or name in inference:
            return "Baselines"
        return name

    baseline_color = "#999999"

    # Single x column (Aggregate Score)
    xa = 0.0

    fig, ax = plt.subplots(1, 1, figsize=(5, 4.2))
    s = 260
    rng = np.random.RandomState(0)

    # Deterministic horizontal offsets for the highlighted markers so they
    # never overlap each other.
    hl_offsets = {name: off for name, off in zip(
        highlight, np.linspace(-0.28, 0.28, len(highlight)))}

    # --- Gray baseline cloud (drawn first, behind) ---
    baseline_vals = [r["agg"] for r in rows
                     if plot_name(r["name"]) == "Baselines"
                     and not (isinstance(r["agg"], float) and math.isnan(r["agg"]))]
    jitter = rng.uniform(-0.3, 0.3, size=len(baseline_vals))
    ax.scatter([xa + j for j in jitter], baseline_vals, marker="o", s=s * 0.8,
               facecolor=baseline_color, edgecolor="black", linewidth=0.5,
               alpha=0.55, zorder=2, label="Baselines")

    # --- Highlighted methods (drawn on top, full colour) ---
    for name in highlight:
        row = next((r for r in rows if r["name"] == name), None)
        if row is None:
            continue
        val = row["agg"]
        if isinstance(val, float) and math.isnan(val):
            continue
        ax.scatter(xa + hl_offsets[name], val,
                   marker=markers[name], s=s,
                   facecolor=palette[name], edgecolor="black", linewidth=1.2,
                   alpha=1.0, zorder=3, label=name)

    ax.set_xticks([])
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylabel("Aggregate Score")
    ax.margins(y=0.08)

    # Two-row legend on top. Interleaved so matplotlib's column-major fill yields:
    #   row 1: Target, Retrain, Baselines
    #   row 2: Linear DD, Rank DD, Distill DD
    legend_order = ["Target", "Linear DD", "Retrain",
                    "Rank DD", "Baselines", "Distill DD"]
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ordered = [(label_to_handle[l], l) for l in legend_order if l in label_to_handle]
    if ordered:
        fig.legend(
            [h for h, _ in ordered], [l for _, l in ordered],
            loc="upper center", ncol=3, frameon=False,
            handletextpad=0.4, columnspacing=0.8,
            bbox_to_anchor=(0.5, 1.11),
        )

    plt.tight_layout()
    plt.savefig("results/tofu_front_page_plot.png", dpi=600, bbox_inches="tight")
    plt.savefig("results/tofu_front_page_plot.pdf", dpi=600, bbox_inches="tight")
    print("\nSaved tofu_front_page_plot.png and tofu_front_page_plot.pdf")


if __name__ == "__main__":
    # Front-page categorical scatter (Agg / Agg. w/o Privacy)
    tofu_front_page_plot()

    # Original front-page strip plot (zero jitter) for side-by-side comparison
    tofu_front_page_scatter()

    # Per-family line plot (Retrain reference + per-method points w/ 99% CIs)
    tofu_front_page_lineplot()

    # Generate the tables with bootstrap confidence intervals
    generate_tofu_tables()

    # Render table as PNG with category-highlighted rows
    render_tofu_table_png()

    # Generate REBEL Leak@ attack table
    #generate_rebel_table()

    # Generate comprehensive appendix tables
    #appendix_table()

    # Print distillation sweep results
    #print_distill_sweep_results()

    # Plot distillation sweep heatmap
    #plot_distill_sweep_heatmap()

    # Generate the model scaling plot
    #tofu_model_scaling_plot()

    # Generate the privacy plot
    #tofu_privacy_plot()

    # Generate the aggregate score plots
    #tofu_agg_plot(with_privacy=True)
    #tofu_agg_plot(with_privacy=False)