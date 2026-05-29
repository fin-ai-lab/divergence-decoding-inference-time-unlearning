import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
import os
from matplotlib.ticker import PercentFormatter
from scipy.stats import t

# Default to point estimates (no bootstrap SEs / error bars): the per-index
# MUSE_EVAL.json files are gitignored (too large to commit), so plots run from
# MUSE_SUMMARY.json alone. Set SKIP_BOOTSTRAP = False (with EVAL.json present) to
# compute the 99% bootstrap CIs.
SKIP_BOOTSTRAP = True

# SNS tab10 color palette - using same colors for same model sizes
palette = {
    "Target": "#1f77b4",
    "Retrain": "#ff7f0e",
    "GradDiff": "#bcbd22",
    "GradAscent": "#ffbb78",
    "RMU": "#9edae5",
    "NPO": "#9467bd",
    "SimNPO": "#2ca02c",
    "Linear DD": "#17becf",
    "Rank DD": "#d62728",
    "Distill DD": "#8c564b",
    "$\\delta$-Unlearning": "#7f7f7f",
    "ULD": "#aec7e8",
    "UNDIAL": "#ff9896",
    "WHP": "#c49c94",
    "GUARD": "#e377c2",
    "ECO": "#dbdb8d",
    "LUNAR": "#f7b6d2",
    "OLMo Linear CT-DD": "#98df8a",
    "OLMo Rank CT-DD": "#98df8a",
    "Gemma Linear CT-DD": "#ff9896",
    "Gemma Rank CT-DD": "#ff9896",
    "Qwen Linear CT-DD": "#c5b0d5",
    "Qwen Rank CT-DD": "#c5b0d5",
    # Category labels for overview plot
    "Gradient": "#999999",
    "Inference-Time": "#bbbbbb",
    "DD Cross-Tokenizer": "#777777",
    "Baselines": "#999999",
}

# Different markers for linear vs rank methods
markers = {
    "Target": "s",
    "Retrain": "s",
    "GradDiff": "o",
    "GradAscent": "o",
    "RMU": "o",
    "NPO": "o",
    "SimNPO": "o",
    "Linear DD": "X",
    "Rank DD": "X",
    "Distill DD": "X",
    "$\\delta$-Unlearning": "o",
    "ULD": "o",
    "UNDIAL": "o",
    "WHP": "o",
    "GUARD": "o",
    "ECO": "o",
    "LUNAR": "o",
    "OLMo Linear CT-DD": "^",
    "OLMo Rank CT-DD": "v",
    "Gemma Linear CT-DD": "^",
    "Gemma Rank CT-DD": "v",
    "Qwen Linear CT-DD": "^",
    "Qwen Rank CT-DD": "v",
    # Category markers for overview plot
    "Gradient": "v",
    "Inference-Time": "^",
    "DD Cross-Tokenizer": "^",
    "Baselines": "o",
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

alpha_values_model = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 3.0]
topk_values_model = [1, 5, 20, 50, 200, 1000]
alpha_values_trigram = [5, 10, 15, 20, 25, 30]
topk_values_trigram = [1, 2, 3, 5, 10]

learning_rates = [2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4, 1.25e-4, 1.5e-4]
epochs = [5]
temperatures = [0.25, 0.5, 1.0, 1.5, 2.0]

# Cross-tokenizer models: label -> (short_name, [lr_values])
cross_tok_models = {
    "OLMo": ("OLMo-2-0425-1B", ["3e-5", "5e-5", "8e-5"]),
    "Gemma": ("gemma-3-1b-pt", ["3e-5", "5e-5", "8e-5"]),
    "Qwen": ("Qwen3-1.7B-Base", ["3e-5", "5e-5", "8e-5"]),
}
cross_tok_alphas = [round(x * 0.1, 1) for x in range(0, 31)]
cross_tok_topks = [1, 5, 20, 100, 200, 500, 1000]


# ── Result-path helpers (see SAVES_LAYOUT.md) ────────────────────────────────
# Baselines live under muse/baselines/{target,retrain}; neural DD sweeps under
# muse/dd_linear and muse/dd_rank keyed by verifier size; the trigram DD sweep
# under muse/dd_trigram (no size token).

def baseline_folder(name):
    """saves/eval folder for a reference baseline ('Target' / 'Retrain')."""
    return f"muse/baselines/{name.lower()}"


def dd_alpha_folder(model_size, alpha):
    """saves/eval folder for a Linear-DD (alpha) sweep point."""
    if model_size == "Trigram":
        return f"muse/dd_trigram/alpha-{alpha}"
    return f"muse/dd_linear/{model_size}-alpha-{alpha}"


def dd_topk_folder(model_size, topk):
    """saves/eval folder for a Rank-DD (top-k) sweep point."""
    if model_size == "Trigram":
        return f"muse/dd_trigram/topk-{topk}"
    return f"muse/dd_rank/{model_size}-topk-{topk}"


def find_optimal_cross_tok_configs():
    """Find optimal cross-tokenizer DD configurations for MUSE."""
    retrain_info = json.load(open("saves/eval/muse/baselines/retrain/MUSE_SUMMARY.json"))
    retrain_scores = {
        'forget_verbmem_ROUGE': retrain_info['forget_verbmem_ROUGE'] * 100,
        'forget_knowmem_ROUGE': retrain_info['forget_knowmem_ROUGE'] * 100,
        'retain_knowmem_ROUGE': retrain_info['retain_knowmem_ROUGE'] * 100,
    }

    def _distance(point, metric='verbmem'):
        forget_key = f'forget_{metric}_ROUGE'
        fd = point[forget_key] - retrain_scores[forget_key]
        rd = point['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE']
        return (fd**2 + rd**2)**0.5

    optimal = {}
    for label, (short, lrs) in cross_tok_models.items():
        optimal[label] = {}
        for lr in lrs:
            for sweep_name, values in [
                ("alpha", cross_tok_alphas),
                ("topk", cross_tok_topks),
            ]:
                points = []
                for val in values:
                    # Only use lr-tagged paths
                    folder = f"muse/cross_tok/{short}-lr{lr}-{sweep_name}-{val}"
                    path = f"saves/eval/{folder}/MUSE_SUMMARY.json"
                    try:
                        info = json.load(open(path))
                        points.append({
                            'value': val,
                            'lr': lr,
                            'folder': folder,
                            'forget_verbmem_ROUGE': info['forget_verbmem_ROUGE'] * 100,
                            'forget_knowmem_ROUGE': info['forget_knowmem_ROUGE'] * 100,
                            'retain_knowmem_ROUGE': info['retain_knowmem_ROUGE'] * 100,
                        })
                    except (FileNotFoundError, KeyError):
                        pass

                for metric in ['verbmem', 'knowmem']:
                    key = f'{sweep_name}_{metric}'
                    best_dist = float('inf')
                    best_point = None
                    for p in points:
                        d = _distance(p, metric)
                        if d < best_dist:
                            best_dist = d
                            best_point = p
                    if best_point:
                        # Keep the best across LRs
                        if key not in optimal[label] or best_dist < optimal[label][key]['distance']:
                            optimal[label][key] = {
                                'value': best_point['value'],
                                'lr': best_point['lr'],
                                'folder': best_point['folder'],
                                'distance': best_dist,
                                'scores': {k: best_point[k] for k in ['forget_verbmem_ROUGE', 'forget_knowmem_ROUGE', 'retain_knowmem_ROUGE']},
                            }
                            print(f"  Cross-tok {label} best {sweep_name} (lr={lr}) for {metric}: {best_point['value']} (dist={best_dist:.4f})")

    return optimal


def find_optimal_configurations():
    """
    Find the optimal alpha and topk configurations for each model size.
    Now finds separate optimal configurations for verbmem and knowmem metrics.
    Returns a dictionary with the optimal settings for hardcoding.
    """
    
    # Load baselines for distance calculation
    retrain_target = ["Target", "Retrain"]
    baseline_scores = {}
    
    for name in retrain_target:
        info = json.load(open(f"saves/eval/{baseline_folder(name)}/MUSE_SUMMARY.json"))
        baseline_scores[name] = {
            'forget_verbmem_ROUGE': info['forget_verbmem_ROUGE'] * 100,
            'forget_knowmem_ROUGE': info['forget_knowmem_ROUGE'] * 100,
            'retain_knowmem_ROUGE': info['retain_knowmem_ROUGE'] * 100
        }
    
    retrain_scores = baseline_scores['Retrain']
    
    def calculate_distance_verbmem(point):
        """Calculate euclidean distance from retrain baseline using verbmem and retain metrics"""
        forget_verbmem_diff = point['forget_verbmem_ROUGE'] - retrain_scores['forget_verbmem_ROUGE']
        retain_diff = point['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE']
        
        return (forget_verbmem_diff**2 + retain_diff**2)**0.5
    
    def calculate_distance_knowmem(point):
        """Calculate euclidean distance from retrain baseline using knowmem and retain metrics"""
        forget_knowmem_diff = point['forget_knowmem_ROUGE'] - retrain_scores['forget_knowmem_ROUGE']
        retain_diff = point['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE']
        
        return (forget_knowmem_diff**2 + retain_diff**2)**0.5
    
    model_sizes = ["1.3b", "2.7b", "Trigram", ] #"7b"
    optimal_configs = {}
    
    print("FINDING OPTIMAL CONFIGURATIONS (SEPARATE FOR VERBMEM AND KNOWMEM)")
    print("="*80)
    
    for model_size in model_sizes:
        print(f"\nProcessing {model_size}...")
        optimal_configs[model_size] = {}
        
        # Process alpha-based models
        alpha_data = []
        if model_size == "Trigram":
            alpha_values = alpha_values_trigram
            topk_values = topk_values_trigram
        else:
            alpha_values = alpha_values_model
            topk_values = topk_values_model

        for alpha in alpha_values:
            folder_name = dd_alpha_folder(model_size, alpha)
            info = json.load(open(f"saves/eval/{folder_name}/MUSE_SUMMARY.json"))
            alpha_data.append({
                'alpha': alpha,
                'folder': folder_name,
                'forget_verbmem_ROUGE': info['forget_verbmem_ROUGE'] * 100,
                'forget_knowmem_ROUGE': info['forget_knowmem_ROUGE'] * 100,
                'retain_knowmem_ROUGE': info['retain_knowmem_ROUGE'] * 100
            })

        # Process topk-based models
        topk_data = []
        for topk in topk_values:
            folder_name = dd_topk_folder(model_size, topk)
            info = json.load(open(f"saves/eval/{folder_name}/MUSE_SUMMARY.json"))
            topk_data.append({
                'topk': topk,
                'folder': folder_name,
                'forget_verbmem_ROUGE': info['forget_verbmem_ROUGE'] * 100,
                'forget_knowmem_ROUGE': info['forget_knowmem_ROUGE'] * 100,
                'retain_knowmem_ROUGE': info['retain_knowmem_ROUGE'] * 100
            })
        
        # Find optimal alpha for verbmem
        if alpha_data:
            best_alpha_distance_verbmem = float('inf')
            best_alpha_config_verbmem = None
            
            for point in alpha_data:
                distance = calculate_distance_verbmem(point)
                if distance < best_alpha_distance_verbmem:
                    best_alpha_distance_verbmem = distance
                    best_alpha_config_verbmem = point
            
            if best_alpha_config_verbmem:
                optimal_configs[model_size]['alpha_verbmem'] = {
                    'value': best_alpha_config_verbmem['alpha'],
                    'folder': best_alpha_config_verbmem['folder'],
                    'distance': best_alpha_distance_verbmem,
                    'scores': {
                        'forget_verbmem_ROUGE': best_alpha_config_verbmem['forget_verbmem_ROUGE'],
                        'forget_knowmem_ROUGE': best_alpha_config_verbmem['forget_knowmem_ROUGE'],
                        'retain_knowmem_ROUGE': best_alpha_config_verbmem['retain_knowmem_ROUGE']
                    }
                }
                print(f"  Best alpha for verbmem: {best_alpha_config_verbmem['alpha']} (distance: {best_alpha_distance_verbmem:.4f})")
        
        # Find optimal alpha for knowmem
        if alpha_data:
            best_alpha_distance_knowmem = float('inf')
            best_alpha_config_knowmem = None
            
            for point in alpha_data:
                distance = calculate_distance_knowmem(point)
                if distance < best_alpha_distance_knowmem:
                    best_alpha_distance_knowmem = distance
                    best_alpha_config_knowmem = point
            
            if best_alpha_config_knowmem:
                optimal_configs[model_size]['alpha_knowmem'] = {
                    'value': best_alpha_config_knowmem['alpha'],
                    'folder': best_alpha_config_knowmem['folder'],
                    'distance': best_alpha_distance_knowmem,
                    'scores': {
                        'forget_verbmem_ROUGE': best_alpha_config_knowmem['forget_verbmem_ROUGE'],
                        'forget_knowmem_ROUGE': best_alpha_config_knowmem['forget_knowmem_ROUGE'],
                        'retain_knowmem_ROUGE': best_alpha_config_knowmem['retain_knowmem_ROUGE']
                    }
                }
                print(f"  Best alpha for knowmem: {best_alpha_config_knowmem['alpha']} (distance: {best_alpha_distance_knowmem:.4f})")
        
        # Find optimal topk for verbmem
        if topk_data:
            best_topk_distance_verbmem = float('inf')
            best_topk_config_verbmem = None
            
            for point in topk_data:
                distance = calculate_distance_verbmem(point)
                if distance < best_topk_distance_verbmem:
                    best_topk_distance_verbmem = distance
                    best_topk_config_verbmem = point
            
            if best_topk_config_verbmem:
                optimal_configs[model_size]['topk_verbmem'] = {
                    'value': best_topk_config_verbmem['topk'],
                    'folder': best_topk_config_verbmem['folder'],
                    'distance': best_topk_distance_verbmem,
                    'scores': {
                        'forget_verbmem_ROUGE': best_topk_config_verbmem['forget_verbmem_ROUGE'],
                        'forget_knowmem_ROUGE': best_topk_config_verbmem['forget_knowmem_ROUGE'],
                        'retain_knowmem_ROUGE': best_topk_config_verbmem['retain_knowmem_ROUGE']
                    }
                }
                print(f"  Best topk for verbmem: {best_topk_config_verbmem['topk']} (distance: {best_topk_distance_verbmem:.4f})")
        
        # Find optimal topk for knowmem
        if topk_data:
            best_topk_distance_knowmem = float('inf')
            best_topk_config_knowmem = None
            
            for point in topk_data:
                distance = calculate_distance_knowmem(point)
                if distance < best_topk_distance_knowmem:
                    best_topk_distance_knowmem = distance
                    best_topk_config_knowmem = point
            
            if best_topk_config_knowmem:
                optimal_configs[model_size]['topk_knowmem'] = {
                    'value': best_topk_config_knowmem['topk'],
                    'folder': best_topk_config_knowmem['folder'],
                    'distance': best_topk_distance_knowmem,
                    'scores': {
                        'forget_verbmem_ROUGE': best_topk_config_knowmem['forget_verbmem_ROUGE'],
                        'forget_knowmem_ROUGE': best_topk_config_knowmem['forget_knowmem_ROUGE'],
                        'retain_knowmem_ROUGE': best_topk_config_knowmem['retain_knowmem_ROUGE']
                    }
                }
                print(f"  Best topk for knowmem: {best_topk_config_knowmem['topk']} (distance: {best_topk_distance_knowmem:.4f})")
    
    # Print detailed summary for hardcoding
    print(f"\n{'='*80}")
    print("OPTIMAL CONFIGURATIONS FOR HARDCODING")
    print(f"{'='*80}")
    
    for model_size, configs in optimal_configs.items():
        print(f"\n{model_size.upper()}:")
        if 'alpha_verbmem' in configs:
            alpha_config = configs['alpha_verbmem']
            print(f"  Linear DD (alpha) for verbmem: {alpha_config['value']} -> folder: {alpha_config['folder']}")
            scores = alpha_config['scores']
            print(f"    Scores: verbmem={scores['forget_verbmem_ROUGE']:.2f}%, knowmem={scores['forget_knowmem_ROUGE']:.2f}%, retain={scores['retain_knowmem_ROUGE']:.2f}%")
        
        if 'alpha_knowmem' in configs:
            alpha_config = configs['alpha_knowmem']
            print(f"  Linear DD (alpha) for knowmem: {alpha_config['value']} -> folder: {alpha_config['folder']}")
            scores = alpha_config['scores']
            print(f"    Scores: verbmem={scores['forget_verbmem_ROUGE']:.2f}%, knowmem={scores['forget_knowmem_ROUGE']:.2f}%, retain={scores['retain_knowmem_ROUGE']:.2f}%")
        
        if 'topk_verbmem' in configs:
            topk_config = configs['topk_verbmem']
            print(f"  Rank DD (topk) for verbmem: {topk_config['value']} -> folder: {topk_config['folder']}")
            scores = topk_config['scores']
            print(f"    Scores: verbmem={scores['forget_verbmem_ROUGE']:.2f}%, knowmem={scores['forget_knowmem_ROUGE']:.2f}%, retain={scores['retain_knowmem_ROUGE']:.2f}%")
        
        if 'topk_knowmem' in configs:
            topk_config = configs['topk_knowmem']
            print(f"  Rank DD (topk) for knowmem: {topk_config['value']} -> folder: {topk_config['folder']}")
            scores = topk_config['scores']
            print(f"    Scores: verbmem={scores['forget_verbmem_ROUGE']:.2f}%, knowmem={scores['forget_knowmem_ROUGE']:.2f}%, retain={scores['retain_knowmem_ROUGE']:.2f}%")
    
    return optimal_configs


def calculate_ci(directory, level=0.99):
    """
    Calculate confidence intervals using parametric bootstrap method.

    Args:
        directory: Path to MUSE_EVAL.json file
        level: Confidence level (default 0.99 for 99% CI)

    Returns:
        Dictionary with confidence interval half-widths for each metric
    """
    # Skip CI computation by default (EVAL files gitignored) -> no error bars.
    if SKIP_BOOTSTRAP or not os.path.exists(directory):
        return {}
    # Load the evaluation data
    info = json.load(open(directory))
    key = 'rougeL_f1'

    # Convert confidence level to alpha for two-tailed test
    alpha = 1 - level

    # Number of bootstrap samples
    n_samples = 1_000

    # Critical value for t-distribution
    cv = t.ppf(1 - alpha / 2, n_samples - 1)

    # Extract raw scores for each metric
    forget_knowmem_values = [info['forget_knowmem_ROUGE']['value_by_index'][str(i)][key]
                             for i in range(len(info['forget_knowmem_ROUGE']['value_by_index']))]
    forget_verbmem_values = [info['forget_verbmem_ROUGE']['value_by_index'][str(i)][key]
                            for i in range(len(info['forget_verbmem_ROUGE']['value_by_index']))]
    retain_values = [info['retain_knowmem_ROUGE']['value_by_index'][str(i)][key]
                    for i in range(len(info['retain_knowmem_ROUGE']['value_by_index']))]

    # Function to calculate parametric bootstrap CI half-width
    def parametric_bootstrap_ci_halfwidth(values, n_samples, cv):
        """Calculate CI half-width using parametric bootstrap"""
        values_array = np.array(values)
        sample_means = []

        # Use a fixed seed for reproducibility
        rng = np.random.RandomState(42)

        for i in range(n_samples):
            # Bootstrap sample with replacement
            bootstrap_sample = rng.choice(values_array, size=len(values_array), replace=True)
            sample_means.append(np.mean(bootstrap_sample))

        # Calculate standard deviation of bootstrap means
        std_of_means = np.std(sample_means, ddof=1)

        # Calculate CI half-width
        ci_half_width = (std_of_means / np.sqrt(n_samples)) * cv

        # Return CI half-width (already in percentage form, will be multiplied by 100 later)
        return ci_half_width

    # Calculate CI half-widths for each metric
    forget_knowmem_hw = parametric_bootstrap_ci_halfwidth(forget_knowmem_values, n_samples, cv)
    forget_verbmem_hw = parametric_bootstrap_ci_halfwidth(forget_verbmem_values, n_samples, cv)
    retain_hw = parametric_bootstrap_ci_halfwidth(retain_values, n_samples, cv)

    return {
        'forget_knowmem_ROUGE_hw': forget_knowmem_hw,
        'forget_verbmem_ROUGE_hw': forget_verbmem_hw,
        'retain_knowmem_ROUGE_hw': retain_hw,
    }


def calculate_ci_distance(baseline_eval_path, method_eval_path, metric_type='verbmem', level=0.99):
    """
    Calculate confidence intervals for euclidean distance metric using bootstrap.

    Bootstraps the distance by resampling from both baseline and method data,
    then computing the euclidean distance for each bootstrap sample.

    Args:
        baseline_eval_path: Path to baseline MUSE_EVAL.json file (e.g., retrain)
        method_eval_path: Path to method MUSE_EVAL.json file
        metric_type: Either 'verbmem' or 'knowmem' to determine which forget metric to use
        level: Confidence level (default 0.99 for 99% CI)

    Returns:
        Dictionary with 'ci_half_width' for the distance
    """
    if SKIP_BOOTSTRAP or not os.path.exists(method_eval_path):
        return {'ci_half_width': 0.0}
    # Load the evaluation data
    baseline_info = json.load(open(baseline_eval_path))
    method_info = json.load(open(method_eval_path))
    key = 'rougeL_f1'

    # Determine which forget metric to use
    if metric_type == 'verbmem':
        forget_metric = 'forget_verbmem_ROUGE'
    elif metric_type == 'knowmem':
        forget_metric = 'forget_knowmem_ROUGE'
    else:
        raise ValueError(f"metric_type must be 'verbmem' or 'knowmem', got {metric_type}")

    # Extract raw scores for baseline
    baseline_forget = np.array([baseline_info[forget_metric]['value_by_index'][str(i)][key] * 100
                                for i in range(len(baseline_info[forget_metric]['value_by_index']))])
    baseline_retain = np.array([baseline_info['retain_knowmem_ROUGE']['value_by_index'][str(i)][key] * 100
                                for i in range(len(baseline_info['retain_knowmem_ROUGE']['value_by_index']))])

    # Extract raw scores for method
    method_forget = np.array([method_info[forget_metric]['value_by_index'][str(i)][key] * 100
                             for i in range(len(method_info[forget_metric]['value_by_index']))])
    method_retain = np.array([method_info['retain_knowmem_ROUGE']['value_by_index'][str(i)][key] * 100
                             for i in range(len(method_info['retain_knowmem_ROUGE']['value_by_index']))])

    # Bootstrap parameters
    n_samples = 1_000
    alpha = 1 - level
    cv = t.ppf(1 - alpha / 2, n_samples - 1)
    rng = np.random.RandomState(42)

    # Bootstrap distances - always resample both datasets
    bootstrap_distances = []

    for _ in range(n_samples):
        # Resample with replacement from baseline
        baseline_forget_sample = rng.choice(baseline_forget, size=len(baseline_forget), replace=True)
        baseline_retain_sample = rng.choice(baseline_retain, size=len(baseline_retain), replace=True)

        # Resample with replacement from method
        method_forget_sample = rng.choice(method_forget, size=len(method_forget), replace=True)
        method_retain_sample = rng.choice(method_retain, size=len(method_retain), replace=True)

        # Calculate means
        baseline_forget_mean = np.mean(baseline_forget_sample)
        baseline_retain_mean = np.mean(baseline_retain_sample)
        method_forget_mean = np.mean(method_forget_sample)
        method_retain_mean = np.mean(method_retain_sample)

        # Calculate euclidean distance
        forget_diff = method_forget_mean - baseline_forget_mean
        retain_diff = method_retain_mean - baseline_retain_mean
        distance = np.sqrt(forget_diff**2 + retain_diff**2)

        bootstrap_distances.append(distance)

    # Calculate confidence intervals using same method as calculate_ci
    bootstrap_distances = np.array(bootstrap_distances)
    std_of_distances = np.std(bootstrap_distances, ddof=1)

    # Calculate CI half-width using t-distribution approach
    ci_half_width = (std_of_distances / np.sqrt(n_samples)) * cv

    return {
        'ci_half_width': ci_half_width
    }


def calculate_ci_privleak(method_eval_path, retain_eval_path='saves/eval/muse/baselines/retrain/MUSE_EVAL.json', level=0.99):
    """
    Calculate confidence intervals for privleak metric using bootstrap.

    Privleak is calculated as: ((1 - method_auc) - (1 - retain_auc)) / (1 - retain_auc) * 100
    where AUC is computed from MIA min-k scores on forget vs holdout data.

    Args:
        method_eval_path: Path to method MUSE_EVAL.json file
        retain_eval_path: Path to retain model MUSE_EVAL.json file (default: retrain)
        level: Confidence level (default 0.99 for 99% CI)

    Returns:
        Dictionary with 'ci_half_width' for privleak
    """
    from sklearn.metrics import roc_auc_score

    if SKIP_BOOTSTRAP or not os.path.exists(method_eval_path):
        return {'ci_half_width': 0.0}
    # Load the evaluation data
    method_info = json.load(open(method_eval_path))
    retain_info = json.load(open(retain_eval_path))

    # Extract raw MIA scores for method
    method_forget_scores = np.array([method_info['mia_min_k']['forget']['value_by_index'][str(i)]['score']
                                     for i in range(len(method_info['mia_min_k']['forget']['value_by_index']))])
    method_holdout_scores = np.array([method_info['mia_min_k']['holdout']['value_by_index'][str(i)]['score']
                                      for i in range(len(method_info['mia_min_k']['holdout']['value_by_index']))])

    # Extract raw MIA scores for retain model
    retain_forget_scores = np.array([retain_info['mia_min_k']['forget']['value_by_index'][str(i)]['score']
                                     for i in range(len(retain_info['mia_min_k']['forget']['value_by_index']))])
    retain_holdout_scores = np.array([retain_info['mia_min_k']['holdout']['value_by_index'][str(i)]['score']
                                      for i in range(len(retain_info['mia_min_k']['holdout']['value_by_index']))])

    # Bootstrap parameters
    n_samples = 1_000
    alpha = 1 - level
    cv = t.ppf(1 - alpha / 2, n_samples - 1)
    rng = np.random.RandomState(42)

    # Bootstrap privleak values
    bootstrap_privleaks = []

    for _ in range(n_samples):
        # Resample method scores
        method_forget_sample = rng.choice(method_forget_scores, size=len(method_forget_scores), replace=True)
        method_holdout_sample = rng.choice(method_holdout_scores, size=len(method_holdout_scores), replace=True)

        # Resample retain scores
        retain_forget_sample = rng.choice(retain_forget_scores, size=len(retain_forget_scores), replace=True)
        retain_holdout_sample = rng.choice(retain_holdout_scores, size=len(retain_holdout_scores), replace=True)

        # Calculate AUC for method (label=0 for forget, label=1 for holdout)
        method_scores = np.concatenate([method_forget_sample, method_holdout_sample])
        method_labels = np.array([0] * len(method_forget_sample) + [1] * len(method_holdout_sample))
        method_auc = roc_auc_score(method_labels, method_scores)

        # Calculate AUC for retain model
        retain_scores = np.concatenate([retain_forget_sample, retain_holdout_sample])
        retain_labels = np.array([0] * len(retain_forget_sample) + [1] * len(retain_holdout_sample))
        retain_auc = roc_auc_score(retain_labels, retain_scores)

        # Calculate privleak using the formula from se_evals/metrics/privacy.py
        method_score = 1 - method_auc
        retain_score = 1 - retain_auc
        privleak = (method_score - retain_score) / (retain_score + 1e-10) * 100

        bootstrap_privleaks.append(privleak)

    # Calculate confidence intervals
    bootstrap_privleaks = np.array(bootstrap_privleaks)
    std_of_privleaks = np.std(bootstrap_privleaks, ddof=1)
    ci_half_width = (std_of_privleaks / np.sqrt(n_samples)) * cv

    return {
        'ci_half_width': ci_half_width
    }


def main_scatter_plot():
    """
    Updated main scatter plot using optimal configurations.
    Left plot shows optimal for verbmem, right plot shows optimal for knowmem.
    Cleaned up privleak code - no longer prints privleak metrics.
    Now includes error bars for both plots.
    """
    # Find optimal configurations
    print("Finding optimal configurations...")
    optimal_configs = find_optimal_configurations()
    
    data = []
    
    # Load Target and Retrain baselines (7B)
    retrain_target = ["Target", "Retrain"]
    for name in retrain_target:
        info = json.load(open(f"saves/eval/{baseline_folder(name)}/MUSE_SUMMARY.json"))
        info["name"] = name
        info["ci"] = calculate_ci(f"saves/eval/{baseline_folder(name)}/MUSE_EVAL.json")
        data.append(info)

    # Load gradient-based methods
    gradient_methods = {"GradDiff": "graddiff", "NPO": "npo", "SimNPO": "simnpo"}
    for name, dirname in gradient_methods.items():
        info = json.load(open(f"saves/eval/muse/gradient/{dirname}/MUSE_SUMMARY.json"))
        info["name"] = name
        info["ci"] = calculate_ci(f"saves/eval/muse/gradient/{dirname}/MUSE_EVAL.json")
        data.append(info)

    # Load DD methods using optimal configurations for verbmem (left plot)
    model_sizes = ["1.3b"]

    # Prepare separate data for left (verbmem) and right (knowmem) plots
    data_verbmem = data.copy()
    data_knowmem = data.copy()
    
    for model_size in model_sizes:
        if model_size in optimal_configs:
            # For verbmem plot - use verbmem optimal configurations
            if 'alpha_verbmem' in optimal_configs[model_size]:
                folder = optimal_configs[model_size]['alpha_verbmem']['folder']
                name = "Linear DD"
                try:
                    info = json.load(open(f"saves/eval/{folder}/MUSE_SUMMARY.json"))
                    info["name"] = name
                    info["ci"] = calculate_ci(f"saves/eval/{folder}/MUSE_EVAL.json")
                    data_verbmem.append(info)
                    print(f"Loaded {name} (verbmem optimal) from {folder}")
                except FileNotFoundError:
                    print(f"Warning: Could not find {folder}")
            
            if 'topk_verbmem' in optimal_configs[model_size]:
                folder = optimal_configs[model_size]['topk_verbmem']['folder']
                name = "Rank DD"
                try:
                    info = json.load(open(f"saves/eval/{folder}/MUSE_SUMMARY.json"))
                    info["name"] = name
                    info["ci"] = calculate_ci(f"saves/eval/{folder}/MUSE_EVAL.json")
                    data_verbmem.append(info)
                    print(f"Loaded {name} (verbmem optimal) from {folder}")
                except FileNotFoundError:
                    print(f"Warning: Could not find {folder}")
            
            # For knowmem plot - use knowmem optimal configurations
            if 'alpha_knowmem' in optimal_configs[model_size]:
                folder = optimal_configs[model_size]['alpha_knowmem']['folder']
                name = "Linear DD"
                try:
                    info = json.load(open(f"saves/eval/{folder}/MUSE_SUMMARY.json"))
                    info["name"] = name
                    info["ci"] = calculate_ci(f"saves/eval/{folder}/MUSE_EVAL.json")
                    data_knowmem.append(info)
                    print(f"Loaded {name} (knowmem optimal) from {folder}")
                except FileNotFoundError:
                    print(f"Warning: Could not find {folder}")
            
            if 'topk_knowmem' in optimal_configs[model_size]:
                folder = optimal_configs[model_size]['topk_knowmem']['folder']
                name = "Rank DD"
                try:
                    info = json.load(open(f"saves/eval/{folder}/MUSE_SUMMARY.json"))
                    info["name"] = name
                    info["ci"] = calculate_ci(f"saves/eval/{folder}/MUSE_EVAL.json")
                    data_knowmem.append(info)
                    print(f"Loaded {name} (knowmem optimal) from {folder}")
                except FileNotFoundError:
                    print(f"Warning: Could not find {folder}")

    # Load Distill DD using optimal config for each metric (may be different configs)
    distill_configs = find_optimal_distill_configs()

    # Load best verbmem config for verbmem plot
    if distill_configs['best_config_verbmem'] is not None:
        cfg = distill_configs['best_config_verbmem']
        task_name = f"lr-{cfg['lr']}-epoch-{cfg['epoch']}-temp-{cfg['temperature']}"
        folder = f"saves/eval/muse/distill/{task_name}"
        try:
            info = json.load(open(f"{folder}/MUSE_SUMMARY.json"))
            info["name"] = "Distill DD"
            info["ci"] = calculate_ci(f"{folder}/MUSE_EVAL.json")
            data_verbmem.append(info)
            print(f"Loaded Distill DD (verbmem optimal) from {folder}")
        except FileNotFoundError:
            print(f"Warning: Could not find Distill DD at {folder}")

    # Load best knowmem config for knowmem plot (may be different from verbmem)
    if distill_configs['best_config_knowmem'] is not None:
        cfg = distill_configs['best_config_knowmem']
        task_name = f"lr-{cfg['lr']}-epoch-{cfg['epoch']}-temp-{cfg['temperature']}"
        folder = f"saves/eval/muse/distill/{task_name}"
        try:
            info = json.load(open(f"{folder}/MUSE_SUMMARY.json"))
            info["name"] = "Distill DD"
            info["ci"] = calculate_ci(f"{folder}/MUSE_EVAL.json")
            data_knowmem.append(info)
            print(f"Loaded Distill DD (knowmem optimal) from {folder}")
        except FileNotFoundError:
            print(f"Warning: Could not find Distill DD at {folder}")

    # Load Offset Unlearning results
    offset_configs = find_optimal_offset_configs()
    for metric, data_list in [('verbmem', data_verbmem), ('knowmem', data_knowmem)]:
        lr = offset_configs.get(f'best_lr_{metric}')
        if lr:
            folder = f"saves/eval/muse/offset/lr-{lr}"
            try:
                info = json.load(open(f"{folder}/MUSE_SUMMARY.json"))
                info["name"] = "$\\delta$-Unlearning"
                info["ci"] = calculate_ci(f"{folder}/MUSE_EVAL.json")
                data_list.append(info)
                print(f"Loaded Offset ({metric} optimal) lr={lr} from {folder}")
            except FileNotFoundError:
                print(f"Warning: Could not find Offset at {folder}")

    # Load ULD results
    uld_configs = find_optimal_uld_configs()
    for metric, data_list in [('verbmem', data_verbmem), ('knowmem', data_knowmem)]:
        lr = uld_configs.get(f'best_lr_{metric}')
        if lr:
            folder = f"saves/eval/muse/uld/lr-{lr}"
            try:
                info = json.load(open(f"{folder}/MUSE_SUMMARY.json"))
                info["name"] = "ULD"
                info["ci"] = calculate_ci(f"{folder}/MUSE_EVAL.json")
                data_list.append(info)
                print(f"Loaded ULD ({metric} optimal) lr={lr} from {folder}")
            except FileNotFoundError:
                print(f"Warning: Could not find ULD at {folder}")

    # Load UNDIAL results
    undial_configs = find_optimal_undial_configs()
    for metric, data_list in [('verbmem', data_verbmem), ('knowmem', data_knowmem)]:
        best = undial_configs.get(f'best_{metric}')
        if best:
            folder = best['folder']
            try:
                info = json.load(open(f"{folder}/MUSE_SUMMARY.json"))
                info["name"] = "UNDIAL"
                info["ci"] = calculate_ci(f"{folder}/MUSE_EVAL.json")
                data_list.append(info)
                print(f"Loaded UNDIAL ({metric} optimal) lr={best['lr']} cp={best['checkpoint']} from {folder}")
            except FileNotFoundError:
                print(f"Warning: Could not find UNDIAL at {folder}")

    # Load WHP results
    whp_configs = find_optimal_whp_configs()
    for metric, data_list in [('verbmem', data_verbmem), ('knowmem', data_knowmem)]:
        alpha = whp_configs.get(f'best_alpha_{metric}')
        if alpha:
            folder = f"saves/eval/muse/whp/alpha-{alpha}"
            try:
                info = json.load(open(f"{folder}/MUSE_SUMMARY.json"))
                info["name"] = "WHP"
                info["ci"] = calculate_ci(f"{folder}/MUSE_EVAL.json")
                data_list.append(info)
                print(f"Loaded WHP ({metric} optimal) alpha={alpha} from {folder}")
            except FileNotFoundError:
                print(f"Warning: Could not find WHP at {folder}")

    # Load GUARD results
    guard_configs = find_optimal_guard_configs()
    for metric, data_list in [('verbmem', data_verbmem), ('knowmem', data_knowmem)]:
        cfg = guard_configs.get(f'best_config_{metric}')
        if cfg:
            folder = cfg['folder']
            try:
                info = json.load(open(f"{folder}/MUSE_SUMMARY.json"))
                info["name"] = "GUARD"
                info["ci"] = calculate_ci(f"{folder}/MUSE_EVAL.json")
                data_list.append(info)
                print(f"Loaded GUARD ({metric} optimal) lr={cfg['lr']} delta={cfg['delta']} from {folder}")
            except FileNotFoundError:
                print(f"Warning: Could not find GUARD at {folder}")

    # Load ECO results
    eco_configs = find_optimal_eco_configs()
    for metric, data_list in [('verbmem', data_verbmem), ('knowmem', data_knowmem)]:
        cfg = eco_configs.get(f'best_config_{metric}')
        if cfg:
            folder = cfg['folder']
            try:
                info = json.load(open(f"{folder}/MUSE_SUMMARY.json"))
                info["name"] = "ECO"
                info["ci"] = calculate_ci(f"{folder}/MUSE_EVAL.json")
                data_list.append(info)
                print(f"Loaded ECO ({metric} optimal) lr={cfg['lr']} strength={cfg['strength']} from {folder}")
            except FileNotFoundError:
                print(f"Warning: Could not find ECO at {folder}")

    # Load LUNAR results
    lunar_configs = find_optimal_lunar_configs()
    for metric, data_list in [('verbmem', data_verbmem), ('knowmem', data_knowmem)]:
        lr = lunar_configs.get(f'best_lr_{metric}')
        if lr:
            folder = f"saves/eval/muse/lunar/lr-{lr}"
            try:
                info = json.load(open(f"{folder}/MUSE_SUMMARY.json"))
                info["name"] = "LUNAR"
                info["ci"] = calculate_ci(f"{folder}/MUSE_EVAL.json")
                data_list.append(info)
                print(f"Loaded LUNAR ({metric} optimal) lr={lr} from {folder}")
            except FileNotFoundError:
                print(f"Warning: Could not find LUNAR at {folder}")

    # Load cross-tokenizer DD results (best of linear vs rank per model)
    cross_tok_configs = find_optimal_cross_tok_configs()

    # Also keep per-variant data for the detailed cross-tok plot
    cross_tok_detail_verbmem = []
    cross_tok_detail_knowmem = []

    for label, configs in cross_tok_configs.items():
        for metric, data_list, detail_list in [
            ('verbmem', data_verbmem, cross_tok_detail_verbmem),
            ('knowmem', data_knowmem, cross_tok_detail_knowmem),
        ]:
            for sweep in ['alpha', 'topk']:
                key = f'{sweep}_{metric}'
                if key in configs:
                    folder = configs[key]['folder']
                    val = configs[key]['value']
                    lr = configs[key].get('lr', '?')
                    variant = "Linear" if sweep == "alpha" else "Rank"
                    try:
                        info = json.load(open(f"saves/eval/{folder}/MUSE_SUMMARY.json"))
                        info["name"] = f"{label} {variant} CT-DD"
                        info["config"] = f"lr={lr}, {'α' if sweep == 'alpha' else 'k'}={val}"
                        info["ci"] = calculate_ci(f"saves/eval/{folder}/MUSE_EVAL.json")
                        detail_list.append(info)
                        data_list.append(info)
                        print(f"Loaded {label} {variant} CT-DD ({metric} optimal) lr={lr}, {sweep}={val} from {folder}")
                    except FileNotFoundError:
                        print(f"Warning: Could not find {folder}")

    if not data_verbmem:
        print("No data found. Please check file paths.")
        return

    # Create dataframes and scale to percentage
    all_dfs = {}
    all_dfs['verbmem'] = pd.DataFrame(data_verbmem)
    all_dfs['knowmem'] = pd.DataFrame(data_knowmem)

    for df in all_dfs.values():
        df["forget_verbmem_ROUGE"] *= 100
        df["forget_knowmem_ROUGE"] *= 100
        df["retain_knowmem_ROUGE"] *= 100

    # Also build cross-tok detail dataframes (with per-variant Linear/Rank entries)
    cross_tok_detail_dfs = {}
    for key, base_df, detail_data in [
        ('verbmem', all_dfs['verbmem'], cross_tok_detail_verbmem),
        ('knowmem', all_dfs['knowmem'], cross_tok_detail_knowmem),
    ]:
        if detail_data:
            detail_df = pd.DataFrame(detail_data)
            detail_df["forget_verbmem_ROUGE"] *= 100
            detail_df["forget_knowmem_ROUGE"] *= 100
            detail_df["retain_knowmem_ROUGE"] *= 100
            # Combine baselines + DD methods + per-variant CT-DD
            keep = base_df[base_df['name'].isin({"Target", "Retrain", "Linear DD", "Rank DD"})].copy()
            cross_tok_detail_dfs[key] = pd.concat([keep, detail_df], ignore_index=True)
        else:
            cross_tok_detail_dfs[key] = base_df[base_df['name'].isin({"Target", "Retrain", "Linear DD", "Rank DD"})].copy()

    # --- Overview scatter plot (muse_scatter_plot.png) ---
    # Remap non-main methods to gray category labels
    gradient_methods_set = {"GradDiff", "NPO", "SimNPO", "UNDIAL", "LUNAR"}
    inference_methods_set = {"$\\delta$-Unlearning", "WHP", "ULD", "GUARD", "ECO", }
    cross_tok_methods_set = {f"{label} {variant} CT-DD"
                              for label in ["OLMo", "Gemma", "Qwen"]
                              for variant in ["Linear", "Rank"]}
    bold_methods = {"Target", "Retrain", "Linear DD", "Rank DD", "Distill DD"}

    category_map = {}
    for m in gradient_methods_set:
        category_map[m] = "Baselines"
    for m in inference_methods_set:
        category_map[m] = "Baselines"

    overview_dfs = {}
    for key in all_dfs:
        df = all_dfs[key].copy()
        # Exclude DD Cross-Tokenizer results from the overview plot
        df = df[~df["name"].isin(cross_tok_methods_set)].copy()
        df["plot_name"] = df["name"].map(lambda n: category_map.get(n, n))
        overview_dfs[key] = df

    _plot_scatter(
        overview_dfs,
        legend_order=[
            "Target", "Retrain", "Linear DD", "Rank DD", "Distill DD",
            "Baselines",
        ],
        filename="muse_scatter_plot",
        name_col="plot_name",
        ncol=6,
        fade_others=True,
        legend_y=1.07,
    )

    # --- Gradient-based comparison (Distill DD only, no Linear/Rank DD) ---
    gradient_keep = ({"Target", "Retrain", "Distill DD"}) | gradient_methods_set
    gradient_dfs = {}
    for key in all_dfs:
        gradient_dfs[key] = all_dfs[key][all_dfs[key]['name'].isin(gradient_keep)].copy()
        gradient_dfs[key]["plot_name"] = gradient_dfs[key]["name"]

    _plot_scatter(
        gradient_dfs,
        legend_order=[
            "Target", "Retrain", "Distill DD",
            "GradDiff", "NPO", "SimNPO", "UNDIAL", "ULD", "LUNAR",
        ],
        filename="muse_scatter_gradient",
        name_col="plot_name",
        ncol=9,
        legend_y=1.07,
    )

    # --- Inference-time comparison (Linear/Rank DD only, no Distill DD) ---
    inference_keep = ({"Target", "Retrain", "Linear DD", "Rank DD"}) | inference_methods_set
    inference_dfs = {}
    for key in all_dfs:
        inference_dfs[key] = all_dfs[key][all_dfs[key]['name'].isin(inference_keep)].copy()
        inference_dfs[key]["plot_name"] = inference_dfs[key]["name"]

    _plot_scatter(
        inference_dfs,
        legend_order=[
            "Target", "Retrain", "Linear DD", "Rank DD",
            "$\\delta$-Unlearning", "WHP", "GUARD", "ECO",
        ],
        filename="muse_scatter_inference",
        name_col="plot_name",
        ncol=8,
        legend_y=1.07,
    )

    return all_dfs, cross_tok_detail_dfs


def _plot_scatter(dfs_dict, legend_order, filename, name_col="plot_name", ncol=8, fade_others=False, legend_y=1.14):
    """Shared helper to draw a verbmem/knowmem scatter pair."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
    s = 250

    plot_configs = {
        'verbmem': {
            'ax': axes[0],
            'df': dfs_dict['verbmem'],
            'x_col': 'forget_verbmem_ROUGE',
            'y_col': 'retain_knowmem_ROUGE',
            'x_ci_key': 'forget_verbmem_ROUGE_hw',
            'y_ci_key': 'retain_knowmem_ROUGE_hw',
            'xlabel': 'Verbatim Memorization of Forget Set',
            'ylabel': 'Utility on Retain Set',
            'xlim': (10, 60),
        },
        'knowmem': {
            'ax': axes[1],
            'df': dfs_dict['knowmem'],
            'x_col': 'forget_knowmem_ROUGE',
            'y_col': 'retain_knowmem_ROUGE',
            'x_ci_key': 'forget_knowmem_ROUGE_hw',
            'y_ci_key': 'retain_knowmem_ROUGE_hw',
            'xlabel': 'Q&A Knowledge of Forget Set',
            'ylabel': None,
            'xlim': (25, 75),
        }
    }

    for plot_type, config in plot_configs.items():
        ax = config['ax']
        df = config['df']
        x_col = config['x_col']
        y_col = config['y_col']
        x_ci_key = config['x_ci_key']
        y_ci_key = config['y_ci_key']

        # Error bars
        for idx, row in df.iterrows():
            if 'ci' in row and row['ci'] is not None:
                x_hw = row['ci'].get(x_ci_key, None)
                y_hw = row['ci'].get(y_ci_key, None)
                xerr = [[x_hw * 100], [x_hw * 100]] if x_hw is not None else None
                yerr = [[y_hw * 100], [y_hw * 100]] if y_hw is not None else None
                if xerr is not None or yerr is not None:
                    try:
                        ax.errorbar(
                            row[x_col], row[y_col],
                            xerr=xerr, yerr=yerr,
                            fmt='none',
                            ecolor=palette[row[name_col]],
                            elinewidth=1, capsize=0, alpha=1, zorder=1,
                        )
                    except Exception as e:
                        print(f"Error plotting error bar for {row[name_col]} on {plot_type} plot: {e}")

        show_legend = (plot_type == 'knowmem')
        bold_names = {"Target", "Retrain", "Linear DD", "Rank DD", "Distill DD"}

        if fade_others:
            df_bold = df[df[name_col].isin(bold_names)]
            df_other = df[~df[name_col].isin(bold_names)]

            if not df_other.empty:
                sns.scatterplot(
                    data=df_other, x=x_col, y=y_col,
                    hue=name_col, style=name_col,
                    palette=palette, markers=markers,
                    s=s, ax=ax,
                    edgecolor="black", linewidth=0.5,
                    legend="full" if show_legend else False,
                    zorder=2, alpha=0.5,
                )

            if not df_bold.empty:
                sns.scatterplot(
                    data=df_bold, x=x_col, y=y_col,
                    hue=name_col, style=name_col,
                    palette=palette, markers=markers,
                    s=s, ax=ax,
                    edgecolor="black", linewidth=1.2,
                    legend="full" if show_legend else False,
                    zorder=3, alpha=1,
                )
        else:
            # Draw non-bold methods first, then bold DD methods on top
            df_bold = df[df[name_col].isin(bold_names)]
            df_other = df[~df[name_col].isin(bold_names)]

            if not df_other.empty:
                sns.scatterplot(
                    data=df_other, x=x_col, y=y_col,
                    hue=name_col, style=name_col,
                    palette=palette, markers=markers,
                    s=s, ax=ax,
                    edgecolor="black", linewidth=1.2,
                    legend="full" if show_legend else False,
                    zorder=2, alpha=1,
                )

            if not df_bold.empty:
                sns.scatterplot(
                    data=df_bold, x=x_col, y=y_col,
                    hue=name_col, style=name_col,
                    palette=palette, markers=markers,
                    s=s, ax=ax,
                    edgecolor="black", linewidth=1.2,
                    legend="full" if show_legend else False,
                    zorder=3, alpha=1,
                )

        ax.set_xlabel(config['xlabel'])
        ax.set_ylabel(config['ylabel'])
        ax.set_xlim(config['xlim'])

    y_bottom, y_top = 30, 60
    axes[0].set_ylim(y_bottom, y_top)
    yticks = [t for t in range(y_bottom, int(y_top) + 1, 10)]
    axes[0].set_yticks(yticks)

    # Build legend
    handles, labels = axes[1].get_legend_handles_labels()
    if labels and labels[0].lower() in ("name", "plot_name"):
        handles, labels = handles[1:], labels[1:]

    ordered_handles = []
    ordered_labels = []
    for desired_label in legend_order:
        if desired_label in labels:
            idx = labels.index(desired_label)
            ordered_handles.append(handles[idx])
            ordered_labels.append(labels[idx])

    ordered_labels = [lbl.replace(" CT-DD", "") for lbl in ordered_labels]

    fig.legend(
        ordered_handles, ordered_labels,
        loc="upper center",
        ncol=ncol,
        frameon=False,
        handletextpad=0.4,
        columnspacing=0.6,
        borderaxespad=0.5,
        bbox_to_anchor=(0.5, legend_y),
    )

    leg = axes[1].get_legend()
    if leg:
        leg.remove()

    plt.tight_layout()
    plt.savefig(f"results/{filename}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"results/{filename}.pdf", dpi=600, bbox_inches="tight")


def cross_tok_scatter_plot(cross_tok_detail_dfs):
    """Scatter plot with Target, Retrain, Linear DD, Rank DD, and per-variant CT-DD methods."""
    ct_dd_names = {f"{label} {variant} CT-DD"
                   for label in ["OLMo", "Gemma", "Qwen"]
                   for variant in ["Linear", "Rank"]}
    dfs = {}
    for key in cross_tok_detail_dfs:
        dfs[key] = cross_tok_detail_dfs[key].copy()
        dfs[key]["plot_name"] = dfs[key]["name"]

    _plot_scatter(
        dfs,
        legend_order=[
            "Target", "Retrain", "Linear DD", "Rank DD",
            "OLMo Linear CT-DD", "OLMo Rank CT-DD",
            "Gemma Linear CT-DD", "Gemma Rank CT-DD",
            "Qwen Linear CT-DD", "Qwen Rank CT-DD",
        ],
        filename="muse_cross_tok_scatter",
        name_col="plot_name",
        ncol=5,
        legend_y=1.14,
    )


def plot_muse_curve():
    """
    Create scatter plots showing performance curves for MUSE models with different alpha values
    and topk values. Shows separate optimal points for verbmem and knowmem.
    Uses color gradients based on hyperparameter values.
    """
    
    # Define base colors for different model sizes
    base_colors = {
        "Target": "#1f77b4",
        "Retrain": "#ff7f0e", 
        "1.3B": "Reds",
        "2.7B": "Blues", 
        "Trigram": "Greens",
    }
    
    curve_markers = {
        "baseline": "s", 
        "alpha": "o",
        "topk": "X",
    }

    sizes = {
        "baseline": 180,
        "alpha": 100,
        "topk": 100,
    }
    
    def get_color_from_gradient(cmap_name, value, value_range):
        """Get color from matplotlib colormap based on normalized value"""
        rank = value_range.index(value)
        color_intensity = 0.5 + rank/len(value_range) * 0.4
        cmap = plt.cm.get_cmap(cmap_name)
        return cmap(color_intensity)
    
    data = []
    
    # Load Target and Retrain data
    retrain_target = ["Target", "Retrain"]
    for name in retrain_target:
        try:
            info = json.load(open(f"saves/eval/{baseline_folder(name)}/MUSE_SUMMARY.json"))
            info["name"] = name
            info["alpha"] = None
            info["topk"] = None
            info["model_size"] = "baseline"
            info["method"] = "baseline"
            info["is_optimal_verbmem"] = False
            info["is_optimal_knowmem"] = False
            info["color"] = base_colors[name]
            info["ci"] = calculate_ci(f"saves/eval/{baseline_folder(name)}/MUSE_EVAL.json")
            data.append(info)
        except FileNotFoundError:
            continue
    
    # Load data for all model sizes
    model_sizes = ["1.3b", "2.7b", "Trigram",]  #"7b"
    
    for model_size in model_sizes:

        if model_size == "Trigram":
            alpha_values = alpha_values_trigram
            topk_values = topk_values_trigram
        else:
            alpha_values = alpha_values_model
            topk_values = topk_values_model


        model_name = f"{model_size.replace('b', 'B')}"
        cmap_name = base_colors[model_name]
        
        # Load alpha-based models
        for alpha in alpha_values:
            folder_name = dd_alpha_folder(model_size, alpha)
            try:
                info = json.load(open(f"saves/eval/{folder_name}/MUSE_SUMMARY.json"))
                info["name"] = model_name
                info["alpha"] = alpha
                info["topk"] = None
                info["model_size"] = model_size
                info["method"] = "alpha"

                info["color"] = get_color_from_gradient(cmap_name, alpha, alpha_values)
                
                data.append(info)
            except FileNotFoundError:
                continue
        
        # Load topk-based models
        for topk in topk_values:
            folder_name = dd_topk_folder(model_size, topk)
            try:
                info = json.load(open(f"saves/eval/{folder_name}/MUSE_SUMMARY.json"))
                info["name"] = model_name
                info["alpha"] = None
                info["topk"] = topk
                info["model_size"] = model_size
                info["method"] = "topk"
                
                # Color based on topk value
                info["color"] = get_color_from_gradient(cmap_name, topk, topk_values)
                
                data.append(info)
            except FileNotFoundError:
                continue
    
    if not data:
        print("No data found. Please check that saves/eval/ directory exists and contains MUSE model folders.")
        return
    
    df = pd.DataFrame(data)
    
    # Convert to percentages
    df["forget_verbmem_ROUGE"] *= 100
    df["forget_knowmem_ROUGE"] *= 100
    df["retain_knowmem_ROUGE"] *= 100
    df["size"] = df["method"].map(sizes)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    
    # Plot each point individually to control colors
    for idx, row in df.iterrows():
        for ax_idx, x_col in enumerate(["forget_verbmem_ROUGE", "forget_knowmem_ROUGE"]):
            axes[ax_idx].scatter(
                row[x_col], 
                row["retain_knowmem_ROUGE"],
                c=[row["color"]], 
                marker=curve_markers[row["method"]],
                s=sizes[row["method"]],
                edgecolor="black",
                linewidth=0.5,
                alpha=1
            )
    
    axes[0].set_xlabel("Verbatim Memorization of Forget Set")
    axes[0].set_ylabel("Utility on Retain Set")
    #axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Q&A Knowledge of Forget Set")
    axes[1].set_ylabel("")  # shared y-axis
    #axes[1].grid(True, alpha=0.3)
    
    # Set consistent axis limits
    axes[0].set_xlim(10, 60)
    axes[0].set_ylim(20, 60)
    axes[1].set_xlim(20, 70)
    axes[0].set_yticks([30, 40, 50, 60])
    
    # Build legend manually with lightest colors
    legend_handles = []
    legend_labels = []
    
    # Define the order of legend items
    legend_items = ["Target", "Retrain", "Trigram", "1.3B", "2.7B", "alpha", "topk"] #"DD 7b",
    
    for model_name in legend_items:
        if model_name in ["Target", "Retrain"]:
            color = base_colors[model_name]
            marker = curve_markers["baseline"]
        elif model_name in ["Trigram", "1.3B", "2.7B"]:
            # Use lightest color from the colormap (0.3 intensity)
            cmap_name = base_colors[model_name]
            cmap = plt.cm.get_cmap(cmap_name)
            color = cmap(0.75)  # Median Color
            marker = 's'  # Default marker for model representation
        elif model_name == "alpha":
            color = 'gray'
            marker = curve_markers["alpha"]
            model_name = "Linear DD"
        elif model_name == "topk":
            color = 'gray'
            marker = curve_markers["topk"]
            model_name = "Rank DD"
        else:
            color = 'gray'
            marker = 's'
        
        handle = plt.Line2D([0], [0], marker=marker, color='w', 
                           markerfacecolor=color, markersize=10,
                           markeredgecolor='black', markeredgewidth=0.5)
        legend_handles.append(handle)
        legend_labels.append(model_name)

    # Place legend at the top
    fig.legend(
        legend_handles, legend_labels,
        loc="upper center",
        ncol=10,  
        frameon=False,
        handletextpad=0.4,
        columnspacing=0.6,
        borderaxespad=0.5,
        bbox_to_anchor=(0.5, 1.05),
    )
    
    plt.tight_layout()
    plt.savefig("results/muse_alpha_topk_curves.png", dpi=600, bbox_inches="tight")
    plt.savefig("results/muse_alpha_topk_curves.pdf", dpi=600, bbox_inches="tight")

def model_scaling_plot():
    """
    Create a line plot showing how model scaling affects performance for different methods.
    Now shows two lines: best method for Q&A Questions and best method for Verbatim Memorization.
    Combines both alpha and topk methods, selecting the best performing option for each.
    Uses euclidean distance only.
    """
    # Find optimal configurations
    print(f"Finding optimal configurations for model scaling plot...")
    optimal_configs = find_optimal_configurations()

    # Load Target and Retrain baselines
    retrain_target = ["Target", "Retrain"]
    baseline_scores = {}
    
    for name in retrain_target:
        try:
            info = json.load(open(f"saves/eval/{baseline_folder(name)}/MUSE_SUMMARY.json"))
            baseline_scores[name] = {
                'forget_verbmem_ROUGE': info['forget_verbmem_ROUGE'] * 100,
                'forget_knowmem_ROUGE': info['forget_knowmem_ROUGE'] * 100,
                'retain_knowmem_ROUGE': info['retain_knowmem_ROUGE'] * 100
            }
        except FileNotFoundError:
            print(f"Warning: Could not find baseline data for {name}")
            continue
    
    # If we don't have retrain baseline or target, we can't compute distances
    if 'Retrain' not in baseline_scores or 'Target' not in baseline_scores:
        print("Error: Retrain or Target baseline not found. Cannot compute normalized distances.")
        return
    
    retrain_scores = baseline_scores['Retrain']
    target_scores = baseline_scores['Target']
    
    # Load data for all model sizes using optimal configurations
    model_sizes = ["Trigram", "1.3b", "2.7b"] #"7b"
    model_size_x = {"Trigram": 0, "1.3b": 1.3, "2.7b": 2.7}
    
    results = []
    
    # Function to calculate distance for verbmem optimization
    def calculate_distance_verbmem(point):
        forget_verbmem_diff = point['forget_verbmem_ROUGE'] - retrain_scores['forget_verbmem_ROUGE']
        retain_diff = point['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE']
        return (forget_verbmem_diff**2 + retain_diff**2)**0.5
    
    # Function to calculate distance for knowmem optimization
    def calculate_distance_knowmem(point):
        forget_knowmem_diff = point['forget_knowmem_ROUGE'] - retrain_scores['forget_knowmem_ROUGE']
        retain_diff = point['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE']
        return (forget_knowmem_diff**2 + retain_diff**2)**0.5
    
    # Calculate target distances for normalization
    target_distance_verbmem = calculate_distance_verbmem(target_scores)
    target_distance_knowmem = calculate_distance_knowmem(target_scores)
    
    # Function to normalize distances relative to target (target = 100%)
    def normalize_distance_verbmem(distance):
        return (distance / target_distance_verbmem) * 100
    
    def normalize_distance_knowmem(distance):
        return (distance / target_distance_knowmem) * 100
    
    # Prepare data for seaborn
    plot_data = []

    # Dictionary to store CI info: {mean_distance: (ci_lower, ci_upper)}
    error_bars = {}

    for model_size in model_sizes:
        print(f"Processing {model_size}...")

        x_pos = model_size_x[model_size]

        if model_size in optimal_configs:
            # Find best method for verbmem (memorization)
            verbmem_candidates = []

            if 'alpha_verbmem' in optimal_configs[model_size]:
                alpha_scores = optimal_configs[model_size]['alpha_verbmem']['scores']
                alpha_folder = optimal_configs[model_size]['alpha_verbmem']['folder']
                alpha_distance = calculate_distance_verbmem(alpha_scores)
                normalized_alpha_distance = normalize_distance_verbmem(alpha_distance)

                # CI calculations disabled — MUSE_EVAL.json files unavailable
                # alpha_ci = calculate_ci_distance(
                #     baseline_eval_path="saves/eval/muse/baselines/retrain/MUSE_EVAL.json",
                #     method_eval_path=f"saves/eval/{alpha_folder}/MUSE_EVAL.json",
                #     metric_type='verbmem'
                # )
                # normalized_ci_hw = normalize_distance_verbmem(alpha_ci['ci_half_width'])
                # normalized_ci_lower = normalized_alpha_distance - normalized_ci_hw
                # normalized_ci_upper = normalized_alpha_distance + normalized_ci_hw

                verbmem_candidates.append({
                    'method_name': 'alpha',
                    'param_value': optimal_configs[model_size]['alpha_verbmem']['value'],
                    'distance': normalized_alpha_distance,
                })
                print(f"  Alpha verbmem: α={optimal_configs[model_size]['alpha_verbmem']['value']}, distance={normalized_alpha_distance:.2f}%")

            if 'topk_verbmem' in optimal_configs[model_size]:
                topk_scores = optimal_configs[model_size]['topk_verbmem']['scores']
                topk_folder = optimal_configs[model_size]['topk_verbmem']['folder']
                topk_distance = calculate_distance_verbmem(topk_scores)
                normalized_topk_distance = normalize_distance_verbmem(topk_distance)

                # CI calculations disabled — MUSE_EVAL.json files unavailable
                # topk_ci = calculate_ci_distance(
                #     baseline_eval_path="saves/eval/muse/baselines/retrain/MUSE_EVAL.json",
                #     method_eval_path=f"saves/eval/{topk_folder}/MUSE_EVAL.json",
                #     metric_type='verbmem'
                # )
                # normalized_ci_hw = normalize_distance_verbmem(topk_ci['ci_half_width'])
                # normalized_ci_lower = normalized_topk_distance - normalized_ci_hw
                # normalized_ci_upper = normalized_topk_distance + normalized_ci_hw

                verbmem_candidates.append({
                    'method_name': 'topk',
                    'param_value': optimal_configs[model_size]['topk_verbmem']['value'],
                    'distance': normalized_topk_distance,
                })
                print(f"  TopK verbmem: k={optimal_configs[model_size]['topk_verbmem']['value']}, distance={normalized_topk_distance:.2f}%")

            # Select best verbmem method (lowest distance)
            if verbmem_candidates:
                best_verbmem = min(verbmem_candidates, key=lambda x: x['distance'])

                # CI error bars disabled
                # error_bars[round(best_verbmem['distance'], 5)] = (
                #     best_verbmem['ci_lower'],
                #     best_verbmem['ci_upper']
                # )

                for i in range(3):  # Trick seaborn by repeating data
                    plot_data.append({
                        'model_size': model_size,
                        'x_pos': x_pos,
                        'distance': best_verbmem['distance'],
                        'method': 'Verbatim Memorization',
                        'best_method': best_verbmem['method_name'],
                        'best_param': best_verbmem['param_value']
                    })
                print(f"  Best verbmem: {best_verbmem['method_name']} with {best_verbmem['param_value']}, distance={best_verbmem['distance']:.2f}%")

            # Find best method for knowmem (Q&A Questions)
            knowmem_candidates = []

            if 'alpha_knowmem' in optimal_configs[model_size]:
                alpha_scores = optimal_configs[model_size]['alpha_knowmem']['scores']
                alpha_folder = optimal_configs[model_size]['alpha_knowmem']['folder']
                alpha_distance = calculate_distance_knowmem(alpha_scores)
                normalized_alpha_distance = normalize_distance_knowmem(alpha_distance)

                # CI calculations disabled — MUSE_EVAL.json files unavailable
                # alpha_ci = calculate_ci_distance(
                #     baseline_eval_path="saves/eval/muse/baselines/retrain/MUSE_EVAL.json",
                #     method_eval_path=f"saves/eval/{alpha_folder}/MUSE_EVAL.json",
                #     metric_type='knowmem'
                # )
                # normalized_ci_hw = normalize_distance_knowmem(alpha_ci['ci_half_width'])
                # normalized_ci_lower = normalized_alpha_distance - normalized_ci_hw
                # normalized_ci_upper = normalized_alpha_distance + normalized_ci_hw

                knowmem_candidates.append({
                    'method_name': 'alpha',
                    'param_value': optimal_configs[model_size]['alpha_knowmem']['value'],
                    'distance': normalized_alpha_distance,
                })
                print(f"  Alpha knowmem: α={optimal_configs[model_size]['alpha_knowmem']['value']}, distance={normalized_alpha_distance:.2f}%")

            if 'topk_knowmem' in optimal_configs[model_size]:
                topk_scores = optimal_configs[model_size]['topk_knowmem']['scores']
                topk_folder = optimal_configs[model_size]['topk_knowmem']['folder']
                topk_distance = calculate_distance_knowmem(topk_scores)
                normalized_topk_distance = normalize_distance_knowmem(topk_distance)

                # CI calculations disabled — MUSE_EVAL.json files unavailable
                # topk_ci = calculate_ci_distance(
                #     baseline_eval_path="saves/eval/muse/baselines/retrain/MUSE_EVAL.json",
                #     method_eval_path=f"saves/eval/{topk_folder}/MUSE_EVAL.json",
                #     metric_type='knowmem'
                # )
                # normalized_ci_hw = normalize_distance_knowmem(topk_ci['ci_half_width'])
                # normalized_ci_lower = normalized_topk_distance - normalized_ci_hw
                # normalized_ci_upper = normalized_topk_distance + normalized_ci_hw

                knowmem_candidates.append({
                    'method_name': 'topk',
                    'param_value': optimal_configs[model_size]['topk_knowmem']['value'],
                    'distance': normalized_topk_distance,
                })
                print(f"  TopK knowmem: k={optimal_configs[model_size]['topk_knowmem']['value']}, distance={normalized_topk_distance:.2f}%")

            # Select best knowmem method (lowest distance)
            if knowmem_candidates:
                best_knowmem = min(knowmem_candidates, key=lambda x: x['distance'])

                # CI error bars disabled
                # error_bars[round(best_knowmem['distance'], 5)] = (
                #     best_knowmem['ci_lower'],
                #     best_knowmem['ci_upper']
                # )

                for i in range(3):  # Trick seaborn by repeating data
                    plot_data.append({
                        'model_size': model_size,
                        'x_pos': x_pos,
                        'distance': best_knowmem['distance'],
                        'method': 'Q&A Questions (Few-Shot)',
                        'best_method': best_knowmem['method_name'],
                        'best_param': best_knowmem['param_value']
                    })
                print(f"  Best knowmem: {best_knowmem['method_name']} with {best_knowmem['param_value']}, distance={best_knowmem['distance']:.2f}%")
    
    # Convert to DataFrame
    df = pd.DataFrame(plot_data)

    # Create the plot using seaborn
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))

    # Error bar function to retrieve CIs based on mean distance
    def get_error_bars(x):
        mean_val = round(x.mean(), 5)
        return error_bars.get(mean_val)
    print(error_bars)
    # Use seaborn lineplot with markers and error bars
    sns.lineplot(data=df, x='x_pos', y='distance', hue='method',
                marker='s', markersize=8, linewidth=2, ax=ax, errorbar=get_error_bars, err_style="band")

    # Customize the plot
    ax.set_xlabel("Model Size Ratio ({p|q} / P)")
    ax.set_ylabel("Distance")
    ax.set_title("MUSE Model Scaling", fontsize=14)
    
    # Set x-axis labels
    ax.set_xticks([0, 1.3, 2.7]) #, 7.0
    ax.set_xticklabels(['~0%', '18.5%', '38.5%'])#'7b'
    
    ax.set_xlim(-0.3, 3.0)
    
    # Set y-axis limits to show 0% to slightly above 100%
    ax.set_ylim(20, 60)
    ax.invert_yaxis()  
    
    # Add y-axis ticks at 0%, 25%, 50%, 75%, and 100%
    ax.set_yticks([25, 35, 45, 55])
    ax.set_yticklabels(['25%', '35%', '45%', '55%'])
    
    # Position legend
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    
    # Save
    plt.savefig(f"results/model_scaling_plot.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"results/model_scaling_plot.pdf", dpi=600, bbox_inches="tight")


def privleak_plot():
    """
    Create a plot showing privleak vs alpha (bottom axis) and privleak vs topk (top twinx axis on log scale).
    Uses only 1.3b model size. Alpha values from 0 to 3 with 0.5 spacing. Uses 'target' value for alpha=0.
    Now includes bootstrap confidence intervals.
    """

    # Get optimal configurations to mark optimal points
    optimal_configs = find_optimal_configurations()

    alpha_data = []
    topk_data = []
    error_bars = {}  # Map round(privleak, 5) -> (ci_lower, ci_upper)

    # Load target baseline for alpha=0
    target_info = json.load(open("saves/eval/muse/baselines/target/MUSE_SUMMARY.json"))
    target_privleak = target_info.get('privleak', None)
    print(f"Target privleak value: {target_privleak}")

    retrain_info = json.load(open("saves/eval/muse/baselines/retrain/MUSE_SUMMARY.json"))
    retrain_privleak = retrain_info.get('privleak', None)
    print(f"Retrain privleak value: {retrain_privleak}")

    # Model sizes to process (only 1.3b)
    model_sizes = ["1.3b"]

    # Alpha values from 0 to 3 with 0.5 spacing
    alpha_values_full = [round(x * 0.5, 1) for x in range(0, 7)]

    for model_size in model_sizes:
        print(f"Loading privleak data for {model_size}...")

        # Add target value for alpha=0 if available (no CI for target)
        if target_privleak is not None:
            error_bars[round(target_privleak, 5)] = (target_privleak, target_privleak)
            for i in range(3):
                alpha_data.append({
                    'model_size': model_size,
                    'alpha': 0.0,
                    'privleak': target_privleak
                })

        alpha_values = alpha_values_full[1:]  # Skip alpha=0 since we use target

        # Load alpha-based models for this model size
        for alpha in alpha_values:
            folder_name = dd_alpha_folder(model_size, alpha)
            try:
                info = json.load(open(f"saves/eval/{folder_name}/MUSE_SUMMARY.json"))
                privleak_value = info.get('privleak', None)

                if privleak_value is not None:
                    # Calculate bootstrap CI
                    eval_path = f"saves/eval/{folder_name}/MUSE_EVAL.json"
                    try:
                        ci_result = calculate_ci_privleak(eval_path)
                        ci_hw = ci_result['ci_half_width']
                        ci_lower = privleak_value - ci_hw
                        ci_upper = privleak_value + ci_hw
                    except (FileNotFoundError, KeyError):
                        ci_lower = privleak_value
                        ci_upper = privleak_value

                    error_bars[round(privleak_value, 5)] = (ci_lower, ci_upper)

                    # Duplicate data 3 times for seaborn
                    for i in range(3):
                        alpha_data.append({
                            'model_size': model_size,
                            'alpha': alpha,
                            'privleak': privleak_value
                        })

            except FileNotFoundError:
                print(f"Warning: Could not find {folder_name}")
                continue

        # Load topk-based models for this model size
        topk_values = topk_values_model

        for topk in topk_values:
            folder_name = dd_topk_folder(model_size, topk)
            try:
                info = json.load(open(f"saves/eval/{folder_name}/MUSE_SUMMARY.json"))
                privleak_value = info.get('privleak', None)

                if privleak_value is not None:
                    # Calculate bootstrap CI
                    eval_path = f"saves/eval/{folder_name}/MUSE_EVAL.json"
                    try:
                        ci_result = calculate_ci_privleak(eval_path)
                        ci_hw = ci_result['ci_half_width']
                        ci_lower = privleak_value - ci_hw
                        ci_upper = privleak_value + ci_hw
                    except (FileNotFoundError, KeyError):
                        ci_lower = privleak_value
                        ci_upper = privleak_value

                    error_bars[round(privleak_value, 5)] = (ci_lower, ci_upper)

                    # Duplicate data 3 times for seaborn
                    for i in range(3):
                        topk_data.append({
                            'model_size': model_size,
                            'topk': topk,
                            'privleak': privleak_value
                        })

            except FileNotFoundError:
                print(f"Warning: Could not find {folder_name}")
                continue

    if not alpha_data and not topk_data:
        print("No privleak data found. Please check that the MUSE_SUMMARY.json files contain 'privleak' values.")
        return

    # Convert to DataFrames
    df_alpha = pd.DataFrame(alpha_data)
    df_topk = pd.DataFrame(topk_data)

    # Error bar function for seaborn
    def get_error_bars(x):
        mean_val = round(x.mean(), 5)
        return error_bars[mean_val]
    
    # Create the plot with twinx
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 3))

    # Use consistent colors from the palette
    linear_color = palette['Linear DD']
    rank_color = palette['Rank DD']

    # Bottom plot: Alpha vs Privleak (Linear DD)
    if not df_alpha.empty:
        sns.lineplot(
            data=df_alpha,
            x='alpha',
            y='privleak',
            marker='o',
            markersize=8,
            linewidth=2,
            color=linear_color,
            label="Linear DD",
            ax=ax1,
            errorbar=get_error_bars,
            err_style='band'
        )

    ax1.set_xlabel("Alpha (α)")
    ax1.set_ylabel("Privacy Leakage")

    # Create twin axis for topk
    ax2 = ax1.twiny()

    # Top plot: TopK vs Privleak on log scale (Rank DD)
    if not df_topk.empty:
        sns.lineplot(
            data=df_topk,
            x='topk',
            y='privleak',
            marker='X',
            markersize=8,
            linewidth=2,
            linestyle='--',
            color=rank_color,
            label="Rank DD",
            ax=ax2,
            errorbar=get_error_bars,
            err_style='band'
        )

    ax2.set_xlabel("Top-k")
    ax2.set_xscale('log')
    
    # Add retrain baseline
    if retrain_privleak is not None:
        ax1.axhline(retrain_privleak, color='black', linestyle='--', alpha=1, label='Retrain')
    
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
    ax1.set_yticks([0])
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("results/privleak_plot.png", dpi=600, bbox_inches="tight")
    plt.savefig("results/privleak_plot.pdf", dpi=600, bbox_inches="tight")
    
    print(f"Privleak plot saved as privleak_plot.png and privleak_plot.pdf")
    print(f"Processed {len(df_alpha)} alpha data points and {len(df_topk)} topk data points")

def muse_distance_plot(metric='knowmem'):
    """
    Create a plot showing euclidean distance from retrain vs alpha (bottom axis)
    and distance vs topk (top twinx axis on log scale).
    Uses only 1.3b model size.
    Distance is normalized so target = 100%.
    Y-axis is inverted so lower distance is at the top (better).
    Includes bootstrap confidence intervals as error bands.

    Args:
        metric (str): Either 'knowmem' or 'verbmem' to select which forget metric to use
    """

    if metric not in ['knowmem', 'verbmem']:
        raise ValueError("metric must be either 'knowmem' or 'verbmem'")

    alpha_data = []
    topk_data = []

    # Dictionary to store CI info: {mean_distance: (ci_lower, ci_upper)}
    error_bars = {}

    # Determine which forget metric to use
    forget_metric = f'forget_{metric}_ROUGE'

    # Paths for bootstrap CI calculation
    retrain_eval_path = "saves/eval/muse/baselines/retrain/MUSE_EVAL.json"
    target_eval_path = "saves/eval/muse/baselines/target/MUSE_EVAL.json"

    # Load retrain baseline for distance calculation
    retrain_info = json.load(open("saves/eval/muse/baselines/retrain/MUSE_SUMMARY.json"))
    retrain_scores = {
        forget_metric: retrain_info[forget_metric] * 100,
        'retain_knowmem_ROUGE': retrain_info['retain_knowmem_ROUGE'] * 100
    }
    print(f"Retrain {metric}: forget={retrain_scores[forget_metric]:.2f}, retain={retrain_scores['retain_knowmem_ROUGE']:.2f}")

    # Load target baseline for alpha=0
    target_info = json.load(open("saves/eval/muse/baselines/target/MUSE_SUMMARY.json"))
    target_scores = {
        forget_metric: target_info[forget_metric] * 100,
        'retain_knowmem_ROUGE': target_info['retain_knowmem_ROUGE'] * 100
    }

    def calculate_distance(point):
        """Calculate euclidean distance from retrain baseline using forget and retain metrics"""
        forget_diff = point[forget_metric] - retrain_scores[forget_metric]
        retain_diff = point['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE']
        return (forget_diff**2 + retain_diff**2)**0.5

    # Calculate target distance for normalization
    target_distance = calculate_distance(target_scores)
    print(f"Target distance ({metric}): {target_distance:.4f}")

    def normalize_distance(distance):
        """Normalize distance so target = 100%"""
        return (distance / target_distance) * 100

    def normalize_ci_halfwidth(ci_hw):
        """Normalize CI half-width using same scale as distance"""
        return (ci_hw / target_distance) * 100

    # Model size to process (only 1.3b)
    model_size = "1.3b"

    # Alpha values (excluding 0.0 since we add target separately)
    alpha_values = [0.5, 1.0, 1.5, 2.0]

    print(f"Loading distance data for {metric} ({model_size})...")

    # Add target value for alpha=0 with bootstrap CI
    target_normalized = normalize_distance(target_distance)
    try:
        target_ci = calculate_ci_distance(retrain_eval_path, target_eval_path, metric_type=metric)
        target_ci_hw_normalized = normalize_ci_halfwidth(target_ci['ci_half_width'])
        target_ci_lower = target_normalized - target_ci_hw_normalized
        target_ci_upper = target_normalized + target_ci_hw_normalized
        error_bars[round(target_normalized, 5)] = (target_ci_lower, target_ci_upper)
        print(f"  Target: distance = {target_normalized:.2f}% ± {target_ci_hw_normalized:.2f}%")
    except Exception as e:
        print(f"  Warning: Could not compute CI for target: {e}")
        error_bars[round(target_normalized, 5)] = (target_normalized, target_normalized)

    for _ in range(3):  # trick sns with duplicate entries
        alpha_data.append({
            'alpha': 0.0,
            'distance': target_normalized
        })

    # Load alpha-based models
    for alpha in alpha_values:
        folder_name = dd_alpha_folder(model_size, alpha)
        try:
            info = json.load(open(f"saves/eval/{folder_name}/MUSE_SUMMARY.json"))
            point = {
                forget_metric: info[forget_metric] * 100,
                'retain_knowmem_ROUGE': info['retain_knowmem_ROUGE'] * 100
            }
            distance = calculate_distance(point)
            normalized_distance = normalize_distance(distance)

            # Calculate bootstrap CI
            method_eval_path = f"saves/eval/{folder_name}/MUSE_EVAL.json"
            try:
                ci_result = calculate_ci_distance(retrain_eval_path, method_eval_path, metric_type=metric)
                ci_hw_normalized = normalize_ci_halfwidth(ci_result['ci_half_width'])
                ci_lower = normalized_distance - ci_hw_normalized
                ci_upper = normalized_distance + ci_hw_normalized
                error_bars[round(normalized_distance, 5)] = (ci_lower, ci_upper)
                print(f"  Alpha {alpha}: distance = {normalized_distance:.2f}% ± {ci_hw_normalized:.2f}%")
            except Exception as e:
                print(f"  Alpha {alpha}: distance = {normalized_distance:.2f}% (no CI: {e})")
                error_bars[round(normalized_distance, 5)] = (normalized_distance, normalized_distance)

            for _ in range(3):  # trick sns with duplicate entries
                alpha_data.append({
                    'alpha': alpha,
                    'distance': normalized_distance
                })
        except FileNotFoundError:
            print(f"Warning: Could not find {folder_name}")
            continue

    # Load topk-based models
    topk_values = topk_values_model

    for topk in topk_values:
        folder_name = dd_topk_folder(model_size, topk)
        try:
            info = json.load(open(f"saves/eval/{folder_name}/MUSE_SUMMARY.json"))
            point = {
                forget_metric: info[forget_metric] * 100,
                'retain_knowmem_ROUGE': info['retain_knowmem_ROUGE'] * 100
            }
            distance = calculate_distance(point)
            normalized_distance = normalize_distance(distance)

            # Calculate bootstrap CI
            method_eval_path = f"saves/eval/{folder_name}/MUSE_EVAL.json"
            try:
                ci_result = calculate_ci_distance(retrain_eval_path, method_eval_path, metric_type=metric)
                ci_hw_normalized = normalize_ci_halfwidth(ci_result['ci_half_width'])
                ci_lower = normalized_distance - ci_hw_normalized
                ci_upper = normalized_distance + ci_hw_normalized
                error_bars[round(normalized_distance, 5)] = (ci_lower, ci_upper)
                print(f"  TopK {topk}: distance = {normalized_distance:.2f}% ± {ci_hw_normalized:.2f}%")
            except Exception as e:
                print(f"  TopK {topk}: distance = {normalized_distance:.2f}% (no CI: {e})")
                error_bars[round(normalized_distance, 5)] = (normalized_distance, normalized_distance)

            for _ in range(3):  # trick sns with duplicate entries
                topk_data.append({
                    'topk': topk,
                    'distance': normalized_distance
                })
        except FileNotFoundError:
            print(f"Warning: Could not find {folder_name}")
            continue

    if not alpha_data and not topk_data:
        print("No distance data found.")
        return

    # Convert to DataFrames
    df_alpha = pd.DataFrame(alpha_data)
    df_topk = pd.DataFrame(topk_data)

    # Error bar function to retrieve CIs based on mean score
    def get_error_bars(x):
        mean_val = round(x.mean(), 5)
        return error_bars.get(mean_val)

    print(f"Error bars: {error_bars}")

    # Create the plot with twinx
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 3))

    # Use consistent colors from the palette
    linear_color = palette['Linear DD']
    rank_color = palette['Rank DD']

    # Bottom plot: Alpha vs Distance (Linear DD)
    if not df_alpha.empty:
        sns.lineplot(data=df_alpha, x='alpha', y='distance',
                marker='o', markersize=8, linewidth=2,
                color=linear_color,
                label="Linear DD",
                errorbar=get_error_bars, err_style="band", ax=ax1)

    ax1.set_xlabel("Alpha (α)")
    ax1.set_ylabel("Distance")

    # Create twin axis for topk
    ax2 = ax1.twiny()

    # Top plot: TopK vs Distance on log scale (Rank DD)
    if not df_topk.empty:
        sns.lineplot(data=df_topk, x='topk', y='distance',
                marker='X', markersize=8, linewidth=2, linestyle='--',
                color=rank_color,
                label="Rank DD",
                errorbar=get_error_bars, err_style="band", ax=ax2)

    ax2.set_xlabel("Top-k")
    ax2.set_xscale('log')

    # Add retrain baseline (distance = 0%)
    ax1.axhline(0, color='black', linestyle='--', alpha=0.7, label='Retrain')

    # Invert y-axis so lower distance is at the top (better)
    ax1.invert_yaxis()
    ax1.yaxis.set_major_formatter(PercentFormatter())

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
    output_basename = f"muse_distance_{metric}_plot"
    plt.savefig(f"results/{output_basename}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"results/{output_basename}.pdf", dpi=600, bbox_inches="tight")

    print(f"\nMUSE distance ({metric}) plot saved as {output_basename}.png and {output_basename}.pdf")
    print(f"Processed {len(df_alpha)} alpha data points and {len(df_topk)} topk data points")


def find_optimal_distill_configs():
    """
    Find optimal configurations for MUSE DD distillation sweep.
    Uses alpha=0.85 (average of verbmem=0.9 and knowmem=0.8).
    Returns best configs by distance to retrain for both verbmem and knowmem metrics.
    """

    # Load retrain baseline
    try:
        retrain_info = json.load(open("saves/eval/muse/baselines/retrain/MUSE_SUMMARY.json"))
    except FileNotFoundError:
        retrain_info = None

    if retrain_info is None:
        print("Warning: Could not load retrain baseline")
        return {'all_results': [], 'best_config_verbmem': None, 'best_config_knowmem': None,
                'best_config_avg': None, 'best_distance_verbmem': float('inf'),
                'best_distance_knowmem': float('inf'), 'best_distance_avg': float('inf')}

    retrain_scores = {
        'forget_verbmem_ROUGE': retrain_info['forget_verbmem_ROUGE'] * 100,
        'forget_knowmem_ROUGE': retrain_info['forget_knowmem_ROUGE'] * 100,
        'retain_knowmem_ROUGE': retrain_info['retain_knowmem_ROUGE'] * 100
    }

    def calculate_distance_verbmem(scores):
        forget_diff = scores['forget_verbmem_ROUGE'] - retrain_scores['forget_verbmem_ROUGE']
        retain_diff = scores['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE']
        return (forget_diff**2 + retain_diff**2)**0.5

    def calculate_distance_knowmem(scores):
        forget_diff = scores['forget_knowmem_ROUGE'] - retrain_scores['forget_knowmem_ROUGE']
        retain_diff = scores['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE']
        return (forget_diff**2 + retain_diff**2)**0.5

    results = {
        'all_results': [],
        'best_config_verbmem': None,
        'best_config_knowmem': None,
        'best_config_avg': None,
        'best_distance_verbmem': float('inf'),
        'best_distance_knowmem': float('inf'),
        'best_distance_avg': float('inf'),
        'retrain_scores': retrain_scores
    }

    for temp in temperatures:
        for lr in learning_rates:
            for epoch in epochs:
                try:
                    # New path format: saves/eval/muse/distill/lr-{lr}-epoch-{epoch}-temp-{temp}/
                    eval_path = f"saves/eval/muse/distill/lr-{lr}-epoch-{epoch}-temp-{temp}/MUSE_SUMMARY.json"
                    info = json.load(open(eval_path))

                    scores = {
                        'forget_verbmem_ROUGE': info['forget_verbmem_ROUGE'] * 100,
                        'forget_knowmem_ROUGE': info['forget_knowmem_ROUGE'] * 100,
                        'retain_knowmem_ROUGE': info['retain_knowmem_ROUGE'] * 100
                    }

                    dist_verbmem = calculate_distance_verbmem(scores)
                    dist_knowmem = calculate_distance_knowmem(scores)
                    dist_avg = (dist_verbmem + dist_knowmem) / 2

                    result = {
                        'temperature': temp,
                        'lr': lr,
                        'epoch': epoch,
                        'distance_verbmem': dist_verbmem,
                        'distance_knowmem': dist_knowmem,
                        'distance_avg': dist_avg,
                        'forget_verbmem_ROUGE': scores['forget_verbmem_ROUGE'],
                        'forget_knowmem_ROUGE': scores['forget_knowmem_ROUGE'],
                        'retain_knowmem_ROUGE': scores['retain_knowmem_ROUGE']
                    }
                    results['all_results'].append(result)

                    if dist_verbmem < results['best_distance_verbmem']:
                        results['best_distance_verbmem'] = dist_verbmem
                        results['best_config_verbmem'] = {'lr': lr, 'epoch': epoch, 'temperature': temp}

                    if dist_knowmem < results['best_distance_knowmem']:
                        results['best_distance_knowmem'] = dist_knowmem
                        results['best_config_knowmem'] = {'lr': lr, 'epoch': epoch, 'temperature': temp}

                    if dist_avg < results['best_distance_avg']:
                        results['best_distance_avg'] = dist_avg
                        results['best_config_avg'] = {'lr': lr, 'epoch': epoch, 'temperature': temp}

                except (FileNotFoundError, KeyError, json.JSONDecodeError):
                    pass

    return results


def find_optimal_offset_configs():
    """Find optimal Offset Unlearning LR for MUSE (closest to retrain)."""
    lrs = ["1e-6", "5e-6", "1e-5", "2e-5", "3e-5", "5e-5"]

    try:
        retrain_info = json.load(open("saves/eval/muse/baselines/retrain/MUSE_SUMMARY.json"))
    except FileNotFoundError:
        return {'best_lr_verbmem': None, 'best_lr_knowmem': None}

    retrain_scores = {
        'forget_verbmem_ROUGE': retrain_info['forget_verbmem_ROUGE'] * 100,
        'forget_knowmem_ROUGE': retrain_info['forget_knowmem_ROUGE'] * 100,
        'retain_knowmem_ROUGE': retrain_info['retain_knowmem_ROUGE'] * 100,
    }

    best = {'best_lr_verbmem': None, 'best_lr_knowmem': None,
            'best_dist_verbmem': float('inf'), 'best_dist_knowmem': float('inf')}

    for lr in lrs:
        folder = f"saves/eval/muse/offset/lr-{lr}"
        try:
            info = json.load(open(f"{folder}/MUSE_SUMMARY.json"))
            scores = {k: info[k] * 100 for k in retrain_scores}

            dv = ((scores['forget_verbmem_ROUGE'] - retrain_scores['forget_verbmem_ROUGE'])**2 +
                  (scores['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE'])**2)**0.5
            dk = ((scores['forget_knowmem_ROUGE'] - retrain_scores['forget_knowmem_ROUGE'])**2 +
                  (scores['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE'])**2)**0.5

            if dv < best['best_dist_verbmem']:
                best['best_dist_verbmem'] = dv
                best['best_lr_verbmem'] = lr
            if dk < best['best_dist_knowmem']:
                best['best_dist_knowmem'] = dk
                best['best_lr_knowmem'] = lr
        except (FileNotFoundError, KeyError):
            pass

    return best


def find_optimal_uld_configs():
    """Find optimal ULD LR for MUSE (closest to retrain)."""
    lrs = ["1e-4", "5e-4", "1e-3", "2e-3", "3e-3", "5e-3"]

    try:
        retrain_info = json.load(open("saves/eval/muse/baselines/retrain/MUSE_SUMMARY.json"))
    except FileNotFoundError:
        return {'best_lr_verbmem': None, 'best_lr_knowmem': None}

    retrain_scores = {
        'forget_verbmem_ROUGE': retrain_info['forget_verbmem_ROUGE'] * 100,
        'forget_knowmem_ROUGE': retrain_info['forget_knowmem_ROUGE'] * 100,
        'retain_knowmem_ROUGE': retrain_info['retain_knowmem_ROUGE'] * 100,
    }

    best = {'best_lr_verbmem': None, 'best_lr_knowmem': None,
            'best_dist_verbmem': float('inf'), 'best_dist_knowmem': float('inf')}

    for lr in lrs:
        folder = f"saves/eval/muse/uld/lr-{lr}"
        try:
            info = json.load(open(f"{folder}/MUSE_SUMMARY.json"))
            scores = {k: info[k] * 100 for k in retrain_scores}

            dv = ((scores['forget_verbmem_ROUGE'] - retrain_scores['forget_verbmem_ROUGE'])**2 +
                  (scores['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE'])**2)**0.5
            dk = ((scores['forget_knowmem_ROUGE'] - retrain_scores['forget_knowmem_ROUGE'])**2 +
                  (scores['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE'])**2)**0.5

            if dv < best['best_dist_verbmem']:
                best['best_dist_verbmem'] = dv
                best['best_lr_verbmem'] = lr
            if dk < best['best_dist_knowmem']:
                best['best_dist_knowmem'] = dk
                best['best_lr_knowmem'] = lr
        except (FileNotFoundError, KeyError):
            pass

    return best


def find_optimal_undial_configs():
    """Load the MUSE UNDIAL gradient baseline result (single optimal eval)."""
    best = {'best_verbmem': None, 'best_knowmem': None}

    folder = "saves/eval/muse/gradient/undial"
    try:
        info = json.load(open(f"{folder}/MUSE_SUMMARY.json"))
        if len(info) < 5:
            return best
        config = {'lr': '1e-5', 'checkpoint': '12', 'folder': folder}
        best['best_verbmem'] = config
        best['best_knowmem'] = config
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass

    return best


def find_optimal_guard_configs():
    """Find optimal GUARD (lr, delta) for MUSE (closest to retrain)."""
    lrs = ["1e-3", "5e-4", "1e-4"]
    deltas = ["0_3", "0_5", "0_7"]

    try:
        retrain_info = json.load(open("saves/eval/muse/baselines/retrain/MUSE_SUMMARY.json"))
    except FileNotFoundError:
        return {'best_config_verbmem': None, 'best_config_knowmem': None}

    retrain_scores = {
        'forget_verbmem_ROUGE': retrain_info['forget_verbmem_ROUGE'] * 100,
        'forget_knowmem_ROUGE': retrain_info['forget_knowmem_ROUGE'] * 100,
        'retain_knowmem_ROUGE': retrain_info['retain_knowmem_ROUGE'] * 100,
    }

    best = {'best_config_verbmem': None, 'best_config_knowmem': None,
            'best_dist_verbmem': float('inf'), 'best_dist_knowmem': float('inf')}

    for lr in lrs:
        for delta in deltas:
            folder = f"saves/eval/muse/guard/lr-{lr}_delta-{delta}"
            try:
                info = json.load(open(f"{folder}/MUSE_SUMMARY.json"))
                scores = {k: info[k] * 100 for k in retrain_scores}

                dv = ((scores['forget_verbmem_ROUGE'] - retrain_scores['forget_verbmem_ROUGE'])**2 +
                      (scores['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE'])**2)**0.5
                dk = ((scores['forget_knowmem_ROUGE'] - retrain_scores['forget_knowmem_ROUGE'])**2 +
                      (scores['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE'])**2)**0.5

                if dv < best['best_dist_verbmem']:
                    best['best_dist_verbmem'] = dv
                    best['best_config_verbmem'] = {'lr': lr, 'delta': delta, 'folder': folder}
                if dk < best['best_dist_knowmem']:
                    best['best_dist_knowmem'] = dk
                    best['best_config_knowmem'] = {'lr': lr, 'delta': delta, 'folder': folder}
            except (FileNotFoundError, KeyError):
                pass

    return best


def find_optimal_whp_configs():
    """Find optimal WHP alpha for MUSE (closest to retrain)."""
    alphas = ["0_5", "1_0", "1_5", "2_0", "3_0", "5_0"]

    try:
        retrain_info = json.load(open("saves/eval/muse/baselines/retrain/MUSE_SUMMARY.json"))
    except FileNotFoundError:
        return {'best_alpha_verbmem': None, 'best_alpha_knowmem': None}

    retrain_scores = {
        'forget_verbmem_ROUGE': retrain_info['forget_verbmem_ROUGE'] * 100,
        'forget_knowmem_ROUGE': retrain_info['forget_knowmem_ROUGE'] * 100,
        'retain_knowmem_ROUGE': retrain_info['retain_knowmem_ROUGE'] * 100,
    }

    best = {'best_alpha_verbmem': None, 'best_alpha_knowmem': None,
            'best_dist_verbmem': float('inf'), 'best_dist_knowmem': float('inf')}

    for alpha in alphas:
        folder = f"saves/eval/muse/whp/alpha-{alpha}"
        try:
            info = json.load(open(f"{folder}/MUSE_SUMMARY.json"))
            scores = {k: info[k] * 100 for k in retrain_scores}

            dv = ((scores['forget_verbmem_ROUGE'] - retrain_scores['forget_verbmem_ROUGE'])**2 +
                  (scores['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE'])**2)**0.5
            dk = ((scores['forget_knowmem_ROUGE'] - retrain_scores['forget_knowmem_ROUGE'])**2 +
                  (scores['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE'])**2)**0.5

            if dv < best['best_dist_verbmem']:
                best['best_dist_verbmem'] = dv
                best['best_alpha_verbmem'] = alpha
            if dk < best['best_dist_knowmem']:
                best['best_dist_knowmem'] = dk
                best['best_alpha_knowmem'] = alpha
        except (FileNotFoundError, KeyError):
            pass

    return best


def find_optimal_eco_configs():
    """Find optimal ECO (classifier lr, strength) for MUSE (closest to retrain)."""
    lrs = ["1e-5", "2e-5", "5e-5"]
    strengths = [50, 100, 200]

    try:
        retrain_info = json.load(open("saves/eval/muse/baselines/retrain/MUSE_SUMMARY.json"))
    except FileNotFoundError:
        return {'best_config_verbmem': None, 'best_config_knowmem': None}

    retrain_scores = {
        'forget_verbmem_ROUGE': retrain_info['forget_verbmem_ROUGE'] * 100,
        'forget_knowmem_ROUGE': retrain_info['forget_knowmem_ROUGE'] * 100,
        'retain_knowmem_ROUGE': retrain_info['retain_knowmem_ROUGE'] * 100,
    }

    best = {'best_config_verbmem': None, 'best_config_knowmem': None,
            'best_dist_verbmem': float('inf'), 'best_dist_knowmem': float('inf')}

    for lr in lrs:
        for strength in strengths:
            folder = f"saves/eval/muse/eco/lr-{lr}_str-{strength}"
            try:
                info = json.load(open(f"{folder}/MUSE_SUMMARY.json"))
                scores = {k: info[k] * 100 for k in retrain_scores}

                dv = ((scores['forget_verbmem_ROUGE'] - retrain_scores['forget_verbmem_ROUGE'])**2 +
                      (scores['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE'])**2)**0.5
                dk = ((scores['forget_knowmem_ROUGE'] - retrain_scores['forget_knowmem_ROUGE'])**2 +
                      (scores['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE'])**2)**0.5

                if dv < best['best_dist_verbmem']:
                    best['best_dist_verbmem'] = dv
                    best['best_config_verbmem'] = {'lr': lr, 'strength': strength, 'folder': folder}
                if dk < best['best_dist_knowmem']:
                    best['best_dist_knowmem'] = dk
                    best['best_config_knowmem'] = {'lr': lr, 'strength': strength, 'folder': folder}
            except (FileNotFoundError, KeyError):
                pass

    return best


def find_optimal_lunar_configs():
    """Find optimal LUNAR lr for MUSE (closest to retrain)."""
    lrs = ["0001", "0005", "001", "005", "01"]

    try:
        retrain_info = json.load(open("saves/eval/muse/baselines/retrain/MUSE_SUMMARY.json"))
    except FileNotFoundError:
        return {'best_lr_verbmem': None, 'best_lr_knowmem': None}

    retrain_scores = {
        'forget_verbmem_ROUGE': retrain_info['forget_verbmem_ROUGE'] * 100,
        'forget_knowmem_ROUGE': retrain_info['forget_knowmem_ROUGE'] * 100,
        'retain_knowmem_ROUGE': retrain_info['retain_knowmem_ROUGE'] * 100,
    }

    best = {'best_lr_verbmem': None, 'best_lr_knowmem': None,
            'best_dist_verbmem': float('inf'), 'best_dist_knowmem': float('inf')}

    for lr in lrs:
        folder = f"saves/eval/muse/lunar/lr-{lr}"
        try:
            info = json.load(open(f"{folder}/MUSE_SUMMARY.json"))
            scores = {k: info[k] * 100 for k in retrain_scores}

            dv = ((scores['forget_verbmem_ROUGE'] - retrain_scores['forget_verbmem_ROUGE'])**2 +
                  (scores['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE'])**2)**0.5
            dk = ((scores['forget_knowmem_ROUGE'] - retrain_scores['forget_knowmem_ROUGE'])**2 +
                  (scores['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE'])**2)**0.5

            if dv < best['best_dist_verbmem']:
                best['best_dist_verbmem'] = dv
                best['best_lr_verbmem'] = lr
            if dk < best['best_dist_knowmem']:
                best['best_dist_knowmem'] = dk
                best['best_lr_knowmem'] = lr
        except (FileNotFoundError, KeyError):
            pass

    return best


def plot_distill_sweep_heatmaps():
    """Plot heatmaps for verbmem and knowmem distances side-by-side in one figure."""
    from matplotlib.colors import LinearSegmentedColormap

    results = find_optimal_distill_configs()
    all_results = results['all_results']

    if not all_results:
        print(f"No distillation results found. Run muse_distill_sweep.py first.")
        return

    # Format learning rate labels nicely
    def format_lr(lr):
        if lr >= 1e-4:
            return f'{lr*1e4:.2g}e-4'
        else:
            return f'{lr*1e5:.0f}e-5'
    lr_labels = [format_lr(lr) for lr in learning_rates]

    # Continuous green-to-red colormap (green=low/good, red=high/bad)
    cmap = LinearSegmentedColormap.from_list('GnRd', ['#2ecc71', '#f1c40f', '#e74c3c'])

    # Create figure with two subplots side by side + colorbar axis
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), gridspec_kw={'width_ratios': [1, 1, 0.05]})

    # Collect all data to find global min/max for consistent color scale
    all_data = []
    data_matrices = []

    for metric in ['verbmem', 'knowmem']:
        data = np.full((len(temperatures), len(learning_rates)), np.nan)
        distance_key = f'distance_{metric}'

        for r in all_results:
            temp = r['temperature']
            lr = r['lr']
            if temp in temperatures and lr in learning_rates:
                row_idx = temperatures.index(temp)
                col_idx = learning_rates.index(lr)
                data[row_idx, col_idx] = r[distance_key]

        data_matrices.append(data)
        all_data.extend(data[~np.isnan(data)].tolist())

    # Set color range from actual data min/max
    vmin, vmax = min(all_data), max(all_data)

    for idx, metric in enumerate(['verbmem', 'knowmem']):
        ax = axes[idx]
        data = data_matrices[idx]

        # Plot the heatmap with continuous colormap (no annotations, no individual colorbars)
        sns.heatmap(
            data,
            ax=ax,
            annot=False,
            cmap=cmap,
            xticklabels=lr_labels,
            yticklabels=[str(t) for t in temperatures],
            cbar=False,
            vmin=vmin,
            vmax=vmax,
            linewidths=1,
            linecolor='white'
        )

        metric_label = {'verbmem': 'Verbmem', 'knowmem': 'Knowmem'}[metric]
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Temperature' if idx == 0 else '')
        ax.set_title(f'{metric_label} Performance')
        ax.tick_params(axis='both', which='both', length=0)

        # Rotate x-axis labels for readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Add shared colorbar on the dedicated axis
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=axes[2])
    cbar.set_label('Distance (lower is better)')

    plt.tight_layout()

    # Save combined figure
    output_basename = "muse_distill_heatmap_combined"
    plt.savefig(f"results/{output_basename}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"results/{output_basename}.pdf", dpi=300, bbox_inches="tight")
    print(f"\nCombined heatmap saved as {output_basename}.png and {output_basename}.pdf")
    plt.close()


# Main execution
if __name__ == "__main__":

    # Run all plotting functions
    result = main_scatter_plot()
    if result is not None:
        all_dfs, cross_tok_detail_dfs = result
        cross_tok_scatter_plot(cross_tok_detail_dfs)
    #plot_muse_curve()
    #model_scaling_plot()
    #privleak_plot()
    #muse_distance_plot(metric='knowmem')
    #muse_distance_plot(metric='verbmem')

    # Distillation sweep results
    #plot_distill_sweep_heatmaps()