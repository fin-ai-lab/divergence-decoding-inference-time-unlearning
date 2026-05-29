"""
Bootstrap confidence intervals for TOFU metrics.

This module implements proper layer-by-layer bootstrapping:
1. Resampling BASE metrics at the per-sample level (value_by_index)
2. RECOMPUTING composite metrics from the resampled base metrics
3. Propagating uncertainty correctly through the metric hierarchy

Key insight: If metric A depends on metrics B and C, we must:
- Resample B and C at the per-sample level
- Recompute A from the resampled B and C
- NOT just resample A's stored per-sample scores
"""

import json
import numpy as np
import yaml
from tqdm.auto import tqdm
from typing import Dict, Any, List, Tuple
import warnings
from sklearn.metrics import roc_auc_score


class TOFUBootstrapper:
    """Bootstrap confidence intervals for TOFU evaluation metrics."""

    def __init__(self, eval_json_path: str, config_yaml_path: str,
                 retrain_summary_path: str = None, target_summary_path: str = None):
        """
        Initialize bootstrapper with evaluation data and configuration.

        Args:
            eval_json_path: Path to TOFU_EVAL.json (contains per-sample scores)
            config_yaml_path: Path to tofu_config.yaml (contains metric definitions)
            retrain_summary_path: Path to retrain TOFU_SUMMARY.json (for privacy scores)
            target_summary_path: Path to target TOFU_SUMMARY.json (for utility normalization)
        """
        # Load evaluation data (per-sample scores)
        with open(eval_json_path, 'r') as f:
            self.eval_data = json.load(f)

        # Load config
        with open(config_yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load retrain baseline for privacy calculations
        self.retrain_priv_scores = {}
        if retrain_summary_path:
            with open(retrain_summary_path, 'r') as f:
                retrain_data = json.load(f)
                privacy_keys = ["mia_min_k_plus_plus", "mia_min_k", "mia_loss", "mia_zlib"]
                for key in privacy_keys:
                    if key in retrain_data:
                        self.retrain_priv_scores[key] = retrain_data[key]

        # Load target baseline for utility normalization
        self.target_util = None
        if target_summary_path:
            with open(target_summary_path, 'r') as f:
                target_data = json.load(f)
                if "model_utility" in target_data and "forget_Q_A_gibberish" in target_data:
                    self.target_util = 2 / ((1 / target_data["model_utility"]) +
                                           (1 / target_data["forget_Q_A_gibberish"]))

    def get_all_indices(self) -> List[str]:
        """Extract all unique sample indices from the evaluation data."""
        all_indices = set()

        for metric_name, metric_data in self.eval_data.items():
            if isinstance(metric_data, dict) and "value_by_index" in metric_data:
                value_by_index = metric_data["value_by_index"]
                all_indices.update(value_by_index.keys())

        return sorted(all_indices)

    def resample_metric(self, metric_name: str, resampled_indices: List[str]) -> Dict[str, Any]:
        """
        Resample a base metric's per-sample scores and recompute aggregation.

        Args:
            metric_name: Name of the metric to resample
            resampled_indices: List of resampled indices (with replacement)

        Returns:
            Dict with 'agg_value' and 'value_by_index' for the resampled metric
        """
        if metric_name not in self.eval_data:
            warnings.warn(f"Metric {metric_name} not found in eval data")
            return None

        metric_data = self.eval_data[metric_name]
        if "value_by_index" not in metric_data:
            warnings.warn(f"Metric {metric_name} has no value_by_index")
            return None

        value_by_index = metric_data["value_by_index"]

        # Resample the per-sample scores
        resampled_values = []
        resampled_value_by_index = {}

        for i, idx in enumerate(resampled_indices):
            if idx in value_by_index:
                sample_data = value_by_index[idx]
                resampled_value_by_index[str(i)] = sample_data

                # Extract the score/value for aggregation
                if isinstance(sample_data, dict):
                    # Try different keys that might contain the score
                    if "score" in sample_data:
                        val = sample_data["score"]
                    elif "prob" in sample_data:
                        val = sample_data["prob"]
                    else:
                        # Use first numeric value
                        val = next((v for v in sample_data.values()
                                  if isinstance(v, (int, float))), None)

                    if val is not None:
                        resampled_values.append(val)

        # Recompute aggregation (mean for most metrics)
        if resampled_values:
            agg_value = np.mean(resampled_values)
        else:
            agg_value = None

        return {
            "agg_value": agg_value,
            "value_by_index": resampled_value_by_index
        }

    def compute_probability_w_options(self, correct_metric: Dict, wrong_metric: Dict) -> Dict[str, Any]:
        """
        Recompute probability_w_options from resampled correct/wrong probabilities.

        This metric normalizes correct answer probabilities against wrong answer probabilities.

        Returns:
            Dict with 'agg_value' and 'value_by_index'
        """
        if not correct_metric or not wrong_metric:
            return None

        correct_vbi = correct_metric.get("value_by_index", {})
        wrong_vbi = wrong_metric.get("value_by_index", {})

        probs = []
        probs_by_index = {}

        for idx in correct_vbi.keys():
            if idx in wrong_vbi:
                correct_prob = correct_vbi[idx].get("prob")
                wrong_prob = wrong_vbi[idx].get("prob")

                if correct_prob is not None and wrong_prob is not None:
                    # Handle if correct_prob is an array
                    if isinstance(correct_prob, (list, np.ndarray)):
                        correct_prob = np.mean(correct_prob)

                    # Handle if wrong_prob is an array (sum across wrong answers)
                    if isinstance(wrong_prob, (list, np.ndarray)):
                        wrong_sum = np.sum(wrong_prob)
                    else:
                        wrong_sum = wrong_prob

                    prob = correct_prob / (correct_prob + wrong_sum + 1e-10)
                    probs.append(prob)
                    probs_by_index[idx] = {"prob": prob}

        if not probs:
            return None

        return {
            "agg_value": np.mean(probs),
            "value_by_index": probs_by_index
        }

    def compute_truth_ratio(self, correct_metric: Dict, wrong_metric: Dict,
                           aggregator: str = "closer_to_1_better") -> Dict[str, Any]:
        """
        Recompute truth_ratio from resampled correct/wrong probabilities.

        Truth ratio = wrong_prob / correct_prob

        Note: wrong answers may have multiple values (arrays) per sample.

        Returns:
            Dict with 'agg_value' and 'value_by_index'
        """
        if not correct_metric or not wrong_metric:
            return None

        correct_vbi = correct_metric.get("value_by_index", {})
        wrong_vbi = wrong_metric.get("value_by_index", {})

        truth_ratios = []
        truth_ratios_by_index = {}

        for idx in correct_vbi.keys():
            if idx in wrong_vbi:
                correct_loss = correct_vbi[idx].get("avg_loss")
                wrong_loss = wrong_vbi[idx].get("avg_loss")

                if correct_loss is not None and wrong_loss is not None:
                    # Handle if losses are arrays (multiple wrong answers)
                    correct_loss = np.mean(correct_loss) if isinstance(correct_loss, (list, np.ndarray)) else correct_loss
                    wrong_loss = np.mean(wrong_loss) if isinstance(wrong_loss, (list, np.ndarray)) else wrong_loss

                    correct_prob = np.exp(-correct_loss)
                    wrong_prob = np.exp(-wrong_loss)
                    tr = wrong_prob / (correct_prob + 1e-10)
                    truth_ratios.append(tr)
                    truth_ratios_by_index[idx] = {"score": tr}

        if not truth_ratios:
            return None

        truth_ratios = np.array(truth_ratios)

        # Apply aggregator
        if aggregator == "closer_to_1_better":
            # For forget data: better if false and true equally likely
            agg_value = np.mean(np.minimum(truth_ratios, 1 / (truth_ratios + 1e-10)))
        elif aggregator == "true_better":
            # For non-forget data: better if tr is lower
            agg_value = np.mean(np.maximum(0, 1 - truth_ratios))
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")

        return {
            "agg_value": agg_value,
            "value_by_index": truth_ratios_by_index
        }

    def compute_hm_aggregate(self, pre_compute_metrics: Dict[str, Dict]) -> float:
        """
        Recompute harmonic mean aggregate from pre-computed metrics.

        Args:
            pre_compute_metrics: Dict mapping metric names to their resampled results
        """
        values = []
        for metric_result in pre_compute_metrics.values():
            if metric_result and "agg_value" in metric_result:
                val = metric_result["agg_value"]
                if val is not None:
                    values.append(val)

        if not values:
            return None

        # Harmonic mean: n / sum(1/x_i)
        from scipy.stats import hmean
        return hmean(values)

    def resample_mia_metric(self, metric_name: str, resampled_indices: List[str]) -> Dict[str, Any]:
        """
        Resample an MIA metric and recompute AUC.

        MIA metrics have separate forget and holdout sets with per-sample scores.
        We resample each set independently and recompute the AUC.

        Args:
            metric_name: Name of the MIA metric (e.g., "mia_min_k")
            resampled_indices: Not used for MIA (MIA has its own forget/holdout split)

        Returns:
            Dict with recomputed 'agg_value' (AUC)
        """
        if metric_name not in self.eval_data:
            return None

        metric_data = self.eval_data[metric_name]

        # Check if this is an MIA metric (has forget and holdout keys)
        if "forget" not in metric_data or "holdout" not in metric_data:
            return None

        forget_data = metric_data["forget"]
        holdout_data = metric_data["holdout"]

        if "value_by_index" not in forget_data or "value_by_index" not in holdout_data:
            return None

        # Get all indices for forget and holdout sets
        forget_indices = list(forget_data["value_by_index"].keys())
        holdout_indices = list(holdout_data["value_by_index"].keys())

        # Resample forget and holdout sets independently (with replacement)
        n_forget = len(forget_indices)
        n_holdout = len(holdout_indices)

        resampled_forget_indices = np.random.choice(forget_indices, size=n_forget, replace=True)
        resampled_holdout_indices = np.random.choice(holdout_indices, size=n_holdout, replace=True)

        # Extract scores for resampled indices
        forget_scores = [
            forget_data["value_by_index"][idx]["score"]
            for idx in resampled_forget_indices
        ]
        holdout_scores = [
            holdout_data["value_by_index"][idx]["score"]
            for idx in resampled_holdout_indices
        ]

        # Recompute AUC (following the same logic as mia/utils.py:mia_auc)
        # Label convention: 0 for forget (member), 1 for holdout (non-member)
        # AUC = 1 when forget scores are much lower than holdout scores
        scores = np.array(forget_scores + holdout_scores)
        labels = np.array([0] * len(forget_scores) + [1] * len(holdout_scores))

        auc_value = roc_auc_score(labels, scores)

        return {"agg_value": auc_value}

    def bootstrap_iteration(self, seed: int) -> Dict[str, float]:
        """
        Perform one bootstrap iteration: resample indices and recompute all metrics.

        This implements proper layer-by-layer bootstrapping:
        1. Resample BASE metrics at per-sample level
        2. Resample MIA metrics (forget/holdout sets) and recompute AUC
        3. RECOMPUTE composite metrics from resampled base metrics

        Args:
            seed: Random seed for this iteration

        Returns:
            Dict mapping metric names to their resampled results (with agg_value)
        """
        np.random.seed(seed)

        # Get all indices and resample with replacement
        all_indices = self.get_all_indices()
        n = len(all_indices)
        resampled_indices = np.random.choice(all_indices, size=n, replace=True)

        results = {}

        # ==== LAYER 1: Resample base metrics (no dependencies) ====
        base_metrics = [
            # Memorization metrics
            "exact_memorization",
            "extraction_strength",

            # Base probabilities for forget set
            "forget_Q_A_PARA_Prob",
            "forget_Q_A_PERT_Prob",

            # ROUGE and gibberish for forget set
            "forget_Q_A_ROUGE",
            "forget_Q_A_gibberish",

            # Base probabilities for retain set
            "retain_Q_A_Prob",
            "retain_Q_A_PARA_Prob",
            "retain_Q_A_PERT_Prob",
            "retain_Q_A_ROUGE",

            # Base probabilities for real authors (ra) set
            "ra_Q_A_Prob",
            "ra_Q_A_PERT_Prob",
            "ra_Q_A_ROUGE",

            # Base probabilities for world facts (wf) set
            "wf_Q_A_Prob",
            "wf_Q_A_PERT_Prob",
            "wf_Q_A_ROUGE",
        ]

        for metric_name in base_metrics:
            results[metric_name] = self.resample_metric(metric_name, resampled_indices)

        # Resample MIA metrics separately (they use their own forget/holdout split)
        mia_metrics = ["mia_min_k_plus_plus", "mia_min_k", "mia_loss", "mia_zlib"]
        for metric_name in mia_metrics:
            results[metric_name] = self.resample_mia_metric(metric_name, resampled_indices)

        # ==== LAYER 2: Recompute composite metrics from resampled base metrics ====

        # Truth ratios (depend on PARA and PERT probabilities)
        if "forget_Q_A_PARA_Prob" in results and "forget_Q_A_PERT_Prob" in results:
            results["forget_truth_ratio"] = self.compute_truth_ratio(
                results["forget_Q_A_PARA_Prob"],
                results["forget_Q_A_PERT_Prob"],
                aggregator="closer_to_1_better"
            )

        if "retain_Q_A_PARA_Prob" in results and "retain_Q_A_PERT_Prob" in results:
            results["retain_Truth_Ratio"] = self.compute_truth_ratio(
                results["retain_Q_A_PARA_Prob"],
                results["retain_Q_A_PERT_Prob"],
                aggregator="true_better"
            )

        if "ra_Q_A_Prob" in results and "ra_Q_A_PERT_Prob" in results:
            results["ra_Truth_Ratio"] = self.compute_truth_ratio(
                results["ra_Q_A_Prob"],
                results["ra_Q_A_PERT_Prob"],
                aggregator="true_better"
            )

        if "wf_Q_A_Prob" in results and "wf_Q_A_PERT_Prob" in results:
            results["wf_Truth_Ratio"] = self.compute_truth_ratio(
                results["wf_Q_A_Prob"],
                results["wf_Q_A_PERT_Prob"],
                aggregator="true_better"
            )

        # Probability normalized (depend on correct and wrong probabilities)
        if "ra_Q_A_Prob" in results and "ra_Q_A_PERT_Prob" in results:
            results["ra_Q_A_Prob_normalised"] = self.compute_probability_w_options(
                results["ra_Q_A_Prob"],
                results["ra_Q_A_PERT_Prob"]
            )

        if "wf_Q_A_Prob" in results and "wf_Q_A_PERT_Prob" in results:
            results["wf_Q_A_Prob_normalised"] = self.compute_probability_w_options(
                results["wf_Q_A_Prob"],
                results["wf_Q_A_PERT_Prob"]
            )

        # ==== LAYER 3: Recompute model_utility (harmonic mean aggregate) ====
        # model_utility is hm_aggregate of retain, ra, wf utility metrics
        utility_components = {}

        # For each component, get the relevant metrics
        # retain utility: retain_Q_A_Prob and retain_Q_A_ROUGE and retain_Truth_Ratio
        if all(k in results for k in ["retain_Q_A_Prob", "retain_Q_A_ROUGE", "retain_Truth_Ratio"]):
            retain_util_values = [
                results["retain_Q_A_Prob"]["agg_value"],
                results["retain_Q_A_ROUGE"]["agg_value"],
                results["retain_Truth_Ratio"]["agg_value"]
            ]
            if all(v is not None for v in retain_util_values):
                from scipy.stats import hmean
                utility_components["retain"] = {"agg_value": hmean(retain_util_values)}

        # ra utility: ra_Q_A_Prob_normalised, ra_Q_A_ROUGE, ra_Truth_Ratio
        if all(k in results for k in ["ra_Q_A_Prob_normalised", "ra_Q_A_ROUGE", "ra_Truth_Ratio"]):
            ra_util_values = [
                results["ra_Q_A_Prob_normalised"]["agg_value"],
                results["ra_Q_A_ROUGE"]["agg_value"],
                results["ra_Truth_Ratio"]["agg_value"]
            ]
            if all(v is not None for v in ra_util_values):
                from scipy.stats import hmean
                utility_components["ra"] = {"agg_value": hmean(ra_util_values)}

        # wf utility: wf_Q_A_Prob_normalised, wf_Q_A_ROUGE, wf_Truth_Ratio
        if all(k in results for k in ["wf_Q_A_Prob_normalised", "wf_Q_A_ROUGE", "wf_Truth_Ratio"]):
            wf_util_values = [
                results["wf_Q_A_Prob_normalised"]["agg_value"],
                results["wf_Q_A_ROUGE"]["agg_value"],
                results["wf_Truth_Ratio"]["agg_value"]
            ]
            if all(v is not None for v in wf_util_values):
                from scipy.stats import hmean
                utility_components["wf"] = {"agg_value": hmean(wf_util_values)}

        # Compute overall model_utility as harmonic mean of components
        if utility_components:
            model_util_val = self.compute_hm_aggregate(utility_components)
            if model_util_val is not None:
                results["model_utility"] = {"agg_value": model_util_val}

        return results

    def compute_aggregate_scores(self, metric_results: Dict[str, Dict]) -> Dict[str, float]:
        """
        Compute high-level aggregate scores from base metrics.

        Uses the same logic as tofu_scores.py calculate_info function.
        """
        # Create a data dict that mimics what calculate_info expects
        # (mapping metric names to their aggregate values)
        data = {}
        for metric_name, result in metric_results.items():
            if result and isinstance(result, dict) and "agg_value" in result:
                data[metric_name] = result["agg_value"]

        # Use the exact same logic as tofu_scores.py
        memorization_keys = ["extraction_strength", "exact_memorization",
                            "forget_Q_A_PARA_Prob", "forget_truth_ratio"]
        utility_keys = ["model_utility", "forget_Q_A_gibberish"]
        privacy_keys = ["mia_min_k_plus_plus", "mia_min_k", "mia_loss", "mia_zlib"]

        # Memorization score (harmonic mean of 1 - metric_value)
        try:
            mem_values = [1 - data[key] for key in memorization_keys if key in data and data[key] is not None]
            if mem_values and all(v > 0 for v in mem_values):
                memorization_score = len(mem_values) / sum(1 / v for v in mem_values)
            else:
                memorization_score = None
        except (KeyError, ZeroDivisionError, ValueError):
            memorization_score = None

        # Utility score (harmonic mean, normalized by target)
        try:
            util_values = [data[key] for key in utility_keys if key in data and data[key] is not None]
            if util_values and all(v > 0 for v in util_values) and self.target_util:
                utility_score = (len(util_values) / sum(1 / v for v in util_values)) / self.target_util
            else:
                utility_score = None
        except (KeyError, ZeroDivisionError, ValueError):
            utility_score = None

        # Privacy score (harmonic mean of 1 - |difference from retrain|)
        try:
            priv_values = []
            for key in privacy_keys:
                if key in data and data[key] is not None and key in self.retrain_priv_scores:
                    val = data[key]
                    retrain_val = self.retrain_priv_scores[key]
                    rel_diff = 1 - abs(val - retrain_val)
                    if rel_diff > 0:
                        priv_values.append(rel_diff)

            if priv_values and all(v > 0 for v in priv_values):
                privacy_score = len(priv_values) / sum(1 / v for v in priv_values)
            else:
                privacy_score = None
        except (KeyError, ZeroDivisionError, ValueError):
            privacy_score = None

        # Aggregate without privacy (harmonic mean of memorization and utility)
        try:
            if memorization_score and utility_score and memorization_score > 0 and utility_score > 0:
                agg_wo_privacy = 2 / ((1 / memorization_score) + (1 / utility_score))
            else:
                agg_wo_privacy = None
        except (ZeroDivisionError, ValueError):
            agg_wo_privacy = None

        # Aggregate with privacy (harmonic mean of memorization, privacy, and utility)
        try:
            if (memorization_score and privacy_score and utility_score and
                memorization_score > 0 and privacy_score > 0 and utility_score > 0):
                agg = 3 / ((1 / memorization_score) + (1 / privacy_score) + (1 / utility_score))
            else:
                agg = None
        except (ZeroDivisionError, ValueError):
            agg = None

        return {
            "memorization_score": memorization_score,
            "privacy_score": privacy_score,
            "utility_score": utility_score,
            "agg_wo_privacy": agg_wo_privacy,
            "agg": agg
        }

    def bootstrap(self, n_samples: int = 1000, alpha: float = 0.01,
                  seed: int = 42) -> Dict[str, Tuple[float, float, float]]:
        """
        Perform bootstrap resampling and compute confidence intervals.

        MIA metrics are now properly bootstrapped by resampling forget/holdout sets
        and recomputing AUC, so all 5 key metrics have full bootstrap CIs.

        Args:
            n_samples: Number of bootstrap samples
            alpha: Significance level (default 0.01 for 99% CI)
            seed: Initial random seed

        Returns:
            Dict mapping metric names to confidence interval statistics
        """
        bootstrap_results = []

        print(f"Performing {n_samples} bootstrap iterations...")
        for i in tqdm(range(n_samples)):
            # Perform one bootstrap iteration (includes MIA resampling)
            metric_results = self.bootstrap_iteration(seed + i)

            # Compute aggregate scores (includes privacy and agg)
            agg_scores = self.compute_aggregate_scores(metric_results)

            # Store results
            bootstrap_results.append({**metric_results, **agg_scores})

        # Compute confidence intervals
        confidence_intervals = {}

        # Get all metric names
        all_metric_names = set()
        for result in bootstrap_results:
            all_metric_names.update(result.keys())

        for metric_name in all_metric_names:
            # Extract values for this metric across all bootstrap samples
            values = []
            for result in bootstrap_results:
                if metric_name in result:
                    if isinstance(result[metric_name], dict):
                        val = result[metric_name].get("agg_value")
                    else:
                        val = result[metric_name]

                    if val is not None and not np.isnan(val):
                        values.append(val)

            if len(values) > 0:
                values = np.array(values)
                mean = np.mean(values)

                # Compute CI using percentile method
                ci_lower = np.percentile(values, 100 * alpha / 2)
                ci_upper = np.percentile(values, 100 * (1 - alpha / 2))

                confidence_intervals[metric_name] = {
                    "mean": mean,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "ci_half_width": (ci_upper - ci_lower) / 2
                }

        return confidence_intervals


def main():
    """Example usage of the bootstrapper."""
    # Paths
    eval_json = "example_TOFU_EVAL.json"
    config_yaml = "tofu_config.yaml"
    retrain_summary = "saves/eval/tofu/baselines/retrain/TOFU_SUMMARY.json"
    target_summary = "saves/eval/tofu/baselines/target/TOFU_SUMMARY.json"

    # Create bootstrapper
    bootstrapper = TOFUBootstrapper(
        eval_json_path=eval_json,
        config_yaml_path=config_yaml,
        retrain_summary_path=retrain_summary,
        target_summary_path=target_summary
    )

    # Run bootstrap
    results = bootstrapper.bootstrap(n_samples=1000, alpha=0.01)

    # Print results
    print("\n" + "="*80)
    print("Bootstrap Confidence Intervals (99%)")
    print("="*80)

    # Order: aggregate scores first, then individual metrics
    score_order = [
        "agg", "agg_wo_privacy", "memorization_score", "privacy_score", "utility_score",
        "exact_memorization", "extraction_strength", "forget_Q_A_PARA_Prob", "forget_truth_ratio",
        "model_utility", "forget_Q_A_gibberish",
        "mia_min_k_plus_plus", "mia_min_k", "mia_loss", "mia_zlib"
    ]

    for metric_name in score_order:
        if metric_name in results:
            r = results[metric_name]
            print(f"{metric_name:30s}: {r['mean']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}] "
                  f"(±{r['ci_half_width']:.4f})")

    # Save results
    output_file = "bootstrap_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
