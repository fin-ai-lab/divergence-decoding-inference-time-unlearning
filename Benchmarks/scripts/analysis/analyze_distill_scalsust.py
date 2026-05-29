#!/usr/bin/env python3
"""
Analyze MUSE distillation results for scaling & sustainability experiments.

Shows distance-to-retrain for DD teacher and distilled student, normalized
by target's distance to retrain (lower = closer to retrain = better).

Usage: python analyze_distill_scalsust.py
"""

import json
from pathlib import Path
from tabulate import tabulate

# Scaling and sustainability evals are split into separate trees; base_step1
# (the shared Step 1) lives under scaling/. See scripts/scaling/muse_distill_scalsust.py.
SCALING_DIR = Path("saves/eval/muse/scaling")
SUST_DIR = Path("saves/eval/muse/sustainability")

# Retrain and target reference baselines.
BASELINE_DIR = Path("saves/eval/muse/baselines")

SCALING_STEPS = [
    ("Step 1", "base_step1"),
    ("Step 2", "scal_step2"),
    ("Step 3", "scal_step3"),
    ("Step 4", "scal_step4"),
]

SUSTAINABILITY_STEPS = [
    ("Step 1", "base_step1"),
    ("Step 2", "sust_step2"),
    ("Step 3", "sust_step3"),
    ("Step 4", "sust_step4"),
]


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def eval_dir_for(config_name):
    """Scaling vs sustainability tree for a config (base_step1 -> scaling)."""
    return SUST_DIR if config_name.startswith("sust_") else SCALING_DIR


def load_summary(prefix, config_name):
    """Load a {dd,distill}_<config> summary from the right tree."""
    subdir = f"{prefix}_{config_name}"
    return load_json(eval_dir_for(config_name) / subdir / "MUSE_SUMMARY.json")


def euclidean_distance(point, retrain, forget_key):
    """Euclidean distance from retrain in (forget_metric, retain) space. Values in 0-100."""
    fd = point[forget_key] * 100 - retrain[forget_key] * 100
    rd = point["retain_knowmem_ROUGE"] * 100 - retrain["retain_knowmem_ROUGE"] * 100
    return (fd**2 + rd**2) ** 0.5


def main():
    retrain = load_json(BASELINE_DIR / "retrain" / "MUSE_SUMMARY.json")
    target = load_json(BASELINE_DIR / "target" / "MUSE_SUMMARY.json")
    if not retrain or not target:
        print(f"ERROR: Missing retrain/target baselines in {BASELINE_DIR}/")
        return

    # Target distances (used for normalization)
    target_dist_vm = euclidean_distance(target, retrain, "forget_verbmem_ROUGE")
    target_dist_km = euclidean_distance(target, retrain, "forget_knowmem_ROUGE")

    for title, steps in [
        ("SCALING (increasing forget data)", SCALING_STEPS),
        ("SUSTAINABILITY (accumulated forget data)", SUSTAINABILITY_STEPS),
    ]:
        headers = ["Step", "Verbmem", "Knowmem"]
        rows = []
        for label, config_name in steps:
            dd = load_summary("dd", config_name)
            dist = load_summary("distill", config_name)
            row = [label]
            for forget_key, target_d in [
                ("forget_verbmem_ROUGE", target_dist_vm),
                ("forget_knowmem_ROUGE", target_dist_km),
            ]:
                if dd and dist:
                    dd_d = euclidean_distance(dd, retrain, forget_key)
                    dist_d = euclidean_distance(dist, retrain, forget_key)
                    extra_pct = (dist_d - dd_d) / target_d * 100
                    row.append(f"{extra_pct:+.1f}%")
                else:
                    row.append("—")
            rows.append(row)

        print(f"\n### {title}")
        print(f"*Extra distance from DD (as % of target distance); 0 = same as DD, negative = closer to retrain.*\n")
        print(tabulate(rows, headers=headers, tablefmt="pipe", stralign="right"))


if __name__ == "__main__":
    main()
