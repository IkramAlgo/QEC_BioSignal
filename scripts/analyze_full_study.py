"""Computes subject-level paired statistics directly from
outputs/full_study_results.csv, the authoritative source, rather than
reading percentages back out of the console log, which is easy to
misread across fifteen runs worth of scrolled output.

Usage:
  python scripts/analyze_full_study.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMBINED_PATH = PROJECT_ROOT / "outputs" / "full_study_results.csv"


def subject_level_means(df, method):
    sub = df[df["method"] == method]
    return sub.groupby("subject")[["mse_vs_ideal", "wasserstein_vs_ideal"]].mean()


def paired_report(a_df, b_df, label_a, label_b, metric):
    merged = a_df[[metric]].join(b_df[[metric]], lsuffix="_a", rsuffix="_b")
    a = merged[f"{metric}_a"].values
    b = merged[f"{metric}_b"].values
    diff = a - b
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    direction = "all same sign" if (np.all(diff > 0) or np.all(diff < 0)) else "mixed signs"
    print(f"  {label_a} vs {label_b}, {metric} (n={len(a)} subjects): mean diff = {diff.mean():.6f} ({direction})")
    print(f"    paired t-test: p={t_p:.4f}   Wilcoxon: p={w_p:.4f}")


def main():
    if not COMBINED_PATH.exists():
        print(f"{COMBINED_PATH} not found, run run_full_study.py first.")
        return

    df = pd.read_csv(COMBINED_PATH)
    print(f"Loaded {len(df)} rows from {COMBINED_PATH}\n")

    methods = ["no_qec", "uniform_repetition_d3", "application_aware", "classical_averaged_50_shots"]
    means = {m: subject_level_means(df, m) for m in methods}

    for metric in ["mse_vs_ideal", "wasserstein_vs_ideal"]:
        print(f"=== {metric} ===")
        paired_report(means["application_aware"], means["no_qec"], "app_aware", "no_qec", metric)
        paired_report(means["application_aware"], means["uniform_repetition_d3"], "app_aware", "uniform", metric)
        paired_report(means["application_aware"], means["classical_averaged_50_shots"], "app_aware", "classical", metric)
        paired_report(means["classical_averaged_50_shots"], means["no_qec"], "classical", "no_qec", metric)
        paired_report(means["classical_averaged_50_shots"], means["uniform_repetition_d3"], "classical", "uniform", metric)
        print()


if __name__ == "__main__":
    main()