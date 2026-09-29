"""Computes downstream classification metrics from
outputs/downstream_predictions_combined.csv.

PRIMARY metric: subject-level averages. Each subject's own metrics
are computed first (across its 3 seeds), then averaged across the 5
subjects. This treats each subject as one independent piece of
evidence, consistent with how every other statistic in this project
(the paired significance tests on MSE and Wasserstein) already works.
It also avoids one subject with a larger test set (e.g. EPCTL04,
which needed its window budget expanded to 400) silently dominating
the result.

SECONDARY view: raw pooled predictions across everything, combined
into one big test set. Kept for reference, but not the headline
number, since subjects here have very different label diversity and
class balance (EPCTL04 is 384 wake windows against 15 N1 windows;
EPCTL03 and EPCTL05 never see more than 2-3 stages at all in their
usable range), so a raw pool is not really comparing like with like.

Usage:
  python scripts/analyze_downstream_classification.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "downstream_predictions_combined.csv"


def compute_metrics(y_true, y_pred, all_labels):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=all_labels, average=None, zero_division=0)
    f1_by_class = dict(zip(all_labels, per_class_f1))
    n1_f1 = f1_by_class.get("N1", float("nan"))
    return acc, bal_acc, macro_f1, n1_f1


def main():
    if not PREDICTIONS_PATH.exists():
        print(f"{PREDICTIONS_PATH} not found, run run_downstream_classification.py first.")
        return

    df = pd.read_csv(PREDICTIONS_PATH)
    subjects = sorted(df["subject"].unique())
    seeds = sorted(df["seed"].unique())
    methods = df["method"].unique()
    print(f"Loaded {len(df)} predictions across {len(subjects)} subjects and {len(seeds)} seeds\n")

    all_labels = sorted(set(df["y_true"]) | set(df["y_pred"]))

    # ---------- PRIMARY: subject-level averages ----------
    print("=" * 70)
    print("PRIMARY RESULT: subject-level averages (each subject = 1 data point)")
    print("=" * 70)

    per_subject_rows = []
    for subject in subjects:
        for seed in seeds:
            sub = df[(df["subject"] == subject) & (df["seed"] == seed)]
            if len(sub) == 0:
                continue
            for method in methods:
                m = sub[sub["method"] == method]
                if len(m) == 0:
                    continue
                acc, bal_acc, macro_f1, n1_f1 = compute_metrics(m["y_true"], m["y_pred"], all_labels)
                per_subject_rows.append({
                    "subject": subject, "seed": seed, "method": method,
                    "n_test_windows": len(m), "accuracy": acc,
                    "balanced_accuracy": bal_acc, "macro_f1": macro_f1, "n1_f1": n1_f1,
                })
    per_run_df = pd.DataFrame(per_subject_rows)
    per_run_df.to_csv(PROJECT_ROOT / "outputs" / "downstream_classification_per_run.csv", index=False)

    # average over seeds first (per subject), then average across subjects
    subject_means = per_run_df.groupby(["subject", "method"])[["accuracy", "balanced_accuracy", "macro_f1", "n1_f1"]].mean()
    primary_summary = subject_means.groupby("method").agg(["mean", "std"])
    print(primary_summary.to_string())
    primary_summary.to_csv(PROJECT_ROOT / "outputs" / "downstream_classification_subject_level.csv")
    print(f"\nSaved outputs/downstream_classification_subject_level.csv (this is the number to report)")
    print(f"Saved outputs/downstream_classification_per_run.csv (per subject, per seed detail)")

    # ---------- SECONDARY: raw pooled predictions ----------
    print("\n" + "=" * 70)
    print("SECONDARY VIEW: raw pooled predictions (subjects with more test windows dominate)")
    print("=" * 70)
    rows = []
    for method in methods:
        sub = df[df["method"] == method]
        y_true = sub["y_true"].values
        y_pred = sub["y_pred"].values
        acc, bal_acc, macro_f1, n1_f1 = compute_metrics(y_true, y_pred, all_labels)
        rows.append({
            "method": method, "n_pooled_windows": len(sub),
            "accuracy": acc, "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1, "n1_f1": n1_f1,
        })
        print(f"\n=== {method} (n={len(sub)} pooled test windows, includes every subject's raw predictions) ===")
        print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))

    pooled_table = pd.DataFrame(rows).sort_values("n1_f1", ascending=False)
    pooled_table.to_csv(PROJECT_ROOT / "outputs" / "downstream_classification_pooled.csv", index=False)
    print(f"\nSaved outputs/downstream_classification_pooled.csv (secondary, do not lead with this)")
    print(pooled_table.to_string(index=False))


if __name__ == "__main__":
    main()