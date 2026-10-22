"""Statistics for the downstream sleep stage check.

Reads outputs/downstream_results.csv and outputs/downstream_predictions.csv
written by run_downstream.py and prints:
  * mean macro F1, N1 F1 and balanced accuracy per method, over subjects
  * the share of the clean generator performance each method retains
  * subject level paired tests between methods (t test and Wilcoxon)
  * pooled N1 F1 per method with a subject level bootstrap interval

Usage:
  python scripts/analyze_downstream.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
ORDER = ["raw_features", "ideal", "no_qec", "uniform", "rep03", "plan_A", "rule"]
LABELS = {
    "raw_features": "Raw 8 features, no quantum circuit",
    "ideal": "Clean generator output",
    "no_qec": "No QEC",
    "uniform": "Uniform repetition",
    "rep03": "Repetition on qubits 0 and 3",
    "plan_A": "Fixed plan A",
    "rule": "Error aware rule",
}
PAIRS = [("rule", "no_qec"), ("rule", "uniform"), ("rep03", "no_qec"), ("rule", "rep03"), ("ideal", "rule")]


def sem(x):
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def n1_counts(df):
    tp = int(((df["true"] == "N1") & (df["pred"] == "N1")).sum())
    fp = int(((df["true"] != "N1") & (df["pred"] == "N1")).sum())
    fn = int(((df["true"] == "N1") & (df["pred"] != "N1")).sum())
    return tp, fp, fn


def f1_from(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0


def main():
    res = pd.read_csv(OUT_DIR / "downstream_results.csv")
    pred = pd.read_csv(OUT_DIR / "downstream_predictions.csv")
    methods = [m for m in ORDER if m in set(res["method"])]
    sub = res.groupby(["method", "subject"])[["macro_f1", "n1_f1", "balanced_acc"]].mean().reset_index()
    subjects = sorted(sub["subject"].unique())
    print(f"Subjects: {len(subjects)}, windows per subject test set: {int(res['n_test'].median())}, chance macro F1 about 0.20\n")

    ideal = sub[sub["method"] == "ideal"].set_index("subject")["macro_f1"]
    print(f"{'method':38s} {'macro F1':>14s} {'N1 F1':>14s} {'balanced acc':>14s} {'retained':>9s}")
    for m in methods:
        g = sub[sub["method"] == m].set_index("subject")
        kept = (g["macro_f1"] / ideal).mean() if m != "raw_features" else float("nan")
        print(f"{LABELS[m]:38s} {g['macro_f1'].mean():7.3f}+-{sem(g['macro_f1']):.3f} "
              f"{g['n1_f1'].mean():7.3f}+-{sem(g['n1_f1']):.3f} "
              f"{g['balanced_acc'].mean():7.3f}+-{sem(g['balanced_acc']):.3f} "
              f"{kept:9.2f}")

    print("\nSubject level paired tests, first method minus second")
    for a, b in PAIRS:
        if a not in methods or b not in methods:
            continue
        for col, name in [("macro_f1", "macro F1"), ("n1_f1", "N1 F1")]:
            x = sub[sub["method"] == a].set_index("subject")[col]
            y = sub[sub["method"] == b].set_index("subject")[col]
            d = x - y
            t = stats.ttest_rel(x, y).pvalue if len(d) > 1 else float("nan")
            try:
                w = stats.wilcoxon(x, y).pvalue
            except ValueError:
                w = float("nan")
            print(f"  {LABELS[a]:30s} minus {LABELS[b]:30s} {name:9s} mean diff {d.mean():+.3f}  "
                  f"better in {int((d > 0).sum())}/{len(d)}  t p={t:.4f}  Wilcoxon p={w:.4f}")

    rng = np.random.default_rng(0)
    counts = {m: {s: n1_counts(pred[(pred['method'] == m) & (pred['subject'] == s)]) for s in subjects} for m in methods}
    print("\nPooled N1 F1 over all test windows, 95% subject level bootstrap interval")
    boots = {m: [] for m in methods}
    for _ in range(2000):
        pick = rng.choice(subjects, size=len(subjects), replace=True)
        for m in methods:
            tp = sum(counts[m][s][0] for s in pick)
            fp = sum(counts[m][s][1] for s in pick)
            fn = sum(counts[m][s][2] for s in pick)
            boots[m].append(f1_from(tp, fp, fn))
    for m in methods:
        tp = sum(c[0] for c in counts[m].values()); fp = sum(c[1] for c in counts[m].values()); fn = sum(c[2] for c in counts[m].values())
        lo, hi = np.percentile(boots[m], [2.5, 97.5])
        print(f"  {LABELS[m]:38s} {f1_from(tp, fp, fn):.3f}  [{lo:.3f}, {hi:.3f}]")
    if "rule" in methods and "no_qec" in methods:
        diff = np.array(boots["rule"]) - np.array(boots["no_qec"])
        lo, hi = np.percentile(diff, [2.5, 97.5])
        print(f"  rule minus no QEC, pooled N1 F1: [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()