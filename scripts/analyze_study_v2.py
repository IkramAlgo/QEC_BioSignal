"""Statistics for run_study_v2.py: reconstruction, fidelity and sleep staging.

Usage:
  python scripts/analyze_study_v2.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
FID_ORDER = ["no_qec", "classical", "uniform", "rep03", "plan_A", "rule"]
ALL_ORDER = ["raw_features", "ideal"] + FID_ORDER
LABELS = {
    "raw_features": "Raw 8 features, no circuit", "ideal": "Clean generator output", "no_qec": "No QEC",
    "classical": "Classical 50 shot average", "uniform": "Uniform repetition",
    "rep03": "Repetition on qubits 0 and 3", "plan_A": "Fixed plan A", "rule": "Error aware rule",
}
FID_PAIRS = [("rule", "no_qec"), ("rule", "classical"), ("rule", "uniform"), ("rep03", "no_qec"),
             ("rep03", "classical"), ("rule", "rep03")]
F1_PAIRS = [("rule", "no_qec"), ("rule", "uniform"), ("rule", "rep03"), ("rep03", "no_qec"),
            ("classical", "no_qec"), ("ideal", "rule")]


def sem(x):
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def paired(x, y):
    d = x - y
    t = stats.ttest_rel(x, y).pvalue if len(d) > 1 else float("nan")
    try:
        w = stats.wilcoxon(x, y).pvalue
    except ValueError:
        w = float("nan")
    return d, t, w


def n1_counts(df):
    tp = int(((df["true"] == "N1") & (df["pred"] == "N1")).sum())
    fp = int(((df["true"] != "N1") & (df["pred"] == "N1")).sum())
    fn = int(((df["true"] == "N1") & (df["pred"] != "N1")).sum())
    return tp, fp, fn


def f1_from(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0


def main():
    res = pd.read_csv(OUT_DIR / "v2_results.csv")
    rec = pd.read_csv(OUT_DIR / "v2_reconstruction.csv")
    pred = pd.read_csv(OUT_DIR / "v2_predictions.csv")
    methods_f = [m for m in FID_ORDER if m in set(res["method"])]
    methods_a = [m for m in ALL_ORDER if m in set(res["method"])]
    sub = res.groupby(["method", "subject"])[["mse_vs_ideal", "wasserstein_vs_ideal", "macro_f1", "n1_f1", "balanced_acc"]].mean().reset_index()
    subjects = sorted(sub["subject"].unique())
    print(f"Subjects: {len(subjects)}, test windows per run about {int(res['n_test'].median())}\n")

    print("=== Reconstruction of the generator target, held out windows, mean over subjects ===")
    r = rec.groupby("subject")[["test_mse", "predict_mean_mse", "r2", "corr"]].mean()
    print(f"  error per output {r['test_mse'].mean():.4f} against {r['predict_mean_mse'].mean():.4f} for predicting the mean, "
          f"R2 {r['r2'].mean():+.2f}, correlation {r['corr'].mean():.2f}")
    print(f"  subjects with R2 above zero: {int((r['r2'] > 0).sum())} of {len(r)}\n")

    print("=== Fidelity to the clean generator output (lower is better), mean and standard error over subjects ===")
    ref = sub[sub["method"] == "no_qec"].set_index("subject")
    print(f"{'method':32s} {'MSE':>20s} {'Wasserstein':>20s} {'MSE vs no QEC':>14s} {'W vs no QEC':>12s}")
    for m in methods_f:
        g = sub[sub["method"] == m].set_index("subject")
        print(f"{LABELS[m]:32s} {g['mse_vs_ideal'].mean():11.6f}+-{sem(g['mse_vs_ideal']):.6f} "
              f"{g['wasserstein_vs_ideal'].mean():11.5f}+-{sem(g['wasserstein_vs_ideal']):.5f} "
              f"{100 * (g['mse_vs_ideal'].mean() / ref['mse_vs_ideal'].mean() - 1):+13.0f}% "
              f"{100 * (g['wasserstein_vs_ideal'].mean() / ref['wasserstein_vs_ideal'].mean() - 1):+11.0f}%")

    print("\nSubject level paired tests on fidelity, first minus second (negative is better for the first)")
    for a, b in FID_PAIRS:
        if a not in methods_f or b not in methods_f:
            continue
        for col, name in [("mse_vs_ideal", "MSE"), ("wasserstein_vs_ideal", "Wasserstein")]:
            x = sub[sub["method"] == a].set_index("subject")[col]
            y = sub[sub["method"] == b].set_index("subject")[col]
            d, t, w = paired(x, y)
            print(f"  {LABELS[a]:30s} vs {LABELS[b]:30s} {name:11s} {100 * d.mean() / y.mean():+6.1f}%  "
                  f"better in {int((d < 0).sum())}/{len(d)}  t p={t:.2e}  Wilcoxon p={w:.4f}")

    print("\n=== Sleep staging with the frozen classifier (higher is better) ===")
    ideal = sub[sub["method"] == "ideal"].set_index("subject")["macro_f1"]
    print(f"{'method':32s} {'macro F1':>14s} {'N1 F1':>14s} {'balanced acc':>14s} {'retained':>9s}")
    for m in methods_a:
        g = sub[sub["method"] == m].set_index("subject")
        kept = (g["macro_f1"] / ideal).mean() if m != "raw_features" else float("nan")
        print(f"{LABELS[m]:32s} {g['macro_f1'].mean():7.3f}+-{sem(g['macro_f1']):.3f} "
              f"{g['n1_f1'].mean():7.3f}+-{sem(g['n1_f1']):.3f} "
              f"{g['balanced_acc'].mean():7.3f}+-{sem(g['balanced_acc']):.3f} {kept:9.2f}")

    print("\nSubject level paired tests on macro F1, first minus second")
    for a, b in F1_PAIRS:
        if a not in methods_a or b not in methods_a:
            continue
        x = sub[sub["method"] == a].set_index("subject")["macro_f1"]
        y = sub[sub["method"] == b].set_index("subject")["macro_f1"]
        d, t, w = paired(x, y)
        print(f"  {LABELS[a]:30s} minus {LABELS[b]:30s} {d.mean():+.3f}  better in {int((d > 0).sum())}/{len(d)}  "
              f"t p={t:.4f}  Wilcoxon p={w:.4f}")

    rng = np.random.default_rng(0)
    counts = {m: {s: n1_counts(pred[(pred["method"] == m) & (pred["subject"] == s)]) for s in subjects} for m in methods_a}
    boots = {m: [] for m in methods_a}
    for _ in range(2000):
        pick = rng.choice(subjects, size=len(subjects), replace=True)
        for m in methods_a:
            boots[m].append(f1_from(*[sum(counts[m][s][i] for s in pick) for i in range(3)]))
    print("\nPooled N1 F1, 95% subject level bootstrap interval")
    for m in methods_a:
        tot = [sum(c[i] for c in counts[m].values()) for i in range(3)]
        lo, hi = np.percentile(boots[m], [2.5, 97.5])
        print(f"  {LABELS[m]:32s} {f1_from(*tot):.3f}  [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()