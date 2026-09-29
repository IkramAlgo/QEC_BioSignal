"""Decision preservation: how often does each method predict the SAME
label as the ideal (noiseless) circuit on the same test window?

This isolates noise resilience from classifier quality. Accuracy vs the
true stage mixes both, and the ideal circuit itself only reaches about
50% accuracy, so there is little for QEC to preserve on that metric.

Reads outputs/downstream_predictions_combined.csv, no rerun needed.
Usage: python scripts/analyze_decision_agreement.py
"""

from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PATH = PROJECT_ROOT / "outputs" / "downstream_predictions_combined.csv"


def main():
    df = pd.read_csv(PATH)
    key = ["subject", "seed", "window"]
    pred = df.set_index(key + ["method"])["y_pred"].unstack("method")
    methods = [m for m in pred.columns if m != "ideal"]

    agree = pd.DataFrame({m: (pred[m] == pred["ideal"]).astype(float) for m in methods})

    # seeds first, then subjects, so each subject counts once
    per_run = agree.groupby(["subject", "seed"]).mean()
    per_subject = per_run.groupby("subject").mean()

    print("Agreement with ideal predictions, per subject (1.0 = identical decisions)\n")
    print(per_subject.round(4).to_string())
    print("\nAcross subjects (mean, std):")
    print(pd.DataFrame({"mean": per_subject.mean(), "std": per_subject.std()}).round(4).to_string())

    print("\nSubjects where application_aware agrees with ideal MORE than:")
    for other in ["no_qec", "uniform_repetition_d3"]:
        if other in per_subject:
            wins = (per_subject["application_aware"] > per_subject[other]).sum()
            ties = (per_subject["application_aware"] == per_subject[other]).sum()
            print(f"  {other}: {wins} of {len(per_subject)} subjects ({ties} ties)")

    print("\nWindow-level exact McNemar (application_aware vs other), pooled over all runs.")
    print("Caveat: windows are not fully independent (3 windows per epoch, overlapping seeds),")
    print("so treat p-values as indicative, not final.")
    for other in ["no_qec", "uniform_repetition_d3"]:
        if other not in agree:
            continue
        a_only = int(((agree["application_aware"] == 1) & (agree[other] == 0)).sum())
        o_only = int(((agree["application_aware"] == 0) & (agree[other] == 1)).sum())
        n = a_only + o_only
        p = binomtest(a_only, n, 0.5).pvalue if n > 0 else float("nan")
        print(f"  vs {other}: app_aware-only agree {a_only}, {other}-only agree {o_only}, p = {p:.4g}")

    per_subject.to_csv(PROJECT_ROOT / "outputs" / "decision_agreement_per_subject.csv")
    print("\nSaved outputs/decision_agreement_per_subject.csv")


if __name__ == "__main__":
    main()