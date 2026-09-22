"""Downstream task: does the protected signal still support sleep
stage classification, not just raw fidelity to the ideal circuit.

Design:
  - Logistic regression, matching the precedent set in the earlier
    QGAN paper. This experiment tests whether the signal preserves
    classifiable structure, not whether a stronger classifier can be
    built.
  - Trained ONCE on ideal circuit outputs from the training windows,
    then frozen and evaluated identically against every method's test
    outputs, no_qec, uniform_repetition_d3, classical_averaged_50_shots,
    application_aware. This is the realistic scenario: a classifier
    trained on clean data, deployed against whatever signal it
    actually receives.
  - Uses a GROUP AWARE split (see src/data_split.py), NOT the plain
    window level split used by the fidelity experiments elsewhere in
    this project. A hypnogram epoch spans 3 windows; splitting by
    window instead of by epoch would let near duplicate signal leak
    between train and test.
  - 'L' labeled epochs (lights off / movement) are dropped entirely.
  - Reports accuracy and balanced accuracy for context, macro F1
    across all stages, and N1 F1 as the headline metric, consistent
    with the minority class framing already established in the prior
    QGAN paper.

Usage:
  python scripts/run_downstream_classification.py data/raw/EPCTL01.edf data/raw/EPCTL01.txt --channel "C3" --max_windows 100 --seed 0 --weights_path outputs/trained_weights.npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_eeg_channel
from src.preprocessing import preprocess_and_segment
from src.features import extract_feature_matrix
from src.quantum_model import make_ideal_qnode, normalize_features
from src.repetition_code import generate_storage_no_qec_batch, generate_repetition_qec_batch
from src.application_aware_qec import ApplicationAwareQEC, build_tiers_from_ranking
from src.classical_baseline import generate_classical_averaged_batch
from src.hypnogram import load_hypnogram, epochs_to_window_labels
from src.data_split import group_train_test_split_indices


def normalize_to_range(feature_matrix, lo, hi):
    mins = feature_matrix.min(axis=0, keepdims=True)
    maxs = feature_matrix.max(axis=0, keepdims=True)
    span = np.where(maxs > mins, maxs - mins, 1.0)
    return (feature_matrix - mins) / span * (hi - lo) + lo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("edf_path")
    parser.add_argument("hypnogram_path")
    parser.add_argument("--channel", required=True)
    parser.add_argument("--window_sec", type=float, default=10.0)
    parser.add_argument("--max_windows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--noise_prob", type=float, default=0.02)
    parser.add_argument("--n_shots", type=int, default=50)
    parser.add_argument("--weights_path", required=True)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    print(f"Loading channel '{args.channel}' from {args.edf_path} ...")
    signal, sfreq = load_eeg_channel(args.edf_path, args.channel)
    windows = preprocess_and_segment(signal, sfreq, window_sec=args.window_sec)
    windows = windows[: args.max_windows]
    raw_features = extract_feature_matrix(windows, sfreq)
    encode_features_all = normalize_features(raw_features)

    print(f"Loading hypnogram from {args.hypnogram_path} ...")
    epochs = load_hypnogram(args.hypnogram_path)
    window_label, epoch_groups = epochs_to_window_labels(
        epochs, window_sec=int(args.window_sec), max_windows=len(raw_features), drop_labels=("L",)
    )
    print(
        f"{len(epoch_groups)} complete, labeled epochs covering {len(window_label)} windows "
        f"('L' epochs and any epoch not fully inside max_windows are dropped)"
    )
    label_counts = pd.Series(list(window_label.values())).value_counts()
    print(f"Window label distribution:\n{label_counts.to_string()}")

    train_idx, test_idx = group_train_test_split_indices(
        epoch_groups, test_fraction=args.test_fraction, seed=args.seed
    )
    print(
        f"Epoch-grouped split (a DIFFERENT split from the main QEC fidelity results, "
        f"grouped so no epoch is split across train and test): "
        f"{len(train_idx)} training windows, {len(test_idx)} test windows"
    )

    y_train = np.array([window_label[w] for w in train_idx])
    y_test = np.array([window_label[w] for w in test_idx])

    weights = np.load(args.weights_path)
    ideal_qnode = make_ideal_qnode()

    print("Computing ideal circuit outputs for training windows ...")
    X_train = np.array([ideal_qnode(encode_features_all[w], weights) for w in train_idx])

    print("Training logistic regression classifier on ideal training outputs only ...")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)

    test_features = encode_features_all[test_idx]

    ranking_path = out_dir / "sensitivity_ranking.csv"
    if not ranking_path.exists():
        raise FileNotFoundError(
            f"{ranking_path} not found, run run_sensitivity_ranking.py on this same "
            "subject and seed first."
        )
    ranking_df = pd.read_csv(ranking_path)
    tiers = build_tiers_from_ranking(ranking_df)
    app_aware = ApplicationAwareQEC(tiers)

    conditions = {}
    print("Computing ideal test outputs (reference condition) ...")
    conditions["ideal"] = np.array([ideal_qnode(f, weights) for f in test_features])
    print("Computing no_qec test outputs ...")
    conditions["no_qec"] = generate_storage_no_qec_batch(
        test_features, weights, args.noise_prob, "depolarizing", args.n_shots
    )
    print("Computing uniform_repetition_d3 test outputs ...")
    conditions["uniform_repetition_d3"] = generate_repetition_qec_batch(
        test_features, weights, args.noise_prob, "depolarizing", args.n_shots
    )
    print("Computing application_aware test outputs ...")
    conditions["application_aware"] = app_aware.generate_batch(
        test_features, weights, args.noise_prob, "depolarizing", args.n_shots
    )
    raw_test_features_for_classical = raw_features[test_idx]
    classical_targets = normalize_to_range(raw_test_features_for_classical, -1, 1)
    print("Computing classical_averaged_50_shots test outputs ...")
    conditions["classical_averaged_50_shots"] = generate_classical_averaged_batch(
        classical_targets, args.noise_prob, args.n_shots
    )

    all_labels = sorted(set(y_train) | set(y_test))
    rows = []
    for name, X_test in conditions.items():
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, labels=all_labels, average="macro", zero_division=0)
        per_class_f1 = f1_score(y_test, y_pred, labels=all_labels, average=None, zero_division=0)
        f1_by_class = dict(zip(all_labels, per_class_f1))
        n1_f1 = f1_by_class.get("N1", float("nan"))
        rows.append(
            {
                "method": name,
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "macro_f1": macro_f1,
                "n1_f1": n1_f1,
            }
        )
        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred, labels=all_labels, zero_division=0))

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "downstream_classification_results.csv", index=False)
    print(f"\nSaved {out_dir / 'downstream_classification_results.csv'}")
    print(table.to_string(index=False))
    print(
        "\nHeadline metric is n1_f1 (minority class F1), consistent with the prior "
        "QGAN paper's framing. macro_f1 and balanced_accuracy are secondary context, "
        "raw accuracy is included only because reviewers expect to see it, not because "
        "it is meaningful on its own given the class imbalance."
    )


if __name__ == "__main__":
    main()