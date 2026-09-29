"""Downstream task: does the protected signal still support sleep
stage classification, not just raw fidelity to the ideal circuit.

THIS IS THE SINGLE AUTHORITATIVE VERSION of this file, combining
everything built across this project: wake-period window expansion,
group-aware epoch splitting, the raw-output sanity check, and saving
predictions to outputs/downstream_predictions_combined.csv for
pooling across subjects and seeds. Do not let another tool or session
rebuild this file from a partial view, it has silently dropped pieces
of this before.

Design:
  - Logistic regression, matching the precedent set in the earlier
    QGAN paper.
  - Trained ONCE on ideal circuit outputs from the training windows,
    then frozen and evaluated identically against every method's test
    outputs.
  - Window budget starts at --max_windows AFTER skipping the leading
    unlabeled/'L' prefix, and auto-expands if that budget contains
    fewer than 2 distinct sleep stages (some subjects, e.g. EPCTL04,
    stay awake far longer after lights-off than others).
  - GROUP AWARE split (src/data_split.py), not the plain window level
    split used by the fidelity experiments, so no 30-second epoch is
    split across train and test.
  - 'L' labeled epochs are dropped entirely.
  - Every run's raw predictions are appended to
    outputs/downstream_predictions_combined.csv, so
    analyze_downstream_classification.py can pool across every
    subject and seed, this is what run_downstream_study.py's
    resume/skip logic checks against too.

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
from src.hypnogram import load_hypnogram, epochs_to_window_labels, find_diverse_window_range
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

    print(f"Loading hypnogram from {args.hypnogram_path} ...")
    epochs = load_hypnogram(args.hypnogram_path)
    offset_windows, search_windows = find_diverse_window_range(
        epochs, window_sec=int(args.window_sec), min_windows=args.max_windows,
        drop_labels=("L",), min_classes=2,
    )
    print(
        f"First usable (non-L) epoch starts at window {offset_windows} "
        f"({offset_windows * args.window_sec:.0f}s into the recording)."
    )
    if search_windows > args.max_windows:
        print(
            f"First {args.max_windows} windows after that were all one sleep stage "
            f"(e.g. still awake) -- expanded to {search_windows} windows to get "
            f"at least 2 distinct stages for the classifier to learn from."
        )

    windows = preprocess_and_segment(signal, sfreq, window_sec=args.window_sec)
    windows = windows[offset_windows : offset_windows + search_windows]
    raw_features = extract_feature_matrix(windows, sfreq)
    encode_features_all = normalize_features(raw_features)

    window_label, epoch_groups = epochs_to_window_labels(
        epochs,
        window_sec=int(args.window_sec),
        max_windows=len(raw_features),
        drop_labels=("L",),
        offset_windows=offset_windows,
    )
    print(
        f"{len(epoch_groups)} complete, labeled epochs covering {len(window_label)} windows "
        f"('L' epochs and any epoch not fully inside the window budget are dropped)"
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

    # Sanity check: confirm methods that might produce identical
    # classification results are NOT producing literally identical raw
    # outputs, which would indicate a real bug rather than the benign
    # explanation that small residual differences do not cross the
    # classifier's decision boundary.
    for pair in [("ideal", "no_qec"), ("uniform_repetition_d3", "application_aware")]:
        a, b = conditions[pair[0]], conditions[pair[1]]
        if np.allclose(a, b, atol=1e-9):
            print(
                f"\nWARNING: {pair[0]} and {pair[1]} raw outputs are numerically identical, "
                "not just producing the same predicted labels. This suggests a real bug, "
                "investigate before trusting any result from this run."
            )
        else:
            max_diff = np.max(np.abs(a - b))
            print(
                f"Sanity check: {pair[0]} vs {pair[1]} raw outputs differ (max abs diff "
                f"{max_diff:.6f}), any identical classification result between them is the "
                "classifier's decision boundary, not identical underlying data."
            )

    all_labels = sorted(set(y_train) | set(y_test))
    rows = []
    prediction_rows = []
    subject_label = Path(args.edf_path).stem

    for name, X_test in conditions.items():
        y_pred = clf.predict(X_test)

        for true_label, pred_label, w in zip(y_test, y_pred, test_idx):
            prediction_rows.append({
                "subject": subject_label,
                "seed": args.seed,
                "method": name,
                "window": int(w),
                "y_true": true_label,
                "y_pred": pred_label,
            })

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, labels=all_labels, average="macro", zero_division=0)
        per_class_f1 = f1_score(y_test, y_pred, labels=all_labels, average=None, zero_division=0)
        f1_by_class = dict(zip(all_labels, per_class_f1))
        n1_f1 = f1_by_class.get("N1", float("nan"))
        rows.append({
            "method": name, "accuracy": acc, "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1, "n1_f1": n1_f1,
        })
        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred, labels=all_labels, zero_division=0))

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "downstream_classification_results.csv", index=False)
    print(f"\nSaved {out_dir / 'downstream_classification_results.csv'}")
    print(table.to_string(index=False))

    predictions_path = out_dir / "downstream_predictions_combined.csv"
    new_predictions = pd.DataFrame(prediction_rows)
    if predictions_path.exists():
        existing = pd.read_csv(predictions_path)
        combined_predictions = pd.concat([existing, new_predictions], ignore_index=True)
    else:
        combined_predictions = new_predictions
    combined_predictions.to_csv(predictions_path, index=False)
    print(f"Appended this run's predictions to {predictions_path} ({len(combined_predictions)} rows total)")

    print(
        "\nHeadline metric is n1_f1 (minority class F1), consistent with the prior "
        "QGAN paper's framing. macro_f1 and balanced_accuracy are secondary context, "
        "raw accuracy is included only because reviewers expect to see it, not because "
        "it is meaningful on its own given the class imbalance."
    )


if __name__ == "__main__":
    main()