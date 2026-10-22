"""Per logical qubit noise sensitivity ranking.

UPDATED: the ranking is now measured on the TRAINING windows by default
(--ranking_split train), not on the held out test windows. Previously the
tier allocation was chosen using the same 20 test windows that the final
application aware score was then computed on, which lets the allocation
tune itself to the evaluation data. With the default below, the test
windows are never touched until the final evaluation in
run_application_aware_experiment.py.

Use --ranking_split test only to reproduce the old behaviour.

Uses the same --max_windows, --test_fraction, and --seed as
train_generator.py so the train and test split is identical everywhere.

Usage:
  python scripts/run_sensitivity_ranking.py data/raw/YOUR_FILE.edf --channel "C3" --max_windows 100 --weights_path outputs\\trained_weights.npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_eeg_channel
from src.preprocessing import preprocess_and_segment
from src.features import extract_feature_matrix
from src.quantum_model import make_ideal_qnode, random_weights, normalize_features
from src.repetition_code import generate_storage_no_qec_batch
from src.data_split import train_test_split_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("edf_path")
    parser.add_argument("--channel", required=True)
    parser.add_argument("--window_sec", type=float, default=10.0)
    parser.add_argument("--max_windows", type=int, default=200)
    parser.add_argument("--noise_prob", type=float, default=0.02)
    parser.add_argument("--n_shots", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument(
        "--ranking_split",
        default="train",
        choices=["train", "test"],
        help="Which windows the ranking is measured on. Default train, so the evaluation "
        "windows are never used to choose the tier allocation.",
    )
    parser.add_argument(
        "--weights_path",
        default=None,
        help="Path to trained weights .npy (from train_generator.py). Falls back to random seed-0 weights if omitted.",
    )
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    print(f"Loading channel '{args.channel}' from {args.edf_path} ...")
    signal, sfreq = load_eeg_channel(args.edf_path, args.channel)

    windows = preprocess_and_segment(signal, sfreq, window_sec=args.window_sec)
    windows = windows[: args.max_windows]
    print(f"{len(windows)} windows total")

    raw_features = extract_feature_matrix(windows, sfreq)
    all_features = normalize_features(raw_features)

    train_idx, test_idx = train_test_split_indices(
        len(raw_features), test_fraction=args.test_fraction, seed=args.seed
    )
    if args.ranking_split == "train":
        rank_idx = train_idx
        print(f"Ranking measured on {len(rank_idx)} TRAINING windows (test windows untouched)")
    else:
        rank_idx = test_idx
        print(f"Ranking measured on {len(rank_idx)} HELD OUT test windows (old behaviour, leaks into evaluation)")
    features = all_features[rank_idx]

    if args.weights_path:
        weights = np.load(args.weights_path)
        print(f"Loaded trained weights from {args.weights_path}")
    else:
        weights = random_weights(seed=0)
        print("No --weights_path given, using random seed-0 weights (untrained baseline)")

    print("Running ideal circuit ...")
    ideal_qnode = make_ideal_qnode()
    ideal_outputs = np.array([ideal_qnode(f, weights) for f in features])

    print(f"Running noisy circuit, no QEC, depolarizing p={args.noise_prob} ...")
    noisy_outputs = generate_storage_no_qec_batch(
        features, weights, args.noise_prob, "depolarizing", args.n_shots
    )

    n_logical = ideal_outputs.shape[1] // 2

    rows = []
    for lq in range(n_logical):
        z_col = lq
        x_col = lq + n_logical
        z_mse = np.mean((ideal_outputs[:, z_col] - noisy_outputs[:, z_col]) ** 2)
        x_mse = np.mean((ideal_outputs[:, x_col] - noisy_outputs[:, x_col]) ** 2)
        combined_mse = (z_mse + x_mse) / 2
        rows.append(
            {
                "logical_qubit": lq,
                "feature_pair": f"features[{2*lq}], features[{2*lq+1}]",
                "z_mse": z_mse,
                "x_mse": x_mse,
                "combined_mse": combined_mse,
            }
        )

    table = pd.DataFrame(rows).sort_values("combined_mse", ascending=False).reset_index(drop=True)
    table.insert(0, "sensitivity_rank", range(1, len(table) + 1))

    table.to_csv(out_dir / "sensitivity_ranking.csv", index=False)
    print(f"\nSaved {out_dir / 'sensitivity_ranking.csv'}")
    print(table.to_string(index=False))
    print(
        "\nRank 1 is the most noise sensitive logical qubit, give it the heaviest "
        f"protection. This ranking was measured on {args.ranking_split} windows only."
    )


if __name__ == "__main__":
    main()