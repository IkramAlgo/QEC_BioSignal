"""Per logical qubit noise sensitivity ranking.

Runs the plain 4 qubit generator, ideal versus noisy (no QEC at all),
and measures degradation SEPARATELY for each of the 4 logical qubits
instead of one aggregated number. That per qubit ranking is what
decides which logical qubits get heavy protection (Shor, 9 physical
qubits) and which get light or no protection, in the application aware
allocation. This only ever touches the existing 4 qubit circuit, so it
carries none of the memory cost the full Shor run has.

Usage:
  python scripts/run_sensitivity_ranking.py data/raw/YOUR_FILE.edf --channel "C3" --max_windows 20
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("edf_path")
    parser.add_argument("--channel", required=True)
    parser.add_argument("--window_sec", type=float, default=10.0)
    parser.add_argument("--max_windows", type=int, default=200)
    parser.add_argument("--noise_prob", type=float, default=0.02)
    parser.add_argument("--n_shots", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    print(f"Loading channel '{args.channel}' from {args.edf_path} ...")
    signal, sfreq = load_eeg_channel(args.edf_path, args.channel)

    windows = preprocess_and_segment(signal, sfreq, window_sec=args.window_sec)
    windows = windows[: args.max_windows]
    print(f"{len(windows)} windows of {args.window_sec}s")

    raw_features = extract_feature_matrix(windows, sfreq)
    features = normalize_features(raw_features)
    weights = random_weights(seed=0)

    print("Running ideal circuit ...")
    ideal_qnode = make_ideal_qnode()
    ideal_outputs = np.array([ideal_qnode(f, weights) for f in features])  # shape (n_windows, 8)

    print(f"Running noisy circuit, no QEC, depolarizing p={args.noise_prob} ...")
    noisy_outputs = generate_storage_no_qec_batch(
        features, weights, args.noise_prob, "depolarizing", args.n_shots
    )  # shape (n_windows, 8)

    n_logical = ideal_outputs.shape[1] // 2  # 4 logical qubits, columns are [Z0..Z3, X0..X3]

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
        "protection (Shor, 9 physical qubits). Lowest ranked qubits can take the "
        "repetition code (3 physical qubits) or no protection at all."
    )


if __name__ == "__main__":
    main()