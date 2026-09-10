"""Experiment A (ideal) vs Experiment B (noisy, multiple noise types) — no QEC yet.

Usage:
  python scripts/run_baseline_experiment.py data/raw/YOUR_FILE.edf --channel "C3"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_eeg_channel
from src.preprocessing import preprocess_and_segment
from src.features import extract_feature_matrix
from src.quantum_model import (
    make_ideal_qnode,
    make_noisy_qnode,
    random_weights,
    normalize_features,
)
from src.metrics import summarize

NOISE_TYPES = ["depolarizing", "bitflip", "phaseflip", "readout"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("edf_path")
    parser.add_argument("--channel", required=True, help="Exact EEG channel name from inspect_edf.py")
    parser.add_argument("--window_sec", type=float, default=10.0)
    parser.add_argument("--max_windows", type=int, default=200, help="Cap windows for a fast first run")
    parser.add_argument("--noise_prob", type=float, default=0.02)
    parser.add_argument(
        "--noise_types",
        nargs="+",
        default=NOISE_TYPES,
        choices=NOISE_TYPES,
        help="Which noise models to compare in this run",
    )
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    print(f"Loading channel '{args.channel}' from {args.edf_path} ...")
    signal, sfreq = load_eeg_channel(args.edf_path, args.channel)

    print("Filtering + windowing ...")
    windows = preprocess_and_segment(signal, sfreq, window_sec=args.window_sec)
    windows = windows[: args.max_windows]
    print(f"  {len(windows)} windows of {args.window_sec}s")

    print("Extracting 8-dim feature vectors ...")
    raw_features = extract_feature_matrix(windows, sfreq)
    features = normalize_features(raw_features)  # scaled to [0, pi] for rotation angles

    print("Running Experiment A (ideal) ...")
    ideal_qnode = make_ideal_qnode()
    weights = random_weights(seed=0)
    ideal_outputs = np.array([ideal_qnode(f, weights) for f in features])

    rows = [{"method": "ideal", "mse_vs_ideal": 0.0, "wasserstein_vs_ideal": 0.0}]
    noisy_outputs_by_type = {}

    for noise_type in args.noise_types:
        print(f"Running Experiment B — noise: {noise_type} (p={args.noise_prob}) ...")
        noisy_qnode = make_noisy_qnode(noise_type=noise_type, prob=args.noise_prob)
        noisy_outputs = np.array([noisy_qnode(f, weights) for f in features])
        noisy_outputs_by_type[noise_type] = noisy_outputs

        results = summarize(ideal_outputs, noisy_outputs)
        print(f"  mse: {results['mse']:.6f}   mean_wasserstein: {results['mean_wasserstein']:.6f}")
        rows.append(
            {
                "method": f"no_qec_{noise_type}",
                "mse_vs_ideal": results["mse"],
                "wasserstein_vs_ideal": results["mean_wasserstein"],
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "baseline_results.csv", index=False)
    print(f"\nSaved {out_dir / 'baseline_results.csv'}")
    print(table.to_string(index=False))

    # Plot: mse and wasserstein per noise type, so it's visually obvious
    # which noise model hits this circuit hardest
    noisy_rows = table[table["method"] != "ideal"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(noisy_rows["method"], noisy_rows["mse_vs_ideal"])
    ax[0].set_title("MSE vs ideal, by noise type")
    ax[0].tick_params(axis="x", rotation=30)

    ax[1].bar(noisy_rows["method"], noisy_rows["wasserstein_vs_ideal"])
    ax[1].set_title("Mean Wasserstein vs ideal, by noise type")
    ax[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(out_dir / "baseline_plot.png", dpi=150)
    print(f"Saved {out_dir / 'baseline_plot.png'}")


if __name__ == "__main__":
    main()