"""Experiment: ideal vs no QEC vs uniform repetition code vs application
aware mixed protection, all under the same post generation depolarizing
noise. This is the comparison that backs both halves of the paper
title at once, noise resilient and application aware, in one table.

Usage:
  python scripts/run_application_aware_experiment.py data/raw/YOUR_FILE.edf --channel "C3" --max_windows 20
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_eeg_channel
from src.preprocessing import preprocess_and_segment
from src.features import extract_feature_matrix
from src.quantum_model import make_ideal_qnode, random_weights, normalize_features
from src.repetition_code import generate_storage_no_qec_batch, generate_repetition_qec_batch
from src.application_aware_qec import generate_application_aware_qec_batch, verify_identity
from src.metrics import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("edf_path")
    parser.add_argument("--channel", required=True)
    parser.add_argument("--window_sec", type=float, default=10.0)
    parser.add_argument("--max_windows", type=int, default=200)
    parser.add_argument(
        "--noise_type", default="depolarizing", choices=["depolarizing", "bitflip", "phaseflip", "readout"]
    )
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

    print("Verifying application aware circuit identity at zero noise ...")
    if not verify_identity(features[0], weights):
        print("Identity check failed, stopping before running the noisy experiment.")
        return

    print("Running ideal circuit ...")
    ideal_qnode = make_ideal_qnode()
    ideal_outputs = np.array([ideal_qnode(f, weights) for f in features])

    print(f"Running no QEC, {args.noise_type} noise, p={args.noise_prob} ...")
    start = time.perf_counter()
    no_qec_outputs = generate_storage_no_qec_batch(features, weights, args.noise_prob, args.noise_type, args.n_shots)
    print(f"  took {time.perf_counter() - start:.1f}s")

    print("Running uniform repetition code, distance 3 on all 4 logical qubits ...")
    start = time.perf_counter()
    uniform_rep_outputs = generate_repetition_qec_batch(
        features, weights, args.noise_prob, args.noise_type, args.n_shots
    )
    print(f"  took {time.perf_counter() - start:.1f}s")

    print("Running application aware mixed protection (Shor / repetition / bare, per sensitivity) ...")
    start = time.perf_counter()
    aa_outputs = generate_application_aware_qec_batch(
        features, weights, args.noise_prob, args.noise_type, args.n_shots
    )
    print(f"  took {time.perf_counter() - start:.1f}s")

    no_qec_results = summarize(ideal_outputs, no_qec_outputs)
    uniform_rep_results = summarize(ideal_outputs, uniform_rep_outputs)
    aa_results = summarize(ideal_outputs, aa_outputs)

    table = pd.DataFrame(
        [
            {"method": "ideal", "physical_qubits": 4, "mse_vs_ideal": 0.0, "wasserstein_vs_ideal": 0.0},
            {
                "method": "no_qec",
                "physical_qubits": 4,
                "mse_vs_ideal": no_qec_results["mse"],
                "wasserstein_vs_ideal": no_qec_results["mean_wasserstein"],
            },
            {
                "method": "uniform_repetition_d3",
                "physical_qubits": 12,
                "mse_vs_ideal": uniform_rep_results["mse"],
                "wasserstein_vs_ideal": uniform_rep_results["mean_wasserstein"],
            },
            {
                "method": "application_aware",
                "physical_qubits": 14,
                "mse_vs_ideal": aa_results["mse"],
                "wasserstein_vs_ideal": aa_results["mean_wasserstein"],
            },
        ]
    )
    table.to_csv(out_dir / "application_aware_results.csv", index=False)
    print(f"\nSaved {out_dir / 'application_aware_results.csv'}")
    print(table.to_string(index=False))

    noisy_rows = table[table["method"] != "ideal"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].bar(noisy_rows["method"], noisy_rows["mse_vs_ideal"], color=["indianred", "steelblue", "seagreen"])
    ax[0].set_title("MSE vs ideal")
    ax[0].tick_params(axis="x", rotation=20)
    ax[1].bar(noisy_rows["method"], noisy_rows["wasserstein_vs_ideal"], color=["indianred", "steelblue", "seagreen"])
    ax[1].set_title("Wasserstein vs ideal")
    ax[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "application_aware_plot.png", dpi=150)
    print(f"Saved {out_dir / 'application_aware_plot.png'}")


if __name__ == "__main__":
    main()