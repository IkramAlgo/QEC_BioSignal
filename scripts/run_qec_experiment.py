"""Experiment A (ideal) vs B (no QEC) vs C (Uniform Partial QEC, repetition code).

Both B and C now expose the SAME post-generation noise (storage/readout
noise on the generator's output) — C protects it with the repetition
code, B doesn't. This is a fair, well-posed comparison; see
src/repetition_code.py's module docstring for why.

Usage:
  python scripts/run_qec_experiment.py data/raw/YOUR_FILE.edf --channel "C3" --max_windows 20
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
    parser.add_argument("--n_shots", type=int, default=50, help="Trajectories averaged per window")
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
    features = normalize_features(raw_features)

    weights = random_weights(seed=0)

    print("Running Experiment A (ideal) ...")
    ideal_qnode = make_ideal_qnode()
    ideal_outputs = np.array([ideal_qnode(f, weights) for f in features])

    print(f"Running Experiment B (no QEC, post-generation {args.noise_type} noise, p={args.noise_prob}) ...")
    start = time.perf_counter()
    no_qec_outputs = generate_storage_no_qec_batch(features, weights, args.noise_prob, args.noise_type, args.n_shots)
    print(f"  took {time.perf_counter() - start:.1f}s")

    print(f"Running Experiment C (Uniform QEC, distance-3 repetition code, {args.n_shots} shots/window) ...")
    start = time.perf_counter()
    uniform_qec_outputs = generate_repetition_qec_batch(
        features, weights, args.noise_prob, args.noise_type, args.n_shots
    )
    print(f"  took {time.perf_counter() - start:.1f}s")

    no_qec_results = summarize(ideal_outputs, no_qec_outputs)
    uniform_qec_results = summarize(ideal_outputs, uniform_qec_outputs)

    table = pd.DataFrame(
        [
            {"method": "ideal", "mse_vs_ideal": 0.0, "wasserstein_vs_ideal": 0.0},
            {
                "method": "no_qec",
                "mse_vs_ideal": no_qec_results["mse"],
                "wasserstein_vs_ideal": no_qec_results["mean_wasserstein"],
            },
            {
                "method": "uniform_qec_repetition_d3",
                "mse_vs_ideal": uniform_qec_results["mse"],
                "wasserstein_vs_ideal": uniform_qec_results["mean_wasserstein"],
            },
        ]
    )
    table.to_csv(out_dir / "qec_results.csv", index=False)
    print(f"\nSaved {out_dir / 'qec_results.csv'}")
    print(table.to_string(index=False))

    improvement_mse = (
        (1 - uniform_qec_results["mse"] / no_qec_results["mse"]) * 100 if no_qec_results["mse"] > 0 else 0.0
    )
    improvement_w = (
        (1 - uniform_qec_results["mean_wasserstein"] / no_qec_results["mean_wasserstein"]) * 100
        if no_qec_results["mean_wasserstein"] > 0
        else 0.0
    )
    print(f"\nUniform QEC changed mse by {improvement_mse:.1f}% and wasserstein by {improvement_w:.1f}% vs no QEC")
    print("(positive % = improvement; negative % means QEC is still hurting, not helping)")

    noisy_rows = table[table["method"] != "ideal"]
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].bar(noisy_rows["method"], noisy_rows["mse_vs_ideal"], color=["indianred", "seagreen"])
    ax[0].set_title("MSE vs ideal")
    ax[0].tick_params(axis="x", rotation=20)
    ax[1].bar(noisy_rows["method"], noisy_rows["wasserstein_vs_ideal"], color=["indianred", "seagreen"])
    ax[1].set_title("Wasserstein vs ideal")
    ax[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "qec_plot.png", dpi=150)
    print(f"Saved {out_dir / 'qec_plot.png'}")


if __name__ == "__main__":
    main()