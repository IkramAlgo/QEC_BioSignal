"""Classical baseline comparison, same noise probability, same shot
averaging concept, same held out windows and same ground truth target
as the quantum pipeline.

IMPORTANT LABELING NOTE: every quantum method in this project, no_qec
included, already averages 50 noisy trajectories internally, that has
been true since generate_storage_no_qec_batch was first written.
classical_single_measurement below is ONE noisy sample, not fifty, so
it is NOT a fair comparison against any quantum column, it is included
only as context for how bad one uncorrected sample looks.

classical_averaged_50_shots is the fair baseline, same shot count as
every quantum method, and is what should actually be compared against
quantum no_qec, quantum uniform_repetition_d3, and quantum
application_aware.

Run train_generator.py, run_sensitivity_ranking.py, and
run_application_aware_experiment.py first (same subject, same seed),
so this script can read that subject's quantum numbers and print the
real, fair comparison automatically.

Usage:
  python scripts/run_classical_baseline.py data/raw/EPCTL01.edf --channel "C3" --max_windows 100 --seed 0
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
from src.data_split import train_test_split_indices
from src.classical_baseline import generate_classical_no_correction_batch, generate_classical_averaged_batch
from src.metrics import summarize


def normalize_to_range(feature_matrix, lo, hi):
    mins = feature_matrix.min(axis=0, keepdims=True)
    maxs = feature_matrix.max(axis=0, keepdims=True)
    span = np.where(maxs > mins, maxs - mins, 1.0)
    return (feature_matrix - mins) / span * (hi - lo) + lo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("edf_path")
    parser.add_argument("--channel", required=True)
    parser.add_argument("--window_sec", type=float, default=10.0)
    parser.add_argument("--max_windows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--noise_prob", type=float, default=0.02)
    parser.add_argument("--n_shots", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    print(f"Loading channel '{args.channel}' from {args.edf_path} ...")
    signal, sfreq = load_eeg_channel(args.edf_path, args.channel)
    windows = preprocess_and_segment(signal, sfreq, window_sec=args.window_sec)
    windows = windows[: args.max_windows]
    print(f"{len(windows)} windows total")

    raw_features = extract_feature_matrix(windows, sfreq)

    train_idx, test_idx = train_test_split_indices(
        len(raw_features), test_fraction=args.test_fraction, seed=args.seed
    )
    print(f"Evaluating on {len(test_idx)} HELD OUT test windows, same split as the quantum pipeline")

    target_features_full = normalize_to_range(raw_features, -1, 1)
    target_features = target_features_full[test_idx]

    print(f"Running classical single measurement (context only, NOT shot matched), p={args.noise_prob} ...")
    classical_single = generate_classical_no_correction_batch(target_features, args.noise_prob)

    print(f"Running classical averaged, {args.n_shots} shots (the FAIR baseline, matches quantum shot count) ...")
    classical_averaged = generate_classical_averaged_batch(target_features, args.noise_prob, args.n_shots)

    single_results = summarize(target_features, classical_single)
    avg_results = summarize(target_features, classical_averaged)

    table = pd.DataFrame(
        [
            {
                "method": "classical_ideal",
                "mse_vs_ideal": 0.0,
                "wasserstein_vs_ideal": 0.0,
                "note": "ground truth, zero error by definition",
            },
            {
                "method": "classical_single_measurement",
                "mse_vs_ideal": single_results["mse"],
                "wasserstein_vs_ideal": single_results["mean_wasserstein"],
                "note": "1 shot, NOT comparable to any quantum column, all of which average 50 shots",
            },
            {
                "method": "classical_averaged_50_shots",
                "mse_vs_ideal": avg_results["mse"],
                "wasserstein_vs_ideal": avg_results["mean_wasserstein"],
                "note": "FAIR baseline, same shot count as every quantum method",
            },
        ]
    )
    table.to_csv(out_dir / "classical_baseline_results.csv", index=False)
    print(f"\nSaved {out_dir / 'classical_baseline_results.csv'}")
    print(table.to_string(index=False))

    quantum_path = out_dir / "application_aware_results_testsplit.csv"
    if not quantum_path.exists():
        print(
            "\nNo application_aware_results_testsplit.csv found, run the quantum pipeline "
            "on this same subject and seed first for the fair comparison below."
        )
        return

    quantum_table = pd.read_csv(quantum_path)
    quantum_table = quantum_table.set_index("method")

    print(
        "\nQuantum results (confirm this file is from the SAME subject and seed as this run):"
    )
    print(quantum_table[["mse_vs_ideal", "wasserstein_vs_ideal"]].to_string())

    print(
        "\n=== The fair comparison: classical_averaged_50_shots vs each quantum method ===\n"
        "(all four of these use the same 50 shot averaging, this is the real 'why quantum' test)"
    )
    classical_mse = avg_results["mse"]
    classical_wass = avg_results["mean_wasserstein"]
    for method in ["no_qec", "uniform_repetition_d3", "application_aware"]:
        if method not in quantum_table.index:
            continue
        q_mse = quantum_table.loc[method, "mse_vs_ideal"]
        q_wass = quantum_table.loc[method, "wasserstein_vs_ideal"]
        mse_pct = (1 - q_mse / classical_mse) * 100 if classical_mse > 0 else 0.0
        wass_pct = (1 - q_wass / classical_wass) * 100 if classical_wass > 0 else 0.0
        better_worse_mse = "better than" if mse_pct > 0 else "worse than"
        better_worse_wass = "better than" if wass_pct > 0 else "worse than"
        print(
            f"  quantum {method}: mse {abs(mse_pct):.1f}% {better_worse_mse} classical averaged, "
            f"wasserstein {abs(wass_pct):.1f}% {better_worse_wass} classical averaged"
        )


if __name__ == "__main__":
    main()