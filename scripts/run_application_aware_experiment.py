"""Experiment: ideal vs no QEC vs uniform repetition code vs application
aware mixed protection, evaluated on the HELD OUT TEST split, with the
application aware tier allocation built DYNAMICALLY from this subject's
own sensitivity ranking (outputs/sensitivity_ranking.csv, produced by
running run_sensitivity_ranking.py immediately before this script, same
subject, same seed).

This replaces the earlier frozen allocation approach, which did not
generalize across subjects, see outputs/multi_subject_results.csv and
outputs/per_subject_breakdown.png.

Usage:
  python scripts/run_sensitivity_ranking.py data/raw/SUBJECT.edf --channel "C3" --max_windows 100 --seed 0 --weights_path outputs/trained_weights.npy
  python scripts/run_application_aware_experiment.py data/raw/SUBJECT.edf --channel "C3" --max_windows 100 --seed 0 --weights_path outputs/trained_weights.npy
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
from src.application_aware_qec import ApplicationAwareQEC, build_tiers_from_ranking
from src.metrics import summarize
from src.data_split import train_test_split_indices


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--n_shor", type=int, default=1)
    parser.add_argument("--n_repetition", type=int, default=2)
    parser.add_argument("--n_bare", type=int, default=1)
    parser.add_argument(
        "--weights_path",
        default=None,
        help="Path to trained weights .npy (from train_generator.py). Falls back to random seed-0 weights if omitted.",
    )
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    ranking_path = out_dir / "sensitivity_ranking.csv"
    if not ranking_path.exists():
        raise FileNotFoundError(
            f"{ranking_path} not found. Run run_sensitivity_ranking.py on this same subject, "
            "same --max_windows, --test_fraction, and --seed, BEFORE this script, so the "
            "allocation is built from this subject's own ranking, not a leftover from a "
            "previous subject's run."
        )
    ranking_df = pd.read_csv(ranking_path)
    tiers = build_tiers_from_ranking(
        ranking_df, n_shor=args.n_shor, n_repetition=args.n_repetition, n_bare=args.n_bare
    )
    print(f"Tier allocation for this subject, built from its own ranking: {tiers}")

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
    print(f"Evaluating on {len(test_idx)} HELD OUT test windows (not seen during training)")
    features = all_features[test_idx]

    if args.weights_path:
        weights = np.load(args.weights_path)
        print(f"Loaded trained weights from {args.weights_path}")
    else:
        weights = random_weights(seed=0)
        print("No --weights_path given, using random seed-0 weights (untrained baseline)")

    app_aware = ApplicationAwareQEC(tiers)

    print("Verifying application aware circuit identity at zero noise ...")
    if not app_aware.verify_identity(features[0], weights):
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

    print(f"Running application aware mixed protection, {app_aware.n_physical} physical qubits ...")
    start = time.perf_counter()
    aa_outputs = app_aware.generate_batch(features, weights, args.noise_prob, args.noise_type, args.n_shots)
    print(f"  took {time.perf_counter() - start:.1f}s")

    no_qec_results = summarize(ideal_outputs, no_qec_outputs)
    uniform_rep_results = summarize(ideal_outputs, uniform_rep_outputs)
    aa_results = summarize(ideal_outputs, aa_outputs)

    table = pd.DataFrame(
        [
            {"method": "ideal", "physical_qubits": 4, "mse_vs_ideal": 0.0, "wasserstein_vs_ideal": 0.0, "tiers": ""},
            {
                "method": "no_qec",
                "physical_qubits": 4,
                "mse_vs_ideal": no_qec_results["mse"],
                "wasserstein_vs_ideal": no_qec_results["mean_wasserstein"],
                "tiers": "",
            },
            {
                "method": "uniform_repetition_d3",
                "physical_qubits": 12,
                "mse_vs_ideal": uniform_rep_results["mse"],
                "wasserstein_vs_ideal": uniform_rep_results["mean_wasserstein"],
                "tiers": "",
            },
            {
                "method": "application_aware",
                "physical_qubits": app_aware.n_physical,
                "mse_vs_ideal": aa_results["mse"],
                "wasserstein_vs_ideal": aa_results["mean_wasserstein"],
                "tiers": str(tiers),
            },
        ]
    )
    table.to_csv(out_dir / "application_aware_results_testsplit.csv", index=False)
    print(f"\nSaved {out_dir / 'application_aware_results_testsplit.csv'}")
    print(table.to_string(index=False))

    noisy_rows = table[table["method"] != "ideal"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].bar(noisy_rows["method"], noisy_rows["mse_vs_ideal"], color=["indianred", "steelblue", "seagreen"])
    ax[0].set_title("MSE vs ideal (held out test windows)")
    ax[0].tick_params(axis="x", rotation=20)
    ax[1].bar(noisy_rows["method"], noisy_rows["wasserstein_vs_ideal"], color=["indianred", "steelblue", "seagreen"])
    ax[1].set_title("Wasserstein vs ideal (held out test windows)")
    ax[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "application_aware_plot_testsplit.png", dpi=150)
    print(f"Saved {out_dir / 'application_aware_plot_testsplit.png'}")


if __name__ == "__main__":
    main()