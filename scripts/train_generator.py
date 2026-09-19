"""Train the quantum generator's variational weights via reconstruction loss.

UPDATED: now trains only on the training split. The held out test
split is never touched here, run_sensitivity_ranking.py and
run_application_aware_experiment.py evaluate on that held out portion
using the same --test_fraction and --seed, so the reported numbers
are no longer measured on data the generator already saw.

Usage:
  python scripts/train_generator.py data/raw/EPCTL01.edf --channel "C3"

IMPORTANT: use the same --max_windows, --test_fraction, and --seed on
this script and on run_sensitivity_ranking.py and
run_application_aware_experiment.py, otherwise the three scripts will
not agree on which windows are held out as test data.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_eeg_channel
from src.preprocessing import preprocess_and_segment
from src.features import extract_feature_matrix
from src.quantum_model import make_ideal_qnode, N_QUBITS, N_LAYERS
from src.data_split import train_test_split_indices


def normalize_to_range(feature_matrix: np.ndarray, lo: float, hi: float) -> np.ndarray:
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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    print(f"Loading channel '{args.channel}' from {args.edf_path} ...")
    signal, sfreq = load_eeg_channel(args.edf_path, args.channel)

    print("Filtering + windowing ...")
    windows = preprocess_and_segment(signal, sfreq, window_sec=args.window_sec)
    windows = windows[: args.max_windows]
    print(f"  {len(windows)} windows total")

    print("Extracting features ...")
    raw_features = extract_feature_matrix(windows, sfreq)

    train_idx, test_idx = train_test_split_indices(
        len(raw_features), test_fraction=args.test_fraction, seed=args.seed
    )
    print(
        f"Split: {len(train_idx)} training windows, {len(test_idx)} held out test windows "
        f"(test_fraction={args.test_fraction}, seed={args.seed})"
    )
    print(
        "Remember: run_sensitivity_ranking.py and run_application_aware_experiment.py "
        "must use the SAME --max_windows, --test_fraction, and --seed as this run, "
        "so they hold out the same windows."
    )

    train_raw_features = raw_features[train_idx]

    input_features = normalize_to_range(train_raw_features, 0, np.pi)
    target_features = normalize_to_range(train_raw_features, -1, 1)

    ideal_qnode = make_ideal_qnode()

    rng = np.random.default_rng(args.seed)
    weights = pnp.array(
        rng.uniform(0, 2 * np.pi, size=(N_LAYERS, N_QUBITS, 2)), requires_grad=True
    )

    opt = qml.AdamOptimizer(stepsize=args.lr)

    batch_size = min(16, len(input_features))

    def batch_loss(w, x_batch, t_batch):
        total = 0.0
        for x_in, x_target in zip(x_batch, t_batch):
            out = pnp.stack(ideal_qnode(x_in, w))
            total = total + pnp.sum((out - x_target) ** 2)
        return total / len(x_batch)

    print(
        f"Training {args.epochs} epochs on {len(input_features)} TRAINING windows only, "
        f"mini batch size {batch_size} (reduces peak memory per gradient step) ..."
    )
    losses = []
    n_samples = len(input_features)
    for epoch in range(args.epochs):
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_samples, batch_size):
            batch_idx = perm[start : start + batch_size]
            x_batch = input_features[batch_idx]
            t_batch = target_features[batch_idx]
            weights, loss = opt.step_and_cost(lambda w: batch_loss(w, x_batch, t_batch), weights)
            epoch_loss += loss
            n_batches += 1
        epoch_loss /= n_batches
        losses.append(float(epoch_loss))
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:4d}  loss {epoch_loss:.6f}")

    weights_path = out_dir / "trained_weights.npy"
    np.save(weights_path, np.array(weights))
    print(f"\nSaved trained weights to {weights_path}")

    losses_path = out_dir / "training_loss.csv"
    pd.DataFrame({"epoch": range(len(losses)), "loss": losses}).to_csv(losses_path, index=False)
    print(f"Saved loss curve to {losses_path}")

    improvement = (1 - losses[-1] / losses[0]) * 100 if losses[0] > 0 else 0.0
    print(f"\nFinal loss: {losses[-1]:.6f}  (started at {losses[0]:.6f})")
    print(f"Loss reduced by {improvement:.1f}% over training, on training windows only")
    print(
        "\nNEXT: rerun run_sensitivity_ranking.py and run_application_aware_experiment.py "
        "with matching --max_windows, --test_fraction, and --seed, so evaluation runs on "
        "the held out test windows, not the training windows."
    )


if __name__ == "__main__":
    main()