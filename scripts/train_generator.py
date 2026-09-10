"""Train the quantum generator's variational weights via reconstruction loss.

Objective: the circuit's 8 output expectation values (4 <Z> + 4 <X>,
range [-1,1]) should reconstruct a [-1,1]-scaled version of the SAME 8
raw biosignal features that were encoded as [0,pi] rotation angles into
the circuit. This is a plain reconstruction task, not GAN-style
adversarial training — simpler, verifiable (loss decreasing is an
honest signal, no mode collapse to debug), and it directly matches the
"does the model preserve meaningful structure" claim the paper cares
about.

Trained against the IDEAL (noiseless) circuit only. Noise/QEC evaluation
happens afterward, reusing these learned weights the same way the
random seed-0 weights were used for the untrained baseline.

Usage:
  python scripts/train_generator.py data/raw/EPCTL01.edf --channel "C3"
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
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    print(f"Loading channel '{args.channel}' from {args.edf_path} ...")
    signal, sfreq = load_eeg_channel(args.edf_path, args.channel)

    print("Filtering + windowing ...")
    windows = preprocess_and_segment(signal, sfreq, window_sec=args.window_sec)
    windows = windows[: args.max_windows]
    print(f"  {len(windows)} windows")

    print("Extracting features ...")
    raw_features = extract_feature_matrix(windows, sfreq)

    # Input encoding stays [0, pi] — unchanged from the untrained pipeline.
    input_features = normalize_to_range(raw_features, 0, np.pi)
    # Reconstruction TARGET is a separate [-1, 1] scaling of the same raw
    # features, matching the circuit's expectation-value output range.
    target_features = normalize_to_range(raw_features, -1, 1)

    ideal_qnode = make_ideal_qnode()

    rng = np.random.default_rng(args.seed)
    weights = pnp.array(
        rng.uniform(0, 2 * np.pi, size=(N_LAYERS, N_QUBITS, 2)), requires_grad=True
    )

    opt = qml.AdamOptimizer(stepsize=args.lr)

    def batch_loss(w):
        total = 0.0
        for x_in, x_target in zip(input_features, target_features):
            out = pnp.stack(ideal_qnode(x_in, w))
            total = total + pnp.sum((out - x_target) ** 2)
        return total / len(input_features)

    print(f"Training {args.epochs} epochs on {len(input_features)} windows ...")
    losses = []
    for epoch in range(args.epochs):
        weights, loss = opt.step_and_cost(batch_loss, weights)
        losses.append(float(loss))
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:4d}  loss {loss:.6f}")

    weights_path = out_dir / "trained_weights.npy"
    np.save(weights_path, np.array(weights))
    print(f"\nSaved trained weights to {weights_path}")

    losses_path = out_dir / "training_loss.csv"
    pd.DataFrame({"epoch": range(len(losses)), "loss": losses}).to_csv(losses_path, index=False)
    print(f"Saved loss curve to {losses_path}")

    improvement = (1 - losses[-1] / losses[0]) * 100 if losses[0] > 0 else 0.0
    print(f"\nFinal loss: {losses[-1]:.6f}  (started at {losses[0]:.6f})")
    print(f"Loss reduced by {improvement:.1f}% over training")
    print(
        "\nIMPORTANT next step: rerun sensitivity ranking and rebuild the "
        "application-aware tier allocation using these trained weights — "
        "the untrained ranking is not assumed to hold after training."
    )


if __name__ == "__main__":
    main()