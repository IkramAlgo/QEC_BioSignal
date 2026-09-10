"""Benchmark: how fast is a 12-qubit noisy circuit on THIS machine?

Uses TRAJECTORY-based (Monte Carlo) noise simulation on default.qubit,
not exact density-matrix simulation on default.mixed. Each "shot" samples
whether each noise channel fires (by its probability) and applies the
corresponding Pauli if so, using only a 2^12-element statevector. Average
many shots to estimate the same expectation values the density-matrix
approach computes exactly. This is the standard way to make noisy
simulation tractable past ~10 qubits — full density-matrix tracking of a
4096x4096 matrix through 40+ gates (including Toffolis) is what made the
previous version hang.

Usage:
  python scripts/benchmark_12qubit.py
  python scripts/benchmark_12qubit.py --n_samples 20 --n_shots 200
"""

import argparse
import time

import numpy as np
import pennylane as qml

N_QUBITS = 12  # 4 logical qubits x 3 physical qubits each (repetition code)


def sample_pauli_noise(prob: float, rng: np.random.Generator, wire: int):
    """With probability `prob`, apply a random Pauli (X/Y/Z) to `wire`; else nothing.
    This is the trajectory-sampled equivalent of a depolarizing channel.
    """
    if rng.random() < prob:
        pauli = rng.choice(["X", "Y", "Z"])
        {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}[pauli](wires=wire)


def build_trajectory_circuit():
    dev = qml.device("default.qubit", wires=N_QUBITS)  # pure state, not density matrix

    def circuit(features, weights, prob, seed):
        rng = np.random.default_rng(seed)

        for logical_q in range(4):
            phys = logical_q * 3
            qml.RY(features[2 * logical_q], wires=phys)
            qml.RZ(features[2 * logical_q + 1], wires=phys)

        for logical_q in range(4):
            phys = logical_q * 3
            qml.CNOT(wires=[phys, phys + 1])
            qml.CNOT(wires=[phys, phys + 2])

        for q in range(N_QUBITS):
            sample_pauli_noise(prob, rng, q)

        for layer in range(2):
            for logical_q in range(4):
                phys = logical_q * 3
                qml.RY(weights[layer, logical_q, 0], wires=phys)
                sample_pauli_noise(prob, rng, phys)
                qml.RZ(weights[layer, logical_q, 1], wires=phys)
                sample_pauli_noise(prob, rng, phys)
            for logical_q in range(4):
                phys = logical_q * 3
                next_phys = ((logical_q + 1) % 4) * 3
                qml.CNOT(wires=[phys, next_phys])
                sample_pauli_noise(prob, rng, phys)

        for logical_q in range(4):
            phys = logical_q * 3
            qml.Toffoli(wires=[phys + 1, phys + 2, phys])

        return [qml.expval(qml.PauliZ(logical_q * 3)) for logical_q in range(4)] + [
            qml.expval(qml.PauliX(logical_q * 3)) for logical_q in range(4)
        ]

    return qml.QNode(circuit, dev)


def run_trajectory_average(circuit, features, weights, prob, n_shots, base_seed=0):
    """Average n_shots independent noise trajectories to estimate the mixed-state result."""
    outputs = [circuit(features, weights, prob, base_seed + i) for i in range(n_shots)]
    return np.mean(outputs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10, help="How many EEG-window evaluations to time")
    parser.add_argument("--n_shots", type=int, default=100, help="Trajectories averaged per evaluation")
    args = parser.parse_args()

    circuit = build_trajectory_circuit()
    rng = np.random.default_rng(0)
    features = rng.uniform(0, np.pi, size=8)
    weights = rng.uniform(0, 2 * np.pi, size=(2, 4, 2))
    prob = 0.02

    print(f"Timing {args.n_samples} window-evaluations, {args.n_shots} trajectories each ...")
    print("(warmup call first, not timed)")

    run_trajectory_average(circuit, features, weights, prob, n_shots=5)  # warmup

    start = time.perf_counter()
    for i in range(args.n_samples):
        run_trajectory_average(circuit, features, weights, prob, n_shots=args.n_shots, base_seed=i * args.n_shots)
    elapsed = time.perf_counter() - start

    per_sample = elapsed / args.n_samples
    print(f"\n{args.n_samples} evaluations took {elapsed:.2f}s  ->  {per_sample:.3f}s per evaluation")
    print(f"(each evaluation = {args.n_shots} trajectories averaged)")
    print("\nProjected times for the real experiment:")
    print(f"  200 windows, single pass: {per_sample * 200:.1f}s")
    print(f"  200 windows x 3 ZNE scale factors: {per_sample * 200 * 3:.1f}s")
    print(f"  200 windows x 3 scales x 4 noise types: {per_sample * 200 * 3 * 4:.1f}s")


if __name__ == "__main__":
    main()