"""4-qubit variational quantum generator.

For the first experiment this is NOT trained — the goal is only to confirm
that (a) an 8-dim EEG feature vector can be encoded, run through a small
circuit, and decoded back into an 8-dim vector, and (b) noise measurably
degrades that reconstruction. Training the generator to actually match the
real feature distribution comes after the ideal/noisy baseline is solid.
"""

import numpy as np
import pennylane as qml

N_QUBITS = 4
N_LAYERS = 2


def _circuit(features: np.ndarray, weights: np.ndarray):
    """features: 8-dim vector -> 2 rotation angles (RY, RZ) per qubit.
    weights: shape (N_LAYERS, N_QUBITS, 2) trainable entangling-layer rotations.
    """
    # Encoding: feature[2i] -> RY on qubit i, feature[2i+1] -> RZ on qubit i
    for q in range(N_QUBITS):
        qml.RY(features[2 * q], wires=q)
        qml.RZ(features[2 * q + 1], wires=q)

    # Variational layers: single-qubit rotations + a ring of entangling CNOTs
    for layer in range(N_LAYERS):
        for q in range(N_QUBITS):
            qml.RY(weights[layer, q, 0], wires=q)
            qml.RZ(weights[layer, q, 1], wires=q)
        for q in range(N_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % N_QUBITS])

    # 8 measurements to reconstruct an 8-dim "generated feature" vector
    return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)] + [
        qml.expval(qml.PauliX(q)) for q in range(N_QUBITS)
    ]


def make_ideal_qnode():
    dev = qml.device("default.qubit", wires=N_QUBITS)
    return qml.QNode(_circuit, dev)


def make_noisy_qnode(noise_type: str = "depolarizing", prob: float = 0.02):
    """Noisy simulator with a selectable noise channel, injected after every gate.

    noise_type: one of "depolarizing", "bitflip", "phaseflip", "readout".
    "readout" applies an ideal circuit but flips each measurement outcome's
    sign with probability `prob` (models imperfect measurement, not gate
    noise). The others inject the named channel after every single- and
    two-qubit gate, same placement pattern as the original depolarizing
    version.

    default.mixed supports density-matrix simulation with channels, which
    is why it's used here instead of default.qubit.
    """
    dev = qml.device("default.mixed", wires=N_QUBITS)

    channel_map = {
        "depolarizing": qml.DepolarizingChannel,
        "bitflip": qml.BitFlip,
        "phaseflip": qml.PhaseFlip,
    }

    def noisy_circuit(features, weights):
        apply_channel = noise_type != "readout" and noise_type in channel_map
        channel = channel_map.get(noise_type)

        for q in range(N_QUBITS):
            qml.RY(features[2 * q], wires=q)
            if apply_channel:
                channel(prob, wires=q)
            qml.RZ(features[2 * q + 1], wires=q)
            if apply_channel:
                channel(prob, wires=q)

        for layer in range(N_LAYERS):
            for q in range(N_QUBITS):
                qml.RY(weights[layer, q, 0], wires=q)
                if apply_channel:
                    channel(prob, wires=q)
                qml.RZ(weights[layer, q, 1], wires=q)
                if apply_channel:
                    channel(prob, wires=q)
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
                if apply_channel:
                    channel(prob, wires=q)

        return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)] + [
            qml.expval(qml.PauliX(q)) for q in range(N_QUBITS)
        ]

    qnode = qml.QNode(noisy_circuit, dev)

    if noise_type != "readout":
        return qnode

    # Readout error: run the ideal-equivalent circuit, then flip each
    # measurement's sign with probability `prob` — applied as a post-hoc
    # wrapper since it acts on outcomes, not gates.
    rng = np.random.default_rng(0)

    def readout_wrapper(features, weights):
        raw = qnode(features, weights)
        flips = rng.choice([1.0, -1.0], size=len(raw), p=[1 - prob, prob])
        return np.array(raw) * flips

    return readout_wrapper


def random_weights(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 2 * np.pi, size=(N_LAYERS, N_QUBITS, 2))


def normalize_features(feature_matrix: np.ndarray) -> np.ndarray:
    """Scale each feature column to [0, pi] so it's usable as a rotation angle."""
    mins = feature_matrix.min(axis=0, keepdims=True)
    maxs = feature_matrix.max(axis=0, keepdims=True)
    span = np.where(maxs > mins, maxs - mins, 1.0)
    return (feature_matrix - mins) / span * np.pi