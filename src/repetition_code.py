"""3-qubit bit-flip repetition code protecting the GENERATOR'S OUTPUT from
post-generation noise (storage/transmission/readout) — not gate-level
noise threaded through the generator itself.

Why this framing: a repetition code only supports transversal LOGICAL
PAULI operations cleanly. Arbitrary continuous rotations (RY/RZ) do not
act as valid logical gates when applied transversally to an encoded
qubit — an earlier version of this file tried that and it doesn't
preserve the codespace correctly. So the generator's rotations and
entangling gates run FIRST, on raw ("logical") qubits — identical in
structure to the ideal/no-QEC circuit — and QEC protects the generator's
OUTPUT while it sits exposed to noise before readout. This also matches
the project pipeline: Generator -> Noise -> No QEC / QEC -> Output. QEC
comes after the generator, not wrapped around every internal gate.

A useful mathematical property of this design: at prob=0 (no noise),
encode -> (no error) -> decode is provably an identity operation, so
the QEC circuit reduces EXACTLY to the ideal circuit. The earlier
transversal-rotation design did not have this guarantee.
"""

import numpy as np
import pennylane as qml

from src.quantum_model import N_QUBITS as N_LOGICAL, N_LAYERS

DISTANCE = 3  # physical qubits per logical qubit
N_PHYSICAL = N_LOGICAL * DISTANCE


def _generator_gates(features, weights, wires):
    """Identical gate sequence to quantum_model._circuit's generator, but
    parameterized over arbitrary `wires` (e.g. the primary physical qubit
    of each triplet) instead of assuming wires 0..3.
    """
    for i, w in enumerate(wires):
        qml.RY(features[2 * i], wires=w)
        qml.RZ(features[2 * i + 1], wires=w)
    for layer in range(N_LAYERS):
        for i, w in enumerate(wires):
            qml.RY(weights[layer, i, 0], wires=w)
            qml.RZ(weights[layer, i, 1], wires=w)
        for i, w in enumerate(wires):
            next_w = wires[(i + 1) % len(wires)]
            qml.CNOT(wires=[w, next_w])


def _sample_pauli_noise(prob, rng, wire, noise_type):
    """Trajectory-sampled equivalent of a single-qubit noise channel."""
    if noise_type == "readout":
        return
    if rng.random() >= prob:
        return
    if noise_type == "bitflip":
        qml.PauliX(wires=wire)
    elif noise_type == "phaseflip":
        qml.PauliZ(wires=wire)
    else:  # depolarizing
        pauli = rng.choice(["X", "Y", "Z"])
        {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}[pauli](wires=wire)


# ---- No-QEC baseline: generator, then noise directly on the 4 output qubits ----

def _build_storage_noise_no_qec_circuit():
    dev = qml.device("default.qubit", wires=N_LOGICAL)

    def circuit(features, weights, prob, noise_type, seed):
        rng = np.random.default_rng(seed)
        _generator_gates(features, weights, wires=list(range(N_LOGICAL)))
        for w in range(N_LOGICAL):
            _sample_pauli_noise(prob, rng, w, noise_type)
        return [qml.expval(qml.PauliZ(w)) for w in range(N_LOGICAL)] + [
            qml.expval(qml.PauliX(w)) for w in range(N_LOGICAL)
        ]

    return qml.QNode(circuit, dev)


# ---- Uniform QEC: generator -> encode -> noise -> decode ----

def _build_repetition_qec_circuit():
    dev = qml.device("default.qubit", wires=N_PHYSICAL)
    primary_wires = [lq * DISTANCE for lq in range(N_LOGICAL)]

    def circuit(features, weights, prob, noise_type, seed):
        rng = np.random.default_rng(seed)

        # Generator runs on the primary physical qubit of each triplet —
        # same gate sequence as the no-QEC / ideal circuit.
        _generator_gates(features, weights, wires=primary_wires)

        # Encode each logical qubit's generator output into its triplet
        for lq in range(N_LOGICAL):
            phys = lq * DISTANCE
            qml.CNOT(wires=[phys, phys + 1])
            qml.CNOT(wires=[phys, phys + 2])

        # Storage/transmission/readout noise on every physical qubit
        for w in range(N_PHYSICAL):
            _sample_pauli_noise(prob, rng, w, noise_type)

        # Decode: syndrome extraction, then majority-vote correction
        for lq in range(N_LOGICAL):
            phys = lq * DISTANCE
            qml.CNOT(wires=[phys, phys + 1])
            qml.CNOT(wires=[phys, phys + 2])
            qml.Toffoli(wires=[phys + 1, phys + 2, phys])

        return [qml.expval(qml.PauliZ(lq * DISTANCE)) for lq in range(N_LOGICAL)] + [
            qml.expval(qml.PauliX(lq * DISTANCE)) for lq in range(N_LOGICAL)
        ]

    return qml.QNode(circuit, dev)


_no_qec_circuit = _build_storage_noise_no_qec_circuit()
_qec_circuit = _build_repetition_qec_circuit()


def _average_trajectories(circuit, features, weights, prob, noise_type, n_shots, base_seed):
    outputs = [circuit(features, weights, prob, noise_type, base_seed + i) for i in range(n_shots)]
    result = np.mean(outputs, axis=0)
    if noise_type == "readout":
        rng = np.random.default_rng(base_seed)
        flips = rng.choice([1.0, -1.0], size=len(result), p=[1 - prob, prob])
        result = result * flips
    return result


def generate_storage_no_qec(features, weights, prob, noise_type="depolarizing", n_shots=50, base_seed=0):
    return _average_trajectories(_no_qec_circuit, features, weights, prob, noise_type, n_shots, base_seed)


def generate_repetition_qec(features, weights, prob, noise_type="depolarizing", n_shots=50, base_seed=0):
    return _average_trajectories(_qec_circuit, features, weights, prob, noise_type, n_shots, base_seed)


def generate_storage_no_qec_batch(feature_matrix, weights, prob, noise_type="depolarizing", n_shots=50):
    return np.stack(
        [
            generate_storage_no_qec(f, weights, prob, noise_type, n_shots, base_seed=i * n_shots)
            for i, f in enumerate(feature_matrix)
        ]
    )


def generate_repetition_qec_batch(feature_matrix, weights, prob, noise_type="depolarizing", n_shots=50):
    return np.stack(
        [
            generate_repetition_qec(f, weights, prob, noise_type, n_shots, base_seed=i * n_shots)
            for i, f in enumerate(feature_matrix)
        ]
    )