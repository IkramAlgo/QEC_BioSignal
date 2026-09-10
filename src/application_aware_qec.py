"""Application aware QEC: protection allocated per logical qubit based
on the measured sensitivity ranking from run_sensitivity_ranking.py.

This is version 2 of the tier allocation, rebuilt after training the
generator, since the untrained ranking did NOT hold after training,
it nearly inverted. If you retrain again, subject, weights, or data
changes, rerun run_sensitivity_ranking.py and update
TIER_BY_LOGICAL_QUBIT below to match, do not assume any allocation
carries over.

Ranking used for THIS allocation (trained weights, 100 windows,
outputs/trained_weights.npy):
  logical qubit 3: combined_mse 0.001131  (most sensitive)
  logical qubit 0: combined_mse 0.001052  (close second)
  logical qubit 2: combined_mse 0.000885
  logical qubit 1: combined_mse 0.000595  (least sensitive)

Allocation:
  logical qubit 3 -> Shor code,       9 physical qubits (heaviest)
  logical qubit 0 -> repetition code, 3 physical qubits (moderate,
                                        close second in sensitivity)
  logical qubit 2 -> repetition code, 3 physical qubits (moderate)
  logical qubit 1 -> bare,            1 physical qubit  (no protection)

Total physical qubits: 9 + 3 + 3 + 1 = 16. Two to the sixteen
amplitudes is trivial memory.

Wire layout is now computed automatically from TIER_BY_LOGICAL_QUBIT
at module load time, in logical qubit order, so changing a tier only
means editing the dictionary below, nothing else in this file.
"""

import numpy as np
import pennylane as qml

from src.quantum_model import N_QUBITS as N_LOGICAL, N_LAYERS

TIER_SIZES = {"bare": 1, "repetition": 3, "shor": 9}

TIER_BY_LOGICAL_QUBIT = {
    0: "repetition",
    1: "bare",
    2: "repetition",
    3: "shor",
}


def _build_wire_layout(tier_by_logical_qubit, n_logical):
    """Assigns physical wire ranges to each logical qubit, in order,
    based on its tier size. Returns the list of primary wires (one per
    logical qubit, in logical qubit order) and the total wire count.
    """
    primary_wires = []
    cursor = 0
    for lq in range(n_logical):
        tier = tier_by_logical_qubit[lq]
        size = TIER_SIZES[tier]
        primary_wires.append(cursor)
        cursor += size
    return primary_wires, cursor


PRIMARY_WIRES, N_PHYSICAL = _build_wire_layout(TIER_BY_LOGICAL_QUBIT, N_LOGICAL)


def _generator_gates(features, weights, wires):
    """Identical gate sequence to every other module in this project."""
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
    """Identical to repetition_code.py and shor_code.py, kept the same
    on purpose so results are comparable across all three modules.
    """
    if noise_type == "readout":
        return
    if rng.random() >= prob:
        return
    if noise_type == "bitflip":
        qml.PauliX(wires=wire)
    elif noise_type == "phaseflip":
        qml.PauliZ(wires=wire)
    else:
        pauli = rng.choice(["X", "Y", "Z"])
        {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}[pauli](wires=wire)


def _shor_encode(block_base):
    outer_primaries = [block_base, block_base + 3, block_base + 6]
    qml.CNOT(wires=[outer_primaries[0], outer_primaries[1]])
    qml.CNOT(wires=[outer_primaries[0], outer_primaries[2]])
    for p in outer_primaries:
        qml.Hadamard(wires=p)
    for p in outer_primaries:
        qml.CNOT(wires=[p, p + 1])
        qml.CNOT(wires=[p, p + 2])


def _shor_decode(block_base):
    outer_primaries = [block_base, block_base + 3, block_base + 6]
    for p in outer_primaries:
        qml.CNOT(wires=[p, p + 1])
        qml.CNOT(wires=[p, p + 2])
        qml.Toffoli(wires=[p + 1, p + 2, p])
    for p in outer_primaries:
        qml.Hadamard(wires=p)
    qml.CNOT(wires=[outer_primaries[0], outer_primaries[1]])
    qml.CNOT(wires=[outer_primaries[0], outer_primaries[2]])
    qml.Toffoli(wires=[outer_primaries[1], outer_primaries[2], outer_primaries[0]])


def _repetition_encode(primary):
    qml.CNOT(wires=[primary, primary + 1])
    qml.CNOT(wires=[primary, primary + 2])


def _repetition_decode(primary):
    qml.CNOT(wires=[primary, primary + 1])
    qml.CNOT(wires=[primary, primary + 2])
    qml.Toffoli(wires=[primary + 1, primary + 2, primary])


def _all_physical_wires_for(lq):
    """All physical wires belonging to a given logical qubit, used for
    applying noise to every physical qubit in that qubit's block.
    """
    tier = TIER_BY_LOGICAL_QUBIT[lq]
    primary = PRIMARY_WIRES[lq]
    size = TIER_SIZES[tier]
    return list(range(primary, primary + size))


def _build_application_aware_circuit():
    dev = qml.device("default.qubit", wires=N_PHYSICAL)

    def circuit(features, weights, prob, noise_type, seed):
        rng = np.random.default_rng(seed)

        _generator_gates(features, weights, wires=PRIMARY_WIRES)

        for lq in range(N_LOGICAL):
            tier = TIER_BY_LOGICAL_QUBIT[lq]
            primary = PRIMARY_WIRES[lq]
            if tier == "shor":
                _shor_encode(primary)
            elif tier == "repetition":
                _repetition_encode(primary)
            # bare tier: nothing to encode

        for lq in range(N_LOGICAL):
            for w in _all_physical_wires_for(lq):
                _sample_pauli_noise(prob, rng, w, noise_type)

        for lq in range(N_LOGICAL):
            tier = TIER_BY_LOGICAL_QUBIT[lq]
            primary = PRIMARY_WIRES[lq]
            if tier == "shor":
                _shor_decode(primary)
            elif tier == "repetition":
                _repetition_decode(primary)
            # bare tier: nothing to decode

        return [qml.expval(qml.PauliZ(p)) for p in PRIMARY_WIRES] + [
            qml.expval(qml.PauliX(p)) for p in PRIMARY_WIRES
        ]

    return qml.QNode(circuit, dev)


_application_aware_circuit = _build_application_aware_circuit()


def _average_trajectories(circuit, features, weights, prob, noise_type, n_shots, base_seed):
    outputs = [circuit(features, weights, prob, noise_type, base_seed + i) for i in range(n_shots)]
    result = np.mean(outputs, axis=0)
    if noise_type == "readout":
        rng = np.random.default_rng(base_seed)
        flips = rng.choice([1.0, -1.0], size=len(result), p=[1 - prob, prob])
        result = result * flips
    return result


def generate_application_aware_qec(features, weights, prob, noise_type="depolarizing", n_shots=50, base_seed=0):
    return _average_trajectories(
        _application_aware_circuit, features, weights, prob, noise_type, n_shots, base_seed
    )


def generate_application_aware_qec_batch(feature_matrix, weights, prob, noise_type="depolarizing", n_shots=50):
    return np.stack(
        [
            generate_application_aware_qec(f, weights, prob, noise_type, n_shots, base_seed=i * n_shots)
            for i, f in enumerate(feature_matrix)
        ]
    )


def verify_identity(features, weights, n_trials=5):
    """Same discipline as every other module here, confirm encode then
    decode is an identity at zero noise before trusting any noisy
    result. Rerun this every time the tier allocation changes, it does
    not depend on which tiers are assigned, only on whether the encode
    and decode gates are wired correctly for those tiers.
    """
    from src.quantum_model import make_ideal_qnode

    ideal_qnode = make_ideal_qnode()
    ideal_out = np.array(ideal_qnode(features, weights))

    for seed in range(n_trials):
        aa_out = generate_application_aware_qec(
            features, weights, prob=0.0, noise_type="depolarizing", n_shots=1, base_seed=seed
        )
        if not np.allclose(ideal_out, aa_out, atol=1e-6):
            print(f"IDENTITY CHECK FAILED at seed {seed}")
            print(f"  ideal: {ideal_out}")
            print(f"  application_aware: {aa_out}")
            return False
    print(f"Identity check passed across {n_trials} trials, ideal matches application aware circuit at zero noise.")
    return True