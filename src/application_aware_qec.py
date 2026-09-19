"""Application aware QEC, now with DYNAMIC tier allocation.

Earlier version of this file hardcoded TIER_BY_LOGICAL_QUBIT as a fixed
dictionary, built once from EPCTL01's sensitivity ranking, and reused
for every subject. Running across 5 subjects showed this does not
generalize, see outputs/multi_subject_results.csv, uniform QEC itself
was inconsistent across subjects too, and the frozen allocation only
helped on 2 of 5.

This version builds the tier allocation at runtime, from whatever
sensitivity ranking is passed in, per subject, per run. Nothing is
frozen anymore. Build a new ApplicationAwareQEC instance whenever the
ranking changes, e.g. once per subject.
"""

import numpy as np
import pennylane as qml

from src.quantum_model import N_QUBITS as N_LOGICAL, N_LAYERS

TIER_SIZES = {"bare": 1, "repetition": 3, "shor": 9}


def build_tiers_from_ranking(ranking_df, n_shor=1, n_repetition=2, n_bare=1):
    """Given a sensitivity ranking DataFrame with 'logical_qubit' and
    'combined_mse' columns (as run_sensitivity_ranking.py saves to
    outputs/sensitivity_ranking.csv), returns a tier dict
    {logical_qubit: tier_name}, most sensitive qubits getting the
    heaviest protection.

    n_shor + n_repetition + n_bare must equal the number of logical
    qubits, default 1 + 2 + 1 = 4, matching every run so far.
    """
    total = n_shor + n_repetition + n_bare
    if total != len(ranking_df):
        raise ValueError(
            f"tier counts must sum to the number of logical qubits, "
            f"got {n_shor}+{n_repetition}+{n_bare}={total}, expected {len(ranking_df)}"
        )
    sorted_df = ranking_df.sort_values("combined_mse", ascending=False).reset_index(drop=True)
    tiers = {}
    for i, row in sorted_df.iterrows():
        lq = int(row["logical_qubit"])
        if i < n_shor:
            tiers[lq] = "shor"
        elif i < n_shor + n_repetition:
            tiers[lq] = "repetition"
        else:
            tiers[lq] = "bare"
    return tiers


def _build_wire_layout(tier_by_logical_qubit, n_logical):
    primary_wires = []
    cursor = 0
    for lq in range(n_logical):
        size = TIER_SIZES[tier_by_logical_qubit[lq]]
        primary_wires.append(cursor)
        cursor += size
    return primary_wires, cursor


def _generator_gates(features, weights, wires):
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


class ApplicationAwareQEC:
    """One compiled circuit for one specific tier allocation. Build a
    new instance whenever the allocation changes, e.g. a new subject
    with a new sensitivity ranking, rather than reusing one instance
    across different allocations.
    """

    def __init__(self, tier_by_logical_qubit):
        self.tiers = tier_by_logical_qubit
        self.primary_wires, self.n_physical = _build_wire_layout(tier_by_logical_qubit, N_LOGICAL)
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        dev = qml.device("default.qubit", wires=self.n_physical)
        tiers = self.tiers
        primary_wires = self.primary_wires

        def circuit(features, weights, prob, noise_type, seed):
            rng = np.random.default_rng(seed)

            _generator_gates(features, weights, wires=primary_wires)

            for lq in range(N_LOGICAL):
                tier = tiers[lq]
                primary = primary_wires[lq]
                if tier == "shor":
                    _shor_encode(primary)
                elif tier == "repetition":
                    _repetition_encode(primary)

            for lq in range(N_LOGICAL):
                tier = tiers[lq]
                primary = primary_wires[lq]
                size = TIER_SIZES[tier]
                for w in range(primary, primary + size):
                    _sample_pauli_noise(prob, rng, w, noise_type)

            for lq in range(N_LOGICAL):
                tier = tiers[lq]
                primary = primary_wires[lq]
                if tier == "shor":
                    _shor_decode(primary)
                elif tier == "repetition":
                    _repetition_decode(primary)

            return [qml.expval(qml.PauliZ(p)) for p in primary_wires] + [
                qml.expval(qml.PauliX(p)) for p in primary_wires
            ]

        return qml.QNode(circuit, dev)

    def _average_trajectories(self, features, weights, prob, noise_type, n_shots, base_seed):
        outputs = [self.circuit(features, weights, prob, noise_type, base_seed + i) for i in range(n_shots)]
        result = np.mean(outputs, axis=0)
        if noise_type == "readout":
            rng = np.random.default_rng(base_seed)
            flips = rng.choice([1.0, -1.0], size=len(result), p=[1 - prob, prob])
            result = result * flips
        return result

    def generate(self, features, weights, prob, noise_type="depolarizing", n_shots=50, base_seed=0):
        return self._average_trajectories(features, weights, prob, noise_type, n_shots, base_seed)

    def generate_batch(self, feature_matrix, weights, prob, noise_type="depolarizing", n_shots=50):
        return np.stack(
            [
                self.generate(f, weights, prob, noise_type, n_shots, base_seed=i * n_shots)
                for i, f in enumerate(feature_matrix)
            ]
        )

    def verify_identity(self, features, weights, n_trials=5):
        from src.quantum_model import make_ideal_qnode

        ideal_qnode = make_ideal_qnode()
        ideal_out = np.array(ideal_qnode(features, weights))

        for seed in range(n_trials):
            aa_out = self.generate(features, weights, prob=0.0, noise_type="depolarizing", n_shots=1, base_seed=seed)
            if not np.allclose(ideal_out, aa_out, atol=1e-6):
                print(f"IDENTITY CHECK FAILED at seed {seed}")
                print(f"  ideal: {ideal_out}")
                print(f"  application_aware: {aa_out}")
                return False
        print(f"Identity check passed across {n_trials} trials for tiers {self.tiers}.")
        return True