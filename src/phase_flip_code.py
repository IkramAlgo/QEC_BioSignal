"""Phase flip repetition code as an extra tier, plus a QEC class that accepts it.

Copy this file to  src/phase_flip_code.py

The bit flip code you already have stores a qubit as a|000> + b|111> and corrects one X error.
The phase flip code is the same idea in the Hadamard basis: a|+++> + b|--->. It corrects one
Z error and turns an X error into an uncorrected logical phase error. It uses the same number of
physical qubits (3), the same noise placement and the same readout as your existing tiers.

Tiers understood here: "bare" (1 qubit), "repetition" (3, bit flip code), "phaserep" (3, phase
flip code), "shor" (9). The generator, noise sampling, encoders and decoders of the existing
tiers are reused from src.application_aware_qec, so the new code is compared like for like.

Usage:
  from src.phase_flip_code import TierQEC
  qec = TierQEC({0: "phaserep", 1: "phaserep", 2: "phaserep", 3: "phaserep"})
  qec.verify_identity(features, weights)
  out = qec.generate_batch(feature_matrix, weights, prob, "phaseflip", n_shots=50)
"""

import numpy as np
import pennylane as qml

import src.application_aware_qec as aa

N_LOGICAL = aa.N_LOGICAL
TIER_SIZES = dict(aa.TIER_SIZES)
TIER_SIZES["phaserep"] = 3


def _phase_encode(primary):
    qml.CNOT(wires=[primary, primary + 1])
    qml.CNOT(wires=[primary, primary + 2])
    for w in (primary, primary + 1, primary + 2):
        qml.Hadamard(wires=w)


def _phase_decode(primary):
    for w in (primary, primary + 1, primary + 2):
        qml.Hadamard(wires=w)
    qml.CNOT(wires=[primary, primary + 1])
    qml.CNOT(wires=[primary, primary + 2])
    qml.Toffoli(wires=[primary + 1, primary + 2, primary])


ENCODERS = {"shor": aa._shor_encode, "repetition": aa._repetition_encode, "phaserep": _phase_encode}
DECODERS = {"shor": aa._shor_decode, "repetition": aa._repetition_decode, "phaserep": _phase_decode}


class TierQEC(aa.ApplicationAwareQEC):
    """Same behaviour as ApplicationAwareQEC, with the extra "phaserep" tier."""

    def __init__(self, tier_by_logical_qubit):
        self.tiers = tier_by_logical_qubit
        primary, cursor = [], 0
        for lq in range(N_LOGICAL):
            primary.append(cursor)
            cursor += TIER_SIZES[tier_by_logical_qubit[lq]]
        self.primary_wires, self.n_physical = primary, cursor
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        dev = qml.device("default.qubit", wires=self.n_physical)
        tiers, primary_wires = self.tiers, self.primary_wires

        def circuit(features, weights, prob, noise_type, seed):
            rng = np.random.default_rng(seed)
            aa._generator_gates(features, weights, wires=primary_wires)
            for lq in range(N_LOGICAL):
                if tiers[lq] in ENCODERS:
                    ENCODERS[tiers[lq]](primary_wires[lq])
            for lq in range(N_LOGICAL):
                for w in range(primary_wires[lq], primary_wires[lq] + TIER_SIZES[tiers[lq]]):
                    aa._sample_pauli_noise(prob, rng, w, noise_type)
            for lq in range(N_LOGICAL):
                if tiers[lq] in DECODERS:
                    DECODERS[tiers[lq]](primary_wires[lq])
            return [qml.expval(qml.PauliZ(p)) for p in primary_wires] + [
                qml.expval(qml.PauliX(p)) for p in primary_wires
            ]

        return qml.QNode(circuit, dev)