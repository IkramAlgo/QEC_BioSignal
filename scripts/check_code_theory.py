"""Single fault tests and closed form check for the repetition, phase flip and Shor tiers.

Copy to  scripts/check_code_theory.py  and run from the project folder:
  python scripts\\check_code_theory.py

Part 1  inject ONE Pauli fault (X, Y or Z) on one physical qubit of logical qubit 0, decode, and
        report what happens to the <Z> and <X> readouts of logical qubit 0.
Part 2  inject TWO faults, to show where each code stops working.
Part 3  closed form prediction of the error ratio (protected / unprotected) for uniform
        repetition, compared with your measured sweep results if they are in outputs.

Noise model, as in your code: one Pauli fault per physical qubit with probability p, applied
once after encoding. Encoding, decoding and syndrome handling are noiseless.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pennylane as qml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import src.application_aware_qec as aa
from src.phase_flip_code import DECODERS, ENCODERS, TIER_SIZES
from src.quantum_model import N_LAYERS, N_QUBITS

PAULI = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}


def build(tier0, faults):
    tiers = {0: tier0, 1: "bare", 2: "bare", 3: "bare"}
    primary, n_phys = [], 0
    for lq in range(N_QUBITS):
        primary.append(n_phys)
        n_phys += TIER_SIZES[tiers[lq]]
    dev = qml.device("default.qubit", wires=n_phys)

    @qml.qnode(dev)
    def circ(features, weights):
        aa._generator_gates(features, weights, wires=primary)
        for lq in range(N_QUBITS):
            if tiers[lq] in ENCODERS:
                ENCODERS[tiers[lq]](primary[lq])
        for pauli, wire in faults:
            PAULI[pauli](wires=wire)
        for lq in range(N_QUBITS):
            if tiers[lq] in DECODERS:
                DECODERS[tiers[lq]](primary[lq])
        return [qml.expval(qml.PauliZ(p)) for p in primary] + [qml.expval(qml.PauliX(p)) for p in primary]

    return circ


def pick_inputs():
    """Random inputs whose clean <Z> and <X> readouts of qubit 0 are clearly non zero."""
    for seed in range(200):
        rng = np.random.default_rng(seed)
        f, w = rng.uniform(0, np.pi, 8), rng.uniform(0, 2 * np.pi, (N_LAYERS, N_QUBITS, 2))
        clean = np.array(build("bare", [])(f, w))
        if abs(clean[0]) > 0.25 and abs(clean[N_QUBITS]) > 0.25:
            return f, w
    raise RuntimeError("could not find test inputs")


def effect(tier0, faults, f, w):
    clean = np.array(build(tier0, [])(f, w))
    out = np.array(build(tier0, faults)(f, w))
    z, x = (clean[0], out[0]), (clean[N_QUBITS], out[N_QUBITS])

    def tag(c, o):
        if abs(o - c) < 1e-7:
            return "ok"
        return "sign flip" if abs(o + c) < 1e-7 else "changed"

    others = np.delete(np.abs(out - clean), [0, N_QUBITS]).max()
    note = "" if others < 1e-7 else "  (other qubits also changed)"
    return tag(*z), tag(*x), note


def part1(f, w):
    print("\nPART 1: one fault on logical qubit 0 (result for its <Z> and <X> readouts)")
    print(f"{'tier':12s}{'fault':8s}{'on qubit':10s}{'<Z> readout':14s}{'<X> readout':14s}")
    for tier in ["bare", "repetition", "phaserep", "shor"]:
        for pauli in "XYZ":
            res = {effect(tier, [(pauli, q)], f, w)[:2] for q in range(TIER_SIZES[tier])}
            for q in range(TIER_SIZES[tier]):
                z, x, note = effect(tier, [(pauli, q)], f, w)
                if tier in ("bare",) or q in (0, 1) or (tier == "shor" and q in (0, 1, 3)):
                    print(f"{tier:12s}{pauli:8s}{q:<10d}{z:14s}{x:14s}{note}")
            if len(res) > 1 and tier != "bare":
                print(f"   note: {tier} {pauli} fault gives different results on different qubits: {sorted(res)}")


def part2(f, w):
    print("\nPART 2: two faults (shows where each code breaks)")
    print(f"{'tier':12s}{'faults':22s}{'<Z> readout':14s}{'<X> readout':14s}")
    cases = [("repetition", [("X", 0), ("X", 1)]), ("repetition", [("Z", 0), ("Z", 1)]),
             ("phaserep", [("Z", 0), ("Z", 1)]), ("phaserep", [("X", 0), ("X", 1)]),
             ("shor", [("X", 0), ("X", 1)]), ("shor", [("X", 0), ("X", 3)]),
             ("shor", [("Z", 0), ("Z", 1)]), ("shor", [("Z", 0), ("Z", 3)])]
    for tier, faults in cases:
        z, x, note = effect(tier, faults, f, w)
        label = " ".join(f"{p}{q}" for p, q in faults)
        print(f"{tier:12s}{label:22s}{z:14s}{x:14s}{note}")


def f_mse(q, n_shots=50):
    """Mean squared error factor of a readout that flips sign with probability q, averaged over n_shots."""
    return q * q + q * (1 - q) / n_shots


def part3():
    print("\nPART 3: closed form for uniform repetition (bit flip code) versus your measured results")
    print("  logical error rate of a 3 qubit repetition block, physical rate p:")
    print("    bit flip noise  : q = 3p^2 - 2p^3            (two or more X faults)")
    print("    phase flip noise: q = 3p(1-p)^2 + p^3        (an odd number of Z faults)")
    print("  predicted error ratio, protected / unprotected = f(q) / f(p), f(q) = q^2 + q(1-q)/50")
    measured = {}
    path = ROOT / "outputs" / "sweep_pauli.csv"
    if path.exists():
        import pandas as pd

        d = pd.read_csv(path)
        for (nt, p), g in d.groupby(["noise_type", "noise_prob"]):
            t = g.groupby(["subject", "method"])["mse_vs_ideal"].mean().unstack("method")
            if "no_qec" in t and "uniform" in t:
                r = (t["uniform"] / t["no_qec"]).dropna()
                measured[(nt, round(float(p), 4))] = float(np.exp(np.log(r.clip(lower=1e-12)).mean()))
    print(f"\n  {'noise':11s}{'p':>6s}{'logical q':>12s}{'predicted':>12s}{'measured':>12s}")
    for nt, qf in [("bitflip", lambda p: 3 * p**2 - 2 * p**3),
                   ("phaseflip", lambda p: 3 * p * (1 - p) ** 2 + p**3)]:
        for p in (0.05, 0.10, 0.20):
            q = qf(p)
            m = measured.get((nt, p))
            print(f"  {nt:11s}{p:6.2f}{q:12.4f}{f_mse(q) / f_mse(p):12.3f}{(f'{m:.3f}' if m else 'n/a'):>12s}")
    print("\n  Shor block (9 qubits), logical error rates from the same style of counting:")
    for p in (0.05, 0.10, 0.20):
        b = 3 * p**2 - 2 * p**3           # inner block fails: two or more X faults in one block
        bit = 3 * b * (1 - b) ** 2 + b**3  # odd number of failed blocks acts as a logical phase error
        c = 3 * p * (1 - p) ** 2 + p**3   # a block flips its sign: odd number of Z faults in the block
        ph = 3 * c * c * (1 - c) + c**3   # outer majority fails
        print(f"    p = {p:.2f}: logical phase error under pure bit flip noise = {bit:.4f}"
              f"   logical error under pure phase flip noise = {ph:.4f}")


def main():
    f, w = pick_inputs()
    part1(f, w)
    part2(f, w)
    part3()


if __name__ == "__main__":
    main()