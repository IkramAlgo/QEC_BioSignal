"""Shor 9 qubit code protecting the generator's output from post
generation noise (storage, transmission, readout), covering both bit
flip and phase flip errors.

Why this design over hand building a Steane code from scratch: Shor's
code is literally two nested applications of the SAME bit flip
repetition block already verified working in repetition_code.py. The
outer layer is that identical block conjugated by Hadamard gates,
which converts bit flip protection into phase flip protection. Reusing
a verified building block twice is lower risk than transcribing a new
stabilizer circuit from memory.

Structure per logical qubit, 9 physical qubits, laid out as 3 outer
blocks of 3:

  outer block 0: physical qubits [0, 1, 2]
  outer block 1: physical qubits [3, 4, 5]
  outer block 2: physical qubits [6, 7, 8]

  the "outer primary" of each block is its first qubit, offsets 0, 3, 6

Sequence:
  1. generator gate runs on physical qubit 0 only (the overall primary)
  2. outer encode: spread qubit 0 to qubit 3 and qubit 6 with CNOT, then
     apply Hadamard to qubits 0, 3, 6. This is the bit flip encode
     conjugated by Hadamard, which is exactly what protects against
     phase flip instead of bit flip.
  3. inner encode: within each outer block, spread the block's primary
     to its other two qubits with CNOT. This is the plain bit flip
     encode, unchanged from repetition_code.py.
  4. noise on all 9 physical qubits
  5. inner decode: within each outer block, syndrome extraction CNOTs
     then Toffoli majority vote, exactly the verified pattern from
     repetition_code.py, corrects bit flip errors inside each block.
  6. outer decode: undo the Hadamards on qubits 0, 3, 6, then syndrome
     extraction CNOTs and Toffoli majority vote across the three block
     primaries, corrects phase flip errors that survived as bit flips
     in the Hadamard basis.

At prob=0 (no noise), encode then decode is provably an identity,
same guarantee as the repetition code redesign. Confirm this with
verify_identity() before trusting any noisy result.
"""

import numpy as np
import pennylane as qml

from src.quantum_model import N_QUBITS as N_LOGICAL, N_LAYERS

INNER_DISTANCE = 3
OUTER_DISTANCE = 3
BLOCK_SIZE = INNER_DISTANCE * OUTER_DISTANCE  # 9 physical qubits per logical qubit
N_PHYSICAL = N_LOGICAL * BLOCK_SIZE


def _generator_gates(features, weights, wires):
    """Identical gate sequence to quantum_model and repetition_code's
    generator, parameterized over arbitrary wires.
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
    """Trajectory sampled equivalent of a single qubit noise channel.
    Same as repetition_code.py, kept identical on purpose.
    """
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


def _shor_encode(block_base):
    """block_base is the index of physical qubit 0 for this logical
    qubit's 9 qubit block. Encodes whatever state is currently on
    block_base into the full Shor code across block_base..block_base+8.
    """
    outer_primaries = [block_base, block_base + 3, block_base + 6]

    # outer encode: spread to the other two block primaries, then
    # Hadamard all three, converting bit flip protection into phase
    # flip protection
    qml.CNOT(wires=[outer_primaries[0], outer_primaries[1]])
    qml.CNOT(wires=[outer_primaries[0], outer_primaries[2]])
    for p in outer_primaries:
        qml.Hadamard(wires=p)

    # inner encode: plain bit flip spread within each outer block
    for p in outer_primaries:
        qml.CNOT(wires=[p, p + 1])
        qml.CNOT(wires=[p, p + 2])


def _shor_decode(block_base):
    outer_primaries = [block_base, block_base + 3, block_base + 6]

    # inner decode: syndrome extraction then Toffoli majority vote,
    # same pattern as repetition_code.py, applied inside each block
    for p in outer_primaries:
        qml.CNOT(wires=[p, p + 1])
        qml.CNOT(wires=[p, p + 2])
        qml.Toffoli(wires=[p + 1, p + 2, p])

    # outer decode: undo the Hadamards, then syndrome extraction and
    # Toffoli majority vote across the three block primaries
    for p in outer_primaries:
        qml.Hadamard(wires=p)
    qml.CNOT(wires=[outer_primaries[0], outer_primaries[1]])
    qml.CNOT(wires=[outer_primaries[0], outer_primaries[2]])
    qml.Toffoli(wires=[outer_primaries[1], outer_primaries[2], outer_primaries[0]])


def _build_shor_qec_circuit():
    dev = qml.device("default.qubit", wires=N_PHYSICAL)
    primary_wires = [lq * BLOCK_SIZE for lq in range(N_LOGICAL)]

    def circuit(features, weights, prob, noise_type, seed):
        rng = np.random.default_rng(seed)

        # generator runs once per logical qubit, on that qubit's
        # overall primary physical qubit, same gate sequence as
        # every other circuit in this project
        _generator_gates(features, weights, wires=primary_wires)

        for lq in range(N_LOGICAL):
            _shor_encode(lq * BLOCK_SIZE)

        for w in range(N_PHYSICAL):
            _sample_pauli_noise(prob, rng, w, noise_type)

        for lq in range(N_LOGICAL):
            _shor_decode(lq * BLOCK_SIZE)

        return [qml.expval(qml.PauliZ(lq * BLOCK_SIZE)) for lq in range(N_LOGICAL)] + [
            qml.expval(qml.PauliX(lq * BLOCK_SIZE)) for lq in range(N_LOGICAL)
        ]

    return qml.QNode(circuit, dev)


_shor_circuit = _build_shor_qec_circuit()


def _average_trajectories(circuit, features, weights, prob, noise_type, n_shots, base_seed):
    outputs = [circuit(features, weights, prob, noise_type, base_seed + i) for i in range(n_shots)]
    result = np.mean(outputs, axis=0)
    if noise_type == "readout":
        rng = np.random.default_rng(base_seed)
        flips = rng.choice([1.0, -1.0], size=len(result), p=[1 - prob, prob])
        result = result * flips
    return result


def generate_shor_qec(features, weights, prob, noise_type="depolarizing", n_shots=50, base_seed=0):
    return _average_trajectories(_shor_circuit, features, weights, prob, noise_type, n_shots, base_seed)


def generate_shor_qec_batch(feature_matrix, weights, prob, noise_type="depolarizing", n_shots=50):
    return np.stack(
        [
            generate_shor_qec(f, weights, prob, noise_type, n_shots, base_seed=i * n_shots)
            for i, f in enumerate(feature_matrix)
        ]
    )


def verify_identity_single_qubit(theta_y, theta_z, n_trials=5):
    """Test the Shor encode and decode building block alone, on ONE
    logical qubit, 9 physical qubits total. This deliberately skips the
    generator's cross qubit entangling ring, since that ring is what
    forces the full 4 logical qubit version to need 36 physical qubits
    and a statevector too large to allocate. Correctness of the code
    itself does not depend on how many logical qubits it is later
    applied to, so this is a fully valid check that costs almost no
    memory, 2**9 amplitudes instead of 2**36.

    theta_y and theta_z are arbitrary rotation angles used to prepare
    some non trivial state on qubit 0 before encoding, so the check is
    not just verifying the all zero state.
    """
    dev = qml.device("default.qubit", wires=BLOCK_SIZE)

    def circuit():
        qml.RY(theta_y, wires=0)
        qml.RZ(theta_z, wires=0)
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))]

    def circuit_encode_decode():
        qml.RY(theta_y, wires=0)
        qml.RZ(theta_z, wires=0)
        _shor_encode(0)
        _shor_decode(0)
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))]

    plain_qnode = qml.QNode(circuit, dev)
    encode_decode_qnode = qml.QNode(circuit_encode_decode, dev)

    plain_out = np.array(plain_qnode())
    encode_decode_out = np.array(encode_decode_qnode())

    if np.allclose(plain_out, encode_decode_out, atol=1e-6):
        print("Single qubit identity check passed, Shor encode then decode matches the unencoded state.")
        print(f"  plain:         {plain_out}")
        print(f"  encode/decode: {encode_decode_out}")
        return True
    else:
        print("SINGLE QUBIT IDENTITY CHECK FAILED")
        print(f"  plain:         {plain_out}")
        print(f"  encode/decode: {encode_decode_out}")
        return False


def verify_identity(features, weights, n_trials=5):
    """Run the encode decode cycle at prob=0 and confirm it matches the
    plain generator exactly, before trusting any noisy result. This is
    the check that should have existed before the first repetition code
    experiment ran. Run this first, every time, on any new code.
    """
    from src.quantum_model import make_ideal_qnode

    ideal_qnode = make_ideal_qnode()
    ideal_out = np.array(ideal_qnode(features, weights))

    for seed in range(n_trials):
        shor_out = generate_shor_qec(features, weights, prob=0.0, noise_type="depolarizing", n_shots=1, base_seed=seed)
        if not np.allclose(ideal_out, shor_out, atol=1e-6):
            print(f"IDENTITY CHECK FAILED at seed {seed}")
            print(f"  ideal: {ideal_out}")
            print(f"  shor:  {shor_out}")
            return False
    print(f"Identity check passed across {n_trials} trials, ideal matches Shor code at zero noise.")
    return True