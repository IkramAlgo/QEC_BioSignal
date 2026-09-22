"""Classical baseline for the 'why does this need to be quantum' question.

Design, chosen for fairness rather than for flattering either side:

  - Same noise probability as every quantum experiment in this project,
    0.02, applied independently to the classical features. It is NOT
    tuned to match the quantum no_qec degradation, that would bias the
    comparison toward whichever side we tuned it to match.

  - Same correction concept as the quantum side: repeated noisy
    measurement averaging, same n_shots (50) as every quantum
    trajectory average in this project. This is the closest classical
    analogue of what generate_repetition_qec_batch and
    generate_application_aware_qec_batch already do, so neither
    pipeline gets a structural advantage from a different correction
    strategy.

  - Same reference target: the [-1, 1] rescaled ground truth features,
    the exact quantity the quantum generator is trained to reconstruct
    in train_generator.py. Both pipelines are scored against the same
    thing, on the same held out windows.

Noise model: with probability `prob`, a feature dimension is replaced
with a fresh uniform random value in [-1, 1], the continuous analogue
of a disruptive flip event, structurally parallel to what a Pauli
error does to a qubit's expectation value, a large, uncorrelated
disruption rather than a small perturbation.
"""

import numpy as np


def _corrupt_once(target, prob, rng):
    """One noisy classical measurement of the target vector."""
    corrupted = target.copy()
    for i in range(len(target)):
        if rng.random() < prob:
            corrupted[i] = rng.uniform(-1, 1)
    return corrupted


def generate_classical_no_correction(target, prob, base_seed=0):
    """Single noisy measurement, no correction. Classical analogue of
    no_qec, one shot, nothing done to fix it.
    """
    rng = np.random.default_rng(base_seed)
    return _corrupt_once(target, prob, rng)


def generate_classical_averaged(target, prob, n_shots=50, base_seed=0):
    """n_shots independent noisy measurements, averaged. Classical
    analogue of the repeated-trajectory averaging already used for
    every quantum QEC method in this project.
    """
    samples = []
    for i in range(n_shots):
        rng = np.random.default_rng(base_seed + i)
        samples.append(_corrupt_once(target, prob, rng))
    return np.mean(samples, axis=0)


def generate_classical_no_correction_batch(target_matrix, prob, base_seed_offset=0):
    return np.stack(
        [
            generate_classical_no_correction(t, prob, base_seed=base_seed_offset + i)
            for i, t in enumerate(target_matrix)
        ]
    )


def generate_classical_averaged_batch(target_matrix, prob, n_shots=50, base_seed_offset=0):
    return np.stack(
        [
            generate_classical_averaged(t, prob, n_shots, base_seed=(base_seed_offset + i) * n_shots)
            for i, t in enumerate(target_matrix)
        ]
    )