"""Consistent train and test split, used across train_generator.py,
run_sensitivity_ranking.py, and run_application_aware_experiment.py,
so the sensitivity ranking and the final comparison are measured on
held out windows, not the same windows the generator trained on.

Every script that touches this split must use the same test_fraction
and seed, otherwise they will not agree on which windows are held out.
Defaults below (0.2, seed 0) are the values used everywhere else in
this project, keep them matched unless you deliberately change all
three scripts together.
"""

import numpy as np


def train_test_split_indices(n_samples, test_fraction=0.2, seed=0):
    """Returns (train_idx, test_idx), sorted, deterministic given the
    same n_samples, test_fraction, and seed.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    n_test = max(1, int(round(n_samples * test_fraction)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return np.sort(train_idx), np.sort(test_idx)