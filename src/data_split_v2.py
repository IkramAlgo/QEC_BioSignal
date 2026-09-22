"""Train and test split utilities.

train_test_split_indices: plain random split by window, used
throughout the QEC fidelity pipeline (train_generator.py,
run_sensitivity_ranking.py, run_application_aware_experiment.py,
run_classical_baseline.py). Keep using this one for those scripts,
it is what all of that project's existing results are built on.

group_train_test_split_indices: GROUP AWARE split, used only by
run_downstream_classification.py. A hypnogram epoch spans 3 windows.
Splitting windows independently at random, as the plain function
above does, could put two windows from the same epoch on opposite
sides of the split, near duplicate signal leaking between train and
test. This function splits whole epochs instead, so every window
belonging to one epoch stays on the same side. These two split
functions are deliberately different, for deliberately different
reasons, do not use one where the other belongs.
"""

import numpy as np


def train_test_split_indices(n_samples, test_fraction=0.2, seed=0):
    """Returns (train_idx, test_idx), sorted, deterministic given the
    same n_samples, test_fraction, and seed. Plain window level split.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    n_test = max(1, int(round(n_samples * test_fraction)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return np.sort(train_idx), np.sort(test_idx)


def group_train_test_split_indices(epoch_groups, test_fraction=0.2, seed=0):
    """Splits at the epoch level, then returns the flattened window
    indices for each side. epoch_groups is a list of lists of window
    indices, one list per epoch, as returned by
    hypnogram.epochs_to_window_labels.
    """
    rng = np.random.default_rng(seed)
    n_epochs = len(epoch_groups)
    perm = rng.permutation(n_epochs)
    n_test_epochs = max(1, int(round(n_epochs * test_fraction)))
    test_epoch_idx = perm[:n_test_epochs]
    train_epoch_idx = perm[n_test_epochs:]
    train_windows = sorted(w for i in train_epoch_idx for w in epoch_groups[i])
    test_windows = sorted(w for i in test_epoch_idx for w in epoch_groups[i])
    return np.array(train_windows), np.array(test_windows)