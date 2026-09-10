"""Metrics for comparing generated-feature distributions (ideal vs noisy vs QEC)."""

import numpy as np
from scipy.stats import wasserstein_distance


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def mean_wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    """Mean 1D Wasserstein distance across output dimensions.

    a, b: shape (n_samples, n_dims). Computed per-dimension then averaged,
    which is a reasonable stand-in for full distribution distance while the
    pipeline is still being validated.
    """
    dims = a.shape[1]
    dists = [wasserstein_distance(a[:, d], b[:, d]) for d in range(dims)]
    return float(np.mean(dists))


def summarize(ideal: np.ndarray, comparison: np.ndarray) -> dict:
    return {
        "mse": mse(ideal, comparison),
        "mean_wasserstein": mean_wasserstein(ideal, comparison),
    }
