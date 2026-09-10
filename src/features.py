"""Reduce each EEG window to an 8-dim feature vector for quantum encoding."""

import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "sigma": (13, 16),
    "beta": (16, 30),
}


def band_power(window: np.ndarray, sfreq: float, band: tuple[float, float]) -> float:
    freqs, psd = welch(window, sfreq, nperseg=min(len(window), 256))
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(np.trapz(psd[mask], freqs[mask])) if mask.any() else 0.0


def spectral_entropy(window: np.ndarray, sfreq: float) -> float:
    freqs, psd = welch(window, sfreq, nperseg=min(len(window), 256))
    psd_norm = psd / (psd.sum() + 1e-12)
    return float(entropy(psd_norm + 1e-12))


def autocorrelation(window: np.ndarray, lag: int = 1) -> float:
    if len(window) <= lag:
        return 0.0
    x = window - window.mean()
    denom = np.dot(x, x)
    if denom == 0:
        return 0.0
    return float(np.dot(x[:-lag], x[lag:]) / denom)


def extract_features(window: np.ndarray, sfreq: float) -> np.ndarray:
    """Returns an 8-dim vector:
    [delta, theta, alpha, sigma, beta, variance, spectral_entropy, autocorr]
    """
    powers = [band_power(window, sfreq, BANDS[b]) for b in ("delta", "theta", "alpha", "sigma", "beta")]
    variance = float(np.var(window))
    sent = spectral_entropy(window, sfreq)
    ac = autocorrelation(window)
    return np.array(powers + [variance, sent, ac], dtype=np.float64)


def extract_feature_matrix(windows: list[np.ndarray], sfreq: float) -> np.ndarray:
    """Shape: (n_windows, 8)."""
    return np.stack([extract_features(w, sfreq) for w in windows])
