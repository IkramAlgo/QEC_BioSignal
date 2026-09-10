"""Basic EEG cleaning and windowing."""

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(
    signal: np.ndarray, sfreq: float, low_hz: float = 0.5, high_hz: float = 40.0, order: int = 4
) -> np.ndarray:
    """Standard EEG bandpass (0.5-40 Hz) to remove drift and high-frequency noise."""
    nyq = sfreq / 2.0
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    return filtfilt(b, a, signal)


def segment_windows(
    signal: np.ndarray, sfreq: float, window_sec: float = 10.0, overlap: float = 0.0
) -> list[np.ndarray]:
    """Split a 1D signal into fixed-length windows.

    overlap is a fraction in [0, 1). 0.0 = non-overlapping windows.
    """
    win_len = int(window_sec * sfreq)
    step = int(win_len * (1 - overlap))
    if step <= 0:
        raise ValueError("overlap too close to 1.0, step size collapsed to 0")

    windows = []
    for start in range(0, len(signal) - win_len + 1, step):
        windows.append(signal[start : start + win_len])
    return windows


def preprocess_and_segment(
    signal: np.ndarray,
    sfreq: float,
    window_sec: float = 10.0,
    overlap: float = 0.0,
) -> list[np.ndarray]:
    """Filter then window — the standard entry point for the rest of the pipeline."""
    filtered = bandpass_filter(signal, sfreq)
    return segment_windows(filtered, sfreq, window_sec=window_sec, overlap=overlap)
