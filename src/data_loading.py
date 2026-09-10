"""EDF inspection and single-channel EEG loading.

Uses pyedflib rather than MNE for header inspection and single-channel
reads. MNE's read_raw_edf tries to parse annotations even with
preload=False, and on wide (many-channel), long recordings that path
allocates an array sized for ALL channels regardless — which is what
blows up memory on files like ANPHY-Sleep (93 channels x several hours).
pyedflib reads the header and individual signals directly, without ever
touching channels you didn't ask for.
"""

import numpy as np
import pyedflib


def inspect_edf(edf_path: str) -> dict:
    """Read EDF header only (no bulk data) and return key metadata.

    Use this first, before writing any preprocessing code, so you know the
    real channel names and sampling rate instead of assuming them.
    """
    f = pyedflib.EdfReader(edf_path)
    try:
        channels = f.getSignalLabels()
        sfreqs = [f.getSampleFrequency(i) for i in range(f.signals_in_file)]
        duration_sec = f.file_duration

        annotations = []
        try:
            onsets, durations, descs = f.readAnnotations()
            annotations = list(zip(onsets, durations, descs))
        except Exception:
            pass  # not all EDFs carry annotations in this call path

        info = {
            "sfreq": sfreqs[0] if sfreqs else None,  # dominant/first channel's rate
            "sfreqs_per_channel": dict(zip(channels, sfreqs)),
            "n_channels": f.signals_in_file,
            "channels": channels,
            "duration_hours": duration_sec / 3600,
            "annotations": annotations,
        }
    finally:
        f.close()
    return info


def load_eeg_channel(edf_path: str, channel_name: str) -> tuple[np.ndarray, float]:
    """Load a single EEG channel as a 1D numpy array.

    Returns (signal, sfreq). Only the requested channel's samples are
    read — the other 90+ channels in a file like ANPHY-Sleep are never
    touched. Raises a clear error if the channel name doesn't match.
    """
    f = pyedflib.EdfReader(edf_path)
    try:
        channels = f.getSignalLabels()
        if channel_name not in channels:
            raise ValueError(
                f"Channel '{channel_name}' not found. Available channels: {channels}"
            )
        idx = channels.index(channel_name)
        signal = f.readSignal(idx)
        sfreq = f.getSampleFrequency(idx)
    finally:
        f.close()
    return signal, sfreq