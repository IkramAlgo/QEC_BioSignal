"""EDF inspection and single-channel EEG loading.

Uses pyedflib rather than MNE for header inspection and single-channel
reads. MNE's read_raw_edf tries to parse annotations even with
preload=False, and on wide (many-channel), long recordings that path
allocates an array sized for ALL channels regardless, which is what
blows up memory on files like ANPHY-Sleep (93 channels x several hours).
pyedflib reads the header and individual signals directly, without ever
touching channels you didn't ask for.

FALLBACK: pyedflib validates the EDF+ annotation channel record by
record when it opens a file. EPCTL10 has a damaged annotation channel
(from record 9528 to the end the annotation bytes contain waveform
data instead of annotation text), so pyedflib refuses to open it even
though the file size and header are fine. When pyedflib raises
OSError, the functions below fall back to reading the header and the
requested channel straight from the raw bytes, which never touches the
annotation channel. This is only used when pyedflib fails, so every
file that opened before still goes through the exact same code path.
"""

import numpy as np
import pyedflib


# ---------------------------------------------------------------------------
# raw byte reader, used only when pyedflib refuses a file
# ---------------------------------------------------------------------------

def _read_raw_header(edf_path: str) -> dict:
    with open(edf_path, "rb") as f:
        h = f.read(256)
        hdr_bytes = int(h[184:192])
        n_rec = int(h[236:244])
        rec_dur = float(h[244:252])
        ns = int(h[252:256])
        sig = f.read(ns * 256)

    sizes = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
    names = ["label", "transducer", "dim", "pmin", "pmax", "dmin", "dmax",
             "prefilter", "nsamp", "reserved"]
    fields = {}
    pos = 0
    for name, sz in zip(names, sizes):
        fields[name] = [sig[pos + i * sz: pos + (i + 1) * sz] for i in range(ns)]
        pos += sz * ns

    return {
        "hdr_bytes": hdr_bytes,
        "n_rec": n_rec,
        "rec_dur": rec_dur,
        "n_signals": ns,
        "labels": [x.decode("ascii", errors="replace").strip() for x in fields["label"]],
        "pmin": [float(x) for x in fields["pmin"]],
        "pmax": [float(x) for x in fields["pmax"]],
        "dmin": [float(x) for x in fields["dmin"]],
        "dmax": [float(x) for x in fields["dmax"]],
        "nsamp": [int(x) for x in fields["nsamp"]],
    }


def _inspect_edf_raw(edf_path: str) -> dict:
    hd = _read_raw_header(edf_path)
    sfreqs = [n / hd["rec_dur"] for n in hd["nsamp"]]
    return {
        "sfreq": sfreqs[0] if sfreqs else None,
        "sfreqs_per_channel": dict(zip(hd["labels"], sfreqs)),
        "n_channels": hd["n_signals"],
        "channels": hd["labels"],
        "duration_hours": hd["n_rec"] * hd["rec_dur"] / 3600,
        "annotations": [],  # annotation channel is not trusted in fallback mode
    }


def _load_channel_raw(edf_path: str, channel_name: str) -> tuple[np.ndarray, float]:
    hd = _read_raw_header(edf_path)
    labels = hd["labels"]
    if channel_name not in labels:
        raise ValueError(
            f"Channel '{channel_name}' not found. Available channels: {labels}"
        )
    idx = labels.index(channel_name)
    nsamp = hd["nsamp"]
    samples_per_record = sum(nsamp)
    start = sum(nsamp[:idx])
    n = nsamp[idx]

    # each record is samples_per_record little endian int16 values
    data = np.memmap(
        edf_path, dtype="<i2", mode="r", offset=hd["hdr_bytes"],
        shape=(hd["n_rec"], samples_per_record),
    )
    digital = np.asarray(data[:, start:start + n]).reshape(-1).astype(np.float64)

    dmin, dmax = hd["dmin"][idx], hd["dmax"][idx]
    pmin, pmax = hd["pmin"][idx], hd["pmax"][idx]
    gain = (pmax - pmin) / (dmax - dmin)
    signal = (digital - dmin) * gain + pmin
    sfreq = n / hd["rec_dur"]
    return signal, sfreq


# ---------------------------------------------------------------------------
# public API, unchanged signatures
# ---------------------------------------------------------------------------

def inspect_edf(edf_path: str) -> dict:
    """Read EDF header only (no bulk data) and return key metadata.

    Use this first, before writing any preprocessing code, so you know the
    real channel names and sampling rate instead of assuming them.
    """
    try:
        f = pyedflib.EdfReader(edf_path)
    except OSError as e:
        print(f"pyedflib could not open {edf_path} ({e}), using raw byte fallback.")
        return _inspect_edf_raw(edf_path)

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
    read, the other 90+ channels in a file like ANPHY-Sleep are never
    touched. Raises a clear error if the channel name doesn't match.
    """
    try:
        f = pyedflib.EdfReader(edf_path)
    except OSError as e:
        print(f"pyedflib could not open {edf_path} ({e}), using raw byte fallback.")
        return _load_channel_raw(edf_path, channel_name)

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