"""Hypnogram parsing and alignment to quantum pipeline windows.

File format, confirmed by inspect_hypnogram.py: tab separated,
label, start_time_seconds, duration_seconds. Duration is always 30
in this dataset, three 10 second windows per epoch.

Alignment uses the epoch's actual start time divided by window_sec,
not an assumption that hypnogram lines and windows are in lockstep
from the first line, which would silently break if either file ever
has a gap or offset.
"""

from pathlib import Path


def load_hypnogram(path):
    """Returns a list of (label, start_sec, duration_sec) tuples."""
    epochs = []
    for line in Path(path).read_text().splitlines():
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue
        label, start, duration = parts
        epochs.append((label, int(start), int(duration)))
    return epochs


def epochs_to_window_labels(epochs, window_sec, max_windows, drop_labels=("L",)):
    """
    Maps hypnogram epochs onto window indices.

    An epoch is kept only if every window it spans is within
    max_windows (the truncated window set used by --max_windows
    elsewhere in this project) and its label is not in drop_labels.

    Returns:
      window_label: dict, window_index -> label, for every window
        belonging to a kept epoch.
      epoch_groups: list of lists of window indices, one list per kept
        epoch. Use this for a GROUP AWARE train/test split, so no
        epoch is split across train and test.
    """
    window_label = {}
    epoch_groups = []
    for label, start_sec, duration_sec in epochs:
        if label in drop_labels:
            continue
        start_window = start_sec // window_sec
        n_windows_this_epoch = duration_sec // window_sec
        window_indices = list(range(start_window, start_window + n_windows_this_epoch))
        if any(w >= max_windows for w in window_indices):
            continue
        for w in window_indices:
            window_label[w] = label
        epoch_groups.append(window_indices)
    return window_label, epoch_groups