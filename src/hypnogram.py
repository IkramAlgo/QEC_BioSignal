"""Hypnogram parsing and alignment to quantum pipeline windows.

File format, confirmed by inspect_hypnogram.py: tab separated,
label, start_time_seconds, duration_seconds. Duration is always 30
in this dataset, three 10 second windows per epoch.

This is the SINGLE authoritative version of this file. It combines
three things that were previously built and lost separately:
  1. Skip the unlabeled / 'L' prefix at the start of a recording,
     rather than assuming labels start at window 0.
  2. Auto-expand the window budget if the initial budget contains
     fewer than 2 distinct sleep stages, since some subjects stay
     awake far longer than others after lights-off (EPCTL04 needed
     400 windows before a second stage appeared at all).
  3. Group-aware epoch splitting, so no 30 second epoch is split
     across train and test.
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


def find_first_usable_window(epochs, window_sec, drop_labels=("L",)):
    """Returns the window index of the first non-dropped epoch, so we
    do not assume labels start at window 0.
    """
    for label, start_sec, duration_sec in sorted(epochs, key=lambda e: e[1]):
        if label not in drop_labels:
            return start_sec // window_sec
    raise ValueError("No usable (non-dropped) epochs found in this hypnogram at all.")


def epochs_to_window_labels(epochs, window_sec, max_windows, drop_labels=("L",), offset_windows=0):
    """
    Maps hypnogram epochs onto window indices, relative to
    offset_windows (from find_first_usable_window).

    An epoch is kept only if every window it spans, after subtracting
    the offset, falls within [0, max_windows) and its label is not in
    drop_labels.

    Returns:
      window_label: dict, window_index -> label (offset-adjusted).
      epoch_groups: list of lists of window indices, one list per kept
        epoch. Use this for group_train_test_split_indices, so no
        epoch is split across train and test.
    """
    window_label = {}
    epoch_groups = []
    for label, start_sec, duration_sec in epochs:
        if label in drop_labels:
            continue
        start_window = start_sec // window_sec - offset_windows
        n_windows_this_epoch = duration_sec // window_sec
        window_indices = list(range(start_window, start_window + n_windows_this_epoch))
        if any(w < 0 or w >= max_windows for w in window_indices):
            continue
        for w in window_indices:
            window_label[w] = label
        epoch_groups.append(window_indices)
    return window_label, epoch_groups


def find_diverse_window_range(epochs, window_sec, min_windows, drop_labels=("L",), min_classes=2):
    """Finds (offset_windows, window_count) covering at least
    min_classes distinct sleep stages, expanding past min_windows if
    the initial budget is not diverse enough.

    Skipping the leading 'L' prefix is not always enough on its own:
    some subjects stay awake for a long time after lights-off, so the
    first min_windows windows after the prefix can still be a single
    stage, which a classifier cannot be trained or evaluated on. This
    doubles the search window until min_classes distinct labels are
    found or the recording (per the hypnogram's own last epoch end
    time) runs out.

    NOTE, a residual limitation worth knowing about: this checks
    diversity across the whole window range, not that the train and
    test split specifically each get 2+ classes after the group split.
    With the small epoch counts here that is a real, if secondary,
    risk, not something this function fully rules out.
    """
    offset = find_first_usable_window(epochs, window_sec, drop_labels)
    last_epoch_end_sec = max(start + dur for _, start, dur in epochs)
    max_available_windows = last_epoch_end_sec // window_sec - offset

    search_windows = min_windows
    while True:
        window_label, _ = epochs_to_window_labels(
            epochs, window_sec=window_sec, max_windows=search_windows,
            drop_labels=drop_labels, offset_windows=offset,
        )
        if len(set(window_label.values())) >= min_classes:
            return offset, search_windows
        if search_windows >= max_available_windows:
            raise ValueError(
                f"Even using the rest of the recording after window {offset}, "
                f"only found these stages: {set(window_label.values())}. "
                f"Need at least {min_classes} distinct stages to train a classifier."
            )
        search_windows = min(search_windows * 2, max_available_windows)