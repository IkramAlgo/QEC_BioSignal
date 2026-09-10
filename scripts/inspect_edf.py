"""Usage: python scripts/inspect_edf.py path/to/file.edf"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import inspect_edf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("edf_path")
    args = parser.parse_args()

    info = inspect_edf(args.edf_path)

    print(f"Sampling frequency: {info['sfreq']} Hz")
    print(f"Number of channels: {info['n_channels']}")
    print(f"Duration: {info['duration_hours']:.2f} hours")
    print("\nChannels:")
    for ch in info["channels"]:
        print(f"  {ch}")
    print(f"\nAnnotations: {len(info['annotations'])}")
    for onset, duration, desc in info["annotations"][:20]:
        print(f"  t={onset:.1f}s  dur={duration:.1f}s  {desc}")
    if len(info["annotations"]) > 20:
        print(f"  ... and {len(info['annotations']) - 20} more")


if __name__ == "__main__":
    main()
