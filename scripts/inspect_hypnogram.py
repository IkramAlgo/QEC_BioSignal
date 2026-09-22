"""Prints the raw structure of a hypnogram file before we write a
parser against it. Run this first, confirm the format, then the real
parser in src/hypnogram.py gets calibrated to match exactly.

Usage:
  python scripts/inspect_hypnogram.py data/raw/EPCTL01.txt
"""

import sys
from pathlib import Path
from collections import Counter

path = Path(sys.argv[1])
lines = path.read_text().splitlines()

print(f"Total lines: {len(lines)}")
print("\nFirst 15 lines, raw:")
for line in lines[:15]:
    print(f"  {line!r}")

print("\nLast 5 lines, raw:")
for line in lines[-5:]:
    print(f"  {line!r}")

print("\nUnique first tokens (split on whitespace), with counts:")
first_tokens = Counter(line.split()[0] if line.split() else "" for line in lines)
for token, count in first_tokens.most_common(20):
    print(f"  {token!r}: {count}")

print("\nUnique full lines (if lines are short, e.g. just a label), with counts:")
full_line_counts = Counter(line.strip() for line in lines)
for line, count in full_line_counts.most_common(20):
    print(f"  {line!r}: {count}")