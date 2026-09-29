"""Runs train_generator, run_sensitivity_ranking, and
run_downstream_classification automatically across every subject and
seed, skipping any subject/seed combination already present in
outputs/downstream_predictions_combined.csv. Safe to leave running
overnight, and safe to rerun if interrupted, it picks up where it
left off.

Fill in SUBJECTS below with your five subject files and their
matching hypnogram files.

Usage:
  python scripts/run_downstream_study.py
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUBJECTS = [
    {"edf": "data/raw/EPCTL01.edf", "hyp": "data/raw/EPCTL01.txt", "channel": "C3", "label": "EPCTL01"},
    {"edf": "data/raw/EPCTL02.edf", "hyp": "data/raw/EPCTL02.txt", "channel": "C3", "label": "EPCTL02"},
    {"edf": "data/raw/EPCTL03.edf", "hyp": "data/raw/EPCTL03.txt", "channel": "C3", "label": "EPCTL03"},
    {"edf": "data/raw/EPCTL04.edf", "hyp": "data/raw/EPCTL04.txt", "channel": "C3", "label": "EPCTL04"},
    {"edf": "data/raw/EPCTL05.edf", "hyp": "data/raw/EPCTL05.txt", "channel": "C3", "label": "EPCTL05"},
]

SEEDS = [0, 1, 2]
MAX_WINDOWS = 100

WEIGHTS_PATH = PROJECT_ROOT / "outputs" / "trained_weights.npy"
PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "downstream_predictions_combined.csv"


def run_step(script_name, args_list):
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / script_name)] + args_list
    print(f"  running {script_name} ...")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed, stopping here so nothing bad gets built on top of it.")


def already_done():
    if not PREDICTIONS_PATH.exists():
        return set()
    df = pd.read_csv(PREDICTIONS_PATH)
    return set(zip(df["subject"], df["seed"]))


def main():
    done = already_done()
    if done:
        print(f"Found {len(done)} subject/seed pairs already in {PREDICTIONS_PATH}, will skip those.\n")

    for subj in SUBJECTS:
        for seed in SEEDS:
            if (subj["label"], seed) in done:
                print(f"Skipping {subj['label']} seed {seed}, already done.")
                continue

            print(f"\n=== {subj['label']}, seed {seed} ===")

            run_step("train_generator.py", [
                subj["edf"], "--channel", subj["channel"],
                "--max_windows", str(MAX_WINDOWS), "--seed", str(seed),
            ])
            run_step("run_sensitivity_ranking.py", [
                subj["edf"], "--channel", subj["channel"],
                "--max_windows", str(MAX_WINDOWS), "--seed", str(seed),
                "--weights_path", str(WEIGHTS_PATH),
            ])
            run_step("run_downstream_classification.py", [
                subj["edf"], subj["hyp"], "--channel", subj["channel"],
                "--max_windows", str(MAX_WINDOWS), "--seed", str(seed),
                "--weights_path", str(WEIGHTS_PATH),
            ])

            print(f"  done with {subj['label']} seed {seed}")

    print("\nAll subjects and seeds complete.")
    print("Run scripts\\analyze_downstream_classification.py next to get the pooled, final numbers.")


if __name__ == "__main__":
    main()