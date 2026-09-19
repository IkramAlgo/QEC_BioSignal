"""Runs the full pipeline, train then rank then evaluate, automatically
across every subject and seed listed below, and combines all results
into one table. This replaces manually retyping the same three
commands over and over for each subject and seed, which is exactly
where today's file mixups and typos came from.

Fill in SUBJECTS below with your actual downloaded EDF files before
running. The channel is usually "C3" for ANPHY-Sleep like EPCTL01, but
if a new subject's file uses different channel naming, run
scripts/inspect_edf.py on it first to confirm, do not assume.

This keeps the tier allocation in application_aware_qec.py FROZEN as
is, the same allocation used for all your seed runs on EPCTL01. We are
deliberately testing whether that one allocation generalizes across
subjects, not rebuilding it per subject, same reasoning as testing it
across seeds.

Usage:
  python scripts/run_multi_subject_study.py
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- FILL THIS IN with your actual subject files before running ----
SUBJECTS = [
    {"edf": "data/raw/EPCTL01.edf", "channel": "C3", "label": "EPCTL01"},
    {"edf": "data/raw/EPCTL02.edf", "channel": "C3", "label": "EPCTL02"},
    {"edf": "data/raw/EPCTL03.edf", "channel": "C3", "label": "EPCTL03"},
    {"edf": "data/raw/EPCTL04.edf", "channel": "C3", "label": "EPCTL04"},
    {"edf": "data/raw/EPCTL05.edf", "channel": "C3", "label": "EPCTL05"},
]

SEEDS = [0, 1, 2]
MAX_WINDOWS = 100

WEIGHTS_PATH = PROJECT_ROOT / "outputs" / "trained_weights.npy"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "application_aware_results_testsplit.csv"
COMBINED_PATH = PROJECT_ROOT / "outputs" / "multi_subject_results.csv"


def run_step(script_name, edf_path, channel, seed, extra_args=None):
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / script_name),
        edf_path,
        "--channel", channel,
        "--max_windows", str(MAX_WINDOWS),
        "--seed", str(seed),
    ]
    if extra_args:
        cmd += extra_args
    print(f"  running {script_name} ...")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed for {edf_path} seed {seed}, stopping the study run.")


def main():
    if len(SUBJECTS) < 2:
        print(
            "Only one subject listed in SUBJECTS. Add your other downloaded "
            "subject files to the SUBJECTS list at the top of this script "
            "before running the multi subject study."
        )
        return

    all_results = []

    for subj in SUBJECTS:
        for seed in SEEDS:
            print(f"\n=== subject {subj['label']}, seed {seed} ===")

            run_step("train_generator.py", subj["edf"], subj["channel"], seed)
            run_step(
                "run_sensitivity_ranking.py",
                subj["edf"], subj["channel"], seed,
                extra_args=["--weights_path", str(WEIGHTS_PATH)],
            )
            run_step(
                "run_application_aware_experiment.py",
                subj["edf"], subj["channel"], seed,
                extra_args=["--weights_path", str(WEIGHTS_PATH)],
            )

            df = pd.read_csv(RESULTS_PATH)
            df["subject"] = subj["label"]
            df["seed"] = seed
            all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(COMBINED_PATH, index=False)

    print(f"\nSaved combined results to {COMBINED_PATH}")
    print(combined.to_string(index=False))

    print("\nSummary, mean and std across all subjects and seeds, per method:")
    summary = combined.groupby("method")[["mse_vs_ideal", "wasserstein_vs_ideal"]].agg(["mean", "std"])
    print(summary.to_string())


if __name__ == "__main__":
    main()