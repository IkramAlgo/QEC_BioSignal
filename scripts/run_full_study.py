"""Runs the full pipeline, train, rank, application aware evaluate,
classical baseline, automatically across every subject and seed
listed below, saving the combined results after EVERY single run,
not just at the end, so an interruption partway through does not
lose the whole combined table the way it did before.

Usage:
  python scripts/run_full_study.py
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
QUANTUM_RESULTS_PATH = PROJECT_ROOT / "outputs" / "application_aware_results_testsplit.csv"
CLASSICAL_RESULTS_PATH = PROJECT_ROOT / "outputs" / "classical_baseline_results.csv"
COMBINED_PATH = PROJECT_ROOT / "outputs" / "full_study_results.csv"


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


def load_existing_combined():
    if COMBINED_PATH.exists():
        return pd.read_csv(COMBINED_PATH).to_dict("records")
    return []


def main():
    all_rows = load_existing_combined()
    already_done = {(r["subject"], r["seed"]) for r in all_rows} if all_rows else set()
    if already_done:
        print(f"Found existing combined results for {len(already_done)} subject/seed pairs, will skip those.")

    for subj in SUBJECTS:
        for seed in SEEDS:
            if (subj["label"], seed) in already_done:
                print(f"Skipping {subj['label']} seed {seed}, already in {COMBINED_PATH}")
                continue

            print(f"\n=== subject {subj['label']}, seed {seed} ===")

            run_step("train_generator.py", subj["edf"], subj["channel"], seed)
            run_step(
                "run_sensitivity_ranking.py", subj["edf"], subj["channel"], seed,
                extra_args=["--weights_path", str(WEIGHTS_PATH)],
            )
            run_step(
                "run_application_aware_experiment.py", subj["edf"], subj["channel"], seed,
                extra_args=["--weights_path", str(WEIGHTS_PATH)],
            )
            run_step("run_classical_baseline.py", subj["edf"], subj["channel"], seed)

            quantum_df = pd.read_csv(QUANTUM_RESULTS_PATH)
            classical_df = pd.read_csv(CLASSICAL_RESULTS_PATH)

            for _, row in quantum_df.iterrows():
                all_rows.append({
                    "subject": subj["label"], "seed": seed, "framework": "quantum",
                    "method": row["method"], "mse_vs_ideal": row["mse_vs_ideal"],
                    "wasserstein_vs_ideal": row["wasserstein_vs_ideal"],
                })
            for _, row in classical_df.iterrows():
                all_rows.append({
                    "subject": subj["label"], "seed": seed, "framework": "classical",
                    "method": row["method"], "mse_vs_ideal": row["mse_vs_ideal"],
                    "wasserstein_vs_ideal": row["wasserstein_vs_ideal"],
                })

            # save after EVERY subject/seed, not just at the end
            pd.DataFrame(all_rows).to_csv(COMBINED_PATH, index=False)
            print(f"  saved progress to {COMBINED_PATH} ({len(all_rows)} rows so far)")

    combined = pd.DataFrame(all_rows)
    print(f"\nFull study complete. Saved {COMBINED_PATH}")
    print("\nSummary, mean and std across all subjects and seeds, per method:")
    summary = combined.groupby(["framework", "method"])[["mse_vs_ideal", "wasserstein_vs_ideal"]].agg(["mean", "std"])
    print(summary.to_string())


if __name__ == "__main__":
    main()