"""Runs the full pipeline, train, rank, application aware evaluate,
classical baseline, automatically across every subject and seed
listed below, saving the combined results after EVERY single run.

UPDATED:
  * --allocation_rule ranking (default) or error_aware. Each rule writes
    its OWN csv, so the two studies can never be mixed up or skipped by
    the resume logic:
      ranking      -> outputs/full_study_results.csv
      error_aware  -> outputs/full_study_results_erroraware.csv
  * The trained weights of every subject and seed are now copied to
    outputs/weights/SUBJECT_seedN.npy, so later experiments (new
    allocation rules, the downstream classifier) do not need retraining.
  * A failed step is retried once after a short pause. This covers the
    out of memory crash that stopped the study at EPCTL11 seed 1, where
    scipy could not allocate 181 MiB. Close other heavy programs while
    the study runs.

  * --subjects and --tag, for confirmation runs on subjects that were not
    part of the original 10. The tag goes into the results file name so the
    confirmation results never mix with the main study. EPCTL10 is read
    through the raw byte loader in src/data_loading.py, and only its first
    2.65 hours are valid, which covers the 100 windows used here.

Usage:
  python scripts/run_full_study.py
  python scripts/run_full_study.py --allocation_rule error_aware
  python scripts/run_full_study.py --allocation_rule error_aware --subjects EPCTL10 --tag confirm
  python scripts/run_full_study.py --allocation_rule ranking --subjects EPCTL10 --tag confirm
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# EPCTL08 is not in this dataset copy and EPCTL10 has a damaged file,
# so 11 and 12 take their place.
SUBJECT_IDS = [f"EPCTL{i:02d}" for i in [1, 2, 3, 4, 5, 6, 7, 9, 11, 12]]
SUBJECTS = [
    {"edf": f"data/raw/{sid}.edf", "channel": "C3", "label": sid}
    for sid in SUBJECT_IDS
]

SEEDS = [0, 1, 2]
MAX_WINDOWS = 100

OUT_DIR = PROJECT_ROOT / "outputs"
WEIGHTS_PATH = OUT_DIR / "trained_weights.npy"
WEIGHTS_ARCHIVE_DIR = OUT_DIR / "weights"
QUANTUM_RESULTS_PATH = OUT_DIR / "application_aware_results_testsplit.csv"
CLASSICAL_RESULTS_PATH = OUT_DIR / "classical_baseline_results.csv"


def run_step(script_name, edf_path, channel, seed, extra_args=None, retries=1):
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
    for attempt in range(retries + 1):
        print(f"  running {script_name} ..." + (f" (retry {attempt})" if attempt else ""))
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if result.returncode == 0:
            return
        if attempt < retries:
            print(f"  {script_name} failed, waiting 20 seconds and trying once more ...")
            time.sleep(20)
    raise RuntimeError(f"{script_name} failed for {edf_path} seed {seed}, stopping the study run.")


def load_existing_combined(path):
    if path.exists():
        return pd.read_csv(path).to_dict("records")
    return []


def tiers_text(row):
    value = row.get("tiers", "")
    return "" if pd.isna(value) else str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allocation_rule", default="ranking", choices=["ranking", "error_aware"])
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Subject ids to run, for example EPCTL10. Default is the 10 subject study.")
    parser.add_argument("--tag", default="", help="Added to the results file name, for example confirm.")
    args = parser.parse_args()

    subjects = SUBJECTS
    if args.subjects:
        subjects = [{"edf": f"data/raw/{sid}.edf", "channel": "C3", "label": sid} for sid in args.subjects]

    base_name = "full_study_results" if args.allocation_rule == "ranking" else "full_study_results_erroraware"
    suffix = f"_{args.tag}" if args.tag else ""
    combined_path = OUT_DIR / f"{base_name}{suffix}.csv"
    print(f"Allocation rule: {args.allocation_rule}, results file: {combined_path}")

    # preflight, fail now instead of hours into the run
    missing = [s["edf"] for s in subjects if not (PROJECT_ROOT / s["edf"]).exists()]
    if missing:
        print("These EDF files were not found, fix the names or paths before running:")
        for m in missing:
            print(f"  {m}")
        return

    WEIGHTS_ARCHIVE_DIR.mkdir(exist_ok=True, parents=True)

    all_rows = load_existing_combined(combined_path)
    already_done = {(r["subject"], r["seed"]) for r in all_rows} if all_rows else set()
    if already_done:
        print(f"Found existing combined results for {len(already_done)} subject/seed pairs, will skip those.")

    for subj in subjects:
        for seed in SEEDS:
            if (subj["label"], seed) in already_done:
                print(f"Skipping {subj['label']} seed {seed}, already in {combined_path}")
                continue

            print(f"\n=== subject {subj['label']}, seed {seed} ===")

            run_step("train_generator.py", subj["edf"], subj["channel"], seed)
            shutil.copy(WEIGHTS_PATH, WEIGHTS_ARCHIVE_DIR / f"{subj['label']}_seed{seed}.npy")

            run_step(
                "run_sensitivity_ranking.py", subj["edf"], subj["channel"], seed,
                extra_args=["--weights_path", str(WEIGHTS_PATH), "--ranking_split", "train"],
            )
            run_step(
                "run_application_aware_experiment.py", subj["edf"], subj["channel"], seed,
                extra_args=["--weights_path", str(WEIGHTS_PATH), "--allocation_rule", args.allocation_rule],
            )
            run_step("run_classical_baseline.py", subj["edf"], subj["channel"], seed)

            quantum_df = pd.read_csv(QUANTUM_RESULTS_PATH)
            classical_df = pd.read_csv(CLASSICAL_RESULTS_PATH)

            for _, row in quantum_df.iterrows():
                all_rows.append({
                    "subject": subj["label"], "seed": seed, "framework": "quantum",
                    "method": row["method"], "mse_vs_ideal": row["mse_vs_ideal"],
                    "wasserstein_vs_ideal": row["wasserstein_vs_ideal"],
                    "tiers": tiers_text(row),
                })
            for _, row in classical_df.iterrows():
                all_rows.append({
                    "subject": subj["label"], "seed": seed, "framework": "classical",
                    "method": row["method"], "mse_vs_ideal": row["mse_vs_ideal"],
                    "wasserstein_vs_ideal": row["wasserstein_vs_ideal"],
                    "tiers": "",
                })

            # save after EVERY subject/seed, not just at the end
            pd.DataFrame(all_rows).to_csv(combined_path, index=False)
            print(f"  saved progress to {combined_path} ({len(all_rows)} rows so far)")

    combined = pd.DataFrame(all_rows)
    print(f"\nFull study complete. Saved {combined_path}")
    print("\nSummary, mean and std across all subjects and seeds, per method:")
    summary = combined.groupby(["framework", "method"])[["mse_vs_ideal", "wasserstein_vs_ideal"]].agg(["mean", "std"])
    print(summary.to_string())


if __name__ == "__main__":
    main()