"""Controls and ablations for the application aware allocation.

Uses the trained weights that run_full_study.py saved in outputs/weights,
so nothing is retrained. For every subject and seed it rebuilds the same
20 held out test windows, evaluates a set of FIXED protection plans under
the same noise (depolarizing, p=0.02, 50 shots) and saves one row per
subject, seed and plan.

Plans (logical qubits 0 to 3):
  plan_A       0 repetition, 1 bare, 2 Shor, 3 repetition. The plan the
               error aware rule picked in 25 of 30 runs. 16 physical qubits.
               This is the "one fixed plan for every subject" control.
  rep_only_03  0 repetition, 1 bare, 2 bare, 3 repetition. No Shor code at
               all, 8 physical qubits. Tests whether the Shor code is needed.
  shor_only_2  0 bare, 1 bare, 2 Shor, 3 bare. 12 physical qubits, the same
               count as uniform repetition. A resource matched comparison.
  plan_q1_rep  0 repetition, 1 repetition, 2 Shor, 3 bare. 16 physical
               qubits. Negative control, repetition on the phase error qubit.

When a run already used a plan in the error aware study, that result is
reused from outputs/full_study_results_erroraware.csv instead of being
simulated again (same weights, same noise seeds, same numbers). As a sanity
check the no QEC error is recomputed for every run and compared with the
saved csv; a mismatch means the weights or windows are not the ones used
in the study.

Usage:
  python scripts/run_controls.py
  python scripts/run_controls.py --plans rep_only_03 shor_only_2 plan_A plan_q1_rep --seeds 0
  python scripts/run_controls.py --subjects EPCTL01 EPCTL07 --seeds 0
"""

import argparse
import ast
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_eeg_channel
from src.preprocessing import preprocess_and_segment
from src.features import extract_feature_matrix
from src.quantum_model import make_ideal_qnode, normalize_features
from src.repetition_code import generate_storage_no_qec_batch
from src.application_aware_qec import ApplicationAwareQEC
from src.metrics import summarize
from src.data_split import train_test_split_indices

R, B, S = "repetition", "bare", "shor"
PLANS = {
    "plan_A": {0: R, 1: B, 2: S, 3: R},
    "rep_only_03": {0: R, 1: B, 2: B, 3: R},
    "shor_only_2": {0: B, 1: B, 2: S, 3: B},
    "plan_q1_rep": {0: R, 1: R, 2: S, 3: B},
}
DEFAULT_PLANS = ["rep_only_03", "shor_only_2", "plan_A"]
DEFAULT_SUBJECTS = [f"EPCTL{i:02d}" for i in [1, 2, 3, 4, 5, 6, 7, 9, 11, 12]]

NOISE_PROB = 0.02
NOISE_TYPE = "depolarizing"
N_SHOTS = 50
MAX_WINDOWS = 100
TEST_FRACTION = 0.2
WINDOW_SEC = 10.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_PATH = OUT_DIR / "controls_results.csv"
STUDY_PATH = OUT_DIR / "full_study_results_erroraware.csv"
WEIGHTS_DIR = OUT_DIR / "weights"


def load_study_reference():
    """Returns (no_qec dict, application_aware dict), each keyed by (subject, seed)."""
    study = pd.read_csv(STUDY_PATH)
    nq = {(r.subject, int(r.seed)): r for r in study[study["method"] == "no_qec"].itertuples()}
    aa = {}
    for r in study[study["method"] == "application_aware"].itertuples():
        aa[(r.subject, int(r.seed))] = r
    return nq, aa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--plans", nargs="+", default=DEFAULT_PLANS, choices=list(PLANS))
    args = parser.parse_args()

    for path in [STUDY_PATH, WEIGHTS_DIR]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run the error aware study first.")

    nq_ref, aa_ref = load_study_reference()

    rows = pd.read_csv(RESULTS_PATH).to_dict("records") if RESULTS_PATH.exists() else []
    done = {(r["subject"], int(r["seed"]), r["plan"]) for r in rows}
    ideal_qnode = make_ideal_qnode()

    for subject in args.subjects:
        todo = [(seed, plan) for seed in args.seeds for plan in args.plans if (subject, seed, plan) not in done]
        if not todo:
            print(f"{subject}: everything already done, skipping")
            continue

        print(f"\n=== {subject}: loading and preparing windows ===")
        signal, sfreq = load_eeg_channel(f"data/raw/{subject}.edf", "C3")
        windows = preprocess_and_segment(signal, sfreq, window_sec=WINDOW_SEC)[:MAX_WINDOWS]
        raw_features = extract_feature_matrix(windows, sfreq)
        all_features = normalize_features(raw_features)
        n_windows = len(raw_features)
        del signal, windows

        for seed in args.seeds:
            seed_todo = [plan for s, plan in todo if s == seed]
            if not seed_todo:
                continue
            weights_path = WEIGHTS_DIR / f"{subject}_seed{seed}.npy"
            if not weights_path.exists():
                print(f"  {weights_path.name} not found, skipping {subject} seed {seed}")
                continue

            weights = np.load(weights_path)
            _, test_idx = train_test_split_indices(n_windows, test_fraction=TEST_FRACTION, seed=seed)
            features = all_features[test_idx]
            ideal = np.array([ideal_qnode(f, weights) for f in features])

            # sanity check against the saved study
            nq_out = generate_storage_no_qec_batch(features, weights, NOISE_PROB, NOISE_TYPE, N_SHOTS)
            nq_res = summarize(ideal, nq_out)
            ref = nq_ref.get((subject, seed))
            if ref is not None:
                rel = abs(nq_res["mse"] - ref.mse_vs_ideal) / ref.mse_vs_ideal
                flag = "OK" if rel < 1e-6 else "MISMATCH, weights or windows differ from the study"
                print(f"  {subject} seed {seed}: no QEC mse {nq_res['mse']:.6f} vs saved {ref.mse_vs_ideal:.6f} ({flag})")

            study_row = aa_ref.get((subject, seed))
            study_tiers = ast.literal_eval(str(study_row.tiers)) if study_row is not None else None

            for plan in seed_todo:
                tiers = PLANS[plan]
                if study_tiers == tiers:
                    mse, wass, reused = study_row.mse_vs_ideal, study_row.wasserstein_vs_ideal, True
                    n_phys = ApplicationAwareQEC(tiers).n_physical
                    print(f"  {subject} seed {seed} {plan}: reused from the study run")
                else:
                    aa = ApplicationAwareQEC(tiers)
                    if not aa.verify_identity(features[0], weights):
                        print(f"  {plan}: identity check failed, skipping")
                        continue
                    start = time.perf_counter()
                    out = aa.generate_batch(features, weights, NOISE_PROB, NOISE_TYPE, N_SHOTS)
                    res = summarize(ideal, out)
                    mse, wass, reused, n_phys = res["mse"], res["mean_wasserstein"], False, aa.n_physical
                    print(f"  {subject} seed {seed} {plan}: mse {mse:.6f} wasserstein {wass:.6f} "
                          f"({n_phys} qubits, {time.perf_counter() - start:.0f}s)")
                rows.append({
                    "subject": subject, "seed": seed, "plan": plan, "physical_qubits": n_phys,
                    "mse_vs_ideal": mse, "wasserstein_vs_ideal": wass, "reused_from_study": reused,
                })
                pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)

    if not rows:
        return
    res = pd.DataFrame(rows)
    study = pd.read_csv(STUDY_PATH)
    print("\n=== Summary: mean over the runs done, all plans and the study baselines ===")
    keep = {(r["subject"], int(r["seed"])) for r in rows}
    base = study[study.apply(lambda r: (r["subject"], int(r["seed"])) in keep, axis=1)]
    for method in ["no_qec", "uniform_repetition_d3", "classical_averaged_50_shots", "application_aware"]:
        b = base[base["method"] == method]
        print(f"  {method:30s} mse {b['mse_vs_ideal'].mean():.6f}  wasserstein {b['wasserstein_vs_ideal'].mean():.6f}  (study)")
    for plan, g in res.groupby("plan"):
        print(f"  {plan:30s} mse {g['mse_vs_ideal'].mean():.6f}  wasserstein {g['wasserstein_vs_ideal'].mean():.6f}  "
              f"({int(g['physical_qubits'].iloc[0])} qubits, {len(g)} runs)")
    print(f"\nSaved {RESULTS_PATH}")


if __name__ == "__main__":
    main()