"""Noise sweep on the repaired generator: when does error correction pay off?

Reuses the generators already trained by run_study_v2.py (outputs/v2_weights),
so nothing is retrained. Same seeded split, same z score scaling, same frozen
classifier as run_study_v2.py. Only the noise level and noise type change.

For every noise setting the rule allocation is recomputed from a sensitivity
ranking on TRAINING windows only, at that noise setting.

Results go to outputs/sweep_results.csv, saved after every method, and the
script resumes from where it stopped.

Usage (from the project folder, one job at a time):
  python scripts\run_noise_sweep.py --methods no_qec uniform rep03
  python scripts\run_noise_sweep.py --methods rule
  python scripts\run_noise_sweep.py --noise_types bitflip phaseflip --probs 0.10
  python scripts\run_noise_sweep.py --summary
  python scripts\run_noise_sweep.py --noise_types bitflip phaseflip --probs 0.05 --methods phase_uniform phase03 --out sweep_pauli.csv

Sanity check first: run one subject at p = 0.02 and compare with v2_results.csv.
  python scripts\run_noise_sweep.py --subjects EPCTL01 --probs 0.02
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_downstream as rd
import run_study_v2 as v2
from src.metrics import summarize
from src.phase_flip_code import TierQEC

# extra protection methods built on the phase flip repetition code (needs src/phase_flip_code.py)
EXTRA_PLANS = {
    "phase_uniform": {q: "phaserep" for q in range(4)},
    "phase03": {0: "phaserep", 1: "bare", 2: "bare", 3: "phaserep"},
}
EXTRA_QUBITS = {"phase_uniform": 12, "phase03": 8}


def noisy(method, features, weights, tiers):
    """Like rd.noisy_outputs, plus the phase flip code methods."""
    if method in EXTRA_PLANS:
        app = TierQEC(EXTRA_PLANS[method])
        if not app.verify_identity(features[0], weights):
            raise RuntimeError(f"identity check failed for method {method}")
        return app.generate_batch(features, weights, rd.NOISE_PROB, rd.NOISE_TYPE, rd.N_SHOTS)
    return rd.noisy_outputs(method, features, weights, tiers)

SWEEP_PATH = rd.OUT_DIR / "sweep_results.csv"
KEY = ["subject", "seed", "noise_type", "noise_prob", "method"]


def print_summary():
    df = pd.read_csv(SWEEP_PATH)
    df = df[df.method != "ideal"]
    for (nt, p), g in df.groupby(["noise_type", "noise_prob"]):
        print(f"\n=== {nt}, p = {p} ===")
        base = g[g.method == "no_qec"].set_index(["subject", "seed"])
        rows = []
        for m, gm in g.groupby("method"):
            gm = gm.set_index(["subject", "seed"])
            common = gm.index.intersection(base.index)
            if len(common) == 0:
                continue
            wins = int((gm.loc[common, "mse_vs_ideal"] < base.loc[common, "mse_vs_ideal"]).sum())
            rows.append({
                "method": m,
                "subjects": len(common),
                "mse": gm.loc[common, "mse_vs_ideal"].mean(),
                "wasserstein": gm.loc[common, "wasserstein_vs_ideal"].mean(),
                "mse_z": gm.loc[common, "mse_z"].mean() if "mse_z" in gm.columns else np.nan,
                "mse_x": gm.loc[common, "mse_x"].mean() if "mse_x" in gm.columns else np.nan,
                "macro_f1": gm.loc[common, "macro_f1"].mean(),
                "beats_no_qec_on_mse": f"{wins}/{len(common)}",
            })
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def main():
    global SWEEP_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=rd.DEFAULT_SUBJECTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--noise_types", nargs="+", default=["depolarizing"])
    parser.add_argument("--probs", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--methods", nargs="+", default=["no_qec", "uniform", "rep03", "rule"],
                        choices=["no_qec", "uniform", "rule"] + list(rd.FIXED_PLANS) + list(EXTRA_PLANS))
    parser.add_argument("--train_per_stage", type=int, default=30)
    parser.add_argument("--test_per_stage", type=int, default=12)
    parser.add_argument("--rank_windows", type=int, default=80)
    parser.add_argument("--channel", default="C3")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--out", default="sweep_results.csv",
                        help="results file name inside outputs, use a new name for a separate run")
    args = parser.parse_args()
    SWEEP_PATH = rd.OUT_DIR / args.out

    if args.summary:
        print_summary()
        return

    rd.OUT_DIR.mkdir(exist_ok=True)
    rows = pd.read_csv(SWEEP_PATH).to_dict("records") if SWEEP_PATH.exists() else []

    def is_done(subject, seed, nt, p, method):
        return any(r["subject"] == subject and int(r["seed"]) == seed and r["noise_type"] == nt
                   and abs(float(r["noise_prob"]) - p) < 1e-9 and r["method"] == method for r in rows)

    for subject in args.subjects:
        jobs = [(s, nt, p, m) for s in args.seeds for nt in args.noise_types for p in args.probs
                for m in args.methods if not is_done(subject, s, nt, p, m)]
        if not jobs:
            print(f"{subject}: nothing left to do, skipping")
            continue

        print(f"\n=== {subject}: loading and filtering the whole night ===")
        windows, sfreq = rd.load_all_windows(str(rd.PROJECT_ROOT / f"data/raw/{subject}.edf"), args.channel)
        epoch_rows = rd.build_epoch_table(rd.PROJECT_ROOT / f"data/raw/{subject}.txt", len(windows))

        for seed in args.seeds:
            seed_jobs = [j for j in jobs if j[0] == seed]
            if not seed_jobs:
                continue
            weights_path = v2.WEIGHTS_DIR / f"{subject}_seed{seed}.npy"
            if not weights_path.exists():
                print(f"  no saved weights at {weights_path}, skipping seed {seed}")
                continue

            # same random sequence as run_study_v2.py, so the windows are identical
            rng = np.random.default_rng(seed)
            train_rows, test_rows = rd.sample_split(epoch_rows, rng, args.train_per_stage, args.test_per_stage)
            tr_idx, y_train, _ = rd.pick_windows(train_rows, rng)
            te_idx, y_test, _ = rd.pick_windows(test_rows, rng)
            print(f"  seed {seed}: {len(tr_idx)} training and {len(te_idx)} test windows")

            raw_train = rd.extract_feature_matrix([windows[i] for i in tr_idx], sfreq)
            raw_test = rd.extract_feature_matrix([windows[i] for i in te_idx], sfreq)
            train_in, train_target, test_in, test_target = v2.zclip_scale(raw_train, raw_test)

            weights = np.load(weights_path)
            qnode = rd.make_ideal_qnode()
            ideal_train = np.array([qnode(f, weights) for f in train_in])
            ideal_test = np.array([qnode(f, weights) for f in test_in])

            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
            clf.fit(ideal_train, y_train)

            if not is_done(subject, seed, "none", 0.0, "ideal"):
                row = {"subject": subject, "seed": seed, "noise_type": "none", "noise_prob": 0.0,
                       "method": "ideal", "physical_qubits": 4, "mse_vs_ideal": np.nan,
                       "wasserstein_vs_ideal": np.nan, "tiers": ""}
                row.update(rd.score(y_test, clf.predict(ideal_test)))
                rows.append(row)

            rng_after_split = copy.deepcopy(rng)

            for nt in args.noise_types:
                for p in args.probs:
                    todo = [m for (s, n, q, m) in seed_jobs if n == nt and abs(q - p) < 1e-9]
                    if not todo:
                        continue
                    rd.NOISE_TYPE, rd.NOISE_PROB = nt, p
                    print(f"  --- {nt}, p = {p} ---")

                    tiers = None
                    if "rule" in todo:
                        tiers = rd.rule_allocation(train_in, weights, args.rank_windows,
                                                   copy.deepcopy(rng_after_split))
                        print(f"    rule allocation from training windows: {tiers}")

                    for method in todo:
                        start = time.perf_counter()
                        out = noisy(method, test_in, weights, tiers)
                        s = summarize(ideal_test, out)
                        row = {"subject": subject, "seed": seed, "noise_type": nt, "noise_prob": p,
                               "method": method, "physical_qubits": EXTRA_QUBITS.get(method) or v2.physical_qubits(method, tiers),
                               "mse_vs_ideal": s["mse"], "wasserstein_vs_ideal": s["mean_wasserstein"],
                               "mse_z": float(np.mean((ideal_test[:, :4] - out[:, :4]) ** 2)),
                               "mse_x": float(np.mean((ideal_test[:, 4:] - out[:, 4:]) ** 2)),
                               "tiers": str(tiers) if method == "rule" else ""}
                        row.update(rd.score(y_test, clf.predict(out)))
                        rows.append(row)
                        pd.DataFrame(rows).to_csv(SWEEP_PATH, index=False)
                        print(f"    {method:10s} mse {s['mse']:.6f}  W {s['mean_wasserstein']:.5f}  "
                              f"macro F1 {row['macro_f1']:.3f}  Zmse {row['mse_z']:.6f}  "
                              f"Xmse {row['mse_x']:.6f}  ({time.perf_counter() - start:.0f}s)")

            pd.DataFrame(rows).to_csv(SWEEP_PATH, index=False)
        del windows

    print("\nDone. Next: python scripts\\run_noise_sweep.py --summary")


if __name__ == "__main__":
    main()