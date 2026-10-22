"""How well does the quantum generator reconstruct its own target features?

The generator is trained to map 8 input features to 8 output expectation
values that should match the same features rescaled to [-1, 1]
(train_generator.py). The noise results in this project measure damage
RELATIVE TO THE CLEAN GENERATOR OUTPUT. They say nothing about how close
that clean output is to the target. This script measures that, on the held
out test windows of the downstream study, using the saved weights, so the
paper can state the reconstruction quality honestly.

For every subject it recomputes exactly the windows run_downstream.py used
(same seed, same sampling), then reports per output and overall:
  r2    1 minus squared error over the variance of the target
  corr  correlation between output and target
  mse   mean squared error per output
  mean_baseline_mse   the error of simply predicting the training mean, the
                      bar any useful reconstruction must beat

Usage:
  python scripts/check_reconstruction.py
  python scripts/check_reconstruction.py --subjects EPCTL01 EPCTL02
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_downstream as rd  # same folder, reuses the exact sampling and scaling

OUT_PATH = rd.OUT_DIR / "reconstruction_check.csv"


def r2_score(pred, target):
    sst = ((target - target.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    sse = ((pred - target) ** 2).sum(axis=0)
    return 1.0 - sse / np.where(sst > 0, sst, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=rd.DEFAULT_SUBJECTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_per_stage", type=int, default=30)
    parser.add_argument("--test_per_stage", type=int, default=12)
    args = parser.parse_args()

    rows = []
    for subject in args.subjects:
        weights_path = rd.WEIGHTS_DIR / f"{subject}_seed{args.seed}.npy"
        if not weights_path.exists():
            print(f"{subject}: {weights_path.name} not found, skipping")
            continue
        print(f"{subject}: loading ...")
        windows, sfreq = rd.load_all_windows(str(rd.PROJECT_ROOT / f"data/raw/{subject}.edf"), "C3")
        epoch_rows = rd.build_epoch_table(rd.PROJECT_ROOT / f"data/raw/{subject}.txt", len(windows))

        rng = np.random.default_rng(args.seed)
        train_rows, test_rows = rd.sample_split(epoch_rows, rng, args.train_per_stage, args.test_per_stage)
        tr_idx, _, _ = rd.pick_windows(train_rows, rng)
        te_idx, _, _ = rd.pick_windows(test_rows, rng)

        raw_train = rd.extract_feature_matrix([windows[i] for i in tr_idx], sfreq)
        raw_test = rd.extract_feature_matrix([windows[i] for i in te_idx], sfreq)
        mins, maxs = raw_train.min(axis=0, keepdims=True), raw_train.max(axis=0, keepdims=True)
        train_target = rd.apply_minmax(raw_train, mins, maxs, -1.0, 1.0)
        test_in = rd.apply_minmax(raw_test, mins, maxs, 0.0, np.pi)
        test_target = rd.apply_minmax(raw_test, mins, maxs, -1.0, 1.0)

        weights = np.load(weights_path)
        qnode = rd.make_ideal_qnode()
        out = np.array([qnode(f, weights) for f in test_in])

        r2 = r2_score(out, test_target)
        corr = np.array([np.corrcoef(out[:, j], test_target[:, j])[0, 1] for j in range(out.shape[1])])
        mean_pred = np.tile(train_target.mean(axis=0, keepdims=True), (len(test_target), 1))
        for j in range(out.shape[1]):
            rows.append({"subject": subject, "output": j, "r2": float(r2[j]), "corr": float(corr[j]),
                         "mse": float(np.mean((out[:, j] - test_target[:, j]) ** 2)),
                         "mean_baseline_mse": float(np.mean((mean_pred[:, j] - test_target[:, j]) ** 2))})
        total_r2 = 1 - ((out - test_target) ** 2).sum() / ((test_target - test_target.mean(axis=0)) ** 2).sum()
        print(f"  overall R2 {total_r2:.3f}, mean output correlation {np.nanmean(corr):.3f}, "
              f"mse {np.mean((out - test_target) ** 2):.4f}, predict-the-mean mse {np.mean((mean_pred - test_target) ** 2):.4f}")
        del windows

    if not rows:
        return
    res = pd.DataFrame(rows)
    res.to_csv(OUT_PATH, index=False)
    per_subject = res.groupby("subject")[["r2", "corr", "mse", "mean_baseline_mse"]].mean()
    print("\nMean over the 8 outputs, per subject:")
    print(per_subject.round(3).to_string())
    print("\nMean over subjects:")
    print(per_subject.mean().round(3).to_string())
    print("\nMean R2 per output over subjects (outputs 0 to 3 are Z, 4 to 7 are X):")
    print(res.groupby("output")["r2"].mean().round(3).to_string())
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()