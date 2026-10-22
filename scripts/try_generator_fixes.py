"""Can the generator reconstruct its target better with a different feature scaling?

check_reconstruction.py showed that with min max scaling the clean generator
output is WORSE than predicting the training mean (error 0.178 against 0.066
per output). Min max scaling puts most windows near one end of the range when
a feature has outliers, and the circuit cannot produce such offsets.

This script trains the generator on the same windows with three scalings of
the inputs and targets and compares them on the held out test windows:
  minmax    the current scaling, inputs to [0, pi], targets to [-1, 1]
  zclip     z scores clipped at 2.5 standard deviations, then mapped to the
            same ranges, so the bulk of the windows sits around the middle
  quantile  each feature mapped through its training empirical distribution,
            so inputs and targets are spread evenly over their ranges

For each it reports reconstruction error against the predict-the-mean error,
R squared, mean correlation, and the macro F1 of a stage classifier on the
clean outputs (so a fix cannot win by losing the stage information).

Usage:
  python scripts/try_generator_fixes.py --subjects EPCTL01 EPCTL09 EPCTL12
  python scripts/try_generator_fixes.py --subjects EPCTL01 --epochs 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_downstream as rd

OUT_PATH = rd.OUT_DIR / "generator_fix_comparison.csv"


def scale_minmax(raw_train, raw_test):
    mins, maxs = raw_train.min(axis=0, keepdims=True), raw_train.max(axis=0, keepdims=True)
    f = lambda raw, lo, hi: rd.apply_minmax(raw, mins, maxs, lo, hi)
    return f(raw_train, 0, np.pi), f(raw_train, -1, 1), f(raw_test, 0, np.pi), f(raw_test, -1, 1)


def scale_zclip(raw_train, raw_test, k=2.5):
    mu, sd = raw_train.mean(axis=0), raw_train.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    z = lambda raw: np.clip((raw - mu) / sd, -k, k)
    inp = lambda raw: (z(raw) + k) / (2 * k) * np.pi
    return inp(raw_train), z(raw_train) / k, inp(raw_test), z(raw_test) / k


def scale_quantile(raw_train, raw_test):
    qt = QuantileTransformer(n_quantiles=min(100, len(raw_train)), output_distribution="uniform").fit(raw_train)
    u_tr, u_te = qt.transform(raw_train), qt.transform(raw_test)
    return u_tr * np.pi, 2 * u_tr - 1, u_te * np.pi, 2 * u_te - 1


VARIANTS = {"minmax": scale_minmax, "zclip": scale_zclip, "quantile": scale_quantile}


def reconstruction_metrics(out, target, train_target):
    mse = float(np.mean((out - target) ** 2))
    mean_mse = float(np.mean((train_target.mean(axis=0, keepdims=True) - target) ** 2))
    sst = ((target - target.mean(axis=0)) ** 2).sum()
    r2 = float(1 - ((out - target) ** 2).sum() / sst)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = float(np.nanmean([np.corrcoef(out[:, j], target[:, j])[0, 1] for j in range(out.shape[1])]))
    return mse, mean_mse, r2, corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=["EPCTL01", "EPCTL09", "EPCTL12"])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--train_per_stage", type=int, default=30)
    parser.add_argument("--test_per_stage", type=int, default=12)
    args = parser.parse_args()

    rows = []
    for subject in args.subjects:
        print(f"\n=== {subject} ===")
        windows, sfreq = rd.load_all_windows(str(rd.PROJECT_ROOT / f"data/raw/{subject}.edf"), "C3")
        epoch_rows = rd.build_epoch_table(rd.PROJECT_ROOT / f"data/raw/{subject}.txt", len(windows))
        rng = np.random.default_rng(args.seed)
        train_rows, test_rows = rd.sample_split(epoch_rows, rng, args.train_per_stage, args.test_per_stage)
        tr_idx, y_train, _ = rd.pick_windows(train_rows, rng)
        te_idx, y_test, _ = rd.pick_windows(test_rows, rng)
        raw_train = rd.extract_feature_matrix([windows[i] for i in tr_idx], sfreq)
        raw_test = rd.extract_feature_matrix([windows[i] for i in te_idx], sfreq)
        del windows

        qnode = rd.make_ideal_qnode()
        for name in args.variants:
            print(f"  variant {name}: training {args.epochs} epochs ...")
            tr_in, tr_target, te_in, te_target = VARIANTS[name](raw_train, raw_test)
            weights, losses = rd.train_generator(tr_in, tr_target, args.epochs, args.lr, args.seed)
            out_tr = np.array([qnode(f, weights) for f in tr_in])
            out_te = np.array([qnode(f, weights) for f in te_in])
            mse, mean_mse, r2, corr = reconstruction_metrics(out_te, te_target, tr_target)
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
            clf.fit(out_tr, y_train)
            f1 = rd.score(y_test, clf.predict(out_te))["macro_f1"]
            rows.append({"subject": subject, "variant": name, "test_mse": mse, "predict_mean_mse": mean_mse,
                         "r2": r2, "corr": corr, "clean_macro_f1": f1, "final_train_loss": losses[-1]})
            print(f"    mse {mse:.4f} (predict the mean {mean_mse:.4f})  R2 {r2:+.2f}  corr {corr:.2f}  "
                  f"clean macro F1 {f1:.3f}")
        pd.DataFrame(rows).to_csv(OUT_PATH, index=False)

    res = pd.DataFrame(rows)
    print("\nMean over subjects:")
    print(res.groupby("variant")[["test_mse", "predict_mean_mse", "r2", "corr", "clean_macro_f1"]].mean().round(3).to_string())
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()