"""One run, repaired generator: reconstruction, noise fidelity and sleep staging.

Replaces the pilot studies. The pilots used min max feature scaling, which left
the generator worse than predicting the mean (see check_reconstruction.py and
try_generator_fixes.py). This script uses clipped z score scaling, which fixed
that, and measures everything in a single pass on the SAME windows:

  per subject and seed
    1. stage balanced windows from the whole night, one window per 30 s epoch,
       test epochs first and training epochs kept one epoch away from them
    2. features scaled with TRAINING statistics only (z score, clipped at 2.5)
    3. generator trained on the training windows only (seeded, reproducible)
    4. reconstruction quality on the test windows (error against the
       predict-the-mean error, R squared, correlation)
    5. noisy outputs of each method on the test windows, scored two ways
         fidelity    MSE and Wasserstein against the clean generator output
         downstream  macro F1, N1 F1 and balanced accuracy of ONE classifier,
                     trained on clean training outputs and then frozen
    6. a classical reference: the same classical noise averaging used in
       run_classical_baseline.py, applied to the clean generator output, so it
       is scored against the same reference and by the same classifier

Methods: no_qec, uniform (repetition on all 4 qubits), rep03 (repetition on
qubits 0 and 3 only), plan_A (fixed plan), rule (error aware allocation from a
sensitivity ranking on TRAINING windows only), classical.

Usage:
  python scripts/run_study_v2.py
  python scripts/run_study_v2.py --subjects EPCTL01 --seeds 0
  python scripts/run_study_v2.py --methods no_qec uniform rep03 plan_A rule
"""

import argparse
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
from src.classical_baseline import generate_classical_averaged_batch
from src.metrics import summarize

CLIP = 2.5
DEFAULT_METHODS = ["no_qec", "uniform", "rep03", "rule"]
RESULTS_PATH = rd.OUT_DIR / "v2_results.csv"
PREDICTIONS_PATH = rd.OUT_DIR / "v2_predictions.csv"
RECON_PATH = rd.OUT_DIR / "v2_reconstruction.csv"
WEIGHTS_DIR = rd.OUT_DIR / "v2_weights"


def zclip_scale(raw_train, raw_test, k=CLIP):
    """Inputs to [0, pi] and targets to [-1, 1] from clipped z scores,
    using training statistics only."""
    mu, sd = raw_train.mean(axis=0), raw_train.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    z = lambda raw: np.clip((raw - mu) / sd, -k, k)
    inp = lambda raw: (z(raw) + k) / (2 * k) * np.pi
    return inp(raw_train), z(raw_train) / k, inp(raw_test), z(raw_test) / k


def reconstruction_row(subject, seed, out, target, train_target):
    mse = float(np.mean((out - target) ** 2))
    mean_mse = float(np.mean((train_target.mean(axis=0, keepdims=True) - target) ** 2))
    sst = ((target - target.mean(axis=0)) ** 2).sum()
    r2 = float(1 - ((out - target) ** 2).sum() / sst)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = float(np.nanmean([np.corrcoef(out[:, j], target[:, j])[0, 1] for j in range(out.shape[1])]))
    return {"subject": subject, "seed": seed, "test_mse": mse, "predict_mean_mse": mean_mse, "r2": r2, "corr": corr}


def physical_qubits(method, tiers):
    sizes = {"bare": 1, "repetition": 3, "shor": 9}
    plan = {"uniform": {q: "repetition" for q in range(4)}, "no_qec": {q: "bare" for q in range(4)},
            "rule": tiers}.get(method, rd.FIXED_PLANS.get(method))
    return int(sum(sizes[plan[q]] for q in range(4))) if plan else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=rd.DEFAULT_SUBJECTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        choices=["no_qec", "uniform", "rule"] + list(rd.FIXED_PLANS))
    parser.add_argument("--train_per_stage", type=int, default=30)
    parser.add_argument("--test_per_stage", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--rank_windows", type=int, default=80)
    parser.add_argument("--channel", default="C3")
    args = parser.parse_args()

    rd.OUT_DIR.mkdir(exist_ok=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    load = lambda p: pd.read_csv(p).to_dict("records") if p.exists() else []
    results, predictions, recon = load(RESULTS_PATH), load(PREDICTIONS_PATH), load(RECON_PATH)
    done = {(r["subject"], int(r["seed"])) for r in recon}

    for subject in args.subjects:
        todo = [s for s in args.seeds if (subject, s) not in done]
        if not todo:
            print(f"{subject}: all seeds already done, skipping")
            continue

        print(f"\n=== {subject}: loading and filtering the whole night ===")
        windows, sfreq = rd.load_all_windows(str(rd.PROJECT_ROOT / f"data/raw/{subject}.edf"), args.channel)
        epoch_rows = rd.build_epoch_table(rd.PROJECT_ROOT / f"data/raw/{subject}.txt", len(windows))

        for seed in todo:
            t0 = time.perf_counter()
            rng = np.random.default_rng(seed)
            train_rows, test_rows = rd.sample_split(epoch_rows, rng, args.train_per_stage, args.test_per_stage)
            tr_idx, y_train, _ = rd.pick_windows(train_rows, rng)
            te_idx, y_test, te_epochs = rd.pick_windows(test_rows, rng)
            print(f"  seed {seed}: {len(tr_idx)} training and {len(te_idx)} test windows")

            raw_train = rd.extract_feature_matrix([windows[i] for i in tr_idx], sfreq)
            raw_test = rd.extract_feature_matrix([windows[i] for i in te_idx], sfreq)
            train_in, train_target, test_in, test_target = zclip_scale(raw_train, raw_test)

            print("  training the generator ...")
            weights, losses = rd.train_generator(train_in, train_target, args.epochs, args.lr, seed)
            np.save(WEIGHTS_DIR / f"{subject}_seed{seed}.npy", weights)

            qnode = rd.make_ideal_qnode()
            ideal_train = np.array([qnode(f, weights) for f in train_in])
            ideal_test = np.array([qnode(f, weights) for f in test_in])

            rec = reconstruction_row(subject, seed, ideal_test, test_target, train_target)
            rec["final_train_loss"] = losses[-1]
            print(f"  reconstruction: mse {rec['test_mse']:.4f} (predict the mean {rec['predict_mean_mse']:.4f}), "
                  f"R2 {rec['r2']:+.2f}, corr {rec['corr']:.2f}")

            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
            clf.fit(ideal_train, y_train)
            raw_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
            raw_clf.fit(train_in, y_train)

            tiers = None
            if "rule" in args.methods:
                tiers = rd.rule_allocation(train_in, weights, args.rank_windows, rng)
                print(f"  rule allocation from training windows: {tiers}")

            outputs = {"classical": generate_classical_averaged_batch(ideal_test, rd.NOISE_PROB, rd.N_SHOTS)}
            for method in args.methods:
                start = time.perf_counter()
                outputs[method] = rd.noisy_outputs(method, test_in, weights, tiers)
                print(f"    {method}: simulated in {time.perf_counter() - start:.0f}s")

            rows = [
                {"method": "ideal", "pred": clf.predict(ideal_test), "mse": np.nan, "w": np.nan, "qubits": 4},
                {"method": "raw_features", "pred": raw_clf.predict(test_in), "mse": np.nan, "w": np.nan, "qubits": 0},
            ]
            for method, out in outputs.items():
                s = summarize(ideal_test, out)
                rows.append({"method": method, "pred": clf.predict(out), "mse": s["mse"],
                             "w": s["mean_wasserstein"], "qubits": physical_qubits(method, tiers)})

            for r in rows:
                sc = rd.score(y_test, r["pred"])
                row = {"subject": subject, "seed": seed, "method": r["method"], "physical_qubits": r["qubits"],
                       "mse_vs_ideal": r["mse"], "wasserstein_vs_ideal": r["w"], "n_train": len(tr_idx),
                       "n_test": len(te_idx), "tiers": str(tiers) if r["method"] == "rule" else ""}
                row.update(sc)
                results.append(row)
                for yt, yp, ep in zip(y_test, r["pred"], te_epochs):
                    predictions.append({"subject": subject, "seed": seed, "method": r["method"],
                                        "epoch": int(ep), "true": yt, "pred": yp})
                print(f"    {r['method']:13s} mse {r['mse']:.6f}  W {r['w']:.5f}  macro F1 {sc['macro_f1']:.3f}  "
                      f"N1 F1 {sc['n1_f1']:.3f}")

            recon.append(rec)
            pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
            pd.DataFrame(predictions).to_csv(PREDICTIONS_PATH, index=False)
            pd.DataFrame(recon).to_csv(RECON_PATH, index=False)
            print(f"  seed {seed} done in {(time.perf_counter() - t0) / 60:.1f} minutes, saved")
        del windows

    print("\nDone. Next: python scripts/analyze_study_v2.py")


if __name__ == "__main__":
    main()