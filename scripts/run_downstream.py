"""Downstream sleep stage check for the protected quantum generator.

Question answered: when the generator output is damaged by noise, does the
error protection keep the output useful for the real job, telling sleep
stages apart?

Protocol, per subject and seed:
  1. Read the hypnogram and the EEG. Every 10 s window lies inside one 30 s
     epoch, so its stage label is the label of that epoch.
  2. Draw a stage balanced sample across the WHOLE night, one window per
     epoch, test epochs first, then training epochs that are not within one
     epoch of any test epoch (a buffer against neighbouring epoch leakage).
  3. Scale features with TRAINING statistics only. Train the quantum
     generator on the training windows only, the same loss as
     train_generator.py but with a seeded batch order.
  4. Compute clean (ideal) generator outputs, train ONE classifier on the
     clean training outputs, then freeze it.
  5. Feed the frozen classifier the outputs of each method on the held out
     test windows under noise: no QEC, uniform repetition, repetition on
     qubits 0 and 3 only, and the error aware rule. The rule allocation is
     chosen from a sensitivity ranking on TRAINING windows only.
  6. Score macro F1, balanced accuracy and N1 F1 against the human labels.
     The clean generator and the raw 8 features are scored as references.

Sampling is stage balanced, so absolute scores are not comparable with
published sleep staging numbers on natural stage frequencies. The result to
report is how much of the clean performance each method retains.

Usage:
  python scripts/run_downstream.py
  python scripts/run_downstream.py --subjects EPCTL01 EPCTL02 --seeds 0
  python scripts/run_downstream.py --methods no_qec uniform rule
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_eeg_channel
from src.preprocessing import preprocess_and_segment
from src.features import extract_feature_matrix
from src.hypnogram import load_hypnogram
from src.quantum_model import make_ideal_qnode, N_QUBITS, N_LAYERS
from src.repetition_code import generate_storage_no_qec_batch, generate_repetition_qec_batch
from src.application_aware_qec import ApplicationAwareQEC
from src.allocation import build_tiers_error_aware

STAGES = ["W", "N1", "N2", "N3", "R"]
EPOCH_SEC = 30
WINDOW_SEC = 10.0
NOISE_PROB = 0.02
NOISE_TYPE = "depolarizing"
N_SHOTS = 50

R, B, S = "repetition", "bare", "shor"
FIXED_PLANS = {
    "plan_A": {0: R, 1: B, 2: S, 3: R},
    "rep03": {0: R, 1: B, 2: B, 3: R},
}
DEFAULT_SUBJECTS = [f"EPCTL{i:02d}" for i in [1, 2, 3, 4, 5, 6, 7, 9, 11, 12]]
DEFAULT_METHODS = ["no_qec", "uniform", "rep03", "rule"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_PATH = OUT_DIR / "downstream_results.csv"
PREDICTIONS_PATH = OUT_DIR / "downstream_predictions.csv"
WEIGHTS_DIR = OUT_DIR / "downstream_weights"


def load_all_windows(edf_path, channel, attempts=3):
    """Filter the whole night once and return every 10 s window, retrying on
    the out of memory error that filtfilt can raise on 6 hour recordings."""
    for attempt in range(attempts):
        try:
            signal, sfreq = load_eeg_channel(edf_path, channel)
            windows = preprocess_and_segment(signal, sfreq, window_sec=WINDOW_SEC)
            return windows, sfreq
        except MemoryError:
            print(f"  out of memory while filtering, waiting 20 seconds (attempt {attempt + 1} of {attempts})")
            time.sleep(20)
    raise MemoryError("Could not filter the recording, close other programs and try again.")


def build_epoch_table(hypnogram_path, n_windows):
    """Rows of (epoch_index, stage, first_window_index) for usable 30 s epochs."""
    per_epoch = int(EPOCH_SEC // WINDOW_SEC)
    rows = []
    for label, start, duration in load_hypnogram(hypnogram_path):
        if label not in STAGES or duration != EPOCH_SEC or start % EPOCH_SEC != 0:
            continue
        first = int(start // WINDOW_SEC)
        if first + per_epoch > n_windows:
            continue
        rows.append((int(start // EPOCH_SEC), label, first))
    return rows


def sample_split(rows, rng, train_per_stage, test_per_stage, buffer=1):
    """Stage balanced split by epoch, test first, then train epochs that are
    not within `buffer` epochs of any test epoch."""
    by_stage = {s: [r for r in rows if r[1] == s] for s in STAGES}
    test_rows = []
    for stage in STAGES:
        avail = by_stage[stage]
        if len(avail) < 4:
            print(f"  stage {stage} has only {len(avail)} usable epochs, skipped")
            continue
        n_test = min(test_per_stage, max(1, len(avail) // 4))
        pick = rng.choice(len(avail), size=n_test, replace=False)
        test_rows += [avail[i] for i in pick]
    blocked = {e + d for e, _, _ in test_rows for d in range(-buffer, buffer + 1)}
    train_rows = []
    for stage in STAGES:
        cands = [r for r in by_stage[stage] if r[0] not in blocked]
        n_train = min(train_per_stage, len(cands))
        if n_train == 0:
            continue
        pick = rng.choice(len(cands), size=n_train, replace=False)
        train_rows += [cands[i] for i in pick]
    return train_rows, test_rows


def pick_windows(rows, rng):
    per_epoch = int(EPOCH_SEC // WINDOW_SEC)
    idx = np.array([r[2] + int(rng.integers(0, per_epoch)) for r in rows])
    labels = np.array([r[1] for r in rows])
    epochs = np.array([r[0] for r in rows])
    return idx, labels, epochs


def apply_minmax(raw, mins, maxs, lo, hi):
    span = np.where(maxs > mins, maxs - mins, 1.0)
    return (raw - mins) / span * (hi - lo) + lo


def train_generator(inputs, targets, epochs, lr, seed):
    """Same loss and optimiser as scripts/train_generator.py, with a seeded
    batch order so the weights are reproducible."""
    import pennylane as qml
    from pennylane import numpy as pnp

    ideal_qnode = make_ideal_qnode()
    rng = np.random.default_rng(seed)
    weights = pnp.array(rng.uniform(0, 2 * np.pi, size=(N_LAYERS, N_QUBITS, 2)), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=lr)
    batch_size = min(16, len(inputs))

    def batch_loss(w, x_batch, t_batch):
        total = 0.0
        for x_in, x_target in zip(x_batch, t_batch):
            out = pnp.stack(ideal_qnode(x_in, w))
            total = total + pnp.sum((out - x_target) ** 2)
        return total / len(x_batch)

    losses = []
    for epoch in range(epochs):
        perm = rng.permutation(len(inputs))
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, len(inputs), batch_size):
            b = perm[start:start + batch_size]
            weights, loss = opt.step_and_cost(lambda w: batch_loss(w, inputs[b], targets[b]), weights)
            epoch_loss += loss
            n_batches += 1
        losses.append(float(epoch_loss / n_batches))
        if epoch % 25 == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch:4d}  loss {losses[-1]:.4f}")
    return np.array(weights), losses


def rule_allocation(train_inputs, weights, n_rank, rng):
    """Sensitivity ranking on training windows only, then the error aware rule."""
    ideal_qnode = make_ideal_qnode()
    pick = rng.choice(len(train_inputs), size=min(n_rank, len(train_inputs)), replace=False)
    feats = train_inputs[pick]
    ideal = np.array([ideal_qnode(f, weights) for f in feats])
    noisy = generate_storage_no_qec_batch(feats, weights, NOISE_PROB, NOISE_TYPE, N_SHOTS)
    n_logical = ideal.shape[1] // 2
    ranking = pd.DataFrame([
        {
            "logical_qubit": lq,
            "z_mse": float(np.mean((ideal[:, lq] - noisy[:, lq]) ** 2)),
            "x_mse": float(np.mean((ideal[:, lq + n_logical] - noisy[:, lq + n_logical]) ** 2)),
        }
        for lq in range(n_logical)
    ])
    return build_tiers_error_aware(ranking)


def noisy_outputs(method, features, weights, tiers):
    if method == "no_qec":
        return generate_storage_no_qec_batch(features, weights, NOISE_PROB, NOISE_TYPE, N_SHOTS)
    if method == "uniform":
        return generate_repetition_qec_batch(features, weights, NOISE_PROB, NOISE_TYPE, N_SHOTS)
    plan = tiers if method == "rule" else FIXED_PLANS[method]
    app = ApplicationAwareQEC(plan)
    if not app.verify_identity(features[0], weights):
        raise RuntimeError(f"identity check failed for method {method}")
    return app.generate_batch(features, weights, NOISE_PROB, NOISE_TYPE, N_SHOTS)


def score(y_true, y_pred):
    present = [s for s in STAGES if s in set(y_true)]
    per_class = f1_score(y_true, y_pred, labels=STAGES, average=None, zero_division=0)
    f1_by_stage = dict(zip(STAGES, per_class))
    return {
        "macro_f1": float(np.mean([f1_by_stage[s] for s in present])),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "n1_f1": float(f1_by_stage["N1"]),
        **{f"f1_{s}": float(f1_by_stage[s]) for s in STAGES},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=["no_qec", "uniform", "rule"] + list(FIXED_PLANS))
    parser.add_argument("--train_per_stage", type=int, default=30)
    parser.add_argument("--test_per_stage", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--rank_windows", type=int, default=80)
    parser.add_argument("--channel", default="C3")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    results = pd.read_csv(RESULTS_PATH).to_dict("records") if RESULTS_PATH.exists() else []
    predictions = pd.read_csv(PREDICTIONS_PATH).to_dict("records") if PREDICTIONS_PATH.exists() else []
    done = {(r["subject"], int(r["seed"])) for r in results}

    for subject in args.subjects:
        todo = [s for s in args.seeds if (subject, s) not in done]
        if not todo:
            print(f"{subject}: all seeds already done, skipping")
            continue

        print(f"\n=== {subject}: loading and filtering the whole night ===")
        windows, sfreq = load_all_windows(str(PROJECT_ROOT / f"data/raw/{subject}.edf"), args.channel)
        epoch_rows = build_epoch_table(PROJECT_ROOT / f"data/raw/{subject}.txt", len(windows))
        counts = pd.Series([r[1] for r in epoch_rows]).value_counts().reindex(STAGES).fillna(0).astype(int)
        print(f"  {len(windows)} windows, usable epochs per stage: {counts.to_dict()}")

        for seed in todo:
            t0 = time.perf_counter()
            rng = np.random.default_rng(seed)
            train_rows, test_rows = sample_split(epoch_rows, rng, args.train_per_stage, args.test_per_stage)
            tr_idx, y_train, _ = pick_windows(train_rows, rng)
            te_idx, y_test, te_epochs = pick_windows(test_rows, rng)
            print(f"  seed {seed}: {len(tr_idx)} training and {len(te_idx)} test windows")

            raw_train = extract_feature_matrix([windows[i] for i in tr_idx], sfreq)
            raw_test = extract_feature_matrix([windows[i] for i in te_idx], sfreq)
            mins, maxs = raw_train.min(axis=0, keepdims=True), raw_train.max(axis=0, keepdims=True)
            train_in = apply_minmax(raw_train, mins, maxs, 0.0, np.pi)
            train_target = apply_minmax(raw_train, mins, maxs, -1.0, 1.0)
            test_in = apply_minmax(raw_test, mins, maxs, 0.0, np.pi)

            print("  training the generator on the training windows ...")
            weights, losses = train_generator(train_in, train_target, args.epochs, args.lr, seed)
            np.save(WEIGHTS_DIR / f"{subject}_seed{seed}.npy", weights)

            ideal_qnode = make_ideal_qnode()
            ideal_train = np.array([ideal_qnode(f, weights) for f in train_in])
            ideal_test = np.array([ideal_qnode(f, weights) for f in test_in])

            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
            clf.fit(ideal_train, y_train)
            raw_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
            raw_clf.fit(train_in, y_train)

            outputs = {"ideal": ideal_test}
            preds = {"ideal": clf.predict(ideal_test), "raw_features": raw_clf.predict(test_in)}

            tiers = None
            if "rule" in args.methods:
                tiers = rule_allocation(train_in, weights, args.rank_windows, rng)
                print(f"  rule allocation from training windows: {tiers}")
            for method in args.methods:
                start = time.perf_counter()
                outputs[method] = noisy_outputs(method, test_in, weights, tiers)
                preds[method] = clf.predict(outputs[method])
                print(f"    {method}: simulated in {time.perf_counter() - start:.0f}s")

            for method, y_pred in preds.items():
                row = {"subject": subject, "seed": seed, "method": method, "n_train": len(tr_idx),
                       "n_test": len(te_idx), "final_loss": losses[-1], "tiers": str(tiers) if method == "rule" else ""}
                row.update(score(y_test, y_pred))
                results.append(row)
                for yt, yp, ep in zip(y_test, y_pred, te_epochs):
                    predictions.append({"subject": subject, "seed": seed, "method": method,
                                        "epoch": int(ep), "true": yt, "pred": yp})
                print(f"    {method:13s} macro F1 {row['macro_f1']:.3f}  N1 F1 {row['n1_f1']:.3f}  "
                      f"balanced acc {row['balanced_acc']:.3f}")

            pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
            pd.DataFrame(predictions).to_csv(PREDICTIONS_PATH, index=False)
            print(f"  seed {seed} done in {(time.perf_counter() - t0) / 60:.1f} minutes, saved {RESULTS_PATH.name}")

        del windows

    print("\nDone. Next: python scripts/analyze_downstream.py")


if __name__ == "__main__":
    main()