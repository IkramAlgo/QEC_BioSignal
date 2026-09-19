# Application-Aware Quantum Error Correction for Noise-Resilient Biosignal Modeling

A framework for studying **application-aware quantum error correction (QEC)** in quantum generative models for biomedical signal features.

The central idea is simple:

> Not every logical qubit contributes equally to the downstream application. Instead of protecting every qubit with the same error-correcting code, allocate stronger protection to the logical qubits that are empirically more sensitive to noise, and recompute that allocation per subject rather than assuming it transfers.

This repository evaluates that idea on EEG-derived features using a small trained variational quantum circuit and classical simulation, validated across 5 subjects and 3 random seeds each.

---

## Current Status

**Stage:** Validated proof of concept, multi-subject

The quantum generator is **trained** (reconstruction objective) with a proper held-out test split, evaluation is reported on windows the generator never saw during training. The application-aware allocation is **computed dynamically per subject** from that subject's own measured sensitivity ranking, not copied from a single reference subject.

The pipeline:

1. Encode an EEG-derived feature vector into a small quantum circuit.
2. Train the generator against a reconstruction objective, held-out split enforced.
3. Measure degradation caused by simulated noise, on held-out windows only.
4. Rank logical qubits by application-level sensitivity, per subject.
5. Allocate QEC strength dynamically from that ranking.
6. Compare application-aware protection against no protection and uniform protection.
7. Repeat across 5 subjects x 3 seeds and report pooled statistics.

---

## Research Question

Can application-aware allocation of quantum error correction provide better protection of application-level outputs than applying the same QEC code uniformly to every logical qubit, and does this hold across different people, not just one recording?

The hypothesis:

> If logical qubits have different sensitivities to noise, protecting them uniformly can waste physical-qubit resources or actively hurt performance. A sensitivity-aware strategy, recalibrated per subject, achieves lower application-level error than uniform protection.

This hypothesis is now supported by evidence across 5 subjects, not just asserted from a single case.

---

## Pipeline

```text
ANPHY-Sleep EDF (5 subjects)
       |
       v
EEG channel selection (C3)
       |
       v
10-second windows
       |
       v
Biomedical feature extraction
       |
       v
8-dimensional feature vector
       |
       v
Train/test split (80/20, per subject, per seed)
       |
       v
Train 4-qubit variational generator (reconstruction loss, training windows only)
       |
       v
Evaluate on HELD-OUT windows only
       |
       +-------------------+
       |                   |
       v                   v
    Ideal             Noisy circuit (depolarizing, p=0.02)
                           |
                           v
                    QEC strategies
                    /      |       \
                   /       |        \
              No QEC   Uniform     Application-
                        Repetition   Aware QEC
                        (d=3)        (per-subject
                                      calibration)
                   \       |        /
                    \      |       /
                     v     v       v
                    Application-level
                       comparison
                           |
                           v
                  MSE + Wasserstein, pooled across
                  5 subjects x 3 seeds
```

---

## Dataset and Signal Representation

The prototype operates on overnight EEG data stored in EDF format, ANPHY-Sleep.

Current experiments use:

* **Subjects:** EPCTL01 through EPCTL05
* **Channel:** C3
* **Window length:** 10 seconds
* **Windows per subject:** 100 (80 training, 20 held out)
* **Seeds per subject:** 0, 1, 2
* **Sampling rate:** 1000 Hz in the underlying EEG data
* **Quantum input:** 8-dimensional feature vector

Each EEG window is converted into eight features:

1. Delta band power
2. Theta band power
3. Alpha band power
4. Sigma band power
5. Beta band power
6. Variance
7. Spectral entropy
8. Autocorrelation

Features are normalized to `[0, π]` for quantum encoding. A separate `[-1, 1]` rescaling of the same raw features is used as the training reconstruction target, matching the circuit's expectation-value output range.

---

## Quantum Generator

A **4-qubit variational quantum circuit**, now trained rather than using fixed random weights.

Each logical qubit receives two input features:

```text
feature[2i]     -> RY
feature[2i + 1] -> RZ
```

Two variational layers, each applying RY, RZ, then a CNOT ring.

Output: eight expectation values, `<Z0..Z3>` and `<X0..X3>`.

**Training objective:** reconstruction. The circuit's 8 output expectation values are trained to reconstruct a `[-1, 1]`-scaled version of the same 8 input features, using the Adam optimizer with mini-batch gradient steps. Trained only on the 80% training split, per subject, per seed. Typical loss reduction: 73-86% over 200 epochs.

Implementation uses PennyLane.

---

## QEC Methods

### 1. Ideal circuit
No noise, no QEC. Reference distribution, MSE and Wasserstein are zero by definition.

### 2. No QEC
Depolarizing noise, `p = 0.02`, applied directly to the 4 physical qubits. No correction.

### 3. Uniform repetition code
Distance-3 bit-flip repetition code applied identically to all 4 logical qubits, 12 physical qubits total.

**Important finding:** a bit-flip repetition code only corrects one of the three error types present in depolarizing noise. Pooled across all 5 subjects, uniform repetition is **32% worse than no protection at all** on MSE. This is a genuine, diagnosed limitation, not a bug, confirmed by testing the same code against bit-flip-only noise (where it works correctly) versus full depolarizing noise (where it does not). This result directly motivates application-aware allocation rather than assuming more QEC is always better.

### 4. Application-aware QEC (dynamic, per subject)

For each subject, independently:

1. Measure per-logical-qubit sensitivity on that subject's held-out windows.
2. Rank the 4 logical qubits by measured MSE degradation.
3. Allocate protection tiers from that ranking: most sensitive qubit gets a 9-qubit Shor code (protects all Pauli error types), next two qubits get a 3-qubit repetition code, least sensitive qubit gets no protection.
4. Total: 16 physical qubits (9 + 3 + 3 + 1).

The allocation is **not fixed**. Training the generator, or evaluating a different subject, changes which qubit is most sensitive; the ranking is recomputed and the allocation rebuilt every time. An earlier version of this project used one allocation frozen from a single subject; testing across 5 subjects showed this does not generalize (see Multi-Subject Results below), which is why allocation is now computed dynamically at runtime from `outputs/sensitivity_ranking.csv`.

Every application-aware circuit passes a zero-noise identity check (encode-then-decode must exactly reproduce the ideal circuit at `p=0`) before any noisy result is trusted.

---

## Multi-Subject Results

Validated across **5 subjects (EPCTL01-EPCTL05) x 3 seeds each = 15 runs**, held-out test windows only, per-subject dynamic allocation.

### Pooled results (mean, all 15 runs)

| Method | Physical Qubits | MSE vs Ideal | Wasserstein vs Ideal |
|---|---:|---:|---:|
| Ideal | 4 | 0.000000 | 0.000000 |
| No QEC | 4 | 0.000847 | 0.016773 |
| Uniform Repetition (d=3) | 12 | 0.001120 | 0.015989 |
| **Application-Aware QEC** | **16** | **0.000638** | **0.012932** |

### Improvement over no QEC

* **Application-aware: 24.6% lower MSE, 22.9% lower Wasserstein distance**
* Uniform repetition: **32.2% worse** MSE, 4.7% better Wasserstein

### Improvement over uniform repetition

* **Application-aware: 43.0% lower MSE, 19.1% lower Wasserstein distance**

Application-aware QEC won on MSE for all 5 subjects. On Wasserstein distance it won for 4 of 5 subjects; EPCTL03 was the one exception, coming in slightly worse than no QEC on that metric only. This is reported plainly as a known exception rather than smoothed over.

This result required fixing an earlier methodology issue: initial single-subject results were partly measured on windows the generator had already trained on. A proper 80/20 train/test split was added, and all reported numbers above are held-out only.

---

## Why Dynamic, Per-Subject Allocation?

An earlier version of this project used one allocation, built from a single subject's sensitivity ranking, applied unchanged to every subject. Testing across 5 subjects showed this does **not** generalize: the frozen allocation only helped on 2 of 5 subjects, and on the other 3, both uniform and application-aware protection actively hurt performance relative to no protection at all.

Rebuilding the allocation per subject, from that subject's own ranking, resolved this: application-aware now wins on MSE for all 5 subjects. This is treated as a central finding of the project, not an implementation detail: **noise-resilient QEC for this kind of generative model needs to be calibrated per subject, not assumed to transfer.**

---

## Resource Comparison

A full Shor-code implementation on all 4 logical qubits simultaneously would require:

```text
4 x 9 = 36 physical qubits
```

Full statevector simulation of 36 qubits requires roughly 1 TB of memory, not feasible on standard hardware. The application-aware configuration instead requires:

```text
9 + 3 + 3 + 1 = 16 physical qubits
```

making the experiment tractable while still giving full Pauli-error protection to the qubit(s) that need it most.

---

## Repository Structure

```text
qec_biosignal/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── data_split.py
│   ├── quantum_model.py
│   ├── repetition_code.py
│   ├── shor_code.py
│   ├── application_aware_qec.py
│   └── metrics.py
│
├── scripts/
│   ├── inspect_edf.py
│   ├── train_generator.py
│   ├── run_sensitivity_ranking.py
│   ├── run_application_aware_experiment.py
│   └── run_multi_subject_study.py
│
├── data/raw/           (EDF files, excluded from version control)
│
└── outputs/
    ├── trained_weights.npy
    ├── training_loss.csv
    ├── sensitivity_ranking.csv
    ├── application_aware_results_testsplit.csv
    ├── application_aware_plot_testsplit.png
    └── multi_subject_results.csv
```

---

## Installation

```bash
python -m venv qec_biosignal_env
```

Windows:
```powershell
qec_biosignal_env\Scripts\activate
```

Linux/macOS:
```bash
source qec_biosignal_env/bin/activate
```

```bash
pip install -r requirements.txt
```

---

## Data

Place EDF recordings in `data/raw/`. Raw EEG recordings are excluded from version control.

Inspect a file's channel names before assuming ANPHY's naming is consistent across subjects:

```bash
python scripts/inspect_edf.py data/raw/YOUR_FILE.edf
```

---

## Running the Experiments

Full pipeline for one subject, one seed:

```bash
python scripts/train_generator.py data/raw/EPCTL01.edf --channel "C3" --max_windows 100 --seed 0
python scripts/run_sensitivity_ranking.py data/raw/EPCTL01.edf --channel "C3" --max_windows 100 --seed 0 --weights_path outputs/trained_weights.npy
python scripts/run_application_aware_experiment.py data/raw/EPCTL01.edf --channel "C3" --max_windows 100 --seed 0 --weights_path outputs/trained_weights.npy
```

`train_generator.py` must run first, same subject and seed, immediately before the other two, since the ranking and evaluation scripts read `outputs/trained_weights.npy` from that run.

Full multi-subject, multi-seed study, edit the `SUBJECTS` list at the top of the script first:

```bash
python scripts/run_multi_subject_study.py
```

This runs the full train/rank/evaluate sequence for every subject and seed automatically and writes `outputs/multi_subject_results.csv`.

---

## Important Limitations

* Statistical testing so far compares means; no paired significance test has been run yet.
* No classical baseline has been evaluated; the case for using a quantum approach specifically is not yet made.
* No downstream clinical task (e.g. sleep-stage classification) has been evaluated yet, despite hypnograms being available for these subjects.
* All noise is simulated (depolarizing channel); no real quantum hardware has been used.
* Current results are on ANPHY-Sleep; the planned move to Sleep-EDF Expanded for the primary reported dataset has not yet happened.
* 5 subjects, 20 held-out windows each; both numbers are planned to increase.

---

## Next Research Steps

1. **Scale to 10 subjects.**
2. **Increase held-out window count per subject** for more stable evaluation.
3. **Statistical testing:** paired test (e.g. Wilcoxon signed-rank) across subjects, reporting p-values alongside percentage improvements.
4. **Classical baseline:** a non-quantum denoising or mitigation method through the same pipeline and metrics.
5. **Downstream task:** sleep-stage classification accuracy under each QEC method, using existing hypnograms.
6. **Hardware validation:** at least a small-scale run on real quantum hardware.
7. **Dataset switch:** move primary results to Sleep-EDF Expanded, keep ANPHY as supporting evidence.

---

## Research Direction

This project explores the intersection of quantum error correction, quantum machine learning, generative modeling, biomedical signal processing, EEG analysis, application-aware resource allocation, and noise-resilient quantum computing.

The broader research question: should QEC be designed only around physical error rates, or should the requirements of the downstream application, and the specific person generating the data, also determine where computational resources are spent?

---

## Citation

If you use this repository or build upon the implementation, please cite the repository and the associated research work when available.

## License

License information will be added as the project matures.