# Application-Aware Quantum Error Correction for Noise-Resilient Biosignal Modeling

A proof-of-concept framework for studying **application-aware quantum error correction (QEC)** in quantum generative models for biomedical signal features.

The central idea is simple:

> Not every logical qubit contributes equally to the downstream application. Instead of protecting every qubit with the same error-correcting code, allocate stronger protection to the logical qubits that are empirically more sensitive to noise.

This repository currently evaluates that idea on EEG-derived features using a small variational quantum circuit and classical simulation.

---

## Current Status

**Stage:** Proof of concept

The current generator is **not trained**. The present experiments focus on:

1. Encoding an EEG-derived feature vector into a small quantum circuit.
2. Measuring degradation caused by simulated noise.
3. Establishing ideal, noisy, and QEC baselines.
4. Ranking logical qubits by application-level sensitivity.
5. Allocating different QEC strengths according to that ranking.
6. Comparing application-aware protection against uniform protection.

Training the quantum generator to learn the real feature distribution is a planned next stage.

---

## Research Question

Can application-aware allocation of quantum error correction provide better protection of application-level outputs than applying the same QEC code uniformly to every logical qubit?

The hypothesis is:

> If logical qubits have different sensitivities to noise, protecting them uniformly can waste physical-qubit resources. A sensitivity-aware strategy may achieve lower application-level error with fewer physical qubits than uniform heavy protection.

---

## Pipeline

```text
ANPHY-Sleep EDF
       |
       v
EEG channel selection
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
4-qubit variational quantum circuit
       |
       +-------------------+
       |                   |
       v                   v
    Ideal             Noisy circuit
                           |
                           v
                    QEC strategies
                    /      |       \
                   /       |        \
              No QEC   Uniform     Application-
                        QEC         Aware QEC
                   \       |        /
                    \      |       /
                     v     v       v
                    Application-level
                       comparison
                           |
                           v
                  MSE + Wasserstein
```

---

## Dataset and Signal Representation

The prototype operates on overnight EEG data stored in EDF format.

The current experiments use:

* **Subject:** EPCTL01
* **Channel:** C3
* **Window length:** 10 seconds
* **Primary experiment:** 100 windows
* **Sampling rate:** 1000 Hz in the underlying EEG data
* **Quantum input:** 8-dimensional feature vector

The raw EEG waveform is **not directly passed to the quantum circuit**.

Instead, each EEG window is converted into eight features:

1. Delta band power
2. Theta band power
3. Alpha band power
4. Sigma band power
5. Beta band power
6. Variance
7. Spectral entropy
8. Autocorrelation

The features are normalized to `[0, π]` before quantum encoding.

---

## Quantum Generator

The current prototype uses a deliberately small **4-qubit variational quantum circuit**.

Each logical qubit receives two input features:

```text
feature[2i]     -> RY
feature[2i + 1] -> RZ
```

The circuit contains two variational layers.

Each layer applies:

```text
RY
RZ
CNOT ring
```

The circuit produces eight expectation values:

```text
<Z0>, <Z1>, <Z2>, <Z3>,
<X0>, <X1>, <X2>, <X3>
```

These eight measurements form the generated feature representation.

The implementation uses PennyLane.

---

# QEC Experiments

## 1. Ideal Circuit

The ideal circuit is used as the reference distribution.

No noise and no QEC are applied.

```text
4 logical qubits
4 physical qubits
```

The ideal result is defined as zero distance from itself:

```text
MSE = 0
Wasserstein = 0
```

---

## 2. No QEC

The noisy baseline applies depolarizing noise with:

```text
p = 0.02
```

No error correction is applied.

```text
4 logical qubits
4 physical qubits
```

This establishes how much the application-level output degrades without QEC.

---

## 3. Uniform Repetition Code

The uniform QEC baseline protects all four logical qubits with a distance-3 repetition code.

```text
4 logical qubits
12 physical qubits
3 physical qubits / logical qubit
```

This provides a useful comparison against the application-aware strategy.

An important observation from the current experiment is that **uniform protection does not automatically produce better application-level results**. In the current configuration, the uniform repetition result is worse than the unprotected baseline.

This motivates the application-aware allocation strategy rather than assuming that more QEC is always better.

---

# 4. Application-Aware QEC

The application-aware strategy first measures the sensitivity of each logical qubit.

The current sensitivity ranking is:

| Rank | Logical qubit | Combined MSE |
| ---: | ------------: | -----------: |
|    1 |       Qubit 1 |    0.0005517 |
|    2 |       Qubit 2 |    0.0001098 |
|    3 |       Qubit 3 |    0.0000424 |
|    4 |       Qubit 0 |    0.0000107 |

The protection allocation is then:

| Logical qubit | Sensitivity | Protection     | Physical qubits |
| ------------- | ----------- | -------------- | --------------: |
| Qubit 1       | Highest     | Shor code      |               9 |
| Qubit 2       | High        | Repetition d=3 |               3 |
| Qubit 3       | Lower       | Bare           |               1 |
| Qubit 0       | Lowest      | Bare           |               1 |
| **Total**     |             |                |          **14** |

Therefore:

```text
4 logical qubits
        |
        +-- Qubit 0 -> bare
        +-- Qubit 1 -> Shor
        +-- Qubit 2 -> repetition
        +-- Qubit 3 -> bare
        |
        v
14 physical qubits
```

This is the central experiment in the repository.

The allocation is derived from the measured sensitivity ranking rather than chosen uniformly.

---

# Current Experimental Results

The main application-aware experiment was run on:

```text
Subject: EPCTL01
Channel: C3
Windows: 100
Window duration: 10 seconds
Noise probability: p = 0.02
```

### Results

| Method                    | Physical Qubits | MSE vs Ideal ↓ | Wasserstein vs Ideal ↓ |
| ------------------------- | --------------: | -------------: | ---------------------: |
| Ideal                     |               4 |       0.000000 |               0.000000 |
| No QEC                    |               4 |     0.00012359 |             0.00527125 |
| Uniform Repetition d=3    |              12 |     0.00044092 |             0.00871476 |
| **Application-Aware QEC** |          **14** | **0.00006999** |         **0.00345390** |

### Application-aware improvement

Relative to the no-QEC baseline:

* **MSE is reduced by approximately 43%**
* **Wasserstein distance is reduced by approximately 34%**

The application-aware result is also substantially better than the uniform repetition result in this experiment.

However, these results should currently be interpreted as a **proof-of-concept result**, not as a general performance claim. The experiment currently uses one subject/channel configuration and an untrained quantum generator.

---

# Why Application-Aware QEC?

A uniform strategy treats all logical qubits as equally important.

That can be inefficient when the downstream application is sensitive to some quantum outputs more than others.

This project instead uses an application-level sensitivity analysis:

```text
Measure sensitivity
       |
       v
Rank logical qubits
       |
       v
Assign QEC strength
       |
       +---- High sensitivity -> stronger code
       |
       +---- Medium sensitivity -> moderate code
       |
       +---- Low sensitivity -> little/no protection
```

The objective is not simply to maximize the amount of error correction.

The objective is:

> **Minimize application-level error while controlling physical-qubit overhead.**

---

# Resource Comparison

The current experiment demonstrates an important simulator constraint.

A full Shor-code implementation on all four logical qubits would require:

```text
4 × 9 = 36 physical qubits
```

A statevector simulation of a 36-qubit circuit is prohibitively expensive for the current local environment.

The application-aware configuration instead requires:

```text
9 + 3 + 1 + 1 = 14 physical qubits
```

This makes the mixed-protection experiment practical while still applying strong protection to the most sensitive logical qubit.

The 14-qubit application-aware circuit also passes a zero-noise identity check before noisy experiments are trusted.

---

# Validation

Before running the application-aware noisy experiment, the implementation verifies that the encoded-and-decoded circuit reproduces the ideal circuit when the noise probability is zero.

The current validation reports:

```text
Identity check passed across 5 trials,
ideal matches application aware circuit at zero noise.
```

This check is important because QEC should not introduce an application-level change in the absence of noise.

---

# Repository Structure

```text
QEC_BioSignal/
│
├── README.md
├── requirements.txt
├── verify_shor.py
│
├── src/
│   ├── __init__.py
│   ├── application_aware_qec.py
│   ├── data_loading.py
│   ├── features.py
│   ├── metrics.py
│   ├── preprocessing.py
│   ├── quantum_model.py
│   ├── repetition_code.py
│   └── shor_code.py
│
├── scripts/
│   ├── inspect_edf.py
│   ├── run_baseline_experiment.py
│   ├── run_qec_experiment.py
│   ├── run_sensitivity_ranking.py
│   ├── run_application_aware_experiment.py
│   └── benchmark_12qubit.py
│
└── outputs/
    ├── application_aware_results.csv
    ├── application_aware_plot.png
    ├── application_aware_journal_figure.png
    ├── baseline_results.csv
    ├── baseline_plot.png
    ├── qec_results.csv
    ├── qec_plot.png
    └── sensitivity_ranking.csv
```

The repository currently contains the baseline, QEC, sensitivity-ranking, and application-aware experiment implementations.

---

# Installation

Create the environment:

```bash
python -m venv qec_biosignal_env
```

### Windows

```powershell
qec_biosignal_env\Scripts\activate
```

### Linux/macOS

```bash
source qec_biosignal_env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Data

Place an EDF recording in:

```text
data/raw/
```

Raw EEG recordings are intentionally excluded from version control.

Inspect an EDF file:

```bash
python scripts/inspect_edf.py data/raw/YOUR_FILE.edf
```

---

# Running the Experiments

## Baseline

```bash
python scripts/run_baseline_experiment.py data/raw/YOUR_FILE.edf --channel "C3"
```

## QEC experiment

```bash
python scripts/run_qec_experiment.py data/raw/YOUR_FILE.edf --channel "C3"
```

## Sensitivity ranking

```bash
python scripts/run_sensitivity_ranking.py data/raw/YOUR_FILE.edf --channel "C3"
```

## Application-aware QEC

Example:

```bash
python scripts/run_application_aware_experiment.py data/raw/EPCTL01.edf --channel "C3" --max_windows 100
```

The application-aware experiment performs:

1. EEG loading
2. Windowing
3. Feature extraction
4. Zero-noise identity verification
5. Ideal simulation
6. No-QEC noisy simulation
7. Uniform repetition-code simulation
8. Application-aware mixed-QEC simulation
9. MSE comparison
10. Wasserstein comparison
11. Result export
12. Plot generation

---

# Output Files

The application-aware experiment produces:

```text
outputs/application_aware_results.csv
outputs/application_aware_plot.png
```

The repository also contains the current sensitivity ranking and experimental figures.

---

# Important Limitations

The current results should not yet be interpreted as evidence that application-aware QEC is universally superior.

Current limitations include:

* The quantum generator is **not trained**.
* The main reported experiment uses one subject/channel configuration.
* The current sensitivity ranking is tied to the specific experiment.
* The application-aware allocation should be recomputed when the dataset, channel, subject, or generator weights change.
* The current noise model is simulated rather than hardware-derived.
* The current study evaluates application-level feature reconstruction rather than a downstream clinical task.
* Statistical significance and multi-subject generalization have not yet been established.
* The current experiment does not yet evaluate hardware execution.

The sensitivity-based allocation is intentionally tied to the measured ranking; it should not be assumed to transfer unchanged to another dataset, channel, subject, or set of generator weights.

---

# Next Research Steps

The next stages of the project are:

### 1. Train the quantum generator

Move from the current untrained variational circuit toward a trained generative model.

Possible objectives include:

* reconstruction loss
* MMD/distribution matching
* adversarial training
* hybrid quantum-classical generative training

### 2. Recompute sensitivity after training

Training may change which logical qubits are most important.

Therefore:

```text
Train generator
      ↓
Measure sensitivity
      ↓
Allocate QEC
      ↓
Evaluate noisy performance
```

should become an iterative experimental pipeline.

### 3. Evaluate additional noise models

The current framework supports:

* depolarizing noise
* bit-flip noise
* phase-flip noise
* readout error

Additional experiments can evaluate different error channels and combinations.

### 4. Multi-subject evaluation

The current proof of concept should be expanded to multiple subjects and EEG channels.

### 5. Statistical evaluation

Future experiments should report:

* mean ± standard deviation
* confidence intervals
* repeated seeds
* statistical tests
* physical-qubit overhead
* runtime
* robustness across subjects

### 6. Hardware validation

After simulator validation, the most promising configurations can be evaluated on available quantum hardware or realistic hardware noise models.

---

# Research Direction

This project explores the intersection of:

* Quantum error correction
* Quantum machine learning
* Generative modeling
* Biomedical signal processing
* EEG analysis
* Application-aware resource allocation
* Noise-resilient quantum computing

The broader research question is whether QEC should be designed only around **physical error rates**, or whether the requirements of the **downstream application** should also determine where computational resources are spent.

---

## Citation

If you use this repository or build upon the implementation, please cite the repository and the associated research work when available.

## License

License information will be added as the project matures.
