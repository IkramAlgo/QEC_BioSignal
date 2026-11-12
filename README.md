# Matching Quantum Error Correction to Noise Type for a Quantum EEG Feature Circuit

**A subject-level simulation study with downstream sleep staging**

[![Repository](https://img.shields.io/badge/repository-GitHub-181717?logo=github)](https://github.com/IkramAlgo/QEC_BioSignal)

This repository contains research code, analysis outputs, and publication figures for a study of quantum error-correction strategies applied to a variational quantum circuit operating on EEG spectral features.

> **Study status:** Manuscript in preparation. All results reported here are from simulation. No experiment on real quantum hardware is claimed.

## Overview

The study examines how small quantum error-correcting codes affect the reconstruction fidelity of a variational quantum feature circuit under different simulated storage-noise channels. The circuit is trained with a **reconstruction objective**: it learns to reconstruct its input features. It is not an adversarially trained generator or a generative sampler.

The experiments compare an unprotected circuit with bit-flip repetition protection, phase-flip repetition protection, and one sensitivity-based allocation rule combining repetition-code and Shor-code blocks. The analysis considers both reconstruction error and the behavior of a downstream sleep-stage classifier evaluated on protected circuit outputs.

The contribution is intentionally scoped to a subject-level simulation study. It examines matched and mismatched repetition-code behavior for an EEG feature circuit and reports a negative result for the particular sensitivity-based allocation rule tested. It does **not** claim to introduce noise-tailored or application-aware quantum error correction in general.

## Study design

| Component | Configuration |
|---|---|
| Dataset | ANPHY-Sleep, a public high-density sleep EEG dataset originally comprising 29 healthy adults |
| Subjects analyzed | 10: `EPCTL01`–`EPCTL07`, `EPCTL09`, `EPCTL11`, and `EPCTL12` |
| Data availability | Two expected recordings in the working subject list were unavailable or damaged; the analysis uses the 10 subjects listed above |
| Signal | EEG channel C3 |
| Windowing | 10-second windows, one window per 30-second epoch, with an epoch buffer between training and test data |
| Test set | Approximately 60 test windows per subject |
| Features | Eight spectral features per window |
| Scaling | Training-set statistics only; z-score scaling clipped at 2.5 |
| Quantum circuit | Four-qubit variational feature circuit, trained separately for each subject and seed for 60 epochs |
| Held-out reconstruction | Reported $R^2$ across the repaired runs: 0.51–0.81 |
| Noise probabilities | $p \in \{0.02, 0.05, 0.10, 0.20\}$ |
| Noise location | Storage stage, after encoding and before decoding/recovery |
| Noise trajectories | 50 trajectories per evaluated condition |

### Noise model and idealizations

Each physical qubit is independently subject to a stochastic Pauli error during the storage stage, with error probability $p$. The bit-flip channel applies $X$, the phase-flip channel applies $Z$, and the depolarizing channel selects $X$, $Y$, or $Z$ with equal probability when an error occurs.

Encoding, decoding, and majority-vote recovery are treated as noiseless. The simulations therefore isolate the specified storage-noise model; they do not estimate end-to-end hardware performance with noisy gates, syndrome extraction, ancilla preparation, or measurements.

### Readout and evaluation

The circuit readout uses expectation values of $X$ and $Z$ on each block's primary qubit, averaged over 50 noise trajectories. Reconstruction is evaluated against the clean circuit output using mean squared error (MSE) and Wasserstein distance.

For downstream evaluation, a sleep-stage classifier is trained on clean circuit outputs and then frozen. Its performance on the protected outputs is reported using macro-F1. Reconstruction fidelity and downstream classification performance are treated as distinct outcomes: lower reconstruction error does not necessarily imply better sleep-stage classification.

## Protection schemes

The method identifiers below match the result files.

| Identifier | Protection scheme | Physical qubits |
|---|---|---:|
| `no_qec` | No error-correcting protection | 4 |
| `rep03` | Bit-flip repetition code on feature blocks 0 and 3 | 8 |
| `uniform` | Bit-flip repetition code on all four feature blocks | 12 |
| `phase03` | Phase-flip repetition code (Hadamard-basis mirror) on blocks 0 and 3 | 8 |
| `phase_uniform` | Phase-flip repetition code on all four feature blocks | 12 |
| `rule` | Sensitivity-based allocation: one Shor block, two repetition-code blocks, and one bare block | 16 |

The sensitivity ranking is computed using training windows only. The `rule` configuration is evaluated as one specific allocation design, not as a general solution to application-aware quantum error correction.

## Results

### Primary bit-flip and phase-flip comparisons at $p=0.05$

The primary bit-flip and phase-flip comparisons use three seeds per subject. The error ratio is the geometric mean across subjects of protected MSE divided by unprotected MSE; values below 1 indicate lower reconstruction error. Because some unprotected baselines are very small, ratios—particularly in mismatched cases—should be interpreted alongside absolute MSE and Wasserstein values.

| Noise channel | Method | MSE ratio vs. no QEC | Subjects with lower error | Macro-F1 difference | Holm-adjusted F1 $p$-value |
|---|---|---:|---:|---:|---:|
| Bit flip | `rep03` | 0.1164 | 10/10 (30/30 subject–seed pairs) | +0.0179 | 0.1172 |
| Bit flip | `uniform` | 0.0619 | 10/10 (30/30) | +0.0324 | **0.0312** |
| Bit flip | `phase03` | 34.4689 | 0/10 (0/30) | −0.0059 | 0.4805 |
| Bit flip | `phase_uniform` | 150.8919 | 0/10 (0/30) | −0.0130 | 0.3359 |
| Phase flip | `rep03` | 1.9187 | 0/10 (0/30) | −0.0139 | 0.6445 |
| Phase flip | `uniform` | 5.1597 | 0/10 (0/30) | −0.0331 | **0.0312** |
| Phase flip | `phase03` | 0.6425 | 10/10 (27/30 pairs) | +0.0083 | 0.6953 |
| Phase flip | `phase_uniform` | 0.0021 | 10/10 (30/30) | +0.0122 | 0.1367 |

For the eight primary reconstruction-error comparisons, each exact subject-level Wilcoxon signed-rank test yielded $p=0.002$; the Holm-adjusted error $p$-value was 0.0156. The eight-comparison family was selected after an initial inspection of the results and was **not preregistered**. This post hoc selection must be considered when interpreting the inferential results.

The downstream F1 comparisons were corrected separately. Two of the eight primary F1 comparisons remain significant after Holm correction: the full bit-flip repetition configuration has a positive difference, while the full phase-flip repetition configuration has a negative difference. These effects are modest and point in opposite directions; they do not support a general claim that improved reconstruction produces improved classification.

### Depolarizing-noise comparisons

The following ratios are taken from the merged final comparison table. Values above 1 indicate higher MSE than the unprotected circuit.

| Noise probability | `rep03` | `uniform` | `phase03` | `phase_uniform` | `rule` |
|---:|---:|---:|---:|---:|---:|
| 0.02 | 1.4689 | 4.1828 | 1.4036 | 4.2326 | 1.3399 |
| 0.05 | 2.0180 | 5.3163 | 1.8911 | 4.9541 | 1.5185 |
| 0.10 | 2.1362 | 5.2944 | 2.1091 | 5.3882 | 1.5790 |
| 0.20 | 1.9633 | 4.7730 | 1.9379 | 4.6734 | 1.6855 |

These results show no overall reconstruction-error improvement over no protection for the tested schemes under the simulated depolarizing channel. Most noise-level sweep results use one seed and are exploratory. At $p=0.02$, the code comparisons use three seeds, while the allocation-rule result uses one seed.

### Sensitivity-based allocation

The tested allocation rule did not show a reliable benefit over no protection under bit-flip noise: its MSE ratio was 1.2709, with lower error in 5 of 10 subjects. This comparison is exploratory. Under depolarizing noise, its MSE ratio was above 1 at every tested probability. The rule also uses 16 physical qubits, compared with 12 for full repetition protection.

This is a negative result for the particular 16-qubit allocation rule and settings evaluated here. It should not be interpreted as evidence that all sensitivity-based or application-aware allocation strategies are ineffective.

## Statistical analysis and interpretation

- The **subject** is the unit of paired inference. Seeds are repeated runs, not independent subjects.
- Primary reconstruction-error comparisons use subject-level paired Wilcoxon signed-rank tests, with Holm correction across eight comparisons.
- F1 comparisons are corrected separately across the same eight primary comparisons.
- Subject-level bootstrap intervals are part of the planned reporting.
- The primary comparison family was selected after initial inspection and was not preregistered.
- Most noise-level sweeps use one seed and should be treated as exploratory.
- Absolute MSE and Wasserstein values should be reported alongside error ratios, especially when the unprotected MSE is very small.

## Figures

Publication-oriented figures are stored in `outputs/final_figures/`. PDF and SVG versions are available; PNG versions are generated for selected figures.

| Figure | Content | File stem | Notes |
|---|---|---|---|
| 1 | Study design | `fig0_study_design` | Shows the 10-subject study and phase-code layout |
| 2 | Noise/protection regime map | `fig1_regime_map` | Stars refer to the eight primary comparisons; seed coverage should be shown |
| 3 | Paired subject-level comparisons | `fig2_paired_subjects` | Primary comparisons at $p=0.05$ |
| 4 | X- versus Z-output error mechanism | `fig4_mechanism_zx` | Depolarizing diagnostic panel uses four subjects; seed coverage should be stated |
| 5 | Sleep-staging forest plot | `fig3_staging_forest` | Downstream results; only primary comparisons should be marked as primary |
| 6 (optional) | Relative error across noise levels | `fig5_relative_sweep` | Mostly one-seed sweeps; consider supplementary placement |
| Supplementary S1 | Cost-benefit comparison | `fig6_cost_benefit` | Qubit overhead versus error reduction or amplification |
| Supplementary S2 | Absolute error | `figS1_absolute_error` | Error bars represent standard errors over subjects |
| Supplementary S3 | Sleep staging across noise levels | `figS2_staging_levels` | Descriptive results; state seed coverage |

The earlier figures `fig1_noise_type_headline`, `fig2_noise_sweep_error`, `fig3_sleep_staging`, and `fig4_z_vs_x_error` are superseded and should not be used in the manuscript.

## Reproducibility

Run commands from the repository root with the project's Python environment activated.

### 1. Set up the environment

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Run the base study

```bash
python scripts/run_study_v2.py
```

### 3. Run or resume a noise sweep

Example: run the phase-repetition configurations under depolarizing noise at $p=0.05$.

```bash
python scripts/run_noise_sweep.py --noise_types depolarizing --probs 0.05 --methods phase03 phase_uniform
```

The sweep runner saves progress to `outputs/sweep_results.csv` and resumes from existing subject/seed/method keys. **Implementation note:** the script currently uses a fixed `SWEEP_PATH` for this output. Do not assume that `--out` redirects the output unless that behavior has been updated and tested.

### 4. Check the code-theory calculations

```bash
python scripts/check_code_theory.py
```

Verify the script output against the implementation and saved results before reporting predicted-versus-measured logical-error agreement.

### 5. Regenerate merged tables and figures

```bash
python scripts/make_final_figures.py
```

This script combines the study and sweep sources documented in the script, then regenerates:

- `outputs/final_tables.csv`
- `outputs/final_summary.csv`
- `outputs/final_summary_seed0.csv`
- Publication figures in `outputs/final_figures/`

Use the merged outputs for the manuscript tables and figures after checking source-row consistency. The `--summary` option of `run_noise_sweep.py` summarizes `outputs/sweep_results.csv` alone; it is **not** the complete merged analysis.

### Reproducibility notes

For each reported result, retain the repository commit, Python and package versions, random seeds, run command, and source-output files. Re-running the experiments may require access to the original ANPHY-Sleep data. The saved outputs are provided to support inspection of the reported analyses; they should not be treated as a substitute for documenting the exact analysis inputs and software environment.

## Repository structure

```text
QEC_BioSignal/
├── README.md
├── requirements.txt
├── src/
│   ├── application_aware_qec.py
│   ├── repetition_code.py
│   └── phase_flip_code.py
├── scripts/
│   ├── run_study_v2.py
│   ├── run_noise_sweep.py
│   ├── make_final_figures.py
│   └── check_code_theory.py
└── outputs/
    ├── v2_results.csv
    ├── sweep_results.csv
    ├── sweep_pauli.csv
    ├── sweep_diag.csv
    ├── final_tables.csv
    ├── final_summary.csv
    ├── final_summary_seed0.csv
    ├── v2_weights/
    └── final_figures/
```

This tree highlights the principal project files; it is not an exhaustive listing. The final figure-generation script may read additional sources documented in `scripts/make_final_figures.py`.

## Data and citation

The study uses the public ANPHY-Sleep dataset. Please cite the dataset paper when describing the source data:

> Wei et al. (2024). “ANPHY-Sleep: an Open Sleep Database from Healthy Adults Using High-Density Scalp Electroencephalogram.” *Scientific Data*, 11, 896. https://doi.org/10.1038/s41597-024-03722-1

The associated research manuscript is in preparation. Do not cite it as a published paper until publication details are available.

Raw EEG data are not included in this repository. Obtain the dataset from its original source and follow its access and usage conditions. Do not commit credentials, restricted data, or files that should not be redistributed.

## Limitations

1. **Simulation only.** Noise is applied during storage; encoding, decoding, and majority-vote recovery are noiseless. No quantum-hardware performance claim is made.
2. **Limited cohort and channel coverage.** The analysis uses 10 of the 29 ANPHY-Sleep participants and only the C3 EEG channel.
3. **Limited seed coverage.** The primary bit-flip and phase-flip comparisons at $p=0.05$ use three seeds; most noise-level sweeps use one seed.
4. **Modest downstream effects.** Macro-F1 changes are generally small. Two of the eight primary F1 comparisons remain significant after the stated Holm correction, and the effects have opposite directions.
5. **Post hoc primary family.** The eight-comparison family was selected after initial inspection and was not preregistered.
6. **Idealized QEC operations.** The simulation excludes noisy encoding/decoding gates, syndrome extraction, ancilla preparation, and measurement errors.
7. **No validated classical baseline.** The current manuscript does not use a validated classical feature-model baseline and makes no claim of quantum advantage.
8. **No noise-aware training or error-mitigation baseline.** These are outside the scope of the current experiments.
9. **Narrow allocation result.** The negative result applies to the specific 16-qubit allocation rule and tested settings, not to all application-aware allocation methods.

## License

A repository license has not yet been specified. Until a license is added, do not assume that the code or other repository contents are licensed for reuse. Confirm the intended license with all relevant authors before adding one.

---

**Research repository:** [IkramAlgo/QEC_BioSignal](https://github.com/IkramAlgo/QEC_BioSignal)
