# Matching Quantum Error Correction to Noise Type for a Quantum EEG Feature Circuit

**A subject-level simulation study with downstream sleep staging**  
Repository: [IkramAlgo/QEC_BioSignal](https://github.com/IkramAlgo/QEC_BioSignal)  
**Status:** Research code and analysis outputs for a manuscript in preparation. Results are simulated; no real quantum-hardware experiment is claimed.

## Project overview

This project studies when small quantum error-correcting codes reduce or increase reconstruction error in a variational quantum circuit applied to EEG spectral features. The circuit is trained with a **reconstruction objective**: it reconstructs its input features. It is not an adversarially trained generator or a generative sampler.

The main comparison is between no protection, bit-flip repetition protection, phase-flip repetition protection, and a sensitivity-based allocation rule that combines repetition and Shor-code blocks. The study asks how the protection scheme interacts with the noise channel and whether any reconstruction-error changes carry through to a downstream sleep-stage classifier.

The contribution is deliberately narrow: a subject-level comparison for an EEG feature circuit, an analysis of matched and mismatched repetition-code behavior, and a negative result for the particular sensitivity-based allocation rule tested. This project does **not** claim to introduce noise-tailored or application-aware QEC in general.

## Current study configuration

- **Dataset:** ANPHY-Sleep, a public high-density sleep EEG dataset originally containing 29 healthy adults.
- **Subjects used:** 10 subjects: `EPCTL01`–`EPCTL07`, `EPCTL09`, `EPCTL11`, and `EPCTL12`. Two files from the original list were missing or damaged.
- **Signal:** EEG channel C3; 10-second windows sampled at one window per 30-second epoch, with an epoch buffer between training and test data. There are approximately 60 test windows per subject.
- **Features:** 8 spectral features per window. Scaling uses training-set statistics only (z-score, clipped at 2.5).
- **Circuit:** 4-qubit variational circuit trained separately per subject and seed for 60 epochs. Held-out reconstruction \(R^2\) across the repaired runs was 0.51–0.81.
- **Noise:** one independent Pauli fault per physical qubit, applied once to the stored circuit output after encoding, with \(p\in\{0.02,0.05,0.10,0.20\}\). Bit-flip noise applies \(X\), phase-flip noise applies \(Z\), and the depolarizing model selects \(X\), \(Y\), or \(Z\) with equal probability.
- **Idealized operations:** encoding, decoding, and majority vote are noiseless. The results therefore isolate this storage-noise model; they do not estimate the performance of an end-to-end hardware implementation with noisy gates and measurements.
- **Readout and metrics:** expectation values of \(X\) and \(Z\) on each block's primary qubit, averaged over 50 noise trajectories. Reconstruction is evaluated against the clean circuit output using MSE and Wasserstein distance. A classifier trained on clean outputs is then frozen and evaluated using macro-F1.

## Protection schemes

| Method name in result files | Description | Total physical qubits |
|---|---|---:|
| `no_qec` | No error-correcting protection | 4 |
| `rep03` | Bit-flip repetition code on feature blocks 0 and 3 | 8 |
| `uniform` | Bit-flip repetition code on all four feature blocks | 12 |
| `phase03` | Phase-flip repetition code (Hadamard-basis mirror) on blocks 0 and 3 | 8 |
| `phase_uniform` | Phase-flip repetition code on all four feature blocks | 12 |
| `rule` | Sensitivity-based allocation: one Shor block, two repetition-code blocks, and one bare block | 16 |

The sensitivity ranking is computed from training windows only. The allocation rule is evaluated as a particular design choice—not as a general solution to application-aware QEC.

## Main results at \(p=0.05\)

The following results use three seeds per subject for the bit-flip and phase-flip primary comparisons. Error ratio is the geometric mean across subjects of protected MSE divided by unprotected MSE; values below 1 mean lower reconstruction error. Large ratios in the mismatched cases start from a very small unprotected baseline, so absolute MSE should always be reported alongside ratios.

| Noise | Method | Error ratio vs. no QEC | Subjects with lower error | Macro-F1 difference | Holm-adjusted F1 p-value |
|---|---|---:|---:|---:|---:|
| Bit flip | `rep03` | 0.1164 | 10/10 (30/30 subject–seed pairs) | +0.0179 | 0.1172 |
| Bit flip | `uniform` | 0.0619 | 10/10 (30/30) | +0.0324 | **0.0312** |
| Bit flip | `phase03` | 34.4689 | 0/10 (0/30) | −0.0059 | 0.4805 |
| Bit flip | `phase_uniform` | 150.8919 | 0/10 (0/30) | −0.0130 | 0.3359 |
| Phase flip | `rep03` | 1.9187 | 0/10 (0/30) | −0.0139 | 0.6445 |
| Phase flip | `uniform` | 5.1597 | 0/10 (0/30) | −0.0331 | **0.0312** |
| Phase flip | `phase03` | 0.6425 | 10/10 (27/30 pairs) | +0.0083 | 0.6953 |
| Phase flip | `phase_uniform` | 0.0021 | 10/10 (30/30) | +0.0122 | 0.1367 |

For the primary family of eight error comparisons, each exact Wilcoxon test has \(p=0.002\) and the Holm-adjusted error p-value is 0.0156. The eight-comparison family was defined after an initial look at the results and was **not preregistered**; this must be disclosed. The F1 results are separate: two of eight primary F1 comparisons remain significant after Holm correction, and these effects are small and in opposite directions.

## Depolarizing-noise results

The phase-flip configurations were subsequently evaluated under depolarizing noise. The ratios below are from the merged final comparison table. Ratios above 1 indicate higher MSE than no protection. The \(p=0.02\) records combine the original study outputs with the sweep records; do not interpret the one-file `--summary` output as the complete merged analysis.

| Noise probability | `rep03` | `uniform` | `phase03` | `phase_uniform` | `rule` |
|---:|---:|---:|---:|---:|---:|
| 0.02 | 1.4689 | 4.1828 | 1.4036 | 4.2326 | 1.3399 |
| 0.05 | 2.0180 | 5.3163 | 1.8911 | 4.9541 | 1.5185 |
| 0.10 | 2.1362 | 5.2944 | 2.1091 | 5.3882 | 1.5790 |
| 0.20 | 1.9633 | 4.7730 | 1.9379 | 4.6734 | 1.6855 |

These results do not show an overall reconstruction-error improvement from either tested repetition-code orientation or from the tested allocation rule under the simulated depolarizing channel. The noise-level sweeps mostly use one seed and should be described as exploratory; the \(p=0.02\) code comparisons use three seeds, while the allocation rule has only one seed.

The separate file `outputs/sweep_phase_depolarizing_p005.csv` contains 20 phase-code rows for \(p=0.05\). A key check found that all 20 subject/seed/method keys already exist in `outputs/sweep_results.csv`, and macro-F1 matched exactly. The MSE and Wasserstein values have **not yet been directly compared in the same check**. Keep the separate file until those metric comparisons are complete; avoid appending duplicate rows.

## Sensitivity-based allocation result

The tested allocation rule did not show a reliable benefit over no protection under bit-flip noise (error ratio 1.2709, 5/10 subjects with lower error; exploratory comparison). Under depolarizing noise its error ratio was above 1 at every tested level. It also used 16 physical qubits, compared with 12 for full repetition protection. We report this as a negative result for this particular rule, not as evidence that all sensitivity-based allocation strategies fail.

## Statistical reporting

- The subject is the unit for paired inference; seeds are repeated runs, **not independent subjects**.
- Primary error comparisons use subject-level paired Wilcoxon signed-rank tests and Holm correction across eight comparisons.
- F1 tests are corrected separately for the same primary family.
- Subject-level bootstrap intervals are part of the planned reporting.
- The primary family was selected after initial inspection and was not preregistered.
- Most noise-level sweeps use one seed; report them as exploratory.
- Report absolute MSE and Wasserstein values with ratios, especially where the unprotected MSE is extremely small.

## Journal figure plan

The current publication figures are in `outputs/final_figures/`. PDF and SVG versions are available; PNG versions are also generated for selected figures. Check the target journal's current figure specifications before submission.

| Planned placement | Figure content | File stem | Notes |
|---|---|---|---|
| Figure 1 | Study design | `fig0_study_design` | State that 10 subjects were used and explain the Hadamard-basis phase-code layout. |
| Figure 2 | Noise/protection regime map | `fig1_regime_map` | Headline figure. Explain that stars refer to the eight primary comparisons only and show seed counts. |
| Figure 3 | Paired subject-level comparisons | `fig2_paired_subjects` | Primary comparison at \(p=0.05\); display subject counts and absolute values where possible. |
| Figure 4 | X- versus Z-output error mechanism | `fig4_mechanism_zx` | Depolarizing diagnostic panel uses four subjects and the diagnostic sweep; accurately label the seed coverage. |
| Figure 5 | Sleep-staging forest plot | `fig3_staging_forest` | Descriptive downstream results; mark only the primary comparisons as primary. |
| Figure 6 (optional) | Relative error across noise levels | `fig5_relative_sweep` | The level sweeps are seed 0 except where stated. Move to the supplement if the journal's figure limit requires it. |
| Supplementary Figure S1 | Cost-benefit comparison | `fig6_cost_benefit` | Compare qubit overhead with error reduction or amplification. |
| Supplementary Figure S2 | Absolute error | `figS1_absolute_error` | State seed coverage and that error bars are standard errors over subjects. |
| Supplementary Figure S3 | Sleep staging across noise levels | `figS2_staging_levels` | Descriptive; state seed coverage. |

Do not use the superseded early figures `fig1_noise_type_headline`, `fig2_noise_sweep_error`, `fig3_sleep_staging`, or `fig4_z_vs_x_error`, or earlier figure outputs created before the Holm-family/layout corrections.

## Reproducing the analysis

Run commands from the repository root with the project's Python environment activated.

### Re-run the base study

```powershell
python scripts\run_study_v2.py
```

### Run or resume a noise sweep

```powershell
python scripts\run_noise_sweep.py --noise_types depolarizing --probs 0.05 --methods phase03 phase_uniform
```

The sweep runner saves progress to `outputs/sweep_results.csv` and resumes from existing keys. **Implementation note:** the script currently defines a fixed `SWEEP_PATH` pointing to `outputs/sweep_results.csv`; do not assume the `--out` option redirects the file until that behavior has been fixed and tested.

### Print the sweep-file summary

```powershell
python scripts\run_noise_sweep.py --summary
```

This summary describes `outputs/sweep_results.csv` by itself. For the final study tables, use the merged analysis below because the figure script draws on multiple input files.

### Check the logical-error calculations

```powershell
python scripts\check_code_theory.py
```

Verify this output against the code and saved results before quoting predicted-versus-measured ratio agreement in the manuscript.

### Regenerate tables and figures

```powershell
python scripts\make_final_figures.py
```

The script combines study and sweep records, regenerates the tables in `outputs/final_tables.csv`, `outputs/final_summary.csv`, and `outputs/final_summary_seed0.csv`, and writes publication figures to `outputs/final_figures/`. The script comments document which source file supplies each noise regime. Use these merged outputs as the primary source for the manuscript, after checking source-row consistency.

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

This is a guide to the key analysis files, not an exhaustive listing of every module or generated artifact. The final figure loader also reads the sources documented in `scripts/make_final_figures.py`.

## Limitations

1. **Simulation only:** noise is applied to stored output qubits, while encoding, decoding, and majority vote are noiseless. No hardware-level performance claim is made.
2. **Limited cohort:** 10 of the 29 ANPHY-Sleep subjects and one EEG channel (C3) were used.
3. **Limited seed coverage:** most sweeps use one seed; the primary bit-flip/phase-flip comparisons at \(p=0.05\) use three seeds.
4. **Downstream effects are small:** sleep-stage macro-F1 changes are generally modest; two of eight primary F1 comparisons survive the stated Holm correction.
5. **Primary family selection:** the eight-comparison family was chosen after an initial look at results and was not preregistered.
6. **No end-to-end noisy QEC operations:** the model excludes noisy encoding/decoding gates, syndrome extraction, ancilla preparation, and measurement errors.
7. **No validated classical baseline in the current manuscript:** a previous baseline produced unexplained results and is excluded rather than used to claim quantum superiority.
8. **No noise-aware training or error-mitigation baseline:** those are separate potential extensions.
9. **Allocation scope:** the negative allocation result applies to this specific 16-qubit rule and tested settings, not to every application-aware allocation method.

## Data and citation

The study uses ANPHY-Sleep. Please cite the dataset paper when describing the source data:

Wei et al., “ANPHY-Sleep: an Open Sleep Database from Healthy Adults Using High-Density Scalp Electroencephalogram,” *Scientific Data* 11, 896 (2024). [https://doi.org/10.1038/s41597-024-03722-1](https://doi.org/10.1038/s41597-024-03722-1)

The associated manuscript is in preparation. Please do not cite the manuscript as a published paper until publication details are available.

## License and data handling

Add an explicit repository license before inviting reuse. Do not commit raw EEG files, credentials, or other restricted data. Preserve the source CSVs and trained weights needed to reproduce the reported tables, while clearly distinguishing canonical files from backups and duplicate diagnostic outputs.
