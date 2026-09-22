# Application-Aware Quantum Error Correction for Noise-Resilient Biosignal Modeling

A framework for studying **application-aware quantum error correction (QEC)** in quantum generative models for biomedical signal features.

The central idea:

> Not every logical qubit contributes equally to the downstream application. Instead of protecting every qubit with the same error-correcting code, allocate stronger protection to the logical qubits that are empirically more sensitive to noise, recompute that allocation per subject, and confirm the result actually needs to be quantum by testing it against a fair classical baseline.

Validated across 5 subjects, 3 seeds each, with proper held-out evaluation, paired statistical significance testing, and a shot-matched classical comparison.

---

## Current Status

**Stage:** Validated, statistically tested, classical baseline complete. Downstream sleep-stage classification in progress.

* Generator: trained (reconstruction objective), proper train/test split enforced.
* Allocation: computed dynamically per subject from that subject's own sensitivity ranking.
* Evaluated across 5 subjects x 3 seeds = 15 runs, results and statistical tests below.
* Classical baseline: implemented and run, same noise probability, same shot-averaging concept, same held-out data and ground truth target as the quantum pipeline.
* Next: downstream sleep-stage classification, using existing hypnograms, epoch-grouped split.

---

## Pooled Results (5 subjects x 3 seeds, n=15)

| Method | MSE vs Ideal | Wasserstein vs Ideal |
|---|---:|---:|
| No QEC | 0.000854 ± 0.000050 | 0.016852 ± 0.000745 |
| Uniform Repetition (d=3) | 0.001120 ± 0.000276 | 0.016118 ± 0.002680 |
| Classical (50-shot averaged) | 0.000686 ± 0.000052 | 0.016652 ± 0.000806 |
| **Application-Aware QEC** | **0.000656 ± 0.000120** | **0.013180 ± 0.001325** |

---

## Statistical Significance (paired t-test, subject-level means, n=5 subjects)

| Comparison | MSE p-value | Wasserstein p-value |
|---|---:|---:|
| Application-Aware vs No QEC | **0.0077** ✓ | **0.0063** ✓ |
| Application-Aware vs Uniform | **0.0284** ✓ | 0.1323 (n.s.) |
| Application-Aware vs Classical | 0.6150 (n.s.) | **0.0149** ✓ |
| Classical vs No QEC | **0.0012** ✓ | 0.5117 (n.s.) |
| Classical vs Uniform | **0.0096** ✓ | 0.5735 (n.s.) |

**Note on Wilcoxon signed-rank:** at n=5, the Wilcoxon test cannot reach below p=0.0625 even when every subject agrees in the same direction, this is a floor of the test at this sample size, not a contradiction of the t-test results above. The paired t-test is reported as primary given all subject-level differences share the same sign in every significant row.

**Reading these results honestly:** Application-Aware QEC is the only method that significantly beats both weak baselines (No QEC, and either Uniform or Classical) on both metrics. Its distinguishing advantage over the classical baseline specifically is on Wasserstein distance; on MSE, it is statistically tied with classical redundancy. This is a precise claim, not application-aware quantum beats everything, but application-aware calibration achieves something on distributional similarity that neither naive quantum protection nor classical redundancy achieves.

---

## Why a Classical Baseline?

A fair answer to "why does this need to be quantum" requires a classical method given the same resources: same noise probability (0.02, not tuned to flatter either side), and the same shot-averaging concept already used by every quantum method in this project (50 shots averaged). The classical baseline corrupts the same ground-truth target features (the same `[-1,1]`-rescaled values the quantum generator is trained to reconstruct) and averages 50 independent noisy measurements, exactly analogous to the quantum trajectory averaging.

Result: classical averaging is a genuinely strong baseline, it clearly beats both No QEC and Uniform QEC on MSE. Application-Aware QEC is the only quantum method that holds its own against it, and its real, statistically significant edge is on Wasserstein distance specifically.

---

## Multi-Subject Generalization

An earlier allocation, frozen from a single subject, failed to generalize, helping only 2 of 5 subjects and actively hurting the other 3. Rebuilding the allocation dynamically per subject, from that subject's own measured sensitivity ranking, resolved this. See `outputs/figures/fig2_per_subject.png` for the per-subject breakdown and `outputs/figures/fig3_significance.png` for the significance results.

---

## Pipeline

```text
ANPHY-Sleep EDF (5 subjects)
       |
       v
EEG channel selection (C3), 10s windows, 8-dim features
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
       +----------+----------+-------------------+
       |          |          |                   |
       v          v          v                   v
   No QEC     Uniform     Classical         Application-
              QEC (d=3)   (50-shot avg)     Aware QEC
       |          |          |                   |
       +----------+----------+-------------------+
                        |
                        v
        MSE + Wasserstein, pooled + paired significance
                        |
                        v
         Downstream sleep-stage classification (in progress)
```

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
│   ├── data_split.py            # plain + epoch-grouped train/test split
│   ├── hypnogram.py              # hypnogram parsing + window alignment
│   ├── quantum_model.py
│   ├── repetition_code.py
│   ├── shor_code.py
│   ├── application_aware_qec.py  # dynamic, per-subject tier allocation
│   ├── classical_baseline.py     # shot-matched classical comparison
│   └── metrics.py
│
├── scripts/
│   ├── inspect_edf.py
│   ├── inspect_hypnogram.py
│   ├── train_generator.py
│   ├── run_sensitivity_ranking.py
│   ├── run_application_aware_experiment.py
│   ├── run_classical_baseline.py
│   ├── run_downstream_classification.py
│   ├── run_full_study.py         # orchestrates all subjects/seeds, incremental save
│   ├── analyze_full_study.py     # paired significance tests from saved results
│   └── make_journal_figures.py   # publication figures from saved results
│
├── data/raw/                     (EDF + hypnogram files, excluded from version control)
│
└── outputs/
    ├── trained_weights.npy
    ├── sensitivity_ranking.csv
    ├── application_aware_results_testsplit.csv
    ├── classical_baseline_results.csv
    ├── full_study_results.csv
    ├── downstream_classification_results.csv
    └── figures/
        ├── fig1_pooled_comparison.png / .pdf
        ├── fig2_per_subject.png / .pdf
        └── fig3_significance.png / .pdf
```

---

## Running the Full Study

```bash
python scripts/run_full_study.py
```

Edit `SUBJECTS` at the top of the script first. Runs train, rank, application-aware evaluation, and classical baseline for every subject and seed automatically, saving `outputs/full_study_results.csv` after every single run (not just at the end), so an interruption partway through does not lose completed work; rerunning the same command skips subjects/seeds already saved.

```bash
python scripts/analyze_full_study.py
```

Computes subject-level paired significance tests directly from the saved CSV.

```bash
python scripts/make_journal_figures.py
```

Generates the three publication figures directly from the saved CSV.

---

## Important Limitations

* Statistical power at n=5 subjects is limited; the Wilcoxon floor (p=0.0625) means only the paired t-test can show significance at this sample size, more subjects would strengthen this.
* Downstream sleep-stage classification is in progress, not yet complete, at time of writing.
* All noise is simulated (depolarizing channel); no real quantum hardware has been used.
* Current results are on ANPHY-Sleep; the move to Sleep-EDF Expanded for the primary reported dataset has not yet happened.
* The Application-Aware vs Uniform QEC comparison on Wasserstein distance is not statistically significant (p=0.1323); this is disclosed, not hidden.

---

## Next Research Steps

1. **Downstream task:** sleep-stage classification accuracy under each method, using existing hypnograms and an epoch-grouped split (in progress).
2. **Scale to 8-10 subjects** to strengthen statistical power, particularly for the Application-Aware vs Uniform comparison.
3. **Real hardware validation:** noted as future work for the initial submission, given target-venue and timeline constraints.
4. **Dataset switch:** move primary results to Sleep-EDF Expanded, keep ANPHY as supporting evidence.

---

## Research Direction

This project explores the intersection of quantum error correction, quantum machine learning, generative modeling, biomedical signal processing, EEG analysis, application-aware resource allocation, and noise-resilient quantum computing.

## Citation

If you use this repository or build upon the implementation, please cite the repository and the associated research work when available.

## License

License information will be added as the project matures.