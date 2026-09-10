# Application-Aware QEC for Noise-Resilient Generative Modeling of Biomedical Signals

Prototype pipeline: EDF -> EEG feature extraction -> small quantum generative
circuit -> ideal vs noisy vs QEC comparison.

## Setup

```bash
python -m venv qec_biosignal_env
source qec_biosignal_env/bin/activate   # Windows: qec_biosignal_env\Scripts\activate
pip install -r requirements.txt
```

Put your ANPHY-Sleep `.edf` file in `data/raw/`.

## Workflow (do these in order — don't skip ahead)

1. **Inspect the EDF** — confirms channel names, sampling rate, duration.
   ```bash
   python scripts/inspect_edf.py data/raw/YOUR_FILE.edf
   ```

2. **Run the baseline experiment** — preprocessing -> feature extraction ->
   ideal quantum circuit -> noisy quantum circuit -> distance metrics.
   ```bash
   python scripts/run_baseline_experiment.py data/raw/YOUR_FILE.edf --channel "EEG channel name"
   ```
   This produces `outputs/baseline_results.csv` and `outputs/baseline_plot.png`.

   No QEC is implemented yet — this step only establishes Experiment A (ideal)
   and Experiment B (noisy). QEC baselines (uniform / application-aware) come
   next, once this runs cleanly on your file.

## Structure

```
qec_biosignal/
├── requirements.txt
├── README.md
├── data/raw/                    <- put your .edf file(s) here
├── src/
│   ├── data_loading.py          EDF inspection + single-channel loading
│   ├── preprocessing.py         filtering + windowing
│   ├── features.py              8-dim biomedical feature vector per window
│   ├── quantum_model.py         small VQC generator, ideal + noisy devices
│   └── metrics.py               distribution-distance metrics (MSE, Wasserstein)
├── scripts/
│   ├── inspect_edf.py           CLI: dump channel/sfreq/duration info
│   └── run_baseline_experiment.py   CLI: run Experiment A + B end to end
└── outputs/                      results land here
```

## Design notes

- Only **one EEG channel** is used for the first prototype. Multi-channel
  comes later.
- The 1000 Hz raw signal is never fed directly into the quantum circuit —
  each window is reduced to an 8-dimensional feature vector
  (delta/theta/alpha/sigma/beta band power, variance, spectral entropy,
  autocorrelation) before quantum encoding.
- The quantum generator is a 4-qubit variational circuit (RY/RZ + CNOT ring),
  kept deliberately small so noise experiments run fast on a simulator.
- Noise is injected via PennyLane's `default.mixed` device with a
  depolarizing channel — swap in bit-flip/phase-flip/readout error the same
  way once this baseline is working.
- QEC (uniform + application-aware) is intentionally not implemented yet.
  Get Experiment A/B solid first — if the noiseless-vs-noisy gap isn't
  measurable and reproducible, nothing built on top of it will mean anything.
