# Privacy-Utility-Fairness Stress Testing for Longitudinal Synthetic EHR and ICU Waveforms

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mtechbro94/synthetic-ehr-benchmark/blob/main/notebooks/Synthetic_EHR_Benchmark_Walkthrough.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Smoke Test](https://github.com/mtechbro94/synthetic-ehr-benchmark/actions/workflows/smoke_test.yml/badge.svg)](https://github.com/mtechbro94/synthetic-ehr-benchmark/actions)
[![Paper Manuscript](https://img.shields.io/badge/Paper-IEEE%20JBHI%20%2F%20JAMIA-purple.svg)](docs/paper.md)
[![Overleaf Package](https://img.shields.io/badge/LaTeX-Overleaf%20Package-teal.svg)](paper_latex/)
[![Privacy Attacks](https://img.shields.io/badge/Privacy-Membership%20%2B%20Attribute%20Inference-orange.svg)](#privacy-stress-testing)
[![Utility Metric](https://img.shields.io/badge/Utility-TSTR%20%2F%20TRTS-green.svg)](#clinical-utility-evaluation)
[![Dataset](https://img.shields.io/badge/Cohort-MIMIC--IV-informational.svg)](https://physionet.org/content/mimiciv/)

A rigorous, modular empirical benchmarking framework for stress-testing **synthetic longitudinal Electronic Health Records (EHR)** and **ICU physiological waveform generators** along three non-negotiable clinical trust axes: **Privacy**, **Utility**, and **Fairness**.

---

## 📌 Overview

Generative models in healthcare (e.g., GANs, diffusion models, and longitudinal transformers) often claim high fidelity and differential privacy guarantees. However, downstream clinical deployment requires independent adversarial stress testing across multiple simultaneous dimensions:

```
                          ┌──────────────────────────┐
                          │   Synthetic EHR / ICU    │
                          │   Waveform Generator     │
                          └─────────────┬────────────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
│     1. PRIVACY        │   │     2. UTILITY        │   │     3. FAIRNESS       │
│ • Shadow-Model MIA    │   │ • TSTR vs. TRTS       │   │ • Subgroup Util Gap   │
│ • Attribute Inference │   │ • Discriminative Dist │   │ • Subgroup MIA Risk   │
│ • Threshold Attacks   │   │ • Temporal Probes     │   │ • Disparity Ratios    │
└───────────────────────┘   └───────────────────────┘   └───────────────────────┘
```

1. **Privacy Stress Testing**: Does the synthetic data leak patient membership or expose sensitive clinical attributes?
2. **Clinical Utility & Temporal Realism**: Can diagnostic models trained on synthetic data perform reliably on real patients (TSTR), and does the generator preserve longitudinal physiology without transition drift?
3. **Algorithmic Fairness**: Are utility and privacy preserved equally across demographic subgroups, or does the generator amplify disparities?

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies in Python 3.10+:

```bash
git clone https://github.com/mtechbro94/synthetic-ehr-benchmark.git
cd synthetic-ehr-benchmark
pip install -r requirements.txt
```

### 2. 1-Click Interactive Google Colab Walkthrough

Run the full interactive benchmark tutorial directly in your browser with pre-rendered visualization plots:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mtechbro94/synthetic-ehr-benchmark/blob/main/notebooks/Synthetic_EHR_Benchmark_Walkthrough.ipynb)

### 3. Run CI Smoke Test Locally

Run the complete pipeline end-to-end on synthetic ICU cohorts:

```bash
python scripts/smoke_test.py
```

The smoke test exercises:
- Cohort ingestion and splitting (`src.data`)
- Shadow-model membership inference attacks (`src.privacy`)
- Attribute disclosure attacks (`src.privacy`)
- TSTR / TRTS downstream utility pipelines (`src.utility`)
- Autocorrelation and transition drift probes (`src.temporal`)
- Demographic subgroup fairness analysis (`src.fairness`)

---

## 🔬 Core Evaluation Modules

### 1. Cohort Construction (`src.data`)
Extracts standardized cohorts from MIMIC-IV clinical databases and formats 3D temporal tensors `(patients × timesteps × vitals/labs)`:

```python
from src.data import CohortConfig, build_cohorts, load_tables

# Define cohort configuration
config = CohortConfig(
    min_age=18,
    obs_window_hours=24,
    gap_hours=6,
    pred_window_hours=24,
)

# Build cohorts with clinical stratification
cohorts = build_cohorts(config, data_dir="data/mimiciv")
train_cohort, test_cohort = cohorts["train"], cohorts["test"]
```

### 2. Privacy Stress Testing (`src.privacy`)
Simulates adversarial attacks against the generator to assess patient re-identification and attribute leakage:

```python
from src.privacy import (
    make_membership_splits,
    shadow_model_attack,
    attribute_inference_attack,
    privacy_report,
)

# Membership Inference: Shadow-model attack
members, non_members, shadow_m, shadow_nm = make_membership_splits(real_records)
mia_results = shadow_model_attack(members, non_members, shadow_m, shadow_nm, [synthetic_records])
print(f"Shadow Model Membership AUROC: {mia_results['auroc']:.3f}")

# Attribute Inference: Predicting sensitive demographics from clinical vitals
attr_results = attribute_inference_attack(
    df_synth=synth_df,
    df_real=real_df,
    sensitive=["gender", "ethnicity"],
    predictors=["age", "heart_rate", "sbp", "glucose"],
)
```

### 3. Clinical Utility & Temporal Probes (`src.utility`, `src.temporal`)
Quantifies machine learning utility through **Train on Synthetic, Test on Real (TSTR)** compared against **Train on Real, Test on Synthetic (TRTS)**:

```python
from src.utility import tstr, trts, utility_report
from src.temporal import temporal_report

# Downstream clinical task evaluation (e.g., ICU deterioration / mortality)
tstr_scores = tstr((X_real_train, y_real_train), (X_real_test, y_real_test), (X_synth, y_synth))
trts_scores = trts((X_real_train, y_real_train), (X_real_test, y_real_test), (X_synth, y_synth))

print("TSTR AUROC:", tstr_scores["logistic"]["auroc"])
print("TRTS AUROC:", trts_scores["logistic"]["auroc"])

# Temporal realism: autocorrelation decay, multi-step transition drift
temp_metrics = temporal_report(real_3d_waveforms, synth_3d_waveforms)
print("Transition Drift:", temp_metrics["transition_drift"])
print("Autocorrelation Error:", temp_metrics["autocorr_mae"])
```

### 4. Algorithmic Fairness (`src.fairness`)
Exposes differential utility and privacy leaks across protected demographic groups:

```python
from src.fairness import fairness_report

# Audit demographic parity in utility and privacy retention
fairness = fairness_report(
    real_data={"features": real_3d, "gender": demographics_real},
    synth_data={"features": synth_3d, "gender": demographics_synth},
    subgroup="gender",
)

print("Subgroup Utility Gap:", fairness["subgroup_utility_gap"])
print("Subgroup Privacy Risk Disparity:", fairness["subgroup_privacy_disparity"])
```

---

## 📊 Benchmark Results Summary

| Generator / Setup | Membership AUROC ↓ | Attribute Acc ↓ | TSTR (Utility) ↑ | TRTS ↑ | Temporal Drift ↓ | Fairness Gap ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ideal Target** | **~0.50** (Random) | **~Chance** | **~TRTR Baseline** | **High** | **< 0.05** | **< 0.03** |
| *Toy Smoke Baseline* | `0.50` | `0.33` | `0.52` | `0.51` | `0.03` | `0.02` |
| *GAN-based Synthesizer* | *evaluating* | *evaluating* | *evaluating* | *evaluating* | *evaluating* | *evaluating* |
| *Diffusion Synthesizer* | *evaluating* | *evaluating* | *evaluating* | *evaluating* | *evaluating* | *evaluating* |
| *Longitudinal Transformer* | *evaluating* | *evaluating* | *evaluating* | *evaluating* | *evaluating* | *evaluating* |

> **Metric Interpretation**:
> - **Lower is better (↓)** for Membership AUROC (0.50 indicates no member re-identification) and Attribute Accuracy.
> - **Higher is better (↑)** for TSTR/TRTS utility scores (approaching real-data baseline).
> - **Lower is better (↓)** for Temporal Drift and Fairness Gaps (near zero indicates unbiased, realistic generation).

---

## 📁 Repository Structure

```text
synthetic-ehr-benchmark/
├── .github/
│   └── workflows/
│       └── smoke_test.yml          # Automated CI pipeline
├── docs/
│   └── paper.md                    # Full Q1 journal research manuscript
├── notebooks/
│   └── Synthetic_EHR_Benchmark_Walkthrough.ipynb  # 1-Click interactive Colab notebook
├── paper_latex/                    # Standalone Overleaf / LaTeX publication package
│   ├── main.tex                    # IEEEtran two-column journal manuscript
│   ├── references.bib              # Complete BibTeX bibliography
│   └── IEEEtran.cls                # Official IEEE class file
├── scripts/
│   ├── build_notebook.py           # Notebook compilation script
│   └── smoke_test.py               # End-to-end toy pipeline validation
├── src/
│   ├── data/                       # MIMIC-IV loaders & cohort construction
│   │   ├── cohort.py
│   │   └── loaders.py
│   ├── privacy/                    # Membership & attribute inference attack suites
│   │   ├── attribute_inference.py
│   │   ├── membership_inference.py
│   │   └── metrics.py
│   ├── utility/                    # TSTR / TRTS clinical evaluation pipeline
│   │   ├── downstream.py
│   │   ├── evaluation.py
│   │   └── metrics.py
│   ├── temporal/                   # Autocorrelation & Markovian transition probes
│   │   ├── probes.py
│   │   └── metrics.py
│   ├── fairness/                   # Subgroup disparity & parity stress tests
│   │   ├── stress.py
│   │   └── metrics.py
│   └── generators/                 # Synthetic generator adapters & interfaces
├── requirements.txt                # Core benchmark dependencies
├── README.md                       # Documentation & API usage guide
└── LICENSE                         # MIT License
```

---

## 📖 Citation

If you use this benchmark in your research or clinical generative AI evaluations, please cite:

```bibtex
@misc{mir2026syntheticehr,
  title={Privacy-Utility-Fairness Stress Testing for Longitudinal Synthetic EHR and ICU Waveform Generation},
  author={Mir, Aaqib Rashid},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/mtechbro94/synthetic-ehr-benchmark}}
}
```

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
