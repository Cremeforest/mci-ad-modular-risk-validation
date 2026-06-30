# Modular longitudinal risk prediction for MCI-to-AD conversion

**ADNI internal development | NACC external validation | Incomplete clinical assessment | Research use only**

This repository presents a retrospective clinical-AI project for predicting conversion from mild cognitive impairment (MCI) to Alzheimer's disease (AD) dementia using longitudinal routine clinical assessments.

The final model is a **summary-augmented feature-level modular longitudinal model** designed for incomplete clinical assessment settings, where some tests may be unavailable across visits or cohorts.

> **Important:** This repository is for research demonstration only. It is not a clinical tool, not medical advice, and not intended for diagnosis, treatment decisions, triage, prognosis communication, or patient-level medical decision-making.

---

## Key contributions

1. **Leakage-aware longitudinal cohort construction**  
   MCI visit histories are converted into landmark-style longitudinal samples with 1/2/3/5-year conversion labels and observed-label masks.

2. **PROMISE-style dynamic token construction**  
   Each patient history is represented using value, missingness mask, time, delta, slope-like trend, and visit-mask components.

3. **Summary-augmented modular longitudinal model**  
   Each clinical feature is modeled as a module combining temporal encoding with module-local trajectory summaries.

4. **Missing-feature-aware external validation**  
   The frozen ADNI-trained model was externally validated on a NACC first-MCI cohort under a no_ADAS13 setting without retraining.

5. **Calibration-shift analysis**  
   Raw frozen probabilities underestimated long-horizon NACC risk; cross-fitted local Platt recalibration improved absolute risk calibration without retraining the prediction model.

---

## Project overview

The project predicts whether an MCI patient will convert to AD dementia within:

```text
1 year
2 years
3 years
5 years
```

The model uses routine clinical variables:

```text
age_at_visit
sex_male
PTEDUCAT
MMSE
ADAS13
CDGLOBAL
CDRSB
FAQTOTAL
```

These variables cover demographics, cognition, global severity, and functional assessment. Biomarkers such as PET, CSF, MRI, and APOE are intentionally not used in the final public-facing model description.

---

## Data processing

Raw ADNI and NACC participant-level data are **not** redistributed in this repository.

The public preprocessing scripts document the internal ADNI pipeline:

```text
scripts/01_tokenize_adni_primary_sequences.py
scripts/02_preprocess_adni_sequences_train_only.py
scripts/03_build_adni_promise_dynamic_tokens.py
```

The final internal token package contains:

```text
X_values_imputed_scaled
X_feature_mask
X_time_scaled
X_delta_from_first
X_slope_from_first
X_elapsed_from_first
X_visit_mask
y_labels
y_observed
sequence_lengths
train_idx
val_idx
test_idx
feature_names
```

See:

```text
docs/DATA_PROCESSING.md
```

for the preprocessing and token-construction description.

---

## Model design

The final model is a **summary-augmented feature-level modular longitudinal model**.

Each clinical feature has its own module:

```text
age_at_visit
sex_male
PTEDUCAT
MMSE
ADAS13
CDGLOBAL
CDRSB
FAQTOTAL
```

A separate visit-process component represents visit timing and missingness structure.

Each feature module combines:

```text
1. temporal encoding of observed visit history
2. module-local trajectory summaries
```

Module-local summaries include:

```text
last observed value
mean observed value
minimum / maximum
standard deviation
last-minus-first change
slope-like change
observed count
observed rate
```

This design allows the model to operate when selected clinical features are unavailable.

---

## Internal ADNI evaluation

Internal full-module performance was evaluated on the ADNI test split with bootstrap 95% confidence intervals.

| Horizon | AUROC | AUPRC | Brier |
|---:|---:|---:|---:|
| 1y | 0.916 (0.880-0.949) | 0.412 (0.287-0.573) | 0.047 (0.036-0.058) |
| 2y | 0.904 (0.869-0.933) | 0.585 (0.483-0.718) | 0.085 (0.071-0.100) |
| 3y | 0.914 (0.880-0.943) | 0.822 (0.746-0.877) | 0.099 (0.081-0.118) |
| 5y | 0.926 (0.899-0.952) | 0.903 (0.860-0.940) | 0.108 (0.088-0.127) |

These internal results should be interpreted together with the external validation below.

---

## NACC external validation

External validation was performed using:

```text
Model: frozen ADNI-trained final model
External cohort: NACC first-MCI cohort
Scenario: no_ADAS13
Sample size: 12,052 unique patients
Retraining: none
```

| Horizon | AUROC | 95% CI | AUPRC | 95% CI | Brier |
|---:|---:|---:|---:|---:|---:|
| 1y | 0.733 | 0.720-0.746 | 0.298 | 0.279-0.318 | 0.122 |
| 2y | 0.740 | 0.729-0.751 | 0.529 | 0.509-0.549 | 0.192 |
| 3y | 0.740 | 0.727-0.750 | 0.664 | 0.646-0.681 | 0.241 |
| 5y | 0.749 | 0.735-0.762 | 0.835 | 0.822-0.848 | 0.286 |

The frozen ADNI-trained model retained moderate external discrimination across horizons under the no_ADAS13 setting.

---

## Calibration under cohort shift

Raw frozen probabilities underestimated long-horizon absolute risk in NACC.

Examples:

```text
3-year observed event rate: 0.425
3-year mean predicted risk: 0.277

5-year observed event rate: 0.643
5-year mean predicted risk: 0.405
```

Cross-fitted NACC local Platt recalibration improved Brier score without retraining the prediction model:

```text
3-year Brier: 0.241 -> 0.203
5-year Brier: 0.286 -> 0.191
```

Interpretation:

> The model retained external risk-stratification ability, but absolute probabilities required local recalibration under cross-cohort shift.

---

## Public script workflow

The public scripts are organized as an end-to-end workflow.

### ADNI internal data processing

```text
01_tokenize_adni_primary_sequences.py
02_preprocess_adni_sequences_train_only.py
03_build_adni_promise_dynamic_tokens.py
```

### Internal model development and evaluation

```text
04_train_final_modular_model.py
05_internal_freeze_and_claims_audit.py
06_internal_bootstrap_calibration.py
```

### NACC external validation

```text
07_nacc_schema_audit.py
08_build_nacc_external_tokens.py
09_prepare_nacc_aligned_tokens.py
10_eval_frozen_model_on_nacc.py
11_external_bootstrap_sensitivity_calibration.py
12_nacc_local_recalibration.py
13_finalize_external_validation_report.py
```

### Patient-style research demo

```text
14_make_patient_demo.py
```

See:

```text
docs/SCRIPT_WORKFLOW.md
```

for a compact script map.

---

## Patient-style research demo

A small Streamlit demo illustrates the intended interface style:

```bash
streamlit run app/streamlit_app.py
```

The demo accepts up to 8 visit records and displays 1/2/3/5-year research-demo risk estimates.

The public demo is not the private frozen model output and is not intended for clinical use.

---

## Repository structure

```text
app/
  streamlit_app.py

docs/
  DATA_PROCESSING.md
  METHODS_AND_RESULTS.md
  CLAIMS_AND_LIMITATIONS.md
  SCRIPT_WORKFLOW.md
  STREAMLIT_DEMO.md
  tables/

scripts/
  01_tokenize_adni_primary_sequences.py
  02_preprocess_adni_sequences_train_only.py
  03_build_adni_promise_dynamic_tokens.py
  04_train_final_modular_model.py
  ...
  14_make_patient_demo.py

README.md
MODEL_CARD.md
NOTICE.md
requirements.txt
```

---

## Quickstart

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the public demo:

```bash
streamlit run app/streamlit_app.py
```

Reproducing the full ADNI/NACC pipeline requires authorized local access to the corresponding datasets.

---

## What is not included

This public repository does not include:

```text
raw ADNI data
raw NACC data
participant-level feature tensors
model checkpoints
patient-level predictions
private paper draft
```

This is intentional for data-access, privacy, and research-integrity reasons.

---

## Limitations

- Retrospective cohort modeling.
- No prospective validation.
- Not clinically deployable.
- NACC external validation was performed under a no_ADAS13 scenario, not full-module external validation.
- Raw frozen probabilities were not externally well calibrated without local recalibration.
- Module weights and masks should not be interpreted as causal feature importance.
- The Streamlit app is a public research demo, not the deployed research model.

---

## Project status

Manuscript-style write-up in preparation.

For now, this repository is intended as a public research portfolio and PhD outreach package.

---

## Notice

Copyright (c) 2026 Cremeforest. All rights reserved unless otherwise stated.

No open-source license is currently granted for reuse, modification, or redistribution of the code.
