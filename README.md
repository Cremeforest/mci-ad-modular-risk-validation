# Modular longitudinal risk prediction for MCI-to-AD conversion

**A summary-augmented modular longitudinal model for low-burden clinical risk prediction, developed on ADNI and externally evaluated on NACC.**

This repository presents a retrospective clinical-AI project for predicting 1-, 2-, 3-, and 5-year conversion from mild cognitive impairment (MCI) to Alzheimer's disease (AD) dementia using routine longitudinal clinical assessments.

The final model showed strong internal performance on ADNI and retained moderate external discrimination on NACC under a cross-cohort missing-feature setting, supporting the feasibility of low-burden longitudinal risk stratification when biomarkers or imaging are unavailable.

---

## Highlights

- **Clinical task:** multi-horizon MCI-to-AD dementia conversion prediction.
- **Inputs:** eight routine clinical variables, without PET, CSF, MRI, or APOE.
- **Model:** summary-augmented feature-level modular longitudinal model.
- **Internal development:** ADNI.
- **External evaluation:** NACC first-MCI external cohort under a no_ADAS13 setting.
- **Main result:** ADNI internal AUROC 0.904-0.926; NACC external AUROC 0.733-0.749.
- **Calibration analysis:** local recalibration improved long-horizon absolute risk calibration without retraining the prediction model.

---

## Main result

The model achieved strong internal discrimination on ADNI and retained stable external discrimination on NACC:

| Evaluation | Horizon range | AUROC range | Main interpretation |
|---|---:|---:|---|
| ADNI internal test | 1/2/3/5y | 0.904-0.926 | strong internal discrimination |
| NACC external evaluation | 1/2/3/5y | 0.733-0.749 | moderate external discrimination under no_ADAS13 |

These results suggest that routine longitudinal cognitive, global-severity, and functional assessments contain transferable risk-stratification signal for MCI-to-AD conversion, even when feature availability differs across cohorts.

---

## Why this matters

Many AD progression models rely on specialized biomarkers or imaging. This project focuses on a lower-burden setting: longitudinal clinical assessments that are more likely to be available in routine memory-clinic workflows, retrospective registries, or resource-limited clinical environments.

The external NACC evaluation provides a practical stress test: the ADNI-trained model was evaluated under a systematic feature mismatch where ADAS13 was unavailable. The model still retained moderate discrimination, while calibration analysis showed that absolute probabilities benefit from local recalibration.

A practical takeaway is:

> A portable longitudinal model can support risk stratification across cohorts, while a lightweight local calibration layer may be needed to align absolute risk estimates to a specific site or registry.

---

## Data and variables

Raw ADNI and NACC participant-level data are **not redistributed** in this repository.

The final model uses eight routine clinical variables:

| Variable | Meaning |
|---|---|
| `age_at_visit` | age at clinical visit |
| `sex_male` | sex indicator |
| `PTEDUCAT` | years of education |
| `MMSE` | Mini-Mental State Examination |
| `ADAS13` | Alzheimer's Disease Assessment Scale, 13-item cognitive score |
| `CDGLOBAL` | Clinical Dementia Rating global score |
| `CDRSB` | Clinical Dementia Rating Sum of Boxes |
| `FAQTOTAL` | Functional Activities Questionnaire total score |

For preprocessing details, see `docs/DATA_PROCESSING.md`.

---

## Method overview

The public workflow follows four stages:

```text
ADNI clinical tables
  -> longitudinal MCI visit-history construction
  -> train-only preprocessing and dynamic token construction
  -> summary-augmented modular longitudinal model
  -> frozen-model NACC external evaluation
```

The model uses one feature module per clinical variable, plus a visit-process component for timing and missingness. Each feature module combines temporal encoding of observed visit history with module-local trajectory summaries such as last value, mean, variability, change, slope-like trend, observed count, and observed rate.

---

## Internal ADNI evaluation

Internal full-module performance was evaluated on the ADNI test split with bootstrap 95% confidence intervals.

| Horizon | AUROC | AUPRC | Brier |
|---:|---:|---:|---:|
| 1y | 0.916 (0.880-0.949) | 0.412 (0.287-0.573) | 0.047 (0.036-0.058) |
| 2y | 0.904 (0.869-0.933) | 0.585 (0.483-0.718) | 0.085 (0.071-0.100) |
| 3y | 0.914 (0.880-0.943) | 0.822 (0.746-0.877) | 0.099 (0.081-0.118) |
| 5y | 0.926 (0.899-0.952) | 0.903 (0.860-0.940) | 0.108 (0.088-0.127) |

---

## NACC external evaluation

External evaluation used the frozen ADNI-trained model without retraining.

| Item | Setting |
|---|---|
| External cohort | NACC first-MCI external evaluation cohort |
| Analysis unit | final evaluable first-MCI landmark samples |
| Scenario | no_ADAS13 |
| Final evaluable N | 9,002 landmark samples |
| Retraining | none |

Here, external N refers to the final evaluable first-MCI landmark samples used for prediction and metric calculation, not the broader number of NACC participants with any MCI history during screening.

| Horizon | AUROC | 95% CI | AUPRC | 95% CI | Brier |
|---:|---:|---:|---:|---:|---:|
| 1y | 0.733 | 0.720-0.746 | 0.298 | 0.279-0.318 | 0.122 |
| 2y | 0.740 | 0.729-0.751 | 0.529 | 0.509-0.549 | 0.192 |
| 3y | 0.740 | 0.727-0.750 | 0.664 | 0.646-0.681 | 0.241 |
| 5y | 0.749 | 0.735-0.762 | 0.835 | 0.822-0.848 | 0.286 |

---

## Calibration analysis

The frozen model preserved useful risk ranking in NACC, but its raw long-horizon probabilities were systematically lower than the observed event rates. A lightweight local Platt recalibration layer improved Brier score without retraining the prediction model.

| Horizon | Observed event rate | Mean frozen predicted risk | Raw Brier | Local recalibrated Brier |
|---:|---:|---:|---:|---:|
| 3y | 0.425 | 0.277 | 0.241 | 0.203 |
| 5y | 0.643 | 0.405 | 0.286 | 0.191 |

This is a common and important issue in clinical prediction: discrimination can transfer across cohorts better than absolute probability calibration.

---

## Repository workflow

The numbered public scripts follow three core stages:

```text
scripts/01-03  ADNI data processing
scripts/04-06  internal model development and evaluation
scripts/07-13  NACC external evaluation
```

The Streamlit demo is kept separately under `app/streamlit_app.py`.

For the full script map, see `docs/SCRIPT_WORKFLOW.md`.

---

## Patient-style research demo

A lightweight Streamlit demo illustrates the intended interface style:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The demo accepts up to 8 visit records and displays 1/2/3/5-year research-demo risk estimates.

The public demo uses a lightweight proxy scoring function for interface illustration. It does not load the full private research checkpoint, raw data, feature tensors, calibration objects, or patient-level predictions.

---

## Repository structure

```text
app/                    Streamlit research demo
docs/                   data processing, workflow, claims, and result tables
scripts/                public data-processing, training, and validation workflow
README.md               project overview
MODEL_CARD.md           intended use and limitations
LICENSE                 MIT license
NOTICE.md               data/model artifact notice
requirements.txt        Python dependencies
```

---

## Availability and limitations

Raw ADNI/NACC data, participant-level feature tensors, model checkpoints, calibration objects, patient-level predictions, and the private paper draft are not redistributed. Reproducing the full pipeline requires authorized local access to ADNI/NACC.

This is a retrospective research project. It has not been prospectively validated and is not intended for clinical deployment. NACC external evaluation was conducted under a no_ADAS13 setting rather than a full-module setting.

---

## Project status

Manuscript-style write-up in preparation.
