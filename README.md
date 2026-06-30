# Modular longitudinal risk prediction for MCI-to-AD conversion

**Routine clinical assessments | ADNI development | NACC external validation | Research use only**

This repository presents a retrospective clinical-AI project for predicting conversion from mild cognitive impairment (MCI) to Alzheimer's disease (AD) dementia using low-burden longitudinal clinical assessments.

The final model is a **summary-augmented feature-level modular longitudinal model** designed for settings where biomarkers or imaging may be unavailable, such as routine memory-clinic assessment, retrospective registry analysis, or resource-limited clinical environments.

> **Research-use notice:** This repository is not a clinical tool, not medical advice, and not intended for diagnosis, treatment decisions, triage, prognosis communication, or patient-level medical decision-making.

---

## Main contribution

This project builds an end-to-end MCI-to-AD risk prediction workflow:

1. **Longitudinal cohort construction** from MCI visit histories with 1/2/3/5-year conversion labels and observed-label masks.
2. **Dynamic clinical token construction** using values, missingness masks, timing, longitudinal change, and visit masks.
3. **Summary-augmented modular modeling**, where each clinical variable is handled by a feature-level module.
4. **NACC external validation** of a frozen ADNI-trained model under a realistic no_ADAS13 missing-feature setting.
5. **Calibration-shift analysis**, showing that external risk ranking transferred better than raw absolute probabilities.

---

## Main finding

The frozen ADNI-trained model retained **moderate external discrimination** on NACC, but raw absolute risks were underestimated under cross-cohort shift.

> Routine cognitive, global-severity, and functional assessments can support cross-cohort **risk ranking**, but absolute risk probabilities are not automatically transportable across cohorts.

This supports a future clinical-AI design pattern: a portable risk-ranking model plus a site- or cohort-specific calibration layer.

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

The final model focuses on routinely available clinical assessments and does not use PET, CSF, MRI, or APOE inputs.

For preprocessing details, see `docs/DATA_PROCESSING.md`.

---

## Method overview

The public workflow follows four stages:

```text
ADNI clinical tables
  -> longitudinal MCI visit-history construction
  -> train-only preprocessing and dynamic token construction
  -> summary-augmented modular longitudinal model
  -> frozen-model NACC external validation
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

These internal results should be interpreted together with the external validation below.

---

## NACC external validation

External validation used the frozen ADNI-trained model without retraining.

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

## Calibration under cohort shift

Raw frozen probabilities underestimated long-horizon absolute risk in NACC.

| Horizon | Observed event rate | Mean frozen predicted risk | Raw Brier | Cross-fitted local recalibrated Brier |
|---:|---:|---:|---:|---:|
| 3y | 0.425 | 0.277 | 0.241 | 0.203 |
| 5y | 0.643 | 0.405 | 0.286 | 0.191 |

Cross-fitted local Platt recalibration improved calibration without retraining the prediction model.

---

## Repository workflow

The numbered public scripts follow three core stages:

```text
scripts/01-03  ADNI data processing
scripts/04-06  internal model development and evaluation
scripts/07-13  NACC external validation
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

The public demo uses a lightweight proxy scoring function for interface illustration. It does **not** load the full private research checkpoint, raw data, feature tensors, calibration objects, or patient-level predictions.

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

## What is not included

This public repository intentionally excludes raw ADNI/NACC data, participant-level feature tensors, model checkpoints, calibration objects, patient-level predictions, and the private paper draft.

Reproducing the full ADNI/NACC pipeline requires authorized local access to the corresponding datasets.

---

## Limitations

- Retrospective cohort modeling.
- No prospective validation.
- Not clinically deployable.
- NACC external validation was no_ADAS13, not full-module external validation.
- Raw frozen probabilities required local recalibration under external cohort shift.
- Module weights and masks should not be interpreted as causal feature importance.
- The Streamlit app is a public proxy demo, not the deployed research model.

---

## Project status

Manuscript-style write-up in preparation.

This repository is currently maintained as a public research portfolio and PhD outreach package.
