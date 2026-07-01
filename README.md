# Modular longitudinal risk prediction for MCI-to-AD conversion

**An externally evaluated modular longitudinal model for low-burden MCI-to-AD risk prediction using routine clinical assessments.**

This repository presents a retrospective clinical-AI pipeline for predicting 1-, 2-, 3-, and 5-year conversion from mild cognitive impairment (MCI) to Alzheimer's disease (AD) dementia.

Developed on ADNI and externally evaluated on NACC, the final summary-augmented modular model achieved strong internal discrimination and retained moderate external discrimination under cross-cohort feature mismatch.

---

## Highlights

- **Task:** 1/2/3/5-year MCI-to-AD dementia conversion prediction.
- **Inputs:** eight routine clinical variables; no PET, CSF, MRI, or APOE.
- **Model:** summary-augmented feature-level modular longitudinal model.
- **Development cohort:** ADNI.
- **External evaluation:** NACC first-MCI cohort with ADAS13 unavailable.
- **Performance:** ADNI internal AUROC 0.904-0.926; NACC external AUROC 0.733-0.749.
- **Calibration:** local recalibration improved long-horizon probability calibration without retraining the prediction model.

---

## Results

The model achieved strong internal discrimination on ADNI and retained moderate external discrimination on NACC.

| Evaluation | Horizon range | AUROC range | Main interpretation |
|---|---:|---:|---|
| ADNI internal test | 1/2/3/5y | 0.904-0.926 | strong internal discrimination |
| NACC external evaluation | 1/2/3/5y | 0.733-0.749 | moderate external discrimination with ADAS13 unavailable |

These results suggest that routine longitudinal cognitive, global-severity, and functional assessments contain transferable risk-stratification signal for MCI-to-AD conversion, even when feature availability differs across cohorts.

---

## Data and variables

Raw ADNI and NACC participant-level data are not redistributed.

The model uses eight routine clinical variables: age at visit, sex, education, MMSE, ADAS13, Clinical Dementia Rating global score, Clinical Dementia Rating Sum of Boxes, and Functional Activities Questionnaire total score.

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

## ADNI internal evaluation

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
| External feature setting | ADAS13 unavailable |
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

The frozen model retained useful risk ranking in NACC, while its raw long-horizon probabilities were conservative. A lightweight local Platt recalibration layer improved probability calibration without retraining the prediction model.

Detailed calibration tables are provided in `docs/tables/`.

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

Raw ADNI/NACC data, participant-level feature tensors, model checkpoints, calibration objects, and patient-level predictions are not redistributed. Full reproduction requires authorized local access to ADNI/NACC.

This is a retrospective research project. It has not been prospectively validated and is not intended for clinical deployment. NACC external evaluation was conducted with ADAS13 unavailable rather than in a full-feature external setting.

---

## Project status

Manuscript-style write-up in preparation.
