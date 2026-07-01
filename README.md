# Low-burden modular longitudinal risk prediction for MCI-to-AD conversion

**A summary-augmented modular model using routine clinical assessments, developed on ADNI and externally evaluated on NACC.**

This repository presents an externally evaluated clinical-AI pipeline for predicting 1-, 2-, 3-, and 5-year conversion from mild cognitive impairment (MCI) to Alzheimer's disease (AD) dementia using routine longitudinal clinical assessments.

Developed on ADNI and evaluated on NACC, the final summary-augmented modular model achieved strong internal discrimination and retained moderate external discrimination, supporting low-burden risk stratification in routine memory-clinic workflows, retrospective registries, and resource-limited clinical environments where PET, CSF, MRI, or genetic biomarkers may be unavailable.

---

## Highlights

- **Task:** 1/2/3/5-year MCI-to-AD dementia conversion prediction.
- **Clinical setting:** low-burden longitudinal risk stratification using routine clinical assessments.
- **Longitudinal design:** leakage-aware MCI landmark construction with multi-horizon labels and observed-label masks.
- **Model:** summary-augmented feature-level modular longitudinal model.
- **Missingness handling:** train-split-only imputation with explicit missingness masks and availability-aware modular fusion.
- **External evaluation:** ADNI development and NACC first-MCI external evaluation.
- **Performance:** ADNI internal AUROC 0.904-0.926; NACC external AUROC 0.733-0.749.

---

## Results

The model achieved strong internal discrimination on ADNI and retained moderate external discrimination on NACC.

| Evaluation | Horizon range | AUROC range | Main interpretation |
|---|---:|---:|---|
| ADNI internal test | 1/2/3/5y | 0.904-0.926 | strong internal discrimination |
| NACC external evaluation | 1/2/3/5y | 0.733-0.749 | moderate external discrimination |

These results suggest that routine longitudinal cognitive, global-severity, and functional assessments contain transferable risk-stratification signal for MCI-to-AD conversion, even when feature availability differs across cohorts.

---

## Quick demo

A lightweight Streamlit demo illustrates the intended interface style:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The demo accepts up to 8 visit records and displays 1/2/3/5-year research-demo risk estimates.

The public demo uses a lightweight proxy scoring function for interface illustration. It does not load the full private research checkpoint, raw data, feature tensors, calibration objects, or patient-level predictions.

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
  -> train-split-only preprocessing and longitudinal token construction
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

The frozen ADNI-trained model was evaluated on the NACC first-MCI external cohort without retraining, with ADAS13 unavailable in the external feature set.
Detailed cohort construction and denominator definitions are documented in `docs/METHODS_AND_RESULTS.md`.

| Horizon | AUROC | 95% CI | AUPRC | 95% CI | Brier |
|---:|---:|---:|---:|---:|---:|
| 1y | 0.733 | 0.720-0.746 | 0.298 | 0.279-0.318 | 0.122 |
| 2y | 0.740 | 0.729-0.751 | 0.529 | 0.509-0.549 | 0.192 |
| 3y | 0.740 | 0.727-0.750 | 0.664 | 0.646-0.681 | 0.241 |
| 5y | 0.749 | 0.735-0.762 | 0.835 | 0.822-0.848 | 0.286 |

---

## Calibration analysis

External calibration was audited as a secondary analysis. A lightweight recalibration layer improved long-horizon probability estimates without retraining the prediction model.

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

## Repository structure

```text
app/                    Streamlit research demo
docs/                   data processing, workflow, claims, and result tables
scripts/                public data-processing, training, and validation workflow
README.md               project overview
MODEL_CARD.md           intended use and limitations
LICENSE                 MIT license
requirements.txt        Python dependencies
```

---

## Availability and limitations

Raw ADNI/NACC data, participant-level feature tensors, model checkpoints, calibration objects, and patient-level predictions are not redistributed. Full reproduction requires authorized local access to ADNI/NACC.

This is a retrospective research project. It has not been prospectively validated and is not intended for clinical deployment.

---

## Project status

Manuscript-style analysis and reporting are in preparation.
