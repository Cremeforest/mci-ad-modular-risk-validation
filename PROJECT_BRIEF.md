# Project brief

## One-sentence summary

I developed and externally validated a summary-augmented modular longitudinal model for predicting MCI-to-Alzheimer's disease conversion under incomplete clinical assessment.

## Motivation

Clinical dementia-risk prediction often faces irregular follow-up, incomplete assessments, and cohort shift. A model trained on one cohort may encounter a different set of available tests in another cohort.

## Technical idea

The final model uses feature-level clinical modules. Each module combines temporal information from observed visits with explicit longitudinal summaries, and the model fuses available modules using an availability-aware gating mechanism.

## External validation

The frozen ADNI-trained model was externally validated on a NACC first-MCI cohort under a no-ADAS13 scenario without retraining. It retained moderate discrimination across 1/2/3/5-year horizons. Raw risks were underestimated under cohort shift, and cross-fitted local recalibration improved calibration without retraining the prediction model.

## Positioning

This is a retrospective, auditable clinical-AI modeling project, not a deployment-ready medical device.
