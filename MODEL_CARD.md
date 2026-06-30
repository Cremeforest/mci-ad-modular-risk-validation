# Model card: Step39 availability-aware modular GRU gated-fusion model

## Model overview

- **Model name:** Step39 availability-aware modular longitudinal risk model
- **Checkpoint:** `results/models/39_modular_availability_aware_model/best_model.pt`
- **Script version:** `v1_modular_availability_aware_gru_gated_fusion`
- **Model family:** module-specific GRU temporal encoders with availability-aware gated fusion
- **Prediction task:** MCI-to-AD dementia conversion risk
- **Prediction horizons:** 1, 2, 3, and 5 years
- **Status:** retrospective research prototype; not clinically validated

## Intended research use

This model is intended for retrospective research on longitudinal clinical risk prediction, missingness-aware modular fusion, and cross-cohort feature mismatch analysis. It can support project demonstration, methods discussion, and internal audit. It is not intended for patient-level clinical decision-making.

## Out-of-scope use

- Diagnosis, treatment, triage, prognosis communication, or patient-level medical decision-making
- Clinical deployment or patient-facing risk communication
- Claiming final external validation or prospective clinical utility
- Claiming causal feature or module importance from gate weights

## Inputs

The locked Step39 model uses longitudinal clinical histories built from current-and-past visits only.

| Module | Variables / components |
|---|---|
| demographics | age_at_visit, sex_male, PTEDUCAT |
| cognition | MMSE, ADAS13 |
| function | FAQTOTAL |
| global_severity | CDGLOBAL, CDRSB |
| timing_missingness | scaled time features plus all feature masks |

Clinical modules include scaled values, feature masks, delta-from-first, and slope-like components. The timing/missingness module concatenates scaled time features and all feature masks.

## Outputs

The model outputs exploratory risk scores for conversion from MCI to AD dementia at 1-, 2-, 3-, and 5-year horizons.

## Training and evaluation data boundary

The model was developed on ADNI-derived longitudinal tensors. Raw ADNI and NACC participant-level data are access-controlled and are not redistributed. NACC results are reported as a strict-load direct external dry-run using ADNI train-only preprocessing, not as definitive external validation.

## Performance summary

| horizon_year | ADNI_AUROC | ADNI_AUPRC | ADNI_Brier | NACC_AUROC | NACC_AUPRC | NACC_Brier |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | 0.9107 | 0.4195 | 0.0481 | 0.6865 | 0.1103 | 0.0509 |
| 2.0000 | 0.8943 | 0.6095 | 0.0863 | 0.7169 | 0.4419 | 0.1582 |
| 3.0000 | 0.9047 | 0.8154 | 0.1032 | 0.7267 | 0.6405 | 0.2083 |
| 5.0000 | 0.8996 | 0.8884 | 0.1300 | 0.7227 | 0.8146 | 0.2301 |

## External feature mismatch

| feature | non_missing_rate |
| --- | --- |
| age_at_visit | 1.0000 |
| sex_male | 0.9999 |
| PTEDUCAT | 0.9966 |
| MMSE | 0.5458 |
| ADAS13 | 0.0000 |
| CDGLOBAL | 1.0000 |
| CDRSB | 1.0000 |
| FAQTOTAL | 0.7009 |

### Module availability

| module | available_fraction |
| --- | --- |
| demographics | 1.0000 |
| cognition | 0.6102 |
| function | 0.7954 |
| global_severity | 1.0000 |
| timing_missingness | 1.0000 |

## Gate-weight behavior

Gate weights are model-behavior summaries and should not be interpreted as causal importance. External gate re-allocation was observed under NACC feature mismatch.

| module | ADNI_internal_test_mean_gate_weight | NACC_external_mean_gate_weight | external_minus_internal_gate_weight | NACC_external_mean_availability |
| --- | --- | --- | --- | --- |
| cognition | 0.3465 | 0.2052 | -0.1413 | 0.6102 |
| demographics | 0.1118 | 0.1498 | 0.0380 | 1.0000 |
| function | 0.2187 | 0.2095 | -0.0093 | 0.7954 |
| global_severity | 0.1622 | 0.2263 | 0.0641 | 1.0000 |
| timing_missingness | 0.1607 | 0.2093 | 0.0486 | 1.0000 |

## Calibration and robustness

Internal calibration and patient-cluster bootstrap robustness artifacts are available from Step40/Step73. These support internal audit, not clinical deployment readiness.

| horizon_year | metric | mean | ci_lower_2.5 | ci_upper_97.5 | n_bootstrap_valid |
| --- | --- | --- | --- | --- | --- |
| 1 | AUROC | 0.9106 | 0.8668 | 0.9492 | 1000 |
| 2 | AUROC | 0.8936 | 0.8463 | 0.9339 | 1000 |
| 3 | AUROC | 0.9041 | 0.8447 | 0.9481 | 1000 |
| 5 | AUROC | 0.8986 | 0.8298 | 0.9523 | 1000 |

## Ethical and clinical safety boundary

This model has not undergone prospective validation, regulatory review, clinical workflow integration testing, or decision-curve based utility evaluation. It should remain a research prototype.

## Known limitations

- NACC lacks ADAS13, and MMSE is only partially observed.
- NACC timing features were reconstructed from a common-feature cohort file rather than native Step39 time tensors.
- External analysis is a direct dry-run, not final locked external validation.
- Calibration is internal; external calibration and clinical utility are not established.
- Gate weights summarize learned fusion behavior, not causal or clinical importance.

