# Public script workflow

The public scripts are organized as a readable end-to-end research pipeline.

## ADNI internal data processing

```text
01_tokenize_adni_primary_sequences.py
02_preprocess_adni_sequences_train_only.py
03_build_adni_promise_dynamic_tokens.py
```

## Internal model development and evaluation

```text
04_train_final_modular_model.py
05_internal_freeze_and_claims_audit.py
06_internal_bootstrap_calibration.py
```

## NACC external validation

```text
07_nacc_schema_audit.py
08_build_nacc_external_tokens.py
09_prepare_nacc_aligned_tokens.py
10_eval_frozen_model_on_nacc.py
11_external_bootstrap_sensitivity_calibration.py
12_nacc_local_recalibration.py
13_finalize_external_validation_report.py
```

## Patient-style research demo

```text
14_make_patient_demo.py
```

Raw data, generated feature tensors, model checkpoints, and patient-level predictions are intentionally not included in this public repository.
