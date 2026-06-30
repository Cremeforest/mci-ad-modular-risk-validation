# Availability-aware modular longitudinal risk prediction for MCI-to-AD conversion

**Research prototype | Longitudinal clinical prediction | Modular missingness-aware fusion | Not clinically validated**

This repository develops a leakage-aware longitudinal model for predicting conversion from mild cognitive impairment (MCI) to Alzheimer's disease dementia over 1-, 2-, 3-, and 5-year horizons. The current primary model is the locked **Step39 availability-aware modular GRU gated-fusion model**.

> **Important.** This is a retrospective research prototype. It is not a clinical calculator and must not be used for diagnosis, treatment, triage, prognosis communication, or patient-level medical decision-making. NACC results are reported as a direct external dry-run, not definitive locked external validation.

## Overview

The model represents each patient's current-and-past visit history as longitudinal clinical tokens. Instead of requiring all cohorts to share a complete feature set, the model separates input history into clinical modules and fuses available modules with a learned gate.

**Modules**

- `demographics`: age at visit, sex, education
- `cognition`: MMSE, ADAS13
- `function`: FAQTOTAL
- `global_severity`: CDGLOBAL, CDRSB
- `timing_missingness`: scaled time features and all feature masks

Clinical modules use scaled values, feature masks, delta-from-first, and slope-like components. Module-specific GRU encoders are combined by availability-aware gated fusion.

## Mainline evidence chain

```text
Step39 locked modular GRU gated-fusion model
→ Step40 internal calibration/robustness audit
→ Step70c strict-load NACC direct external dry-run
→ Step70d internal/external and gate-shift synthesis
→ Step71 claims audit
→ Step74 compact README
```

## Model checkpoint

- Checkpoint: `results/models/39_modular_availability_aware_model/best_model.pt`
- Script version: `v1_modular_availability_aware_gru_gated_fusion`
- Horizons: `[1, 2, 3, 5]`
- Architecture: module-specific GRU temporal encoders with availability-aware gated fusion
- The Step39 primary model is **not** a Transformer

## Results summary

### ADNI internal test and NACC direct external dry-run

| Horizon | ADNI AUROC | ADNI AUPRC | ADNI Brier | NACC dry-run AUROC | NACC dry-run AUPRC | NACC dry-run Brier |
|---:|---:|---:|---:|---:|---:|---:|
| 1y | 0.911 | 0.420 | 0.048 | 0.687 | 0.110 | 0.051 |
| 2y | 0.894 | 0.610 | 0.086 | 0.717 | 0.442 | 0.158 |
| 3y | 0.905 | 0.815 | 0.103 | 0.727 | 0.641 | 0.208 |
| 5y | 0.900 | 0.888 | 0.130 | 0.723 | 0.815 | 0.230 |

The model shows strong ADNI internal discrimination and moderate NACC direct dry-run discrimination. The external drop is expected because NACC has systematic feature mismatch and reconstructed timing inputs.

### External feature mismatch

- ADAS13 token non-missing rate: `0.0000`
- MMSE token non-missing rate: `0.5458`
- FAQTOTAL token non-missing rate: `0.7009`
- Cognition module landmark availability: `0.6102`
- Function module landmark availability: `0.7954`

### Gate re-allocation under external mismatch

| Module | ADNI mean gate | NACC mean gate | Shift | NACC availability |
|---|---:|---:|---:|---:|
| cognition | 0.347 | 0.205 | -0.141 | 0.610 |
| demographics | 0.112 | 0.150 | 0.038 | 1.000 |
| function | 0.219 | 0.209 | -0.009 | 0.795 |
| global_severity | 0.162 | 0.226 | 0.064 | 1.000 |
| timing_missingness | 0.161 | 0.209 | 0.049 | 1.000 |

Gate weights summarize model behavior, not causal importance. In NACC, cognition receives lower gate weight, while global severity, timing/missingness, and demographics receive higher weights, consistent with NACC-specific feature availability.

### Missing-module stress-test headline

Missing-module scenarios are inference-time stress tests, not evidence that modules are clinically interchangeable. The full table is stored in `results/reports/70d_step39_modular_external_dry_run_synthesis/external_missing_module_interpretation_table.csv`.

| Scenario | 3y AUROC | 3y AUPRC | 3y Brier | Interpretation |
|---|---:|---:|---:|---|
| full_modules | 0.727 | 0.641 | 0.208 | Reference full modular scenario. |
| process_only | 0.506 | 0.394 | 0.251 | Timing/missingness process alone is near chance externally, indicating clinical modules carry most discrimination. |
| function_only | 0.669 | 0.568 | 0.218 | Function module is one of the strongest single-module external scenarios, consistent with FAQTOTAL carrying progression signal. |
| severity_only | 0.653 | 0.576 | 0.229 | Global severity is a strong single-module scenario, reflecting high availability of CDGLOBAL/CDRSB in NACC. |
| no_cognition | 0.717 | 0.620 | 0.210 | Removing cognition causes only small loss, likely because ADAS13 is unavailable and MMSE is incomplete in NACC. |
| no_timing_missingness | 0.736 | 0.646 | 0.206 | Removing timing/missingness improves this horizon, suggesting reconstructed timing features may introduce external domain shift. |

## Calibration and robustness

Step40 provides internal calibration and patient-cluster bootstrap artifacts for the locked modular model. These results support internal model auditing but do not establish clinical deployment readiness or final external calibration.

### Calibration metrics

| split | risk_type | horizon_year | AUROC | AUPRC | Brier | calibration_intercept | calibration_slope | observed_n | positive_n | positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | platt | 1 | 0.9107 | 0.4195 | 0.0481 | 0.1284 | 1.2076 | 570 | 37 | 0.0649 |
| test | raw | 1 | 0.9107 | 0.4195 | 0.0481 | 0.3271 | 1.2150 | 570 | 37 | 0.0649 |
| test | platt | 2 | 0.8943 | 0.6095 | 0.0861 | -0.0028 | 1.0536 | 509 | 80 | 0.1572 |
| test | raw | 2 | 0.8943 | 0.6095 | 0.0863 | -0.1252 | 1.0418 | 509 | 80 | 0.1572 |
| test | platt | 3 | 0.9047 | 0.8154 | 0.1036 | -0.0890 | 0.9319 | 459 | 117 | 0.2549 |
| test | raw | 3 | 0.9047 | 0.8154 | 0.1032 | -0.0307 | 1.0052 | 459 | 117 | 0.2549 |
| test | platt | 5 | 0.8996 | 0.8884 | 0.1323 | -0.3216 | 0.9034 | 369 | 148 | 0.4011 |
| test | raw | 5 | 0.8996 | 0.8884 | 0.1300 | -0.1481 | 0.9169 | 369 | 148 | 0.4011 |

### Patient-cluster bootstrap CI headline

| risk_type | horizon_year | metric | mean | median | ci_lower_2.5 | ci_upper_97.5 | n_bootstrap_valid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 1 | AUROC | 0.9106 | 0.9122 | 0.8668 | 0.9492 | 1000 |
| raw | 2 | AUROC | 0.8936 | 0.8944 | 0.8463 | 0.9339 | 1000 |
| raw | 3 | AUROC | 0.9041 | 0.9057 | 0.8447 | 0.9481 | 1000 |
| raw | 5 | AUROC | 0.8986 | 0.9006 | 0.8298 | 0.9523 | 1000 |

Available Step40 artifacts include calibration metrics, Platt parameters, calibration deciles, calibration plots, bootstrap distribution, and patient-cluster bootstrap CIs.

## Reproducibility

These commands assume upstream artifacts already exist. They do not redistribute raw ADNI/NACC data.

```bat
conda activate pytorch
cd /d C:\Users\18142\Desktop\mci_ad_longitudinal_survival

python scripts\70c_strict_load_step39_modular_checkpoint.py
python scripts\70d_summarize_step39_modular_external_dry_run.py
python scripts\71_modular_mainline_freeze_and_claims_audit.py
python scripts\73_summarize_modular_calibration_robustness.py
python scripts\74_generate_compact_modular_first_readme.py
```

## Repository map

- `scripts/39_train_modular_availability_aware_model.py`: Step39 modular model training script
- `results/models/39_modular_availability_aware_model/`: locked Step39 checkpoint, metrics, predictions, gate weights
- `results/reporting/40_modular_model_calibration_robustness/`: calibration and robustness artifacts
- `results/external_validation/70c_step39_modular_nacc_direct_dry_run/`: strict-load NACC direct dry-run outputs
- `results/reports/70d_step39_modular_external_dry_run_synthesis/`: internal/external and gate-shift synthesis
- `results/reports/71_modular_mainline_freeze_and_claims_audit/`: evidence inventory and claims audit
- `results/reports/73_modular_calibration_robustness_synthesis/`: Step40 calibration/robustness synthesis

## What this project can claim

- A locked Step39 modular availability-aware GRU gated-fusion checkpoint exists.
- ADNI internal test performance is strong across 1/2/3/5-year horizons.
- NACC direct dry-run retained moderate discrimination despite feature mismatch.
- External gate re-allocation and missing-module stress tests provide useful model-behavior diagnostics.
- Step40 provides internal calibration and patient-cluster robustness artifacts.

## What this project does not claim

- It does not claim clinical deployment readiness.
- It does not claim prospective validation.
- It does not claim definitive locked external validation on NACC.
- It does not claim causal module importance from gate weights.
- It does not claim state-of-the-art performance without a fair benchmark suite.

## Limitations and next steps

The NACC analysis is a direct dry-run because NACC lacks ADAS13 and the Step45 common-feature file does not preserve the full native Step39 four-dimensional time tensor. Future work should reconstruct external cohorts with native time features, lock calibration/display rules, benchmark against fair baselines, and evaluate clinical utility prospectively.

## Optional web demo

A web interface can be added as an optional research demonstration. It should not define the scientific contribution. The scientific contribution is the modular longitudinal model and its behavior under cross-cohort feature mismatch.

## Disclaimer

Research use only. This model has not undergone prospective validation, regulatory review, or clinical utility evaluation.


## Streamlit patient-style research demo

A patient-style research demo is available:

```bash
streamlit run app/streamlit_app.py
```

The demo accepts routine clinical assessment values and displays 1/2/3/5-year research-demo risk estimates. The public demo estimate is not the private frozen model output and is not intended for clinical use.

## Data processing

The repository includes the upstream ADNI preprocessing scripts used to construct the longitudinal token tensors for the final model:

```text
scripts/01_tokenize_adni_primary_sequences.py
scripts/02_preprocess_adni_sequences_train_only.py
scripts/03_build_adni_promise_dynamic_tokens.py
```

These scripts document the pipeline from ADNI clinical tables to PROMISE-style dynamic tokens with value, missingness mask, time, delta, slope, visit mask, horizon labels, and observed-label masks. Raw ADNI/NACC data and generated participant-level tensors are not redistributed. See `docs/DATA_PROCESSING.md` for details.

## Script workflow

The public scripts are renumbered as a clean end-to-end workflow from ADNI preprocessing to internal evaluation, NACC external validation, and the patient-style research demo. See `docs/SCRIPT_WORKFLOW.md`.
