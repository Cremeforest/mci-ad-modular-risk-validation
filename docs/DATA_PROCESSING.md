# Data processing and token construction

This repository does not redistribute raw ADNI or NACC participant-level data. The public code documents the preprocessing logic used to construct the longitudinal tensors for the final research model.

## Overview

The internal ADNI pipeline converts irregular MCI visit histories into fixed-length longitudinal tensors for 1/2/3/5-year MCI-to-AD conversion risk prediction.

```text
ADNI clinical tables
  -> MCI landmark / visit-history construction
  -> train-only preprocessing
  -> dynamic longitudinal token construction
  -> final model training and evaluation
```

## Public preprocessing scripts

```text
scripts/01_tokenize_adni_primary_sequences.py
scripts/02_preprocess_adni_sequences_train_only.py
scripts/03_build_adni_dynamic_tokens.py
```

### 1. Tokenize primary ADNI sequences

`01_tokenize_adni_primary_sequences.py`

This stage constructs MCI landmark-style longitudinal histories from ADNI-derived clinical tables. It organizes current-and-past visits for each landmark and prepares visit-level metadata for downstream preprocessing.

### 2. Train-only preprocessing

`02_preprocess_adni_sequences_train_only.py`

This stage applies preprocessing using training-split information only. It creates scaled value tensors, feature masks, visit masks, horizon labels, observed-label masks, and split metadata.

The important design choice is that imputation and scaling parameters are estimated from the training split only, then applied to validation/test data. This reduces preprocessing leakage.

### 3. Dynamic longitudinal tokens

`03_build_adni_dynamic_tokens.py`

This stage builds the final longitudinal token package used by the model:

```text
results/features/12_promise_dynamic_tokens_cleaned/primary_promise_dynamic_tokens_k8.npz
```

The final tensor package contains:

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

## Features

The final model uses eight routine clinical variables:

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

## Prediction horizons

The model predicts conversion risk at:

```text
1 year
2 years
3 years
5 years
```

These horizons were chosen to cover short-, intermediate-, and longer-term MCI-to-AD conversion risk while avoiding redundant yearly endpoints.

## Missingness handling

Missing clinical values are represented explicitly using feature masks. Numeric values are imputed and scaled using training-split statistics. The model receives both the values and the masks, so it can distinguish observed values from imputed placeholders.

## Longitudinal components

For each feature, the token package includes:

- current/imputed scaled value
- observed/missing mask
- delta from the first available visit
- slope-like longitudinal change
- visit timing features
- visit-level mask for padding

## Reproducibility boundary

The scripts are included to document and reproduce the processing logic when the user has authorized local access to ADNI/NACC data. Raw participant-level data and generated tensors are intentionally not included in this public repository.
