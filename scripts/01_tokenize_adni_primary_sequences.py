# ---------------------------------------------------------------------------
# AUTO-GENERATED CLEANED TOKENIZATION SCRIPT
# Created by scripts/18e_create_cleaned_tokenization_script.py
#
# Differences from 07_tokenize_primary_sequences.py:
#   - reads clinical_visits_long.csv
#   - writes to results/features/07_primary_sequence_tokens_cleaned
#
# The original script is not modified.
# ---------------------------------------------------------------------------

"""
07_tokenize_primary_sequences.py

Purpose
-------
Tokenize the PRIMARY low-cost clinical longitudinal data into sequence tensors.

Primary model:
- No APOE4
- No PET / MRI / CSF
- No diagnosis columns as model inputs
- No future outcome information as model inputs

Inputs:
- results/features/05b_clinical_visits_long/clinical_visits_long.csv
- results/features/06_feature_availability_audit/landmark_feature_availability_primary.csv

Main subset:
- subset_primary_sequence_ge2 == 1

Outputs:
- X_values_raw_zero_filled.npy
- X_feature_mask.npy
- X_time.npy
- X_visit_mask.npy
- y_labels.npy
- y_observed.npy
- sequence_lengths.npy
- tokenized_landmark_metadata.csv
- primary_sequence_tokens_k8.npz
- tokenization_summary.csv
- tokenized_feature_observation_summary.csv
- tokenized_horizon_label_summary.csv

This step does NOT:
- train / validation / test split
- impute using training data
- standardize features
- train model

It only builds leakage-safe sequence tensors.
"""

from __future__ import annotations

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLINICAL_VISITS_PATH = (
    PROJECT_ROOT
    / "results"
    / "features"
    / "05b_clinical_visits_long"
    / "clinical_visits_long.csv"
)

LANDMARK_AVAILABILITY_PATH = (
    PROJECT_ROOT
    / "results"
    / "features"
    / "06_feature_availability_audit"
    / "landmark_feature_availability_primary.csv"
)

OUT_DIR = PROJECT_ROOT / "results" / "features" / "07_primary_sequence_tokens_cleaned"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

MAX_SEQ_LEN = 8

SUBSET_FLAG = "subset_primary_sequence_ge2"

# Raw clinical features used for the primary model.
# PTGENDER will be encoded as sex_male.
PRIMARY_MODEL_FEATURES = [
    "age_at_visit",
    "sex_male",
    "PTEDUCAT",
    "MMSE",
    "ADAS13",
    "CDGLOBAL",
    "CDRSB",
    "FAQTOTAL",
]

STATE_FEATURES = [
    "MMSE",
    "ADAS13",
    "CDGLOBAL",
    "CDRSB",
    "FAQTOTAL",
]

TIME_FEATURES = [
    "years_from_first_mci",
    "visit_interval_years",
    "time_from_visit_to_landmark_years",
    "visit_index_after_mci",
]

HORIZONS = ["1y", "2y", "3y", "5y"]

FORBIDDEN_INPUT_FEATURES = {
    "APOE4",
    "years_to_outcome",
    "DIAGNOSIS",
    "DXCHANGE",
    "DXCURREN",
    "DXCONV",
    "DXNORM",
    "DXMCI",
    "DXAD",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin1")


def parse_date_safe(series: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(series, errors="coerce")


def save_csv(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(OUT_DIR / filename, index=False, encoding="utf-8-sig")


def save_json(obj: dict, filename: str) -> None:
    with open(OUT_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_fraction(num: int | float, den: int | float) -> float:
    if den == 0:
        return np.nan
    return float(num / den)


def encode_sex_male(value) -> float:
    """
    Encode sex as male=1, female=0.

    Handles common string values:
    - 'Male' / 'M' -> 1
    - 'Female' / 'F' -> 0

    Handles common ADNI numeric coding:
    - 1 -> male
    - 2 -> female
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()

    # Check female before male because "female" contains "male".
    if text in {"female", "f", "woman", "women"}:
        return 0.0

    if text in {"male", "m", "man", "men"}:
        return 1.0

    num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if pd.isna(num):
        return np.nan

    num = float(num)

    if num == 1:
        return 1.0

    if num == 2:
        return 0.0

    if num in {0.0, 1.0}:
        return num

    return np.nan


def summarize_horizon_labels(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for h in HORIZONS:
        label_col = f"label_{h}"
        observed_col = f"observed_{h}"

        if label_col not in metadata.columns or observed_col not in metadata.columns:
            continue

        observed_mask = metadata[observed_col].fillna(0).astype(int) == 1
        observed_df = metadata.loc[observed_mask].copy()

        n_total = int(metadata.shape[0])
        n_observed = int(observed_mask.sum())
        n_unknown = int(n_total - n_observed)
        n_positive = int((observed_df[label_col] == 1).sum())
        n_negative = int((observed_df[label_col] == 0).sum())

        rows.append(
            {
                "horizon": h,
                "n_landmarks_total": n_total,
                "n_observed": n_observed,
                "n_unknown": n_unknown,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "positive_rate_among_observed": safe_fraction(n_positive, n_observed),
                "observed_fraction": safe_fraction(n_observed, n_total),
                "n_patients_total": int(metadata["RID"].nunique()),
                "n_patients_observed": int(observed_df["RID"].nunique()),
                "n_positive_patients": int(
                    observed_df.loc[observed_df[label_col] == 1, "RID"].nunique()
                ),
            }
        )

    return pd.DataFrame(rows)


def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    if not CLINICAL_VISITS_PATH.exists():
        raise FileNotFoundError(f"Missing clinical visits file: {CLINICAL_VISITS_PATH}")

    if not LANDMARK_AVAILABILITY_PATH.exists():
        raise FileNotFoundError(
            f"Missing landmark availability file: {LANDMARK_AVAILABILITY_PATH}"
        )

    forbidden_used = sorted(set(PRIMARY_MODEL_FEATURES + TIME_FEATURES) & FORBIDDEN_INPUT_FEATURES)
    if forbidden_used:
        raise ValueError(f"Forbidden leakage / non-primary features used: {forbidden_used}")

    print(f"[INFO] Reading clinical visits: {CLINICAL_VISITS_PATH}")
    visits = read_csv_robust(CLINICAL_VISITS_PATH)

    print(f"[INFO] Reading primary landmark availability: {LANDMARK_AVAILABILITY_PATH}")
    landmarks = read_csv_robust(LANDMARK_AVAILABILITY_PATH)

    require_columns(
        visits,
        [
            "RID",
            "VISCODE2",
            "visit_date",
            "PTGENDER",
            "age_at_visit",
            "PTEDUCAT",
            "any_state_feature_observed",
            "years_from_first_mci",
            "visit_interval_years",
            "visit_index_after_mci",
        ],
        "clinical_visits_long.csv",
    )

    require_columns(
        landmarks,
        ["RID", "landmark_id", "landmark_date", SUBSET_FLAG],
        "landmark_feature_availability_primary.csv",
    )

    visits["RID"] = pd.to_numeric(visits["RID"], errors="coerce")
    landmarks["RID"] = pd.to_numeric(landmarks["RID"], errors="coerce")

    visits["visit_date"] = parse_date_safe(visits["visit_date"])
    landmarks["landmark_date"] = parse_date_safe(landmarks["landmark_date"])

    # Encode sex for model input.
    visits["sex_male"] = visits["PTGENDER"].apply(encode_sex_male)

    # Ensure primary feature columns exist and are numeric.
    for feature in PRIMARY_MODEL_FEATURES:
        if feature not in visits.columns:
            visits[feature] = np.nan
        visits[feature] = pd.to_numeric(visits[feature], errors="coerce")

    for feature in TIME_FEATURES:
        if feature not in visits.columns and feature != "time_from_visit_to_landmark_years":
            visits[feature] = np.nan
        if feature in visits.columns:
            visits[feature] = pd.to_numeric(visits[feature], errors="coerce")

    # Use only primary modeling subset.
    modeling_landmarks = landmarks.loc[
        landmarks[SUBSET_FLAG].fillna(0).astype(int) == 1
    ].copy()

    modeling_landmarks = modeling_landmarks.sort_values(["RID", "landmark_date", "landmark_id"]).copy()
    modeling_landmarks = modeling_landmarks.reset_index(drop=True)

    n_samples = int(modeling_landmarks.shape[0])
    n_features = len(PRIMARY_MODEL_FEATURES)
    n_time_features = len(TIME_FEATURES)
    n_horizons = len(HORIZONS)

    print(f"[INFO] Tokenizing subset: {SUBSET_FLAG}")
    print(f"[INFO] n_landmarks: {n_samples}")
    print(f"[INFO] n_features: {n_features}")
    print(f"[INFO] max_seq_len: {MAX_SEQ_LEN}")

    X_values = np.zeros((n_samples, MAX_SEQ_LEN, n_features), dtype=np.float32)
    X_feature_mask = np.zeros((n_samples, MAX_SEQ_LEN, n_features), dtype=np.float32)
    X_time = np.zeros((n_samples, MAX_SEQ_LEN, n_time_features), dtype=np.float32)
    X_visit_mask = np.zeros((n_samples, MAX_SEQ_LEN), dtype=np.float32)

    y_labels = np.full((n_samples, n_horizons), np.nan, dtype=np.float32)
    y_observed = np.zeros((n_samples, n_horizons), dtype=np.float32)

    sequence_lengths = np.zeros(n_samples, dtype=np.int32)

    metadata_rows = []

    visits_by_rid = {
        rid: group.sort_values(["visit_date", "VISCODE2"]).copy()
        for rid, group in visits.groupby("RID", dropna=False)
    }

    for i, landmark in modeling_landmarks.iterrows():
        rid = landmark["RID"]
        landmark_date = landmark["landmark_date"]

        patient_visits = visits_by_rid.get(rid, visits.iloc[0:0].copy())

        if pd.isna(landmark_date):
            history = patient_visits.iloc[0:0].copy()
        else:
            history = patient_visits.loc[
                patient_visits["visit_date"].notna()
                & (patient_visits["visit_date"] <= landmark_date)
                & (patient_visits["any_state_feature_observed"].fillna(0).astype(int) == 1)
            ].copy()

        history = history.sort_values(["visit_date", "VISCODE2"]).copy()

        n_available_history_visits = int(history.shape[0])

        # Keep the most recent K visits.
        history_used = history.tail(MAX_SEQ_LEN).copy()
        seq_len = int(history_used.shape[0])
        sequence_lengths[i] = seq_len

        pad_start = MAX_SEQ_LEN - seq_len

        used_visit_dates = []

        for local_idx, (_, visit_row) in enumerate(history_used.iterrows()):
            token_idx = pad_start + local_idx

            X_visit_mask[i, token_idx] = 1.0

            visit_date = visit_row["visit_date"]
            used_visit_dates.append(visit_date)

            # Clinical values and masks.
            for j, feature in enumerate(PRIMARY_MODEL_FEATURES):
                value = visit_row.get(feature, np.nan)

                if pd.notna(value):
                    X_values[i, token_idx, j] = float(value)
                    X_feature_mask[i, token_idx, j] = 1.0
                else:
                    X_values[i, token_idx, j] = 0.0
                    X_feature_mask[i, token_idx, j] = 0.0

            # Non-leakage time features.
            time_values = {
                "years_from_first_mci": visit_row.get("years_from_first_mci", np.nan),
                "visit_interval_years": visit_row.get("visit_interval_years", np.nan),
                "time_from_visit_to_landmark_years": (
                    (landmark_date - visit_date).days / 365.25
                    if pd.notna(landmark_date) and pd.notna(visit_date)
                    else np.nan
                ),
                "visit_index_after_mci": visit_row.get("visit_index_after_mci", np.nan),
            }

            for t, feature in enumerate(TIME_FEATURES):
                value = time_values.get(feature, np.nan)
                X_time[i, token_idx, t] = float(value) if pd.notna(value) else 0.0

        # Labels and observed masks.
        for h_idx, h in enumerate(HORIZONS):
            label_col = f"label_{h}"
            observed_col = f"observed_{h}"

            label_value = landmark.get(label_col, np.nan)
            observed_value = landmark.get(observed_col, 0)

            if pd.notna(label_value):
                y_labels[i, h_idx] = float(label_value)

            if pd.notna(observed_value):
                y_observed[i, h_idx] = float(observed_value)

        if used_visit_dates:
            first_used_visit_date = min(used_visit_dates)
            last_used_visit_date = max(used_visit_dates)
        else:
            first_used_visit_date = pd.NaT
            last_used_visit_date = pd.NaT

        metadata = {
            "token_index": i,
            "RID": rid,
            "landmark_id": landmark.get("landmark_id", ""),
            "landmark_date": landmark_date,
            "n_available_history_state_visits": n_available_history_visits,
            "sequence_length": seq_len,
            "max_seq_len": MAX_SEQ_LEN,
            "first_used_visit_date": first_used_visit_date,
            "last_used_visit_date": last_used_visit_date,
            "truncated_history": int(n_available_history_visits > MAX_SEQ_LEN),
            "subset_flag": SUBSET_FLAG,
        }

        keep_landmark_cols = [
            "event",
            "time_to_event_from_landmark_years",
            "time_to_outcome_from_landmark_years",
            "eligible_mci_state_landmark",
            "n_history_state_feature_dates",
            "n_state_features_available",
            "n_delta_ready_state_features",
        ]

        for col in keep_landmark_cols:
            if col in landmark.index:
                metadata[col] = landmark[col]

        for h in HORIZONS:
            label_col = f"label_{h}"
            observed_col = f"observed_{h}"

            metadata[label_col] = landmark.get(label_col, np.nan)
            metadata[observed_col] = landmark.get(observed_col, np.nan)

        metadata_rows.append(metadata)

    metadata_df = pd.DataFrame(metadata_rows)

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------

    real_token_count = int(X_visit_mask.sum())

    feature_summary_rows = []
    for j, feature in enumerate(PRIMARY_MODEL_FEATURES):
        n_observed_tokens = int(X_feature_mask[:, :, j].sum())
        feature_summary_rows.append(
            {
                "feature": feature,
                "n_real_visit_tokens": real_token_count,
                "n_observed_tokens": n_observed_tokens,
                "token_observation_fraction": safe_fraction(n_observed_tokens, real_token_count),
                "n_landmarks_with_feature_observed": int(
                    (X_feature_mask[:, :, j].sum(axis=1) > 0).sum()
                ),
                "landmark_observation_fraction": safe_fraction(
                    int((X_feature_mask[:, :, j].sum(axis=1) > 0).sum()),
                    n_samples,
                ),
            }
        )

    feature_summary = pd.DataFrame(feature_summary_rows)
    horizon_summary = summarize_horizon_labels(metadata_df)

    summary_dict = {
        "subset": SUBSET_FLAG,
        "n_tokenized_landmarks": n_samples,
        "n_patients": int(metadata_df["RID"].nunique()),
        "max_seq_len": MAX_SEQ_LEN,
        "n_primary_features": n_features,
        "n_time_features": n_time_features,
        "median_sequence_length": float(np.median(sequence_lengths)) if n_samples else np.nan,
        "mean_sequence_length": float(np.mean(sequence_lengths)) if n_samples else np.nan,
        "min_sequence_length": int(np.min(sequence_lengths)) if n_samples else 0,
        "max_sequence_length": int(np.max(sequence_lengths)) if n_samples else 0,
        "n_truncated_landmarks": int(metadata_df["truncated_history"].sum()),
        "truncated_fraction": safe_fraction(int(metadata_df["truncated_history"].sum()), n_samples),
        "real_visit_tokens": real_token_count,
        "padded_visit_tokens": int(n_samples * MAX_SEQ_LEN - real_token_count),
    }

    summary = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in summary_dict.items()]
    )

    # --------------------------------------------------------
    # Save arrays
    # --------------------------------------------------------

    np.save(OUT_DIR / "X_values_raw_zero_filled.npy", X_values)
    np.save(OUT_DIR / "X_feature_mask.npy", X_feature_mask)
    np.save(OUT_DIR / "X_time.npy", X_time)
    np.save(OUT_DIR / "X_visit_mask.npy", X_visit_mask)
    np.save(OUT_DIR / "y_labels.npy", y_labels)
    np.save(OUT_DIR / "y_observed.npy", y_observed)
    np.save(OUT_DIR / "sequence_lengths.npy", sequence_lengths)

    np.savez_compressed(
        OUT_DIR / f"primary_sequence_tokens_k{MAX_SEQ_LEN}.npz",
        X_values_raw_zero_filled=X_values,
        X_feature_mask=X_feature_mask,
        X_time=X_time,
        X_visit_mask=X_visit_mask,
        y_labels=y_labels,
        y_observed=y_observed,
        sequence_lengths=sequence_lengths,
    )

    # --------------------------------------------------------
    # Save tables and config
    # --------------------------------------------------------

    save_csv(metadata_df, "tokenized_landmark_metadata.csv")
    save_csv(summary, "tokenization_summary.csv")
    save_csv(feature_summary, "tokenized_feature_observation_summary.csv")
    save_csv(horizon_summary, "tokenized_horizon_label_summary.csv")

    config = {
        "subset": SUBSET_FLAG,
        "max_seq_len": MAX_SEQ_LEN,
        "primary_model_features": PRIMARY_MODEL_FEATURES,
        "state_features": STATE_FEATURES,
        "time_features": TIME_FEATURES,
        "horizons": HORIZONS,
        "excluded_from_primary_model": [
            "APOE4",
            "PET",
            "MRI",
            "CSF",
            "diagnosis columns",
            "years_to_outcome",
            "post-landmark visits",
        ],
        "padding": {
            "alignment": "left padded; most recent visits are at the end",
            "value_for_missing_features": 0.0,
            "feature_mask": "1 if feature observed, else 0",
            "visit_mask": "1 if real visit token, else 0",
        },
        "notes": [
            "No standardization or imputation is performed here.",
            "Feature scaling and imputation must be fit on the training split only.",
            "Only visits at or before each landmark date are used.",
            "Only primary low-cost clinical features are tokenized.",
        ],
    }

    save_json(config, "tokenization_config.json")

    readme = f"""# Primary Sequence Tokens

This folder contains sequence tensors for the primary low-cost clinical model.

Input files:
- {CLINICAL_VISITS_PATH}
- {LANDMARK_AVAILABILITY_PATH}

Subset:
- {SUBSET_FLAG}

Primary features:
- {", ".join(PRIMARY_MODEL_FEATURES)}

Time features:
- {", ".join(TIME_FEATURES)}

Excluded:
- APOE4
- PET / MRI / CSF
- diagnosis columns
- years_to_outcome
- post-landmark visits

Array shapes:
- X_values_raw_zero_filled: [n_landmarks, max_seq_len, n_primary_features]
- X_feature_mask: [n_landmarks, max_seq_len, n_primary_features]
- X_time: [n_landmarks, max_seq_len, n_time_features]
- X_visit_mask: [n_landmarks, max_seq_len]
- y_labels: [n_landmarks, 4]
- y_observed: [n_landmarks, 4]

Important:
Missing clinical values are filled with 0 only as a placeholder.
The feature mask must be used by downstream models.
Train/validation/test split, imputation, and scaling are not done in this step.
"""

    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    assert X_values.shape == (n_samples, MAX_SEQ_LEN, n_features)
    assert X_feature_mask.shape == (n_samples, MAX_SEQ_LEN, n_features)
    assert X_time.shape == (n_samples, MAX_SEQ_LEN, n_time_features)
    assert X_visit_mask.shape == (n_samples, MAX_SEQ_LEN)
    assert y_labels.shape == (n_samples, n_horizons)
    assert y_observed.shape == (n_samples, n_horizons)

    assert np.all(sequence_lengths >= 2), (
        "Primary sequence_ge2 subset should have at least 2 visits per landmark."
    )

    assert "APOE4" not in PRIMARY_MODEL_FEATURES
    assert "years_to_outcome" not in PRIMARY_MODEL_FEATURES
    assert "years_to_outcome" not in TIME_FEATURES

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("[DONE] Primary sequence tokenization completed.")
    print(f"[DONE] Output folder: {OUT_DIR}")
    print()

    print("[TOKENIZATION SUMMARY]")
    for key, value in summary_dict.items():
        print(f"{key}: {value}")

    print()
    print("[FEATURE TOKEN OBSERVATION]")
    for _, row in feature_summary.iterrows():
        print(
            f"{row['feature']}: "
            f"token_observed={int(row['n_observed_tokens'])}/{int(row['n_real_visit_tokens'])}, "
            f"token_fraction={row['token_observation_fraction']:.4f}, "
            f"landmark_fraction={row['landmark_observation_fraction']:.4f}"
        )

    print()
    print("[TOKENIZED HORIZON LABELS]")
    for _, row in horizon_summary.iterrows():
        print(
            f"{row['horizon']}: "
            f"observed={int(row['n_observed'])}/{int(row['n_landmarks_total'])}, "
            f"unknown={int(row['n_unknown'])}, "
            f"positive={int(row['n_positive'])}, "
            f"negative={int(row['n_negative'])}, "
            f"pos_rate={row['positive_rate_among_observed']:.4f}"
        )

    print()
    print("[NEXT]")
    print("Review tokenization_summary.csv and tokenized_horizon_label_summary.csv.")
    print("If shapes and label distributions look correct, proceed to patient-level train/val/test split.")


if __name__ == "__main__":
    main()
