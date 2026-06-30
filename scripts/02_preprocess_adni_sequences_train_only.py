# ---------------------------------------------------------------------------
# AUTO-GENERATED CLEANED PREPROCESSING SCRIPT
# Created by scripts/18f_create_cleaned_preprocessing_script.py
#
# Differences from 09_preprocess_primary_sequences_train_only.py:
#   - reads results/features/07_primary_sequence_tokens_cleaned
#   - writes results/features/09_primary_train_only_preprocessing_cleaned
#
# The original script is not modified.
# ---------------------------------------------------------------------------

"""
09_preprocess_primary_sequences_train_only.py

Purpose
-------
Fit imputation and scaling parameters on TRAIN split only, then transform
train / validation / test / all primary sequence tensors.

Why this matters:
- Missing value imputation must not use validation / test information.
- Feature standardization must not use validation / test information.
- Patient-level split was already created in Step 08.

Inputs:
- results/features/07_primary_sequence_tokens_cleaned/
  - X_values_raw_zero_filled.npy
  - X_feature_mask.npy
  - X_time.npy
  - X_visit_mask.npy
  - y_labels.npy
  - y_observed.npy
  - sequence_lengths.npy
  - tokenized_landmark_metadata.csv
  - tokenization_config.json
- results/features/08_primary_patient_split/
  - primary_split_indices.npz
  - tokenized_landmark_metadata_with_split.csv

Outputs:
- X_values_imputed_scaled.npy
- X_time_scaled.npy
- X_feature_mask.npy
- X_visit_mask.npy
- y_labels.npy
- y_observed.npy
- sequence_lengths.npy
- primary_preprocessed_k8.npz
- preprocessing_parameters.csv
- preprocessing_summary.csv
- tokenized_landmark_metadata_with_split.csv
- preprocessing_config.json

This step does NOT:
- train a model
- tune hyperparameters
- use validation/test to fit preprocessing parameters
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOKEN_DIR = PROJECT_ROOT / "results" / "features" / "07_primary_sequence_tokens_cleaned"
SPLIT_DIR = PROJECT_ROOT / "results" / "features" / "08_primary_patient_split"

OUT_DIR = PROJECT_ROOT / "results" / "features" / "09_primary_train_only_preprocessing_cleaned"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

EPS = 1e-8

DEFAULT_PRIMARY_FEATURES = [
    "age_at_visit",
    "sex_male",
    "PTEDUCAT",
    "MMSE",
    "ADAS13",
    "CDGLOBAL",
    "CDRSB",
    "FAQTOTAL",
]

DEFAULT_TIME_FEATURES = [
    "years_from_first_mci",
    "visit_interval_years",
    "time_from_visit_to_landmark_years",
    "visit_index_after_mci",
]

HORIZONS = ["1y", "2y", "3y", "5y"]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin1")


def save_csv(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(OUT_DIR / filename, index=False, encoding="utf-8-sig")


def save_json(obj: dict, filename: str) -> None:
    with open(OUT_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_tokenization_config() -> dict:
    path = TOKEN_DIR / "tokenization_config.json"
    if not path.exists():
        return {
            "primary_model_features": DEFAULT_PRIMARY_FEATURES,
            "time_features": DEFAULT_TIME_FEATURES,
        }

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_std(values: np.ndarray) -> float:
    if values.size == 0:
        return 1.0

    std = float(np.nanstd(values))

    if not np.isfinite(std) or std < EPS:
        return 1.0

    return std


def safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0

    mean = float(np.nanmean(values))

    if not np.isfinite(mean):
        return 0.0

    return mean


def safe_median(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0

    median = float(np.nanmedian(values))

    if not np.isfinite(median):
        return 0.0

    return median


def summarize_horizon_by_split(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for split in ["train", "val", "test"]:
        split_df = metadata.loc[metadata["split"] == split].copy()

        for h in HORIZONS:
            label_col = f"label_{h}"
            observed_col = f"observed_{h}"

            if label_col not in split_df.columns or observed_col not in split_df.columns:
                continue

            observed_mask = split_df[observed_col].fillna(0).astype(int) == 1
            observed_df = split_df.loc[observed_mask].copy()

            n_total = int(split_df.shape[0])
            n_observed = int(observed_mask.sum())
            n_positive = int((observed_df[label_col] == 1).sum())
            n_negative = int((observed_df[label_col] == 0).sum())

            rows.append(
                {
                    "split": split,
                    "horizon": h,
                    "n_landmarks_total": n_total,
                    "n_observed": n_observed,
                    "n_unknown": int(n_total - n_observed),
                    "n_positive": n_positive,
                    "n_negative": n_negative,
                    "positive_rate_among_observed": (
                        float(n_positive / n_observed) if n_observed > 0 else np.nan
                    ),
                    "observed_fraction": (
                        float(n_observed / n_total) if n_total > 0 else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    required_files = [
        TOKEN_DIR / "X_values_raw_zero_filled.npy",
        TOKEN_DIR / "X_feature_mask.npy",
        TOKEN_DIR / "X_time.npy",
        TOKEN_DIR / "X_visit_mask.npy",
        TOKEN_DIR / "y_labels.npy",
        TOKEN_DIR / "y_observed.npy",
        TOKEN_DIR / "sequence_lengths.npy",
        SPLIT_DIR / "primary_split_indices.npz",
        SPLIT_DIR / "tokenized_landmark_metadata_with_split.csv",
    ]

    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {missing_files}")

    print("[INFO] Loading token arrays")
    X_values_raw = np.load(TOKEN_DIR / "X_values_raw_zero_filled.npy")
    X_feature_mask = np.load(TOKEN_DIR / "X_feature_mask.npy")
    X_time_raw = np.load(TOKEN_DIR / "X_time.npy")
    X_visit_mask = np.load(TOKEN_DIR / "X_visit_mask.npy")
    y_labels = np.load(TOKEN_DIR / "y_labels.npy")
    y_observed = np.load(TOKEN_DIR / "y_observed.npy")
    sequence_lengths = np.load(TOKEN_DIR / "sequence_lengths.npy")

    split_npz = np.load(SPLIT_DIR / "primary_split_indices.npz")
    train_idx = split_npz["train_idx"]
    val_idx = split_npz["val_idx"]
    test_idx = split_npz["test_idx"]

    metadata = read_csv_robust(SPLIT_DIR / "tokenized_landmark_metadata_with_split.csv")

    config = load_tokenization_config()
    primary_features = config.get("primary_model_features", DEFAULT_PRIMARY_FEATURES)
    time_features = config.get("time_features", DEFAULT_TIME_FEATURES)

    n_samples, max_seq_len, n_features = X_values_raw.shape
    _, _, n_time_features = X_time_raw.shape

    if n_features != len(primary_features):
        raise ValueError(
            f"Feature number mismatch: array has {n_features}, "
            f"config has {len(primary_features)}."
        )

    if n_time_features != len(time_features):
        raise ValueError(
            f"Time feature number mismatch: array has {n_time_features}, "
            f"config has {len(time_features)}."
        )

    if metadata.shape[0] != n_samples:
        raise ValueError(
            f"Metadata rows {metadata.shape[0]} != array samples {n_samples}."
        )

    print("[INFO] Fitting preprocessing parameters on TRAIN split only")

    # --------------------------------------------------------
    # Fit clinical feature imputer/scaler on train observed tokens only
    # --------------------------------------------------------

    X_values_scaled = np.zeros_like(X_values_raw, dtype=np.float32)
    preprocessing_rows = []

    train_visit_mask = X_visit_mask[train_idx] == 1

    for j, feature in enumerate(primary_features):
        train_feature_mask = X_feature_mask[train_idx, :, j] == 1
        train_observed_mask = train_visit_mask & train_feature_mask

        observed_values = X_values_raw[train_idx, :, j][train_observed_mask]

        median_value = safe_median(observed_values)
        mean_value = safe_mean(observed_values)
        std_value = safe_std(observed_values)

        # Impute missing real tokens with train median, then scale.
        feature_raw = X_values_raw[:, :, j].astype(np.float32).copy()
        feature_observed = X_feature_mask[:, :, j] == 1
        real_tokens = X_visit_mask == 1

        feature_imputed = feature_raw.copy()
        feature_imputed[real_tokens & (~feature_observed)] = median_value

        feature_scaled = (feature_imputed - mean_value) / std_value

        # Padded tokens should remain exactly zero.
        feature_scaled[X_visit_mask == 0] = 0.0

        X_values_scaled[:, :, j] = feature_scaled.astype(np.float32)

        all_real_tokens = int(real_tokens.sum())
        all_observed_tokens = int(feature_observed[real_tokens].sum())
        train_real_tokens = int(train_visit_mask.sum())
        train_observed_tokens = int(train_observed_mask.sum())

        preprocessing_rows.append(
            {
                "type": "clinical_feature",
                "name": feature,
                "train_observed_tokens": train_observed_tokens,
                "train_real_tokens": train_real_tokens,
                "train_observed_fraction": (
                    float(train_observed_tokens / train_real_tokens)
                    if train_real_tokens > 0 else np.nan
                ),
                "all_observed_tokens": all_observed_tokens,
                "all_real_tokens": all_real_tokens,
                "all_observed_fraction": (
                    float(all_observed_tokens / all_real_tokens)
                    if all_real_tokens > 0 else np.nan
                ),
                "imputation_value_train_median": median_value,
                "scaling_mean_train_observed": mean_value,
                "scaling_std_train_observed": std_value,
            }
        )

    # --------------------------------------------------------
    # Fit time feature scaler on train real tokens only
    # --------------------------------------------------------

    X_time_scaled = np.zeros_like(X_time_raw, dtype=np.float32)

    for j, feature in enumerate(time_features):
        train_time_values = X_time_raw[train_idx, :, j][train_visit_mask]

        mean_value = safe_mean(train_time_values)
        std_value = safe_std(train_time_values)

        feature_scaled = (X_time_raw[:, :, j].astype(np.float32) - mean_value) / std_value
        feature_scaled[X_visit_mask == 0] = 0.0

        X_time_scaled[:, :, j] = feature_scaled.astype(np.float32)

        preprocessing_rows.append(
            {
                "type": "time_feature",
                "name": feature,
                "train_observed_tokens": int(train_visit_mask.sum()),
                "train_real_tokens": int(train_visit_mask.sum()),
                "train_observed_fraction": 1.0,
                "all_observed_tokens": int(X_visit_mask.sum()),
                "all_real_tokens": int(X_visit_mask.sum()),
                "all_observed_fraction": 1.0,
                "imputation_value_train_median": np.nan,
                "scaling_mean_train_observed": mean_value,
                "scaling_std_train_observed": std_value,
            }
        )

    preprocessing_params = pd.DataFrame(preprocessing_rows)

    # --------------------------------------------------------
    # Split-level summary
    # --------------------------------------------------------

    split_rows = []

    for split_name, idx in [
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx),
    ]:
        split_meta = metadata.iloc[idx].copy()
        split_visit_mask = X_visit_mask[idx] == 1

        split_rows.append(
            {
                "split": split_name,
                "n_landmarks": int(len(idx)),
                "n_patients": int(split_meta["RID"].nunique()),
                "real_visit_tokens": int(split_visit_mask.sum()),
                "median_sequence_length": float(np.median(sequence_lengths[idx])),
                "mean_sequence_length": float(np.mean(sequence_lengths[idx])),
                "min_sequence_length": int(np.min(sequence_lengths[idx])),
                "max_sequence_length": int(np.max(sequence_lengths[idx])),
            }
        )

    split_preprocess_summary = pd.DataFrame(split_rows)

    global_summary = pd.DataFrame(
        [
            {"metric": "n_samples", "value": int(n_samples)},
            {"metric": "max_seq_len", "value": int(max_seq_len)},
            {"metric": "n_primary_features", "value": int(n_features)},
            {"metric": "n_time_features", "value": int(n_time_features)},
            {"metric": "train_landmarks", "value": int(len(train_idx))},
            {"metric": "val_landmarks", "value": int(len(val_idx))},
            {"metric": "test_landmarks", "value": int(len(test_idx))},
            {"metric": "train_patients", "value": int(metadata.iloc[train_idx]["RID"].nunique())},
            {"metric": "val_patients", "value": int(metadata.iloc[val_idx]["RID"].nunique())},
            {"metric": "test_patients", "value": int(metadata.iloc[test_idx]["RID"].nunique())},
            {"metric": "fit_preprocessing_on", "value": "train split only"},
        ]
    )

    horizon_summary = summarize_horizon_by_split(metadata)

    # --------------------------------------------------------
    # Save arrays
    # --------------------------------------------------------

    print("[INFO] Saving preprocessed arrays")

    np.save(OUT_DIR / "X_values_imputed_scaled.npy", X_values_scaled)
    np.save(OUT_DIR / "X_time_scaled.npy", X_time_scaled)
    np.save(OUT_DIR / "X_feature_mask.npy", X_feature_mask.astype(np.float32))
    np.save(OUT_DIR / "X_visit_mask.npy", X_visit_mask.astype(np.float32))
    np.save(OUT_DIR / "y_labels.npy", y_labels.astype(np.float32))
    np.save(OUT_DIR / "y_observed.npy", y_observed.astype(np.float32))
    np.save(OUT_DIR / "sequence_lengths.npy", sequence_lengths.astype(np.int32))

    np.save(OUT_DIR / "train_indices.npy", train_idx)
    np.save(OUT_DIR / "val_indices.npy", val_idx)
    np.save(OUT_DIR / "test_indices.npy", test_idx)

    np.savez_compressed(
        OUT_DIR / "primary_preprocessed_k8.npz",
        X_values_imputed_scaled=X_values_scaled,
        X_time_scaled=X_time_scaled,
        X_feature_mask=X_feature_mask.astype(np.float32),
        X_visit_mask=X_visit_mask.astype(np.float32),
        y_labels=y_labels.astype(np.float32),
        y_observed=y_observed.astype(np.float32),
        sequence_lengths=sequence_lengths.astype(np.int32),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    # --------------------------------------------------------
    # Save tables and config
    # --------------------------------------------------------

    save_csv(metadata, "tokenized_landmark_metadata_with_split.csv")
    save_csv(preprocessing_params, "preprocessing_parameters.csv")
    save_csv(split_preprocess_summary, "split_preprocessing_summary.csv")
    save_csv(global_summary, "preprocessing_summary.csv")
    save_csv(horizon_summary, "preprocessing_horizon_label_summary.csv")

    preprocessing_config = {
        "fit_on": "train split only",
        "clinical_features": primary_features,
        "time_features": time_features,
        "clinical_feature_imputation": "train observed-token median",
        "clinical_feature_scaling": "train observed-token mean/std",
        "time_feature_scaling": "train real-token mean/std",
        "padded_tokens": "kept at zero after scaling",
        "missing_feature_values": "imputed with train median, with original feature mask retained",
        "inputs": {
            "token_dir": str(TOKEN_DIR),
            "split_dir": str(SPLIT_DIR),
        },
        "outputs": {
            "X_values_imputed_scaled": "X_values_imputed_scaled.npy",
            "X_time_scaled": "X_time_scaled.npy",
            "combined": "primary_preprocessed_k8.npz",
        },
        "important": [
            "No validation/test values are used to fit imputation or scaling parameters.",
            "Feature masks and visit masks must be passed to downstream models.",
            "APOE4 is not included in the primary model.",
            "years_to_outcome is not included as an input feature.",
        ],
    }

    save_json(preprocessing_config, "preprocessing_config.json")

    readme = f"""# Primary Train-Only Preprocessing

This folder contains train-only imputed and scaled primary sequence tensors.

Inputs:
- {TOKEN_DIR}
- {SPLIT_DIR}

Outputs:
- X_values_imputed_scaled.npy
- X_time_scaled.npy
- X_feature_mask.npy
- X_visit_mask.npy
- y_labels.npy
- y_observed.npy
- sequence_lengths.npy
- primary_preprocessed_k8.npz

Clinical features:
- {", ".join(primary_features)}

Time features:
- {", ".join(time_features)}

Preprocessing:
- Clinical missing values are imputed using train observed-token medians.
- Clinical features are scaled using train observed-token mean/std.
- Time features are scaled using train real-token mean/std.
- Padded tokens are kept at zero after scaling.

Important:
Validation/test values are not used to fit preprocessing parameters.
Feature masks and visit masks are retained for downstream missingness-aware models.
"""

    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    assert X_values_scaled.shape == X_values_raw.shape
    assert X_time_scaled.shape == X_time_raw.shape
    assert X_feature_mask.shape[:2] == X_visit_mask.shape
    assert y_labels.shape == y_observed.shape

    assert np.all(X_values_scaled[X_visit_mask == 0] == 0.0)
    assert np.all(X_time_scaled[X_visit_mask == 0] == 0.0)

    assert not np.isnan(X_values_scaled).any(), "NaN found in X_values_scaled."
    assert not np.isnan(X_time_scaled).any(), "NaN found in X_time_scaled."

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("[DONE] Train-only preprocessing completed.")
    print(f"[DONE] Output folder: {OUT_DIR}")
    print()

    print("[PREPROCESSING SUMMARY]")
    for _, row in global_summary.iterrows():
        print(f"{row['metric']}: {row['value']}")

    print()
    print("[SPLIT PREPROCESSING SUMMARY]")
    for _, row in split_preprocess_summary.iterrows():
        print(
            f"{row['split']}: "
            f"landmarks={int(row['n_landmarks'])}, "
            f"patients={int(row['n_patients'])}, "
            f"real_tokens={int(row['real_visit_tokens'])}, "
            f"median_seq_len={row['median_sequence_length']:.1f}"
        )

    print()
    print("[PREPROCESSING PARAMETERS]")
    for _, row in preprocessing_params.iterrows():
        print(
            f"{row['type']} | {row['name']}: "
            f"median={row['imputation_value_train_median']}, "
            f"mean={row['scaling_mean_train_observed']:.4f}, "
            f"std={row['scaling_std_train_observed']:.4f}, "
            f"train_obs_fraction={row['train_observed_fraction']:.4f}"
        )

    print()
    print("[NEXT]")
    print("Proceed to baseline models or the first primary temporal survival model.")


if __name__ == "__main__":
    main()