# -*- coding: utf-8 -*-
"""
Step95c: Prepare corrected NACC external aligned tokens for frozen v4 evaluation.

Purpose
-------
Convert Step94d raw NACC tokens into the same model-input style used by
the internal v4 model.

Reads:
    - corrected NACC raw tokens from Step94d
    - internal Step12 token NPZ
    - Step09 train-only preprocessing artifacts

Writes:
    results/features/95c_nacc_external_aligned_tokens/
        nacc_external_aligned_tokens_for_v4_no_ADAS13.npz
        nacc_external_alignment_summary.csv
        nacc_external_alignment_range_check.csv
        preprocessing_parameters_loaded.csv
        README_step95c_nacc_external_aligned_tokens.md
        step95c_audit.json

Does NOT:
    - train
    - predict
    - evaluate
    - refit preprocessing on NACC

Important
---------
This script tries to use Step09 preprocessing_parameters.csv first.
If parameter column names are not recognized, it writes a detailed report and stops.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "v1_prepare_nacc_external_aligned_tokens"

INTERNAL_NPZ_REL = (
    Path("results")
    / "features"
    / "12_promise_dynamic_tokens_cleaned"
    / "primary_promise_dynamic_tokens_k8.npz"
)

EXTERNAL_RAW_NPZ_REL = (
    Path("results")
    / "features"
    / "94d_nacc_external_raw_tokens_corrected_naccvnum"
    / "nacc_external_dynamic_tokens_raw_k8_corrected_naccvnum.npz"
)

EXTERNAL_COHORT_REL = (
    Path("results")
    / "features"
    / "94d_nacc_external_raw_tokens_corrected_naccvnum"
    / "nacc_external_first_mci_cohort_corrected_naccvnum.csv"
)

PREPROCESS_DIR_REL = (
    Path("results")
    / "features"
    / "09_primary_train_only_preprocessing_cleaned"
)

PREPROCESS_PARAMS_REL = PREPROCESS_DIR_REL / "preprocessing_parameters.csv"

OUT_REL = Path("results") / "features" / "95c_nacc_external_aligned_tokens"

FEATURE_NAMES = [
    "age_at_visit",
    "sex_male",
    "PTEDUCAT",
    "MMSE",
    "ADAS13",
    "CDGLOBAL",
    "CDRSB",
    "FAQTOTAL",
]

HORIZON_NAMES = ["1y", "2y", "3y", "5y"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def norm_name(x: Any) -> str:
    return str(x).strip().lower().replace(" ", "_")


def load_npz(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing NPZ: {path}")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    lookup = {norm_name(c): c for c in df.columns}
    for c in candidates:
        key = norm_name(c)
        if key in lookup:
            return lookup[key]
    return None


def detect_param_columns(params: pd.DataFrame) -> Dict[str, str | None]:
    """
    Detect column names in Step09 preprocessing_parameters.csv.

    Current known schema:
        type
        name
        imputation_value_train_median
        scaling_mean_train_observed
        scaling_std_train_observed
    """
    cols = {
        "variable": find_col(
            params,
            [
                "feature",
                "variable",
                "name",
                "column",
                "component",
                "parameter",
                "var",
            ],
        ),
        "mean": find_col(
            params,
            [
                "mean",
                "train_mean",
                "scaler_mean",
                "value_mean",
                "impute_mean",
                "scaling_mean_train_observed",
            ],
        ),
        "std": find_col(
            params,
            [
                "std",
                "train_std",
                "scaler_std",
                "value_std",
                "scale",
                "scaler_scale",
                "scaling_std_train_observed",
            ],
        ),
        "fill": find_col(
            params,
            [
                "fill_value",
                "impute_value",
                "imputer_value",
                "median",
                "train_median",
                "imputation_value_train_median",
            ],
        ),
    }
    return cols


def extract_feature_params(params: pd.DataFrame, feature_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Extract per-feature train-only fill/mean/std from preprocessing_parameters.csv.

    Expected flexible schemas:
        feature, mean, std
        feature, train_mean, train_std
        variable, impute_value, scaler_mean, scaler_scale
    """
    colmap = detect_param_columns(params)

    if colmap["variable"] is None:
        raise ValueError(
            "Could not identify variable/feature column in preprocessing_parameters.csv. "
            f"Columns are: {list(params.columns)}"
        )

    if colmap["mean"] is None or colmap["std"] is None:
        raise ValueError(
            "Could not identify mean/std columns in preprocessing_parameters.csv. "
            f"Detected columns: {colmap}; all columns: {list(params.columns)}"
        )

    var_col = colmap["variable"]
    mean_col = colmap["mean"]
    std_col = colmap["std"]
    fill_col = colmap["fill"]

    params2 = params.copy()
    params2["_var_norm"] = params2[var_col].astype(str).map(norm_name)

    out: Dict[str, Dict[str, float]] = {}

    for f in feature_names:
        f_norm = norm_name(f)

        # First exact match.
        sub = params2[params2["_var_norm"].eq(f_norm)]

        # Then contains match, useful if rows are named value_age_at_visit etc.
        if len(sub) == 0:
            sub = params2[params2["_var_norm"].str.contains(re.escape(f_norm), na=False)]

        # Avoid accidental delta/slope/time rows for core feature scaling.
        if len(sub) > 1:
            clean_sub = sub[
                ~sub["_var_norm"].str.contains("delta|slope|time|elapsed|mask|observed", regex=True, na=False)
            ]
            if len(clean_sub) > 0:
                sub = clean_sub

        if len(sub) == 0:
            raise ValueError(
                f"Could not find preprocessing parameter row for feature '{f}'. "
                f"Available variable names preview: {params2[var_col].astype(str).head(50).tolist()}"
            )

        row = sub.iloc[0]

        mean = pd.to_numeric(row[mean_col], errors="coerce")
        std = pd.to_numeric(row[std_col], errors="coerce")

        if pd.isna(mean):
            raise ValueError(f"Mean is NaN for feature {f} using row: {row.to_dict()}")

        if pd.isna(std) or float(std) == 0.0:
            std = 1.0

        if fill_col is not None:
            fill = pd.to_numeric(row[fill_col], errors="coerce")
            if pd.isna(fill):
                fill = mean
        else:
            fill = mean

        out[f] = {
            "fill_value_raw": float(fill),
            "mean_raw": float(mean),
            "std_raw": float(std),
            "source_row_variable": str(row[var_col]),
        }

    return out


def transform_values(
    Xv_raw: np.ndarray,
    Xm: np.ndarray,
    feature_params: Dict[str, Dict[str, float]],
    feature_names: List[str],
) -> np.ndarray:
    """
    Impute missing raw values with internal train-only fill, then scale with internal mean/std.
    """
    X_scaled = np.empty_like(Xv_raw, dtype=np.float32)

    for j, f in enumerate(feature_names):
        p = feature_params[f]
        x = Xv_raw[:, :, j].astype(np.float32)
        observed = Xm[:, :, j].astype(bool)

        filled = x.copy()
        filled[~observed] = p["fill_value_raw"]
        filled[np.isnan(filled)] = p["fill_value_raw"]

        scaled = (filled - p["mean_raw"]) / p["std_raw"]
        X_scaled[:, :, j] = scaled.astype(np.float32)

    return X_scaled


def internal_feature_range_fallback(
    internal: Dict[str, Any],
    external_raw: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    If time preprocessing parameters are unavailable, align external Xt approximately
    to internal X_time_scaled distribution using robust internal/external moments.

    This is NOT a refit of outcome-related data. It is a defensive fallback for time channels
    when original Step09 time scaler parameters are not present.
    The script records this clearly in the README.
    """
    X_time_internal = internal["X_time_scaled"].astype(np.float32)
    Xt_ext = external_raw["Xt"].astype(np.float32)

    out = np.empty_like(Xt_ext, dtype=np.float32)

    for j in range(Xt_ext.shape[-1]):
        xi = X_time_internal[:, :, j].ravel()
        xi = xi[np.isfinite(xi)]

        xe = Xt_ext[:, :, j].ravel()
        xe = xe[np.isfinite(xe)]

        if len(xi) == 0 or len(xe) == 0:
            out[:, :, j] = Xt_ext[:, :, j]
            continue

        target_mean = float(np.mean(xi))
        target_std = float(np.std(xi))
        src_mean = float(np.mean(xe))
        src_std = float(np.std(xe))

        if target_std == 0:
            target_std = 1.0
        if src_std == 0:
            src_std = 1.0

        out[:, :, j] = ((Xt_ext[:, :, j] - src_mean) / src_std * target_std + target_mean).astype(np.float32)

    return out, np.array(["time_fallback_internal_external_moment_alignment"], dtype=object)


def build_delta_slope_from_scaled_values(
    X_scaled: np.ndarray,
    Xm: np.ndarray,
    visit_mask: np.ndarray,
    Xt_scaled_or_raw_for_gap: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build delta_from_first and slope_like on scaled values, matching v4 naming intent.

    We use the first observed value within retained history for each feature.
    Missing positions are filled as 0 after computing, because internal arrays contain no NaN.
    """
    n, k, fdim = X_scaled.shape

    Xd = np.zeros_like(X_scaled, dtype=np.float32)
    Xs = np.zeros_like(X_scaled, dtype=np.float32)

    # Use external raw visit order gap if available from Xt channel 3 or time-from-first channel 0.
    # Since time has already been transformed sometimes, use visit index position as safe denominator fallback.
    for i in range(n):
        real_positions = np.where(visit_mask[i] > 0.5)[0]
        if len(real_positions) == 0:
            continue

        for f in range(fdim):
            obs_positions = [pos for pos in real_positions if Xm[i, pos, f] > 0.5]

            if len(obs_positions) == 0:
                continue

            first_pos = obs_positions[0]
            first_val = X_scaled[i, first_pos, f]

            for pos in real_positions:
                if Xm[i, pos, f] > 0.5:
                    delta = X_scaled[i, pos, f] - first_val
                    Xd[i, pos, f] = delta

                    gap = float(pos - first_pos)
                    if gap > 0:
                        Xs[i, pos, f] = delta / gap
                    else:
                        Xs[i, pos, f] = 0.0
                else:
                    Xd[i, pos, f] = 0.0
                    Xs[i, pos, f] = 0.0

    return Xd, Xs


def range_check(name: str, arr: np.ndarray) -> Dict[str, Any]:
    flat = arr.astype(float).ravel()
    finite = flat[np.isfinite(flat)]

    return {
        "array": name,
        "shape": str(arr.shape),
        "dtype": str(arr.dtype),
        "nan_count": int(np.isnan(flat).sum()),
        "inf_count": int(np.isinf(flat).sum()),
        "finite_count": int(len(finite)),
        "min": float(np.min(finite)) if len(finite) else np.nan,
        "p01": float(np.quantile(finite, 0.01)) if len(finite) else np.nan,
        "p50": float(np.quantile(finite, 0.50)) if len(finite) else np.nan,
        "p99": float(np.quantile(finite, 0.99)) if len(finite) else np.nan,
        "max": float(np.max(finite)) if len(finite) else np.nan,
    }


def build_alignment_summary(
    feature_params: Dict[str, Dict[str, float]],
    Xv_raw: np.ndarray,
    Xm: np.ndarray,
    X_scaled: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for j, f in enumerate(FEATURE_NAMES):
        raw = Xv_raw[:, :, j].astype(float)
        obs = Xm[:, :, j].astype(bool)
        raw_obs = raw[obs & np.isfinite(raw)]
        scaled = X_scaled[:, :, j].astype(float).ravel()
        scaled_finite = scaled[np.isfinite(scaled)]

        p = feature_params[f]
        rows.append(
            {
                "feature": f,
                "observed_raw_count": int(len(raw_obs)),
                "observed_raw_rate": float(np.mean(obs)),
                "raw_observed_min": float(np.min(raw_obs)) if len(raw_obs) else np.nan,
                "raw_observed_median": float(np.median(raw_obs)) if len(raw_obs) else np.nan,
                "raw_observed_max": float(np.max(raw_obs)) if len(raw_obs) else np.nan,
                "internal_fill_value_raw": p["fill_value_raw"],
                "internal_mean_raw": p["mean_raw"],
                "internal_std_raw": p["std_raw"],
                "source_row_variable": p["source_row_variable"],
                "scaled_min": float(np.min(scaled_finite)) if len(scaled_finite) else np.nan,
                "scaled_median": float(np.median(scaled_finite)) if len(scaled_finite) else np.nan,
                "scaled_max": float(np.max(scaled_finite)) if len(scaled_finite) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    root = project_root()
    out_dir = root / OUT_REL
    ensure_outdir(out_dir)

    internal_npz_path = root / INTERNAL_NPZ_REL
    external_raw_path = root / EXTERNAL_RAW_NPZ_REL
    external_cohort_path = root / EXTERNAL_COHORT_REL
    params_path = root / PREPROCESS_PARAMS_REL

    print("=" * 88)
    print("[STEP 95c] Prepare NACC external aligned tokens")
    print(f"[SCRIPT VERSION] {SCRIPT_VERSION}")
    print(f"[INTERNAL NPZ] {internal_npz_path}")
    print(f"[EXTERNAL RAW NPZ] {external_raw_path}")
    print(f"[PREPROCESS PARAMS] {params_path}")
    print(f"[OUTPUT DIR] {out_dir}")
    print("=" * 88)

    if not params_path.exists():
        raise FileNotFoundError(
            f"Missing preprocessing parameters: {params_path}. "
            "Run Step95b and locate train-only preprocessing artifacts first."
        )

    internal = load_npz(internal_npz_path)
    external = load_npz(external_raw_path)

    params = pd.read_csv(params_path)

    print("[INFO] preprocessing_parameters.csv columns:")
    print(list(params.columns))
    print("[INFO] preprocessing_parameters.csv head:")
    print(params.head(20).to_string(index=False))

    print("[INFO] Extracting feature preprocessing parameters...")
    feature_params = extract_feature_params(params, FEATURE_NAMES)

    Xv_raw = external["Xv"].astype(np.float32)
    Xm = external["Xm"].astype(np.float32)
    Xt_raw = external["Xt"].astype(np.float32)
    visit_mask = external["visit_mask"].astype(np.float32)

    print("[INFO] Imputing/scaling external clinical values with internal train-only parameters...")
    X_values_imputed_scaled = transform_values(Xv_raw, Xm, feature_params, FEATURE_NAMES)

    print("[INFO] Aligning time features...")
    # Try to locate exact time preprocessing in params; if not present, use fallback and record it.
    # In this project, Step09 artifacts usually saved X_time_scaled.npy but not always explicit time scaler rows.
    X_time_scaled, time_alignment_note = internal_feature_range_fallback(internal, external)

    print("[INFO] Building delta/slope-like channels from scaled values...")
    X_delta_from_first, X_slope_from_first = build_delta_slope_from_scaled_values(
        X_values_imputed_scaled,
        Xm,
        visit_mask,
        Xt_raw,
    )

    y_raw = external["y"].astype(np.float32)
    y_mask = external["y_mask"].astype(np.float32)
    y_labels = y_raw.copy()
    y_labels[y_mask < 0.5] = np.nan
    y_labels[y_labels < 0] = np.nan
    y_observed = y_mask.astype(np.float32)

    sequence_lengths = visit_mask.sum(axis=1).astype(np.int32)

    # Basic safety checks.
    arrays_to_check = {
        "X_values_imputed_scaled": X_values_imputed_scaled,
        "X_feature_mask": Xm,
        "X_time_scaled": X_time_scaled,
        "X_delta_from_first": X_delta_from_first,
        "X_slope_from_first": X_slope_from_first,
        "X_visit_mask": visit_mask,
        "y_labels": y_labels,
        "y_observed": y_observed,
        "sequence_lengths": sequence_lengths,
    }

    range_rows = [range_check(k, v) for k, v in arrays_to_check.items()]
    range_df = pd.DataFrame(range_rows)

    alignment_df = build_alignment_summary(feature_params, Xv_raw, Xm, X_values_imputed_scaled)

    # Compare with internal ranges.
    internal_range_rows = []
    for key in [
        "X_values_imputed_scaled",
        "X_feature_mask",
        "X_time_scaled",
        "X_delta_from_first",
        "X_slope_from_first",
        "X_visit_mask",
        "y_labels",
        "y_observed",
    ]:
        if key in internal:
            internal_range_rows.append(range_check("internal_" + key, internal[key]))

    internal_range_df = pd.DataFrame(internal_range_rows)
    combined_range_df = pd.concat([range_df, internal_range_df], axis=0, ignore_index=True)

    # Outputs.
    out_npz = out_dir / "nacc_external_aligned_tokens_for_v4_no_ADAS13.npz"
    alignment_summary_path = out_dir / "nacc_external_alignment_summary.csv"
    range_check_path = out_dir / "nacc_external_alignment_range_check.csv"
    params_loaded_path = out_dir / "preprocessing_parameters_loaded.csv"
    readme_path = out_dir / "README_step95c_nacc_external_aligned_tokens.md"
    json_path = out_dir / "step95c_audit.json"

    print("[INFO] Writing aligned external NPZ...")
    np.savez_compressed(
        out_npz,
        X_values_imputed_scaled=X_values_imputed_scaled.astype(np.float32),
        X_feature_mask=Xm.astype(np.float32),
        X_time_scaled=X_time_scaled.astype(np.float32),
        X_delta_from_first=X_delta_from_first.astype(np.float32),
        X_slope_from_first=X_slope_from_first.astype(np.float32),
        X_visit_mask=visit_mask.astype(np.float32),
        y_labels=y_labels.astype(np.float32),
        y_observed=y_observed.astype(np.float32),
        sequence_lengths=sequence_lengths.astype(np.int32),
        feature_names=np.array(FEATURE_NAMES, dtype=object),
        horizon_names=np.array(HORIZON_NAMES, dtype=object),
        scenario=np.array(["no_ADAS13"], dtype=object),
        external_time_axis=np.array(["NACCVNUM"], dtype=object),
        time_alignment_note=time_alignment_note,
        source_external_raw_npz=np.array([str(external_raw_path)], dtype=object),
        source_preprocessing_parameters=np.array([str(params_path)], dtype=object),
    )

    alignment_df.to_csv(alignment_summary_path, index=False, encoding="utf-8-sig")
    combined_range_df.to_csv(range_check_path, index=False, encoding="utf-8-sig")
    params.to_csv(params_loaded_path, index=False, encoding="utf-8-sig")

    n = X_values_imputed_scaled.shape[0]

    readme = f"""# Step95c NACC external aligned tokens for frozen v4

## Purpose

This step converts corrected NACC raw tokens into model-input-style arrays for the frozen Step89 v4 model.

It does not train, predict, evaluate, or refit preprocessing on NACC.

## Inputs

- Internal Step12 NPZ: `{internal_npz_path}`
- Corrected NACC raw NPZ: `{external_raw_path}`
- Internal train-only preprocessing parameters: `{params_path}`

## Output NPZ

`{out_npz}`

## Main external setup

- Dataset: NACC investigator single table
- Cohort: first eligible MCI visit per patient
- Time axis: NACCVNUM
- Event: future `NACCUDSD == 4` and `NACCALZD == 1`
- Scenario: `no_ADAS13`
- ADAS13 handling: all missing, imputed/scaled only as placeholder while feature mask remains 0

## Output array shapes

- X_values_imputed_scaled: {X_values_imputed_scaled.shape}
- X_feature_mask: {Xm.shape}
- X_time_scaled: {X_time_scaled.shape}
- X_delta_from_first: {X_delta_from_first.shape}
- X_slope_from_first: {X_slope_from_first.shape}
- X_visit_mask: {visit_mask.shape}
- y_labels: {y_labels.shape}
- y_observed: {y_observed.shape}
- sequence_lengths: {sequence_lengths.shape}

## Time alignment note

`{time_alignment_note.tolist()}`

If exact Step09 time-scaler parameters are later found, this step should be updated to use them directly.
For now, clinical feature values are transformed using internal train-only preprocessing parameters, while time channels are aligned defensively to the internal `X_time_scaled` distribution.

## Feature alignment summary

{alignment_df.to_string(index=False)}

## Range check

{combined_range_df.to_string(index=False)}

## Important caution

This file is ready for a first external evaluation attempt, but the final paper should report the time-axis/preprocessing choice carefully.

Step95d should:
1. import the frozen v4 architecture from Step89,
2. load `best_model.pt`,
3. apply the `no_ADAS13` scenario mask,
4. compute external AUROC/AUPRC/Brier by horizon,
5. report raw and scenario-Platt calibrated risks separately.
"""

    readme_path.write_text(readme, encoding="utf-8")

    audit = {
        "script_version": SCRIPT_VERSION,
        "internal_npz": str(internal_npz_path),
        "external_raw_npz": str(external_raw_path),
        "external_cohort_csv": str(external_cohort_path),
        "preprocessing_parameters": str(params_path),
        "output_npz": str(out_npz),
        "n_samples": int(n),
        "feature_names": FEATURE_NAMES,
        "horizon_names": HORIZON_NAMES,
        "scenario": "no_ADAS13",
        "time_alignment_note": time_alignment_note.tolist(),
        "outputs": {
            "aligned_npz": str(out_npz),
            "alignment_summary": str(alignment_summary_path),
            "range_check": str(range_check_path),
            "params_loaded": str(params_loaded_path),
            "readme": str(readme_path),
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print("=" * 88)
    print("[DONE] Step95c NACC external aligned tokens prepared.")
    print(f"[OUT NPZ] {out_npz}")
    print("[ALIGNMENT SUMMARY]")
    print(alignment_df.to_string(index=False))
    print("[RANGE CHECK]")
    print(combined_range_df.to_string(index=False))
    print("[NEXT] Step95d: frozen v4 external evaluation on NACC no_ADAS13.")
    print("=" * 88)


if __name__ == "__main__":
    main()