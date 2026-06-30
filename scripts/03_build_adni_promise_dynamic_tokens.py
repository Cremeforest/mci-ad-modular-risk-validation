# ---------------------------------------------------------------------------
# AUTO-GENERATED CLEANED DYNAMIC-TOKEN SCRIPT
# Created by scripts/18g_create_cleaned_dynamic_token_script_fixed.py
#
# Original script:
#   12_build_promise_dynamic_tokens.py
#
# Differences:
#   - reads results/features/09_primary_train_only_preprocessing_cleaned
#   - writes results/features/12_promise_dynamic_tokens_cleaned
#
# The original script is not modified.
# ---------------------------------------------------------------------------

# scripts/12_build_promise_dynamic_tokens.py

from pathlib import Path
import json
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_NPZ = PROJECT_ROOT / "results/features/09_primary_train_only_preprocessing_cleaned/primary_preprocessed_k8.npz"
OUT_DIR = PROJECT_ROOT / "results/features/12_promise_dynamic_tokens_cleaned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_NPZ = OUT_DIR / "primary_promise_dynamic_tokens_k8.npz"
OUTPUT_META_JSON = OUT_DIR / "primary_promise_dynamic_tokens_metadata.json"
OUTPUT_AUDIT_CSV = OUT_DIR / "dynamic_token_audit_summary.csv"

HORIZONS = [1, 2, 3, 5]
EPS = 1e-6


def find_key(keys, candidates):
    for c in candidates:
        if c in keys:
            return c

    lower_map = {k.lower(): k for k in keys}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    return None


def require_shape(name, arr, expected_ndim=None):
    if expected_ndim is not None and arr.ndim != expected_ndim:
        raise ValueError(f"{name} expected ndim={expected_ndim}, got shape={arr.shape}")


def get_split_arrays(data, keys, n):
    split_arrays = {}

    for name in ["train_idx", "val_idx", "test_idx"]:
        if name in keys:
            idx = np.asarray(data[name]).astype(int)
            if np.any(idx < 0) or np.any(idx >= n):
                raise ValueError(f"{name} contains out-of-range indices.")
            split_arrays[name] = idx

    return split_arrays


def build_first_valid_values(X_values, X_visit_mask):
    """
    For current sequence_ge2 subset, every sample should have at least one valid visit.
    We still implement this safely.

    X_values: (n, k, f)
    X_visit_mask: (n, k)
    """
    n, k, f = X_values.shape
    first_values = np.zeros((n, f), dtype=X_values.dtype)
    first_pos = np.zeros(n, dtype=np.int64)

    for i in range(n):
        valid_positions = np.where(X_visit_mask[i] > 0.5)[0]
        if len(valid_positions) == 0:
            first_pos[i] = 0
            first_values[i] = 0.0
        else:
            p0 = int(valid_positions[0])
            first_pos[i] = p0
            first_values[i] = X_values[i, p0, :]

    return first_values, first_pos


def build_visit_order_time(X_visit_mask):
    """
    Fallback pseudo-time based on retained visit order.

    For each valid visit, time is 0, 1, 2, ...
    Padding positions keep 0.

    This is NOT true calendar time. It is only a safe fallback if raw visit time
    is unavailable in the current NPZ.
    """
    n, k = X_visit_mask.shape
    t = np.zeros((n, k), dtype=np.float32)

    for i in range(n):
        valid_positions = np.where(X_visit_mask[i] > 0.5)[0]
        for order, pos in enumerate(valid_positions):
            t[i, pos] = float(order)

    return t


def try_get_raw_time(data, keys, X_visit_mask):
    """
    Try to find an unscaled time axis suitable for time-normalized slopes.

    Expected shape:
    - (n, k): one raw time coordinate per visit
    or
    - (n, k, d): use first channel as the raw time coordinate

    If no raw time key exists, return visit-order pseudo-time.
    """
    raw_time_key = find_key(keys, [
        "X_time_raw",
        "X_time_years",
        "X_visit_time_years",
        "X_visit_years",
        "X_age_at_visit_raw",
        "X_age_at_visit_years",
        "visit_time_years",
        "time_since_first_visit_years",
        "time_since_first_mci_years",
        "X_time_unscaled",
    ])

    if raw_time_key is None:
        return build_visit_order_time(X_visit_mask), None, "visit_order_fallback"

    raw = np.asarray(data[raw_time_key]).astype(np.float32)

    if raw.ndim == 2:
        if raw.shape != X_visit_mask.shape:
            raise ValueError(f"Raw time key {raw_time_key} shape {raw.shape} does not match visit mask {X_visit_mask.shape}")
        return raw, raw_time_key, "raw_time_2d"

    if raw.ndim == 3:
        if raw.shape[:2] != X_visit_mask.shape:
            raise ValueError(f"Raw time key {raw_time_key} shape {raw.shape} does not match visit mask {X_visit_mask.shape}")
        return raw[:, :, 0], raw_time_key, "raw_time_3d_first_channel"

    raise ValueError(f"Unsupported raw time key {raw_time_key} shape: {raw.shape}")


def build_delta_and_slope(X_values, X_visit_mask, raw_time_2d):
    """
    Build PROMISE-style dynamic features.

    delta_from_first:
        current scaled/imputed value - first retained visit scaled/imputed value

    slope_from_first:
        delta_from_first / elapsed time from first retained visit

    Important:
    - Padding positions are zeroed.
    - First valid visit has delta=0 and slope=0.
    - If raw time is unavailable, elapsed time is visit-order distance.
    """
    n, k, f = X_values.shape

    first_values, first_pos = build_first_valid_values(X_values, X_visit_mask)

    delta = X_values - first_values[:, None, :]
    delta = delta.astype(np.float32)

    first_time = np.zeros(n, dtype=np.float32)
    for i in range(n):
        first_time[i] = raw_time_2d[i, first_pos[i]]

    elapsed = raw_time_2d - first_time[:, None]
    elapsed = elapsed.astype(np.float32)

    # Avoid division by zero for first visit or duplicated time.
    denom = np.maximum(np.abs(elapsed), EPS)
    slope = delta / denom[:, :, None]

    # First valid visit should be exactly zero, not huge due to EPS.
    for i in range(n):
        slope[i, first_pos[i], :] = 0.0
        delta[i, first_pos[i], :] = 0.0

    valid = (X_visit_mask > 0.5).astype(np.float32)
    delta *= valid[:, :, None]
    slope *= valid[:, :, None]

    # Guard against accidental numerical issues.
    delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    slope = np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return delta, slope, elapsed


def summarize_array(name, arr, valid_mask=None):
    if valid_mask is not None:
        if arr.ndim == 3:
            x = arr[valid_mask > 0.5]
        elif arr.ndim == 2:
            x = arr[valid_mask > 0.5]
        else:
            x = arr
    else:
        x = arr.reshape(-1)

    x = pd.Series(np.asarray(x).reshape(-1)).replace([np.inf, -np.inf], np.nan).dropna()

    if len(x) == 0:
        return {
            "array": name,
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "max": np.nan,
        }

    return {
        "array": name,
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "p25": float(x.quantile(0.25)),
        "median": float(x.median()),
        "p75": float(x.quantile(0.75)),
        "max": float(x.max()),
    }


def main():
    if not INPUT_NPZ.exists():
        raise FileNotFoundError(f"Cannot find input NPZ: {INPUT_NPZ}")

    data = np.load(INPUT_NPZ, allow_pickle=True)
    keys = list(data.keys())

    print("[INFO] Loaded:", INPUT_NPZ)
    print("[INFO] Keys:")
    for k0 in keys:
        arr = data[k0]
        print(f"  - {k0}: shape={getattr(arr, 'shape', None)}, dtype={getattr(arr, 'dtype', None)}")

    value_key = find_key(keys, ["X_values_imputed_scaled", "X_values", "values"])
    time_key = find_key(keys, ["X_time_scaled", "X_time", "time_features"])
    feature_mask_key = find_key(keys, ["X_feature_mask", "feature_mask", "X_mask"])
    visit_mask_key = find_key(keys, ["X_visit_mask", "visit_mask"])
    y_key = find_key(keys, ["y_labels", "y", "labels"])
    y_obs_key = find_key(keys, ["y_observed", "observed_mask", "label_mask"])
    seq_len_key = find_key(keys, ["sequence_lengths", "seq_lengths", "lengths"])

    required = {
        "values": value_key,
        "time": time_key,
        "feature_mask": feature_mask_key,
        "visit_mask": visit_mask_key,
        "labels": y_key,
        "label_observed": y_obs_key,
    }

    missing = [name for name, key in required.items() if key is None]
    if missing:
        raise KeyError(f"Missing required arrays: {missing}. Available keys: {keys}")

    X_values = np.asarray(data[value_key]).astype(np.float32)
    X_time = np.asarray(data[time_key]).astype(np.float32)
    X_feature_mask = np.asarray(data[feature_mask_key]).astype(np.float32)
    X_visit_mask = np.asarray(data[visit_mask_key]).astype(np.float32)
    y_labels = np.asarray(data[y_key]).astype(np.float32)
    y_observed = np.asarray(data[y_obs_key]).astype(np.float32)

    require_shape("X_values", X_values, expected_ndim=3)
    require_shape("X_time", X_time, expected_ndim=3)
    require_shape("X_feature_mask", X_feature_mask, expected_ndim=3)
    require_shape("X_visit_mask", X_visit_mask, expected_ndim=2)

    n, k, f = X_values.shape

    if X_feature_mask.shape != X_values.shape:
        raise ValueError(f"X_feature_mask shape {X_feature_mask.shape} does not match X_values {X_values.shape}")

    if X_time.shape[:2] != (n, k):
        raise ValueError(f"X_time shape {X_time.shape} incompatible with X_values {X_values.shape}")

    if X_visit_mask.shape != (n, k):
        raise ValueError(f"X_visit_mask shape {X_visit_mask.shape} incompatible with X_values {X_values.shape}")

    if y_labels.shape != (n, len(HORIZONS)):
        raise ValueError(f"y_labels expected shape {(n, len(HORIZONS))}, got {y_labels.shape}")

    if y_observed.shape != y_labels.shape:
        raise ValueError(f"y_observed shape {y_observed.shape} does not match y_labels {y_labels.shape}")

    raw_time_2d, raw_time_key, raw_time_mode = try_get_raw_time(data, keys, X_visit_mask)

    X_delta_from_first, X_slope_from_first, X_elapsed_from_first = build_delta_and_slope(
        X_values=X_values,
        X_visit_mask=X_visit_mask,
        raw_time_2d=raw_time_2d,
    )

    X_promise_tokens = np.concatenate(
        [
            X_values,
            X_feature_mask,
            X_time,
            X_delta_from_first,
            X_slope_from_first,
        ],
        axis=-1,
    ).astype(np.float32)

    expected_dim = f + f + X_time.shape[-1] + f + f
    if X_promise_tokens.shape != (n, k, expected_dim):
        raise RuntimeError(f"Unexpected promise token shape: {X_promise_tokens.shape}")

    split_arrays = get_split_arrays(data, keys, n)

    save_dict = {
        "X_promise_tokens": X_promise_tokens,
        "X_values_imputed_scaled": X_values,
        "X_feature_mask": X_feature_mask,
        "X_time_scaled": X_time,
        "X_delta_from_first": X_delta_from_first,
        "X_slope_from_first": X_slope_from_first,
        "X_elapsed_from_first": X_elapsed_from_first.astype(np.float32),
        "X_visit_mask": X_visit_mask,
        "y_labels": y_labels,
        "y_observed": y_observed,
    }

    if seq_len_key is not None:
        save_dict["sequence_lengths"] = np.asarray(data[seq_len_key]).astype(np.int32)

    for name, idx in split_arrays.items():
        save_dict[name] = idx

    np.savez_compressed(OUTPUT_NPZ, **save_dict)

    audit_rows = [
        summarize_array("X_values_imputed_scaled_valid_visits", X_values, X_visit_mask),
        summarize_array("X_time_scaled_valid_visits", X_time, X_visit_mask),
        summarize_array("X_delta_from_first_valid_visits", X_delta_from_first, X_visit_mask),
        summarize_array("X_slope_from_first_valid_visits", X_slope_from_first, X_visit_mask),
        summarize_array("X_elapsed_from_first_valid_visits", X_elapsed_from_first, X_visit_mask),
        summarize_array("X_promise_tokens_valid_visits", X_promise_tokens, X_visit_mask),
    ]

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(OUTPUT_AUDIT_CSV, index=False)

    metadata = {
        "input_npz": str(INPUT_NPZ),
        "output_npz": str(OUTPUT_NPZ),
        "n_landmarks": int(n),
        "max_seq_len": int(k),
        "n_primary_features": int(f),
        "n_time_features": int(X_time.shape[-1]),
        "token_components": {
            "values": int(f),
            "feature_masks": int(f),
            "time_features": int(X_time.shape[-1]),
            "delta_from_first": int(f),
            "slope_from_first": int(f),
        },
        "final_token_dim": int(X_promise_tokens.shape[-1]),
        "horizons_years": HORIZONS,
        "raw_time_key_used_for_slope": raw_time_key,
        "raw_time_mode": raw_time_mode,
        "slope_warning": (
            "If raw_time_key_used_for_slope is None, slope_from_first uses retained visit order "
            "rather than true calendar-time-normalized slope. This is leakage-safe but should be "
            "described as visit-order-normalized slope until raw landmark visit times are saved."
        ),
        "leakage_safety_notes": [
            "All dynamic features are computed only within the retained pre-landmark sequence.",
            "Delta is computed from the first retained visit to each retained visit.",
            "No post-landmark visits, diagnosis columns, future outcome time, or years_to_outcome are used.",
            "Values are based on train-only imputed/scaled arrays from Step 09.",
        ],
        "output_arrays": {
            "X_promise_tokens": list(X_promise_tokens.shape),
            "X_values_imputed_scaled": list(X_values.shape),
            "X_feature_mask": list(X_feature_mask.shape),
            "X_time_scaled": list(X_time.shape),
            "X_delta_from_first": list(X_delta_from_first.shape),
            "X_slope_from_first": list(X_slope_from_first.shape),
            "X_elapsed_from_first": list(X_elapsed_from_first.shape),
            "X_visit_mask": list(X_visit_mask.shape),
            "y_labels": list(y_labels.shape),
            "y_observed": list(y_observed.shape),
        },
    }

    with open(OUTPUT_META_JSON, "w", encoding="utf-8") as f_json:
        json.dump(metadata, f_json, indent=2)

    print("[DONE] PROMISE-style dynamic token file saved:")
    print(OUTPUT_NPZ)
    print("")
    print("[INFO] Token shape:")
    print("  X_promise_tokens:", X_promise_tokens.shape)
    print("")
    print("[INFO] Component dims:")
    print(f"  values: {f}")
    print(f"  masks: {f}")
    print(f"  time: {X_time.shape[-1]}")
    print(f"  delta: {f}")
    print(f"  slope: {f}")
    print(f"  total: {X_promise_tokens.shape[-1]}")
    print("")
    print("[INFO] Slope time source:")
    print(f"  raw_time_key: {raw_time_key}")
    print(f"  raw_time_mode: {raw_time_mode}")
    print("")
    print("[NEXT] Check:")
    print(OUTPUT_META_JSON)
    print(OUTPUT_AUDIT_CSV)


if __name__ == "__main__":
    main()