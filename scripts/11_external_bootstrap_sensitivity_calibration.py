# -*- coding: utf-8 -*-
"""
Step95e: NACC external bootstrap CI, calibration audit, and sequence-length sensitivity.

Purpose
-------
Post-process Step95d frozen v4 NACC external predictions.

Inputs
------
results/reports/95d_nacc_external_frozen_v4_no_adas13_eval/
    external_predictions_no_ADAS13.csv
    external_metrics_raw_and_platt_no_ADAS13.csv

Optional:
results/features/95c_nacc_external_aligned_tokens/
    nacc_external_aligned_tokens_for_v4_no_ADAS13.npz

Outputs
-------
results/reports/95e_nacc_external_bootstrap_ci_sensitivity_calibration/
    external_sensitivity_metrics_by_sequence_length.csv
    external_bootstrap_ci_by_sequence_length.csv
    external_calibration_summary_by_sequence_length.csv
    external_calibration_bins_by_sequence_length.csv
    external_sequence_length_distribution.csv
    README_step95e_nacc_external_bootstrap_ci_sensitivity_calibration.md
    step95e_audit.json

Notes
-----
- This step does not train, refit, or re-predict.
- The main analysis remains NACC first-MCI / no_ADAS13 / frozen v4.
- Bootstrap is row-level. Because Step94d used first MCI per patient, this is equivalent to patient-level bootstrap if NACCID is unique.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


SCRIPT_VERSION = "v1_nacc_external_bootstrap_ci_sensitivity_calibration"

PRED_REL = (
    Path("results")
    / "reports"
    / "95d_nacc_external_frozen_v4_no_adas13_eval"
    / "external_predictions_no_ADAS13.csv"
)

METRICS_REL = (
    Path("results")
    / "reports"
    / "95d_nacc_external_frozen_v4_no_adas13_eval"
    / "external_metrics_raw_and_platt_no_ADAS13.csv"
)

ALIGNED_NPZ_REL = (
    Path("results")
    / "features"
    / "95c_nacc_external_aligned_tokens"
    / "nacc_external_aligned_tokens_for_v4_no_ADAS13.npz"
)

OUT_REL = Path("results") / "reports" / "95e_nacc_external_bootstrap_ci_sensitivity_calibration"

HORIZONS = ["1y", "2y", "3y", "5y"]
HORIZON_YEARS = {"1y": 1, "2y": 2, "3y": 3, "5y": 5}

RISK_TYPES = {
    "raw_sigmoid": "risk_raw",
    "scenario_platt": "risk_platt",
}

SEQ_GROUPS = [
    ("all_seq_len_ge_1", 1),
    ("seq_len_ge_2", 2),
    ("seq_len_ge_3", 3),
    ("seq_len_ge_4", 4),
]

N_BOOT = 1000
RANDOM_SEED = 42
N_CAL_BINS = 10


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_metric(metric_name: str, y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan

    try:
        if metric_name == "AUROC":
            return float(roc_auc_score(y, p))
        if metric_name == "AUPRC":
            return float(average_precision_score(y, p))
        if metric_name == "Brier":
            return float(brier_score_loss(y, p))
    except Exception:
        return np.nan

    raise ValueError(metric_name)


def metric_block(df: pd.DataFrame, group_name: str, risk_type: str, risk_prefix: str) -> pd.DataFrame:
    rows = []

    for h in HORIZONS:
        y_col = f"y_{h}"
        known_col = f"known_{h}"
        p_col = f"{risk_prefix}_{h}"

        mask = (
            (df[known_col].astype(float) > 0.5)
            & df[y_col].notna()
            & df[p_col].notna()
        )
        sub = df.loc[mask, [y_col, p_col]].copy()

        y = sub[y_col].astype(int).to_numpy()
        p = sub[p_col].astype(float).to_numpy()

        n = int(len(y))
        n_event = int(np.sum(y == 1))
        n_nonevent = int(np.sum(y == 0))

        rows.append(
            {
                "dataset": "NACC_external",
                "scenario": "no_ADAS13",
                "seq_group": group_name,
                "risk_type": risk_type,
                "horizon": h,
                "horizon_year": HORIZON_YEARS[h],
                "n_known": n,
                "n_event": n_event,
                "n_nonevent": n_nonevent,
                "event_rate": n_event / n if n else np.nan,
                "AUROC": safe_metric("AUROC", y, p),
                "AUPRC": safe_metric("AUPRC", y, p),
                "Brier": safe_metric("Brier", y, p),
                "mean_predicted_risk": float(np.mean(p)) if n else np.nan,
                "median_predicted_risk": float(np.median(p)) if n else np.nan,
            }
        )

    return pd.DataFrame(rows)


def make_sensitivity_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    all_rows = []

    for group_name, min_len in SEQ_GROUPS:
        sub = pred[pred["sequence_length_for_sensitivity"] >= min_len].copy()

        for risk_type, risk_prefix in RISK_TYPES.items():
            all_rows.append(metric_block(sub, group_name, risk_type, risk_prefix))

    return pd.concat(all_rows, axis=0, ignore_index=True)


def bootstrap_ci_for_one(
    df: pd.DataFrame,
    group_name: str,
    risk_type: str,
    risk_prefix: str,
    horizon: str,
    rng: np.random.Generator,
    n_boot: int,
) -> List[Dict[str, Any]]:
    y_col = f"y_{horizon}"
    known_col = f"known_{horizon}"
    p_col = f"{risk_prefix}_{horizon}"

    mask = (
        (df[known_col].astype(float) > 0.5)
        & df[y_col].notna()
        & df[p_col].notna()
    )
    sub = df.loc[mask, [y_col, p_col]].copy()

    y = sub[y_col].astype(int).to_numpy()
    p = sub[p_col].astype(float).to_numpy()
    n = len(y)

    point = {
        "AUROC": safe_metric("AUROC", y, p),
        "AUPRC": safe_metric("AUPRC", y, p),
        "Brier": safe_metric("Brier", y, p),
    }

    boot_values = {
        "AUROC": [],
        "AUPRC": [],
        "Brier": [],
    }

    if n == 0:
        return [
            {
                "dataset": "NACC_external",
                "scenario": "no_ADAS13",
                "seq_group": group_name,
                "risk_type": risk_type,
                "horizon": horizon,
                "horizon_year": HORIZON_YEARS[horizon],
                "metric": m,
                "n_known": 0,
                "point": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n_boot_requested": n_boot,
                "n_boot_valid": 0,
            }
            for m in ["AUROC", "AUPRC", "Brier"]
        ]

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        pp = p[idx]

        # AUROC/AUPRC need both classes. Brier does not.
        for m in ["AUROC", "AUPRC", "Brier"]:
            val = safe_metric(m, yy, pp)
            if np.isfinite(val):
                boot_values[m].append(val)

    rows = []
    for m in ["AUROC", "AUPRC", "Brier"]:
        vals = np.array(boot_values[m], dtype=float)
        if len(vals) > 0:
            ci_low = float(np.quantile(vals, 0.025))
            ci_high = float(np.quantile(vals, 0.975))
        else:
            ci_low = np.nan
            ci_high = np.nan

        rows.append(
            {
                "dataset": "NACC_external",
                "scenario": "no_ADAS13",
                "seq_group": group_name,
                "risk_type": risk_type,
                "horizon": horizon,
                "horizon_year": HORIZON_YEARS[horizon],
                "metric": m,
                "n_known": int(n),
                "n_event": int(np.sum(y == 1)),
                "n_nonevent": int(np.sum(y == 0)),
                "point": point[m],
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_boot_requested": n_boot,
                "n_boot_valid": int(len(vals)),
            }
        )

    return rows


def make_bootstrap_ci(pred: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    total_jobs = len(SEQ_GROUPS) * len(RISK_TYPES) * len(HORIZONS)
    job = 0

    for group_name, min_len in SEQ_GROUPS:
        sub = pred[pred["sequence_length_for_sensitivity"] >= min_len].copy()

        for risk_type, risk_prefix in RISK_TYPES.items():
            for h in HORIZONS:
                job += 1
                print(f"[BOOT] {job}/{total_jobs}: group={group_name}, risk={risk_type}, horizon={h}, n={len(sub)}")
                rows.extend(
                    bootstrap_ci_for_one(
                        sub,
                        group_name=group_name,
                        risk_type=risk_type,
                        risk_prefix=risk_prefix,
                        horizon=h,
                        rng=rng,
                        n_boot=n_boot,
                    )
                )

    return pd.DataFrame(rows)


def calibration_summary_one(
    df: pd.DataFrame,
    group_name: str,
    risk_type: str,
    risk_prefix: str,
    horizon: str,
    n_bins: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    y_col = f"y_{horizon}"
    known_col = f"known_{horizon}"
    p_col = f"{risk_prefix}_{horizon}"

    mask = (
        (df[known_col].astype(float) > 0.5)
        & df[y_col].notna()
        & df[p_col].notna()
    )
    sub = df.loc[mask, [y_col, p_col]].copy()
    sub.columns = ["y", "p"]

    if len(sub) == 0:
        summary = {
            "dataset": "NACC_external",
            "scenario": "no_ADAS13",
            "seq_group": group_name,
            "risk_type": risk_type,
            "horizon": horizon,
            "horizon_year": HORIZON_YEARS[horizon],
            "n_known": 0,
            "observed_event_rate": np.nan,
            "mean_predicted_risk": np.nan,
            "calibration_gap_mean_pred_minus_observed": np.nan,
            "ECE_quantile_bins": np.nan,
            "max_abs_bin_gap": np.nan,
            "Brier": np.nan,
        }
        return summary, pd.DataFrame()

    y = sub["y"].astype(int).to_numpy()
    p = sub["p"].astype(float).to_numpy()

    try:
        sub["bin"] = pd.qcut(sub["p"], q=n_bins, duplicates="drop")
    except Exception:
        sub["bin"] = pd.cut(sub["p"], bins=n_bins)

    bin_rows = []
    ece = 0.0
    max_gap = 0.0

    for b, g in sub.groupby("bin", observed=True):
        n_b = len(g)
        pred_b = float(g["p"].mean())
        obs_b = float(g["y"].mean())
        gap_b = pred_b - obs_b
        abs_gap_b = abs(gap_b)

        ece += (n_b / len(sub)) * abs_gap_b
        max_gap = max(max_gap, abs_gap_b)

        bin_rows.append(
            {
                "dataset": "NACC_external",
                "scenario": "no_ADAS13",
                "seq_group": group_name,
                "risk_type": risk_type,
                "horizon": horizon,
                "horizon_year": HORIZON_YEARS[horizon],
                "bin": str(b),
                "n": int(n_b),
                "mean_predicted_risk": pred_b,
                "observed_event_rate": obs_b,
                "gap_pred_minus_observed": gap_b,
                "abs_gap": abs_gap_b,
                "min_predicted_risk": float(g["p"].min()),
                "max_predicted_risk": float(g["p"].max()),
            }
        )

    summary = {
        "dataset": "NACC_external",
        "scenario": "no_ADAS13",
        "seq_group": group_name,
        "risk_type": risk_type,
        "horizon": horizon,
        "horizon_year": HORIZON_YEARS[horizon],
        "n_known": int(len(sub)),
        "observed_event_rate": float(np.mean(y)),
        "mean_predicted_risk": float(np.mean(p)),
        "median_predicted_risk": float(np.median(p)),
        "calibration_gap_mean_pred_minus_observed": float(np.mean(p) - np.mean(y)),
        "ECE_quantile_bins": float(ece),
        "max_abs_bin_gap": float(max_gap),
        "Brier": safe_metric("Brier", y, p),
    }

    return summary, pd.DataFrame(bin_rows)


def make_calibration_tables(pred: pd.DataFrame, n_bins: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    bins = []

    for group_name, min_len in SEQ_GROUPS:
        sub = pred[pred["sequence_length_for_sensitivity"] >= min_len].copy()

        for risk_type, risk_prefix in RISK_TYPES.items():
            for h in HORIZONS:
                s, b = calibration_summary_one(
                    sub,
                    group_name=group_name,
                    risk_type=risk_type,
                    risk_prefix=risk_prefix,
                    horizon=h,
                    n_bins=n_bins,
                )
                summaries.append(s)
                if len(b):
                    bins.append(b)

    summary_df = pd.DataFrame(summaries)
    bins_df = pd.concat(bins, axis=0, ignore_index=True) if bins else pd.DataFrame()

    return summary_df, bins_df


def sequence_length_distribution(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []

    vc = pred["sequence_length_for_sensitivity"].value_counts(dropna=False).sort_index()
    total = len(pred)

    for length, count in vc.items():
        rows.append(
            {
                "sequence_length": length,
                "n": int(count),
                "rate": int(count) / total if total else np.nan,
            }
        )

    for group_name, min_len in SEQ_GROUPS:
        n = int((pred["sequence_length_for_sensitivity"] >= min_len).sum())
        rows.append(
            {
                "sequence_length": group_name,
                "n": n,
                "rate": n / total if total else np.nan,
            }
        )

    return pd.DataFrame(rows)


def attach_sequence_length(pred: pd.DataFrame, aligned_npz_path: Path) -> pd.DataFrame:
    pred = pred.copy()

    if "n_visits_in_window" in pred.columns:
        pred["sequence_length_for_sensitivity"] = pd.to_numeric(
            pred["n_visits_in_window"], errors="coerce"
        )
        return pred

    if "sequence_lengths" in pred.columns:
        pred["sequence_length_for_sensitivity"] = pd.to_numeric(
            pred["sequence_lengths"], errors="coerce"
        )
        return pred

    if aligned_npz_path.exists():
        data = np.load(aligned_npz_path, allow_pickle=True)
        if "sequence_lengths" in data.files:
            seq = data["sequence_lengths"]
            if len(seq) == len(pred):
                pred["sequence_length_for_sensitivity"] = seq.astype(int)
                return pred

    raise RuntimeError(
        "Could not find sequence length. Need n_visits_in_window in predictions "
        "or sequence_lengths in aligned NPZ."
    )


def main() -> None:
    root = project_root()
    out_dir = root / OUT_REL
    ensure_outdir(out_dir)

    pred_path = root / PRED_REL
    metrics_path = root / METRICS_REL
    aligned_npz_path = root / ALIGNED_NPZ_REL

    print("=" * 88)
    print("[STEP 95e] NACC external bootstrap CI + sensitivity + calibration")
    print(f"[SCRIPT VERSION] {SCRIPT_VERSION}")
    print(f"[PREDICTIONS] {pred_path}")
    print(f"[METRICS] {metrics_path}")
    print(f"[ALIGNED NPZ] {aligned_npz_path}")
    print(f"[OUTPUT DIR] {out_dir}")
    print("=" * 88)

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing Step95d predictions: {pred_path}")

    pred = pd.read_csv(pred_path)
    pred = attach_sequence_length(pred, aligned_npz_path)

    n_total = len(pred)
    n_patients = pred["NACCID"].nunique() if "NACCID" in pred.columns else np.nan
    n_duplicate_patients = (
        int(pred["NACCID"].duplicated().sum()) if "NACCID" in pred.columns else np.nan
    )

    print(f"[INFO] predictions rows: {n_total}")
    print(f"[INFO] unique NACCID: {n_patients}")
    print(f"[INFO] duplicated NACCID rows: {n_duplicate_patients}")

    print("[INFO] Computing sensitivity metrics...")
    sensitivity = make_sensitivity_metrics(pred)

    print("[INFO] Computing calibration tables...")
    cal_summary, cal_bins = make_calibration_tables(pred, n_bins=N_CAL_BINS)

    print("[INFO] Computing sequence length distribution...")
    seq_dist = sequence_length_distribution(pred)

    print(f"[INFO] Computing bootstrap CIs with N_BOOT={N_BOOT}...")
    boot = make_bootstrap_ci(pred, n_boot=N_BOOT, seed=RANDOM_SEED)

    sensitivity_path = out_dir / "external_sensitivity_metrics_by_sequence_length.csv"
    boot_path = out_dir / "external_bootstrap_ci_by_sequence_length.csv"
    cal_summary_path = out_dir / "external_calibration_summary_by_sequence_length.csv"
    cal_bins_path = out_dir / "external_calibration_bins_by_sequence_length.csv"
    seq_dist_path = out_dir / "external_sequence_length_distribution.csv"
    readme_path = out_dir / "README_step95e_nacc_external_bootstrap_ci_sensitivity_calibration.md"
    audit_path = out_dir / "step95e_audit.json"

    sensitivity.to_csv(sensitivity_path, index=False, encoding="utf-8-sig")
    boot.to_csv(boot_path, index=False, encoding="utf-8-sig")
    cal_summary.to_csv(cal_summary_path, index=False, encoding="utf-8-sig")
    cal_bins.to_csv(cal_bins_path, index=False, encoding="utf-8-sig")
    seq_dist.to_csv(seq_dist_path, index=False, encoding="utf-8-sig")

    main_sens = sensitivity[
        (sensitivity["seq_group"] == "all_seq_len_ge_1")
        & (sensitivity["risk_type"].isin(["raw_sigmoid", "scenario_platt"]))
    ].copy()

    main_boot = boot[
        (boot["seq_group"] == "all_seq_len_ge_1")
        & (boot["risk_type"].isin(["raw_sigmoid", "scenario_platt"]))
    ].copy()

    readme = f"""# Step95e NACC external bootstrap CI, calibration, and sequence-length sensitivity

## Purpose

This step post-processes Step95d external predictions from the frozen ADNI-trained v4 model on the NACC first-MCI cohort.

No model training, refitting, or re-prediction is performed.

## Main external setting

- Dataset: NACC investigator single table
- Cohort: first eligible MCI visit per patient
- Scenario: no_ADAS13
- Time axis: NACCVNUM
- Model: frozen Step89 v4
- Samples: {n_total}
- Unique NACCID: {n_patients}
- Duplicated NACCID rows: {n_duplicate_patients}

Because Step94d used first-MCI per patient, row-level bootstrap is equivalent to patient-level bootstrap if `duplicated NACCID rows = 0`.

## Sequence-length distribution

{seq_dist.to_string(index=False)}

## Main sensitivity metrics

{main_sens.to_string(index=False)}

## Main bootstrap CI rows

{main_boot.to_string(index=False)}

## Calibration summary

{cal_summary[cal_summary["seq_group"].eq("all_seq_len_ge_1")].to_string(index=False)}

## Interpretation guidance

The main external result should be reported as:

Frozen v4 external validation on a NACC first-MCI cohort under the no_ADAS13 scenario.

Key caveats:
1. NACC does not provide ADAS13 in the investigator table, so the external analysis is no_ADAS13 rather than full-module.
2. Many NACC first-MCI patients have limited pre-landmark history, so this is a first-landmark / low-history external setting.
3. AUROC/AUPRC reflect discrimination; calibration should be interpreted separately.
4. Scenario-Platt calibration was fit on ADNI validation data, not NACC.

## Output files

- `external_sensitivity_metrics_by_sequence_length.csv`
- `external_bootstrap_ci_by_sequence_length.csv`
- `external_calibration_summary_by_sequence_length.csv`
- `external_calibration_bins_by_sequence_length.csv`
- `external_sequence_length_distribution.csv`
- `README_step95e_nacc_external_bootstrap_ci_sensitivity_calibration.md`
- `step95e_audit.json`
"""

    readme_path.write_text(readme, encoding="utf-8")

    audit = {
        "script_version": SCRIPT_VERSION,
        "predictions_csv": str(pred_path),
        "metrics_csv": str(metrics_path),
        "aligned_npz": str(aligned_npz_path),
        "output_dir": str(out_dir),
        "n_predictions": int(n_total),
        "n_unique_NACCID": None if pd.isna(n_patients) else int(n_patients),
        "n_duplicate_NACCID_rows": None if pd.isna(n_duplicate_patients) else int(n_duplicate_patients),
        "n_boot": N_BOOT,
        "random_seed": RANDOM_SEED,
        "n_cal_bins": N_CAL_BINS,
        "outputs": {
            "sensitivity_metrics": str(sensitivity_path),
            "bootstrap_ci": str(boot_path),
            "calibration_summary": str(cal_summary_path),
            "calibration_bins": str(cal_bins_path),
            "sequence_length_distribution": str(seq_dist_path),
            "readme": str(readme_path),
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print("=" * 88)
    print("[DONE] Step95e finished.")
    print("[SEQUENCE LENGTH DISTRIBUTION]")
    print(seq_dist.to_string(index=False))
    print("[SENSITIVITY METRICS]")
    print(sensitivity.to_string(index=False))
    print("[BOOTSTRAP CI - MAIN ALL SEQ LEN]")
    print(main_boot.to_string(index=False))
    print("[CALIBRATION SUMMARY - MAIN ALL SEQ LEN]")
    print(cal_summary[cal_summary["seq_group"].eq("all_seq_len_ge_1")].to_string(index=False))
    print("[OUTPUTS]")
    print(f"  sensitivity: {sensitivity_path}")
    print(f"  bootstrap: {boot_path}")
    print(f"  calibration summary: {cal_summary_path}")
    print(f"  calibration bins: {cal_bins_path}")
    print(f"  readme: {readme_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()