# -*- coding: utf-8 -*-
"""
Step96: NACC cross-fitted local recalibration audit.

Purpose
-------
Evaluate whether local NACC recalibration can improve absolute risk calibration
without retraining the frozen ADNI-trained v4 prediction model.

Input
-----
Step95d prediction file:
results/reports/95d_nacc_external_frozen_v4_no_adas13_eval/
    external_predictions_no_ADAS13.csv

This file should contain:
    y_1y, known_1y, risk_raw_1y, logit_raw_1y, risk_platt_1y
    ...
    y_5y, known_5y, risk_raw_5y, logit_raw_5y, risk_platt_5y

Outputs
-------
results/reports/96_nacc_crossfit_local_recalibration_audit/
    nacc_crossfit_recalibrated_predictions.csv
    nacc_crossfit_recalibration_metrics.csv
    nacc_crossfit_recalibration_bootstrap_ci.csv
    nacc_crossfit_recalibration_calibration_summary.csv
    nacc_crossfit_recalibration_calibration_bins.csv
    nacc_crossfit_recalibration_parameters.csv
    README_step96_nacc_crossfit_local_recalibration_audit.md
    step96_audit.json

Important
---------
- This does NOT retrain the v4 model.
- This does NOT update model weights.
- This only calibrates output probabilities in a cross-fitted held-out manner.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import KFold


SCRIPT_VERSION = "v1_nacc_crossfit_local_recalibration_audit"

PRED_REL = (
    Path("results")
    / "reports"
    / "95d_nacc_external_frozen_v4_no_adas13_eval"
    / "external_predictions_no_ADAS13.csv"
)

OUT_REL = Path("results") / "reports" / "96_nacc_crossfit_local_recalibration_audit"

HORIZONS = ["1y", "2y", "3y", "5y"]
HORIZON_YEARS = {"1y": 1, "2y": 2, "3y": 3, "5y": 5}

N_SPLITS = 5
RANDOM_SEED = 42
N_BOOT = 1000
N_CAL_BINS = 10

REFERENCE_METHODS = {
    "raw_sigmoid": "risk_raw",
    "adni_scenario_platt": "risk_platt",
}

LOCAL_METHODS = [
    "nacc_intercept_only_raw",
    "nacc_platt_raw",
    "nacc_isotonic_raw",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(float), -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def safe_metric(metric_name: str, y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    p = np.asarray(p)

    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask].astype(int)
    p = p[mask].astype(float)

    if len(y) == 0:
        return np.nan

    try:
        if metric_name == "Brier":
            return float(brier_score_loss(y, p))

        if len(np.unique(y)) < 2:
            return np.nan

        if metric_name == "AUROC":
            return float(roc_auc_score(y, p))

        if metric_name == "AUPRC":
            return float(average_precision_score(y, p))
    except Exception:
        return np.nan

    raise ValueError(metric_name)


def fit_intercept_only_offset(logits_train: np.ndarray, y_train: np.ndarray) -> float:
    """
    Fit b in:
        logit(p_new) = logit(p_old) + b

    This is a one-parameter logistic calibration.
    It preserves ranking and therefore preserves AUROC/AUPRC.
    """
    logits_train = np.asarray(logits_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    mask = np.isfinite(logits_train) & np.isfinite(y_train)
    x = logits_train[mask]
    y = y_train[mask]

    if len(y) < 20 or len(np.unique(y.astype(int))) < 2:
        return 0.0

    b = 0.0

    for _ in range(100):
        p = sigmoid(x + b)
        grad = np.sum(p - y)
        hess = np.sum(p * (1.0 - p))

        if hess <= 1e-12:
            break

        step = grad / hess
        b_new = b - step

        if not np.isfinite(b_new):
            break

        if abs(b_new - b) < 1e-8:
            b = b_new
            break

        b = b_new

    return float(b)


def fit_platt(logits_train: np.ndarray, y_train: np.ndarray) -> Tuple[float, float, str]:
    """
    Fit:
        logit(p_new) = a * logit(p_old) + b
    """
    logits_train = np.asarray(logits_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)

    mask = np.isfinite(logits_train) & np.isfinite(y_train)
    x = logits_train[mask].reshape(-1, 1)
    y = y_train[mask]

    if len(y) < 20 or len(np.unique(y)) < 2:
        return 1.0, 0.0, "fallback_identity_insufficient_classes"

    try:
        clf = LogisticRegression(
            solver="lbfgs",
            C=1e6,
            max_iter=2000,
            random_state=RANDOM_SEED,
        )
        clf.fit(x, y)
        a = float(clf.coef_[0, 0])
        b = float(clf.intercept_[0])
        status = "ok"
        return a, b, status
    except Exception as e:
        return 1.0, 0.0, f"fallback_identity_error_{repr(e)}"


def fit_isotonic(risk_train: np.ndarray, y_train: np.ndarray):
    """
    Fit non-parametric monotone calibration:
        p_new = f(p_old)
    """
    risk_train = np.asarray(risk_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    mask = np.isfinite(risk_train) & np.isfinite(y_train)
    x = risk_train[mask]
    y = y_train[mask]

    if len(y) < 20 or len(np.unique(y.astype(int))) < 2 or len(np.unique(x)) < 2:
        return None, "fallback_identity_insufficient_data"

    try:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(x, y)
        return iso, "ok"
    except Exception as e:
        return None, f"fallback_identity_error_{repr(e)}"


def assign_folds(pred: pd.DataFrame) -> np.ndarray:
    """
    Assign row-level folds. Since Step95e confirmed first-MCI per patient and no duplicated NACCID,
    row-level folds are patient-level folds.
    """
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    folds = np.zeros(len(pred), dtype=int)

    for fold_id, (_, test_idx) in enumerate(kf.split(np.arange(len(pred)))):
        folds[test_idx] = fold_id

    return folds


def crossfit_recalibration(pred: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred_out = pred.copy()
    pred_out["recalibration_fold"] = assign_folds(pred_out)

    param_rows: List[Dict[str, Any]] = []

    for h in HORIZONS:
        y_col = f"y_{h}"
        known_col = f"known_{h}"
        raw_col = f"risk_raw_{h}"
        logit_col = f"logit_raw_{h}"

        if logit_col not in pred_out.columns:
            pred_out[logit_col] = logit(pred_out[raw_col].to_numpy(dtype=float))

        for method in LOCAL_METHODS:
            pred_out[f"risk_{method}_{h}"] = np.nan

        for fold in range(N_SPLITS):
            train_mask = pred_out["recalibration_fold"] != fold
            test_mask = pred_out["recalibration_fold"] == fold

            fit_mask = (
                train_mask
                & (pd.to_numeric(pred_out[known_col], errors="coerce") > 0.5)
                & pred_out[y_col].notna()
                & pred_out[raw_col].notna()
                & pred_out[logit_col].notna()
            )

            test_any_mask = test_mask & pred_out[raw_col].notna() & pred_out[logit_col].notna()

            y_train = pred_out.loc[fit_mask, y_col].astype(int).to_numpy()
            raw_train = pred_out.loc[fit_mask, raw_col].astype(float).to_numpy()
            logit_train = pred_out.loc[fit_mask, logit_col].astype(float).to_numpy()

            raw_test = pred_out.loc[test_any_mask, raw_col].astype(float).to_numpy()
            logit_test = pred_out.loc[test_any_mask, logit_col].astype(float).to_numpy()

            n_fit = int(len(y_train))
            n_event = int(np.sum(y_train == 1)) if n_fit else 0
            n_nonevent = int(np.sum(y_train == 0)) if n_fit else 0

            # 1) Intercept-only
            offset = fit_intercept_only_offset(logit_train, y_train)
            p_intercept = sigmoid(logit_test + offset)
            pred_out.loc[test_any_mask, f"risk_nacc_intercept_only_raw_{h}"] = p_intercept

            param_rows.append(
                {
                    "horizon": h,
                    "horizon_year": HORIZON_YEARS[h],
                    "fold": fold,
                    "method": "nacc_intercept_only_raw",
                    "n_fit": n_fit,
                    "n_event": n_event,
                    "n_nonevent": n_nonevent,
                    "coef": 1.0,
                    "intercept": offset,
                    "status": "ok",
                }
            )

            # 2) Platt
            coef, intercept, status = fit_platt(logit_train, y_train)
            p_platt = sigmoid(coef * logit_test + intercept)
            pred_out.loc[test_any_mask, f"risk_nacc_platt_raw_{h}"] = p_platt

            param_rows.append(
                {
                    "horizon": h,
                    "horizon_year": HORIZON_YEARS[h],
                    "fold": fold,
                    "method": "nacc_platt_raw",
                    "n_fit": n_fit,
                    "n_event": n_event,
                    "n_nonevent": n_nonevent,
                    "coef": coef,
                    "intercept": intercept,
                    "status": status,
                }
            )

            # 3) Isotonic
            iso, iso_status = fit_isotonic(raw_train, y_train)

            if iso is None:
                p_iso = raw_test.copy()
            else:
                p_iso = iso.predict(raw_test)

            p_iso = np.clip(p_iso, 1e-6, 1 - 1e-6)
            pred_out.loc[test_any_mask, f"risk_nacc_isotonic_raw_{h}"] = p_iso

            param_rows.append(
                {
                    "horizon": h,
                    "horizon_year": HORIZON_YEARS[h],
                    "fold": fold,
                    "method": "nacc_isotonic_raw",
                    "n_fit": n_fit,
                    "n_event": n_event,
                    "n_nonevent": n_nonevent,
                    "coef": np.nan,
                    "intercept": np.nan,
                    "status": iso_status,
                }
            )

    params = pd.DataFrame(param_rows)
    return pred_out, params


def method_to_probability_column(method: str, horizon: str) -> str:
    if method == "raw_sigmoid":
        return f"risk_raw_{horizon}"
    if method == "adni_scenario_platt":
        return f"risk_platt_{horizon}"
    if method in LOCAL_METHODS:
        return f"risk_{method}_{horizon}"
    raise ValueError(method)


def compute_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    methods = list(REFERENCE_METHODS.keys()) + LOCAL_METHODS
    rows = []

    for method in methods:
        for h in HORIZONS:
            y_col = f"y_{h}"
            known_col = f"known_{h}"
            p_col = method_to_probability_column(method, h)

            mask = (
                (pd.to_numeric(pred[known_col], errors="coerce") > 0.5)
                & pred[y_col].notna()
                & pred[p_col].notna()
            )

            y = pred.loc[mask, y_col].astype(int).to_numpy()
            p = pred.loc[mask, p_col].astype(float).to_numpy()

            n = int(len(y))
            n_event = int(np.sum(y == 1))
            n_nonevent = int(np.sum(y == 0))

            rows.append(
                {
                    "dataset": "NACC_external",
                    "scenario": "no_ADAS13",
                    "method": method,
                    "method_type": "reference" if method in REFERENCE_METHODS else "crossfit_local_recalibration",
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
                    "calibration_gap_mean_pred_minus_observed": float(np.mean(p) - np.mean(y)) if n else np.nan,
                }
            )

    return pd.DataFrame(rows)


def calibration_summary_and_bins(pred: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    methods = list(REFERENCE_METHODS.keys()) + LOCAL_METHODS
    summary_rows = []
    bin_rows = []

    for method in methods:
        for h in HORIZONS:
            y_col = f"y_{h}"
            known_col = f"known_{h}"
            p_col = method_to_probability_column(method, h)

            mask = (
                (pd.to_numeric(pred[known_col], errors="coerce") > 0.5)
                & pred[y_col].notna()
                & pred[p_col].notna()
            )

            sub = pred.loc[mask, [y_col, p_col]].copy()
            sub.columns = ["y", "p"]

            if len(sub) == 0:
                summary_rows.append(
                    {
                        "dataset": "NACC_external",
                        "scenario": "no_ADAS13",
                        "method": method,
                        "horizon": h,
                        "horizon_year": HORIZON_YEARS[h],
                        "n_known": 0,
                        "observed_event_rate": np.nan,
                        "mean_predicted_risk": np.nan,
                        "calibration_gap_mean_pred_minus_observed": np.nan,
                        "ECE_quantile_bins": np.nan,
                        "max_abs_bin_gap": np.nan,
                    }
                )
                continue

            y = sub["y"].astype(int).to_numpy()
            p = sub["p"].astype(float).to_numpy()

            try:
                sub["bin"] = pd.qcut(sub["p"], q=N_CAL_BINS, duplicates="drop")
            except Exception:
                sub["bin"] = pd.cut(sub["p"], bins=N_CAL_BINS)

            ece = 0.0
            max_gap = 0.0

            for b, g in sub.groupby("bin", observed=True):
                n_b = int(len(g))
                mean_p = float(g["p"].mean())
                obs = float(g["y"].mean())
                gap = mean_p - obs
                abs_gap = abs(gap)

                ece += (n_b / len(sub)) * abs_gap
                max_gap = max(max_gap, abs_gap)

                bin_rows.append(
                    {
                        "dataset": "NACC_external",
                        "scenario": "no_ADAS13",
                        "method": method,
                        "horizon": h,
                        "horizon_year": HORIZON_YEARS[h],
                        "bin": str(b),
                        "n": n_b,
                        "mean_predicted_risk": mean_p,
                        "observed_event_rate": obs,
                        "gap_pred_minus_observed": gap,
                        "abs_gap": abs_gap,
                        "min_predicted_risk": float(g["p"].min()),
                        "max_predicted_risk": float(g["p"].max()),
                    }
                )

            summary_rows.append(
                {
                    "dataset": "NACC_external",
                    "scenario": "no_ADAS13",
                    "method": method,
                    "method_type": "reference" if method in REFERENCE_METHODS else "crossfit_local_recalibration",
                    "horizon": h,
                    "horizon_year": HORIZON_YEARS[h],
                    "n_known": int(len(sub)),
                    "observed_event_rate": float(np.mean(y)),
                    "mean_predicted_risk": float(np.mean(p)),
                    "median_predicted_risk": float(np.median(p)),
                    "calibration_gap_mean_pred_minus_observed": float(np.mean(p) - np.mean(y)),
                    "ECE_quantile_bins": float(ece),
                    "max_abs_bin_gap": float(max_gap),
                    "Brier": safe_metric("Brier", y, p),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(bin_rows)


def bootstrap_ci(pred: pd.DataFrame) -> pd.DataFrame:
    methods = list(REFERENCE_METHODS.keys()) + LOCAL_METHODS
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    total_jobs = len(methods) * len(HORIZONS)
    job = 0

    for method in methods:
        for h in HORIZONS:
            job += 1
            y_col = f"y_{h}"
            known_col = f"known_{h}"
            p_col = method_to_probability_column(method, h)

            mask = (
                (pd.to_numeric(pred[known_col], errors="coerce") > 0.5)
                & pred[y_col].notna()
                & pred[p_col].notna()
            )

            y = pred.loc[mask, y_col].astype(int).to_numpy()
            p = pred.loc[mask, p_col].astype(float).to_numpy()
            n = len(y)

            print(f"[BOOT] {job}/{total_jobs}: method={method}, horizon={h}, n={n}")

            point = {
                "AUROC": safe_metric("AUROC", y, p),
                "AUPRC": safe_metric("AUPRC", y, p),
                "Brier": safe_metric("Brier", y, p),
            }

            boot_values = {"AUROC": [], "AUPRC": [], "Brier": []}

            if n > 0:
                for _ in range(N_BOOT):
                    idx = rng.integers(0, n, size=n)
                    yy = y[idx]
                    pp = p[idx]

                    for metric in ["AUROC", "AUPRC", "Brier"]:
                        val = safe_metric(metric, yy, pp)
                        if np.isfinite(val):
                            boot_values[metric].append(val)

            for metric in ["AUROC", "AUPRC", "Brier"]:
                vals = np.array(boot_values[metric], dtype=float)
                rows.append(
                    {
                        "dataset": "NACC_external",
                        "scenario": "no_ADAS13",
                        "method": method,
                        "method_type": "reference" if method in REFERENCE_METHODS else "crossfit_local_recalibration",
                        "horizon": h,
                        "horizon_year": HORIZON_YEARS[h],
                        "metric": metric,
                        "n_known": int(n),
                        "n_event": int(np.sum(y == 1)) if n else 0,
                        "n_nonevent": int(np.sum(y == 0)) if n else 0,
                        "point": point[metric],
                        "ci_low": float(np.quantile(vals, 0.025)) if len(vals) else np.nan,
                        "ci_high": float(np.quantile(vals, 0.975)) if len(vals) else np.nan,
                        "n_boot_requested": N_BOOT,
                        "n_boot_valid": int(len(vals)),
                    }
                )

    return pd.DataFrame(rows)


def make_delta_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Compare local recalibration methods against raw_sigmoid per horizon.
    """
    rows = []

    base = metrics[metrics["method"].eq("raw_sigmoid")].copy()

    for _, b in base.iterrows():
        h = b["horizon"]
        for method in LOCAL_METHODS + ["adni_scenario_platt"]:
            sub = metrics[
                metrics["method"].eq(method)
                & metrics["horizon"].eq(h)
            ]

            if len(sub) == 0:
                continue

            r = sub.iloc[0]

            rows.append(
                {
                    "horizon": h,
                    "horizon_year": b["horizon_year"],
                    "method": method,
                    "delta_AUROC_vs_raw": r["AUROC"] - b["AUROC"],
                    "delta_AUPRC_vs_raw": r["AUPRC"] - b["AUPRC"],
                    "delta_Brier_vs_raw": r["Brier"] - b["Brier"],
                    "delta_abs_calibration_gap_vs_raw": abs(r["calibration_gap_mean_pred_minus_observed"]) - abs(b["calibration_gap_mean_pred_minus_observed"]),
                    "raw_Brier": b["Brier"],
                    "method_Brier": r["Brier"],
                    "raw_gap": b["calibration_gap_mean_pred_minus_observed"],
                    "method_gap": r["calibration_gap_mean_pred_minus_observed"],
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    root = project_root()
    pred_path = root / PRED_REL
    out_dir = root / OUT_REL
    ensure_outdir(out_dir)

    print("=" * 88)
    print("[STEP 96] NACC cross-fitted local recalibration audit")
    print(f"[SCRIPT VERSION] {SCRIPT_VERSION}")
    print(f"[PREDICTIONS] {pred_path}")
    print(f"[OUTPUT DIR] {out_dir}")
    print("=" * 88)

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction CSV from Step95d: {pred_path}")

    pred = pd.read_csv(pred_path)

    n_rows = len(pred)
    n_unique = int(pred["NACCID"].nunique()) if "NACCID" in pred.columns else None
    n_dup = int(pred["NACCID"].duplicated().sum()) if "NACCID" in pred.columns else None

    print(f"[INFO] prediction rows: {n_rows}")
    print(f"[INFO] unique NACCID: {n_unique}")
    print(f"[INFO] duplicated NACCID rows: {n_dup}")

    print("[INFO] Running cross-fitted local recalibration...")
    pred_recal, params = crossfit_recalibration(pred)

    print("[INFO] Computing metrics...")
    metrics = compute_metrics(pred_recal)

    print("[INFO] Computing calibration summary and bins...")
    cal_summary, cal_bins = calibration_summary_and_bins(pred_recal)

    print("[INFO] Computing bootstrap CIs...")
    boot = bootstrap_ci(pred_recal)

    print("[INFO] Computing delta table vs raw...")
    delta = make_delta_table(metrics)

    pred_out_path = out_dir / "nacc_crossfit_recalibrated_predictions.csv"
    metrics_path = out_dir / "nacc_crossfit_recalibration_metrics.csv"
    boot_path = out_dir / "nacc_crossfit_recalibration_bootstrap_ci.csv"
    cal_summary_path = out_dir / "nacc_crossfit_recalibration_calibration_summary.csv"
    cal_bins_path = out_dir / "nacc_crossfit_recalibration_calibration_bins.csv"
    params_path = out_dir / "nacc_crossfit_recalibration_parameters.csv"
    delta_path = out_dir / "nacc_crossfit_recalibration_delta_vs_raw.csv"
    readme_path = out_dir / "README_step96_nacc_crossfit_local_recalibration_audit.md"
    audit_path = out_dir / "step96_audit.json"

    pred_recal.to_csv(pred_out_path, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    boot.to_csv(boot_path, index=False, encoding="utf-8-sig")
    cal_summary.to_csv(cal_summary_path, index=False, encoding="utf-8-sig")
    cal_bins.to_csv(cal_bins_path, index=False, encoding="utf-8-sig")
    params.to_csv(params_path, index=False, encoding="utf-8-sig")
    delta.to_csv(delta_path, index=False, encoding="utf-8-sig")

    key_methods = [
        "raw_sigmoid",
        "adni_scenario_platt",
        "nacc_intercept_only_raw",
        "nacc_platt_raw",
        "nacc_isotonic_raw",
    ]

    key_metrics = metrics[metrics["method"].isin(key_methods)].copy()
    key_cal = cal_summary[cal_summary["method"].isin(key_methods)].copy()
    key_delta = delta[delta["method"].isin(key_methods)].copy()

    readme = f"""# Step96 NACC cross-fitted local recalibration audit

## Purpose

This step tests whether NACC local recalibration can improve absolute risk calibration
without retraining the frozen ADNI-trained v4 model.

## Design

- Input: Step95d NACC external predictions
- Model weights: frozen, unchanged
- Local recalibration: 5-fold cross-fitted
- Calibration is fit on training folds and evaluated on held-out folds
- Horizons: 1y, 2y, 3y, 5y
- Scenario: no_ADAS13

## Sample check

- Rows: {n_rows}
- Unique NACCID: {n_unique}
- Duplicated NACCID rows: {n_dup}

## Methods compared

1. raw_sigmoid
2. adni_scenario_platt
3. nacc_intercept_only_raw
4. nacc_platt_raw
5. nacc_isotonic_raw

## Metrics

{key_metrics.to_string(index=False)}

## Calibration summary

{key_cal.to_string(index=False)}

## Delta vs raw

{key_delta.to_string(index=False)}

## Interpretation

This analysis should be interpreted as post-hoc local recalibration sensitivity,
not as a fully untouched external validation.

If local recalibration improves Brier score, ECE, and calibration gap while AUROC remains similar,
the correct conclusion is:

The frozen model preserves external risk ranking, but absolute risks require local recalibration
before clinical interpretation.

## Output files

- `nacc_crossfit_recalibrated_predictions.csv`
- `nacc_crossfit_recalibration_metrics.csv`
- `nacc_crossfit_recalibration_bootstrap_ci.csv`
- `nacc_crossfit_recalibration_calibration_summary.csv`
- `nacc_crossfit_recalibration_calibration_bins.csv`
- `nacc_crossfit_recalibration_parameters.csv`
- `nacc_crossfit_recalibration_delta_vs_raw.csv`
- `README_step96_nacc_crossfit_local_recalibration_audit.md`
- `step96_audit.json`
"""

    readme_path.write_text(readme, encoding="utf-8")

    audit = {
        "script_version": SCRIPT_VERSION,
        "predictions_csv": str(pred_path),
        "output_dir": str(out_dir),
        "n_rows": int(n_rows),
        "n_unique_NACCID": n_unique,
        "n_duplicate_NACCID_rows": n_dup,
        "n_splits": N_SPLITS,
        "n_boot": N_BOOT,
        "random_seed": RANDOM_SEED,
        "n_cal_bins": N_CAL_BINS,
        "methods": key_methods,
        "outputs": {
            "predictions": str(pred_out_path),
            "metrics": str(metrics_path),
            "bootstrap_ci": str(boot_path),
            "calibration_summary": str(cal_summary_path),
            "calibration_bins": str(cal_bins_path),
            "parameters": str(params_path),
            "delta_vs_raw": str(delta_path),
            "readme": str(readme_path),
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print("=" * 88)
    print("[DONE] Step96 NACC cross-fitted local recalibration audit finished.")
    print("[METRICS]")
    print(key_metrics.to_string(index=False))
    print("[CALIBRATION SUMMARY]")
    print(key_cal.to_string(index=False))
    print("[DELTA VS RAW]")
    print(key_delta.to_string(index=False))
    print("[OUTPUTS]")
    print(f"  predictions: {pred_out_path}")
    print(f"  metrics: {metrics_path}")
    print(f"  bootstrap: {boot_path}")
    print(f"  calibration summary: {cal_summary_path}")
    print(f"  calibration bins: {cal_bins_path}")
    print(f"  parameters: {params_path}")
    print(f"  delta vs raw: {delta_path}")
    print(f"  readme: {readme_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()