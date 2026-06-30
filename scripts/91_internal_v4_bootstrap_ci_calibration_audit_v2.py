# Step 91: bootstrap confidence intervals and calibration audit for frozen modular v4.
# Purpose:
#   - Add bootstrap 95% CI for v4 metrics.
#   - Add paired bootstrap CI for v4-vs-v3 and v4-vs-best-baseline comparisons when prediction files are available.
#   - Add calibration bins, ECE, calibration slope/intercept, and calibration plots for v4.
#
# Read-only audit. No training. No inference. No external data. No raw-data edits. No deletion.

from __future__ import annotations

from pathlib import Path
import json
import warnings
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression


SCRIPT_VERSION = "v2_internal_v4_bootstrap_ci_calibration_audit_robust_baseline_diff"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

V4_PRED = PROJECT_ROOT / "results" / "models" / "89_summary_augmented_scenario_dropout_modular_v4_internal" / "predictions_by_scenario.csv"
V4_METRICS = PROJECT_ROOT / "results" / "models" / "89_summary_augmented_scenario_dropout_modular_v4_internal" / "metrics_by_scenario_raw_and_scenario_platt.csv"

V3_PRED = PROJECT_ROOT / "results" / "reports" / "85_internal_missing_feature_stress_test_v3" / "predictions_by_scenario.csv"
BASELINE_PRED = PROJECT_ROOT / "results" / "reports" / "88_internal_baseline_missing_feature_stress_comparison" / "baseline_stress_predictions.csv"

STEP90_DIR = PROJECT_ROOT / "results" / "reports" / "90_internal_modular_v4_freeze_and_claims_audit"
STEP90_COMPARISON = STEP90_DIR / "v4_vs_v3_and_best_baseline_key_comparison.csv"
STEP90_FULL = STEP90_DIR / "frozen_v4_full_module_test_metrics.csv"
STEP90_SCENARIO = STEP90_DIR / "frozen_v4_scenario_test_metrics.csv"

OUT_DIR = PROJECT_ROOT / "results" / "reports" / "91_internal_v4_bootstrap_ci_calibration_audit"
PLOT_DIR = OUT_DIR / "calibration_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 2, 3, 5]
BOOTSTRAP_N = 1000
SEED = 42
N_BINS = 10

KEY_SCENARIOS = [
    "full_all_modules",
    "no_ADAS13",
    "no_MMSE",
    "no_MMSE_ADAS13",
    "no_FAQTOTAL",
    "basic_plus_MMSE_CDRSB_FAQ",
    "basic_plus_MMSE_only",
    "basic_only",
    "visit_process_only",
]

METRICS = ["AUROC", "AUPRC", "Brier"]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        warnings.warn(f"Could not read {path}: {e}")
        return pd.DataFrame()


def fmt(x, digits=3) -> str:
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = 1.0 / (1.0 + np.exp(-x))
    return np.clip(np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)


def logit_np(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def metric_value(y: np.ndarray, p: np.ndarray, metric: str) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok]
    p = p[ok]
    if len(y) == 0:
        return np.nan
    if metric in ["AUROC", "AUPRC"] and len(np.unique(y)) < 2:
        return np.nan
    try:
        if metric == "AUROC":
            return float(roc_auc_score(y, p))
        if metric == "AUPRC":
            return float(average_precision_score(y, p))
        if metric == "Brier":
            return float(brier_score_loss(y, p))
    except Exception:
        return np.nan
    raise ValueError(metric)


def ci_from_values(vals: List[float]) -> Tuple[float, float, float, int]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan, 0
    return float(np.nanmean(arr)), float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5)), int(len(arr))


def bootstrap_metric(y: np.ndarray, p: np.ndarray, metric: str, n_boot: int = BOOTSTRAP_N, seed: int = SEED) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok]
    p = p[ok]
    n = len(y)
    point = metric_value(y, p, metric)
    if n == 0 or not np.isfinite(point):
        return {"point": point, "boot_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "boot_valid_n": 0}

    rng = np.random.default_rng(seed)
    vals = []
    idx_all = np.arange(n)
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        vals.append(metric_value(y[idx], p[idx], metric))
    mean, low, high, valid_n = ci_from_values(vals)
    return {"point": point, "boot_mean": mean, "ci_low": low, "ci_high": high, "boot_valid_n": valid_n}


def bootstrap_metric_diff(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, metric: str, n_boot: int = BOOTSTRAP_N, seed: int = SEED) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p_a = np.asarray(p_a).astype(float)
    p_b = np.asarray(p_b).astype(float)
    ok = np.isfinite(y) & np.isfinite(p_a) & np.isfinite(p_b)
    y = y[ok]
    p_a = p_a[ok]
    p_b = p_b[ok]
    n = len(y)
    point_a = metric_value(y, p_a, metric)
    point_b = metric_value(y, p_b, metric)
    point = point_a - point_b if np.isfinite(point_a) and np.isfinite(point_b) else np.nan
    if n == 0 or not np.isfinite(point):
        return {"point_diff": point, "boot_mean_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "boot_valid_n": 0}

    rng = np.random.default_rng(seed)
    vals = []
    idx_all = np.arange(n)
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        ma = metric_value(y[idx], p_a[idx], metric)
        mb = metric_value(y[idx], p_b[idx], metric)
        vals.append(ma - mb if np.isfinite(ma) and np.isfinite(mb) else np.nan)
    mean, low, high, valid_n = ci_from_values(vals)
    return {"point_diff": point, "boot_mean_diff": mean, "ci_low": low, "ci_high": high, "boot_valid_n": valid_n}


def parse_v4_predictions(path: Path) -> pd.DataFrame:
    df = read_csv(path)
    if df.empty:
        return df

    rows = []
    for _, r in df.iterrows():
        split = str(r.get("split", "test"))
        scenario = str(r.get("scenario", "full_all_modules"))
        sample_index = int(r.get("sample_index"))
        for h in HORIZONS:
            risk_col = f"scenario_platt_risk_{h}y"
            raw_col = f"raw_risk_{h}y"
            label_col = f"label_{h}y"
            obs_col = f"observed_{h}y"
            if risk_col not in df.columns or label_col not in df.columns or obs_col not in df.columns:
                continue
            rows.append(
                {
                    "model": "step89_summary_augmented_modular_v4",
                    "family": "summary_augmented_feature_modular_v4",
                    "split": split,
                    "scenario": scenario,
                    "sample_index": sample_index,
                    "horizon_year": h,
                    "risk": float(r[risk_col]),
                    "raw_risk": float(r[raw_col]) if raw_col in df.columns else np.nan,
                    "label": float(r[label_col]),
                    "observed": float(r[obs_col]),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["horizon_year"] = out["horizon_year"].astype(int)
    return out


def parse_long_predictions(df: pd.DataFrame, model_default: str, family_default: str, split_default: str = "test") -> pd.DataFrame:
    if df.empty:
        return df

    d = df.copy()
    if "horizon_year" not in d.columns:
        return pd.DataFrame()

    risk_col = None
    for c in ["platt_risk", "scenario_platt_risk", "risk", "prob", "probability", "pred"]:
        if c in d.columns:
            risk_col = c
            break
    if risk_col is None:
        return pd.DataFrame()

    label_col = "label" if "label" in d.columns else ("y" if "y" in d.columns else None)
    obs_col = "observed" if "observed" in d.columns else ("y_observed" if "y_observed" in d.columns else None)
    if label_col is None or obs_col is None:
        return pd.DataFrame()

    if "sample_index" not in d.columns:
        if "index" in d.columns:
            d["sample_index"] = d["index"]
        else:
            d["sample_index"] = np.arange(len(d))

    if "model" not in d.columns:
        d["model"] = model_default
    if "family" not in d.columns:
        d["family"] = family_default
    if "split" not in d.columns:
        d["split"] = split_default
    if "scenario" not in d.columns:
        d["scenario"] = "full_all_modules"

    out = d.rename(columns={risk_col: "risk", label_col: "label", obs_col: "observed"}).copy()
    keep = ["model", "family", "split", "scenario", "sample_index", "horizon_year", "risk", "label", "observed"]
    out = out[[c for c in keep if c in out.columns]]
    out["horizon_year"] = out["horizon_year"].astype(int)
    return out


def parse_wide_predictions(df: pd.DataFrame, model_default: str, family_default: str, split_default: str = "test") -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    if "sample_index" not in df.columns:
        sample_indices = np.arange(len(df))
    else:
        sample_indices = df["sample_index"].values

    for i, r in df.iterrows():
        model = str(r.get("model", model_default))
        family = str(r.get("family", family_default))
        split = str(r.get("split", split_default))
        scenario = str(r.get("scenario", "full_all_modules"))
        sample_index = int(sample_indices[i])
        for h in HORIZONS:
            candidates = [
                f"platt_risk_{h}y",
                f"scenario_platt_risk_{h}y",
                f"risk_platt_{h}y",
                f"prob_platt_{h}y",
                f"risk_{h}y",
                f"raw_risk_{h}y",
            ]
            risk_col = next((c for c in candidates if c in df.columns), None)
            label_col = next((c for c in [f"label_{h}y", f"y_{h}y", f"target_{h}y"] if c in df.columns), None)
            obs_col = next((c for c in [f"observed_{h}y", f"y_observed_{h}y", f"obs_{h}y"] if c in df.columns), None)
            if risk_col and label_col and obs_col:
                rows.append(
                    {
                        "model": model,
                        "family": family,
                        "split": split,
                        "scenario": scenario,
                        "sample_index": sample_index,
                        "horizon_year": h,
                        "risk": float(r[risk_col]),
                        "label": float(r[label_col]),
                        "observed": float(r[obs_col]),
                    }
                )
    return pd.DataFrame(rows)


def parse_v3_predictions(path: Path) -> pd.DataFrame:
    df = read_csv(path)
    if df.empty:
        return df
    long = parse_long_predictions(df, "step84_feature_modular_v3", "feature_modular_v3", "test")
    if not long.empty:
        long["model"] = "step84_feature_modular_v3"
        long["family"] = "feature_modular_v3"
        return long
    wide = parse_wide_predictions(df, "step84_feature_modular_v3", "feature_modular_v3", "test")
    if not wide.empty:
        wide["model"] = "step84_feature_modular_v3"
        wide["family"] = "feature_modular_v3"
    return wide


def parse_baseline_predictions(path: Path) -> pd.DataFrame:
    df = read_csv(path)
    if df.empty:
        return df
    out = parse_long_predictions(df, "baseline", "baseline", "test")
    if not out.empty:
        return out
    return parse_wide_predictions(df, "baseline", "baseline", "test")


def v4_bootstrap_table(v4_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = v4_long[(v4_long["split"].eq("test")) & (v4_long["observed"] > 0.5)].copy()
    for scenario in sorted(d["scenario"].unique()):
        for h in HORIZONS:
            dh = d[(d["scenario"].eq(scenario)) & (d["horizon_year"].eq(h))]
            if dh.empty:
                continue
            for metric in METRICS:
                res = bootstrap_metric(dh["label"].values, dh["risk"].values, metric, seed=SEED + h + len(scenario) + len(metric))
                rows.append(
                    {
                        "model": "step89_summary_augmented_modular_v4",
                        "scenario": scenario,
                        "horizon_year": h,
                        "metric": metric,
                        "point": res["point"],
                        "boot_mean": res["boot_mean"],
                        "ci_low": res["ci_low"],
                        "ci_high": res["ci_high"],
                        "boot_valid_n": res["boot_valid_n"],
                        "observed_n": int(len(dh)),
                        "positive_n": int(np.sum(dh["label"].values == 1)),
                    }
                )
    return pd.DataFrame(rows)


def paired_diff_table(v4_long: pd.DataFrame, other_long: pd.DataFrame, other_model: str, label: str) -> pd.DataFrame:
    rows = []
    if v4_long.empty or other_long.empty:
        return pd.DataFrame()

    v4 = v4_long[(v4_long["split"].eq("test")) & (v4_long["observed"] > 0.5)].copy()
    other = other_long[(other_long["split"].eq("test")) & (other_long["observed"] > 0.5)].copy()

    if other_model:
        other = other[other["model"].eq(other_model)].copy()

    for scenario in sorted(set(v4["scenario"]).intersection(set(other["scenario"]))):
        for h in HORIZONS:
            a = v4[(v4["scenario"].eq(scenario)) & (v4["horizon_year"].eq(h))]
            b = other[(other["scenario"].eq(scenario)) & (other["horizon_year"].eq(h))]
            if a.empty or b.empty:
                continue
            merged = a[["sample_index", "label", "risk"]].merge(
                b[["sample_index", "risk"]], on="sample_index", how="inner", suffixes=("_v4", "_other")
            )
            if merged.empty:
                continue
            for metric in METRICS:
                res = bootstrap_metric_diff(
                    merged["label"].values,
                    merged["risk_v4"].values,
                    merged["risk_other"].values,
                    metric,
                    seed=SEED + h + len(scenario) + len(metric) + 200,
                )
                rows.append(
                    {
                        "comparison": label,
                        "scenario": scenario,
                        "horizon_year": h,
                        "metric": metric,
                        "point_diff_v4_minus_other": res["point_diff"],
                        "boot_mean_diff": res["boot_mean_diff"],
                        "ci_low": res["ci_low"],
                        "ci_high": res["ci_high"],
                        "boot_valid_n": res["boot_valid_n"],
                        "paired_n": int(len(merged)),
                        "positive_n": int(np.sum(merged["label"].values == 1)),
                        "other_model": other_model,
                    }
                )
    return pd.DataFrame(rows)


def best_baseline_diff_table(v4_long: pd.DataFrame, baseline_long: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    """
    Paired bootstrap CI for v4 minus the best scenario-specific baseline.

    Robust behavior:
    - If comparison contains best_AUROC_baseline / best_AUPRC_baseline / best_Brier_baseline, use them.
    - Otherwise, derive the best baseline directly from baseline_long predictions for each scenario/horizon/metric.
    """
    columns = [
        "comparison", "scenario", "horizon_year", "metric", "best_baseline_model",
        "point_diff_v4_minus_baseline", "boot_mean_diff", "ci_low", "ci_high",
        "boot_valid_n", "paired_n", "positive_n"
    ]
    rows = []
    if v4_long.empty or baseline_long.empty:
        return pd.DataFrame(columns=columns)

    v4 = v4_long[(v4_long["split"].eq("test")) & (v4_long["observed"] > 0.5)].copy()
    b_all = baseline_long[(baseline_long["split"].eq("test")) & (baseline_long["observed"] > 0.5)].copy()

    scenario_set = sorted(set(v4["scenario"]).intersection(set(b_all["scenario"])))
    for scenario in scenario_set:
        for h in HORIZONS:
            a = v4[(v4["scenario"].eq(scenario)) & (v4["horizon_year"].eq(h))]
            b_s = b_all[(b_all["scenario"].eq(scenario)) & (b_all["horizon_year"].eq(h))]
            if a.empty or b_s.empty:
                continue

            for metric in METRICS:
                # First try to get explicit best-baseline model name from comparison tables.
                baseline_model = None
                if comparison is not None and (not comparison.empty):
                    row = comparison[(comparison["scenario"].eq(scenario)) & (comparison["horizon_year"].astype(int).eq(int(h)))]
                    if not row.empty:
                        model_col = {
                            "AUROC": "best_AUROC_baseline",
                            "AUPRC": "best_AUPRC_baseline",
                            "Brier": "best_Brier_baseline",
                        }[metric]
                        if model_col in row.columns and pd.notna(row.iloc[0].get(model_col)):
                            baseline_model = str(row.iloc[0][model_col])

                # If Step90 table dropped model names, derive best baseline directly.
                if baseline_model is None or baseline_model == "":
                    candidates = []
                    for model_name, gm in b_s.groupby("model"):
                        y_tmp = gm["label"].values
                        p_tmp = gm["risk"].values
                        val = metric_value(y_tmp, p_tmp, metric)
                        if np.isfinite(val):
                            candidates.append((model_name, val))
                    if not candidates:
                        continue
                    if metric in ["AUROC", "AUPRC"]:
                        baseline_model = max(candidates, key=lambda x: x[1])[0]
                    else:
                        baseline_model = min(candidates, key=lambda x: x[1])[0]

                b = b_s[b_s["model"].eq(baseline_model)]
                if b.empty:
                    continue

                merged = a[["sample_index", "label", "risk"]].merge(
                    b[["sample_index", "risk"]],
                    on="sample_index",
                    how="inner",
                    suffixes=("_v4", "_baseline"),
                )
                if merged.empty:
                    continue

                res = bootstrap_metric_diff(
                    merged["label"].values,
                    merged["risk_v4"].values,
                    merged["risk_baseline"].values,
                    metric,
                    seed=SEED + h + len(scenario) + len(metric) + 500,
                )
                rows.append(
                    {
                        "comparison": "v4_minus_best_baseline_by_metric",
                        "scenario": scenario,
                        "horizon_year": h,
                        "metric": metric,
                        "best_baseline_model": baseline_model,
                        "point_diff_v4_minus_baseline": res["point_diff"],
                        "boot_mean_diff": res["boot_mean_diff"],
                        "ci_low": res["ci_low"],
                        "ci_high": res["ci_high"],
                        "boot_valid_n": res["boot_valid_n"],
                        "paired_n": int(len(merged)),
                        "positive_n": int(np.sum(merged["label"].values == 1)),
                    }
                )
    return pd.DataFrame(rows, columns=columns)

def calibration_bins_and_summary(v4_long: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = v4_long[(v4_long["split"].eq("test")) & (v4_long["observed"] > 0.5)].copy()
    bin_rows = []
    summary_rows = []

    for scenario in sorted(d["scenario"].unique()):
        for h in HORIZONS:
            dh = d[(d["scenario"].eq(scenario)) & (d["horizon_year"].eq(h))].copy()
            if dh.empty:
                continue
            y = dh["label"].values.astype(int)
            p = np.clip(dh["risk"].values.astype(float), 1e-6, 1 - 1e-6)
            n = len(y)
            pos = int(np.sum(y == 1))
            brier = metric_value(y, p, "Brier")

            # Fixed-width bins for standard calibration reporting.
            bin_ids = np.minimum((p * N_BINS).astype(int), N_BINS - 1)
            ece = 0.0
            mce = 0.0
            for b in range(N_BINS):
                idx = bin_ids == b
                if idx.sum() == 0:
                    continue
                mean_pred = float(np.mean(p[idx]))
                obs_rate = float(np.mean(y[idx]))
                gap = abs(mean_pred - obs_rate)
                ece += (idx.sum() / n) * gap
                mce = max(mce, gap)
                bin_rows.append(
                    {
                        "scenario": scenario,
                        "horizon_year": h,
                        "bin_id": b,
                        "bin_lower": b / N_BINS,
                        "bin_upper": (b + 1) / N_BINS,
                        "n": int(idx.sum()),
                        "positive_n": int(np.sum(y[idx] == 1)),
                        "mean_predicted_risk": mean_pred,
                        "observed_event_rate": obs_rate,
                        "absolute_gap": gap,
                    }
                )

            cal_intercept = np.nan
            cal_slope = np.nan
            if n >= 20 and len(np.unique(y)) >= 2:
                try:
                    clf = LogisticRegression(solver="lbfgs", C=1e6, max_iter=1000)
                    clf.fit(logit_np(p).reshape(-1, 1), y)
                    cal_intercept = float(clf.intercept_[0])
                    cal_slope = float(clf.coef_[0, 0])
                except Exception:
                    pass

            summary_rows.append(
                {
                    "scenario": scenario,
                    "horizon_year": h,
                    "observed_n": n,
                    "positive_n": pos,
                    "event_rate": pos / n if n else np.nan,
                    "Brier": brier,
                    "ECE_fixed_10_bins": float(ece),
                    "MCE_fixed_10_bins": float(mce),
                    "calibration_intercept_logistic": cal_intercept,
                    "calibration_slope_logistic": cal_slope,
                }
            )
    return pd.DataFrame(bin_rows), pd.DataFrame(summary_rows)


def make_plots(bins: pd.DataFrame) -> List[str]:
    paths = []
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        (OUT_DIR / "calibration_plot_note.txt").write_text(f"Matplotlib unavailable; plots skipped: {e}", encoding="utf-8")
        return paths

    d = bins[bins["scenario"].isin(KEY_SCENARIOS)].copy()
    if d.empty:
        return paths

    for scenario in KEY_SCENARIOS:
        for h in HORIZONS:
            dh = d[(d["scenario"].eq(scenario)) & (d["horizon_year"].eq(h))]
            if dh.empty:
                continue
            fig = plt.figure(figsize=(5, 5))
            plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
            plt.plot(dh["mean_predicted_risk"], dh["observed_event_rate"], marker="o")
            plt.xlabel("Mean predicted risk")
            plt.ylabel("Observed event rate")
            plt.title(f"Calibration: {scenario}, {h}y")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            out = PLOT_DIR / f"calibration_{scenario}_{h}y.png"
            fig.tight_layout()
            fig.savefig(out, dpi=180)
            plt.close(fig)
            paths.append(str(out))
    return paths


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_Not available._\n"
    d = df.head(max_rows).copy()
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in d.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(fmt(v, 3))
            else:
                vals.append(str(v).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Only first {max_rows} rows shown; full table has {len(df)} rows._")
    return "\n".join(lines) + "\n"


def make_paper_ci_table(v4_ci: pd.DataFrame, diff_v3: pd.DataFrame, diff_base: pd.DataFrame, cal_summary: pd.DataFrame) -> pd.DataFrame:
    key = v4_ci[
        (v4_ci["scenario"].isin(["full_all_modules", "no_ADAS13", "no_MMSE_ADAS13", "basic_plus_MMSE_CDRSB_FAQ", "basic_plus_MMSE_only", "basic_only"]))
        & (v4_ci["metric"].isin(["AUROC", "Brier"]))
    ].copy()

    rows = []
    for _, r in key.iterrows():
        scenario = r["scenario"]
        h = int(r["horizon_year"])
        metric = r["metric"]
        row = {
            "scenario": scenario,
            "horizon_year": h,
            "metric": metric,
            "v4_point": r["point"],
            "v4_ci_95": f"{fmt(r['ci_low'])}-{fmt(r['ci_high'])}",
        }
        dv3 = pd.DataFrame()
        if diff_v3 is not None and (not diff_v3.empty) and all(c in diff_v3.columns for c in ["scenario", "horizon_year", "metric"]):
            dv3 = diff_v3[(diff_v3["scenario"].eq(scenario)) & (diff_v3["horizon_year"].eq(h)) & (diff_v3["metric"].eq(metric))]
        if not dv3.empty:
            rr = dv3.iloc[0]
            row["v4_minus_v3_point"] = rr["point_diff_v4_minus_other"]
            row["v4_minus_v3_ci_95"] = f"{fmt(rr['ci_low'])}-{fmt(rr['ci_high'])}"

        db = pd.DataFrame()
        if diff_base is not None and (not diff_base.empty) and all(c in diff_base.columns for c in ["scenario", "horizon_year", "metric"]):
            db = diff_base[(diff_base["scenario"].eq(scenario)) & (diff_base["horizon_year"].eq(h)) & (diff_base["metric"].eq(metric))]
        if not db.empty:
            rr = db.iloc[0]
            row["v4_minus_best_baseline_point"] = rr["point_diff_v4_minus_baseline"]
            row["v4_minus_best_baseline_ci_95"] = f"{fmt(rr['ci_low'])}-{fmt(rr['ci_high'])}"
            row["best_baseline_model"] = rr["best_baseline_model"]

        cs = cal_summary[(cal_summary["scenario"].eq(scenario)) & (cal_summary["horizon_year"].eq(h))]
        if not cs.empty:
            row["ECE"] = cs.iloc[0]["ECE_fixed_10_bins"]
            row["calibration_slope"] = cs.iloc[0]["calibration_slope_logistic"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_readme(v4_ci, diff_v3, diff_base, cal_summary, paper_table, plots):
    lines = []
    lines.append("# Step91 internal v4 bootstrap confidence interval and calibration audit\n\n")
    lines.append("## Purpose\n\n")
    lines.append(
        "This read-only audit adds uncertainty estimates and calibration diagnostics to the frozen Step90 v4 mainline. "
        "It does not train a new model, perform inference, use external data, or edit raw data.\n\n"
    )

    lines.append("## What this step adds\n\n")
    lines.append("- Bootstrap 95% confidence intervals for v4 AUROC/AUPRC/Brier.\n")
    lines.append("- Paired bootstrap confidence intervals for v4 minus v3 when v3 predictions are available.\n")
    lines.append("- Paired bootstrap confidence intervals for v4 minus the best scenario-specific baseline when baseline predictions are available.\n")
    lines.append("- Calibration bins, ECE/MCE, calibration intercept/slope, and calibration plots.\n\n")

    lines.append("## Key paper CI table\n\n")
    lines.append(md_table(paper_table, max_rows=160))

    lines.append("\n## V4 full-module bootstrap CI\n\n")
    full = v4_ci[(v4_ci["scenario"].eq("full_all_modules"))].copy()
    lines.append(md_table(full, max_rows=80))

    lines.append("\n## V4 minus v3 bootstrap CI, key scenarios\n\n")
    if not diff_v3.empty:
        d = diff_v3[diff_v3["scenario"].isin(KEY_SCENARIOS)].copy()
        lines.append(md_table(d, max_rows=120))
    else:
        lines.append("_V3 prediction file was not available or could not be parsed._\n")

    lines.append("\n## V4 minus best baseline bootstrap CI, key scenarios\n\n")
    if not diff_base.empty:
        d = diff_base[diff_base["scenario"].isin(KEY_SCENARIOS)].copy()
        lines.append(md_table(d, max_rows=120))
    else:
        lines.append("_Baseline prediction file was not available or could not be parsed._\n")

    lines.append("\n## Calibration summary, key scenarios\n\n")
    lines.append(md_table(cal_summary[cal_summary["scenario"].isin(KEY_SCENARIOS)], max_rows=120))

    lines.append("\n## Interpretation boundary\n\n")
    lines.append(
        "Bootstrap confidence intervals quantify uncertainty around the internal split. They do not replace external validation. "
        "Calibration results are internal and scenario-specific. Gate weights and stress tests remain model-behavior diagnostics, not causal clinical importance.\n\n"
    )
    lines.append("## Calibration plots\n\n")
    if plots:
        for p in plots[:40]:
            lines.append(f"- `{rel(Path(p))}`\n")
    else:
        lines.append("_No plots generated._\n")

    (OUT_DIR / "README_step91_internal_v4_bootstrap_ci_calibration_audit.md").write_text("".join(lines), encoding="utf-8")


def main():
    print("=" * 88)
    print("[STEP 91] Internal v4 bootstrap CI and calibration audit")
    print(f"[SCRIPT VERSION] {SCRIPT_VERSION}")
    print("=" * 88)
    print(f"[PROJECT ROOT] {PROJECT_ROOT}")
    print("[MODE] read-only audit; no training; no inference; no external data")
    print(f"[OUTPUT DIR] {OUT_DIR}")
    print(f"[BOOTSTRAP_N] {BOOTSTRAP_N}")

    v4_long = parse_v4_predictions(V4_PRED)
    v3_long = parse_v3_predictions(V3_PRED)
    baseline_long = parse_baseline_predictions(BASELINE_PRED)
    comparison = read_csv(STEP90_COMPARISON)

    if v4_long.empty:
        raise RuntimeError(f"Could not parse v4 predictions: {V4_PRED}")

    print("\n[PARSED PREDICTIONS]")
    print(f"v4_long={v4_long.shape}")
    print(f"v3_long={v3_long.shape if not v3_long.empty else (0, 0)}")
    print(f"baseline_long={baseline_long.shape if not baseline_long.empty else (0, 0)}")

    v4_ci = v4_bootstrap_table(v4_long)
    diff_v3 = paired_diff_table(v4_long, v3_long, "step84_feature_modular_v3", "v4_minus_v3")
    diff_base = best_baseline_diff_table(v4_long, baseline_long, comparison)

    cal_bins, cal_summary = calibration_bins_and_summary(v4_long)
    plots = make_plots(cal_bins)
    paper = make_paper_ci_table(v4_ci, diff_v3, diff_base, cal_summary)

    v4_ci.to_csv(OUT_DIR / "bootstrap_ci_v4_metrics.csv", index=False)
    diff_v3.to_csv(OUT_DIR / "paired_bootstrap_ci_v4_minus_v3.csv", index=False)
    diff_base.to_csv(OUT_DIR / "paired_bootstrap_ci_v4_minus_best_baseline.csv", index=False)
    cal_bins.to_csv(OUT_DIR / "calibration_bins_v4.csv", index=False)
    cal_summary.to_csv(OUT_DIR / "calibration_summary_v4.csv", index=False)
    paper.to_csv(OUT_DIR / "paper_table_v4_bootstrap_ci_calibration.csv", index=False)

    write_readme(v4_ci, diff_v3, diff_base, cal_summary, paper, plots)

    audit = {
        "script_version": SCRIPT_VERSION,
        "mode": "internal_v4_bootstrap_ci_calibration_audit",
        "no_training": True,
        "no_inference": True,
        "no_external_data": True,
        "no_raw_data_edits": True,
        "project_root": str(PROJECT_ROOT),
        "out_dir": str(OUT_DIR),
        "bootstrap_n": BOOTSTRAP_N,
        "inputs": {
            "v4_predictions": rel(V4_PRED),
            "v3_predictions": rel(V3_PRED),
            "baseline_predictions": rel(BASELINE_PRED),
            "step90_comparison": rel(STEP90_COMPARISON),
        },
        "parsed_shapes": {
            "v4_long": list(v4_long.shape),
            "v3_long": list(v3_long.shape) if not v3_long.empty else [0, 0],
            "baseline_long": list(baseline_long.shape) if not baseline_long.empty else [0, 0],
        },
        "outputs": {
            "bootstrap_ci_v4_metrics": str(OUT_DIR / "bootstrap_ci_v4_metrics.csv"),
            "paired_bootstrap_ci_v4_minus_v3": str(OUT_DIR / "paired_bootstrap_ci_v4_minus_v3.csv"),
            "paired_bootstrap_ci_v4_minus_best_baseline": str(OUT_DIR / "paired_bootstrap_ci_v4_minus_best_baseline.csv"),
            "calibration_bins_v4": str(OUT_DIR / "calibration_bins_v4.csv"),
            "calibration_summary_v4": str(OUT_DIR / "calibration_summary_v4.csv"),
            "paper_table": str(OUT_DIR / "paper_table_v4_bootstrap_ci_calibration.csv"),
            "readme": str(OUT_DIR / "README_step91_internal_v4_bootstrap_ci_calibration_audit.md"),
            "plot_dir": str(PLOT_DIR),
        },
        "n_plots": int(len(plots)),
    }
    (OUT_DIR / "step91_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("\n[V4 BOOTSTRAP CI: FULL MODULE]")
    print(v4_ci[v4_ci["scenario"].eq("full_all_modules")].to_string(index=False))

    print("\n[V4 MINUS V3 BOOTSTRAP CI: KEY ROWS]")
    if not diff_v3.empty:
        print(diff_v3[diff_v3["scenario"].isin(KEY_SCENARIOS)].head(80).to_string(index=False))
    else:
        print("[SKIPPED] v3 predictions unavailable or unparsable.")

    print("\n[V4 MINUS BEST BASELINE BOOTSTRAP CI: KEY ROWS]")
    if not diff_base.empty:
        print(diff_base[diff_base["scenario"].isin(KEY_SCENARIOS)].head(80).to_string(index=False))
    else:
        print("[SKIPPED] baseline predictions unavailable or unparsable.")

    print("\n[CALIBRATION SUMMARY: KEY SCENARIOS]")
    print(cal_summary[cal_summary["scenario"].isin(KEY_SCENARIOS)].head(80).to_string(index=False))

    print("\n[OUTPUTS]")
    for _, path in audit["outputs"].items():
        print(f"  - {path}")
    print(f"  - {OUT_DIR / 'step91_audit.json'}")
    print(f"\n[DONE] Step91 internal v4 bootstrap CI and calibration audit complete. Plots generated: {len(plots)}")


if __name__ == "__main__":
    main()
