# -*- coding: utf-8 -*-
"""
Step95d: Evaluate frozen Step89 v4 on corrected NACC external validation cohort.

Purpose
-------
Load frozen internal Step89 v4 checkpoint and evaluate it on NACC external aligned tokens.

External setup:
    - NACC investigator single table
    - first eligible MCI visit per patient
    - time axis = NACCVNUM
    - event = future NACCUDSD == 4 and NACCALZD == 1
    - scenario = no_ADAS13
    - no retraining, no refitting

Outputs
-------
results/reports/95d_nacc_external_frozen_v4_no_adas13_eval/
    external_metrics_raw_and_platt_no_ADAS13.csv
    external_predictions_no_ADAS13.csv
    external_calibration_bins_no_ADAS13.csv
    README_step95d_nacc_external_frozen_v4_no_ADAS13_eval.md
    step95d_audit.json
"""

from __future__ import annotations

import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


SCRIPT_VERSION = "v1_eval_frozen_v4_on_nacc_external_no_ADAS13"

STEP89_SCRIPT_REL = Path("scripts") / "89_train_summary_augmented_scenario_dropout_modular_v4.py"

MODEL_DIR_REL = Path("results") / "models" / "89_summary_augmented_scenario_dropout_modular_v4_internal"
CKPT_REL = MODEL_DIR_REL / "best_model.pt"
PLATT_REL = MODEL_DIR_REL / "scenario_platt_calibration_parameters.csv"

EXTERNAL_ALIGNED_REL = (
    Path("results")
    / "features"
    / "95c_nacc_external_aligned_tokens"
    / "nacc_external_aligned_tokens_for_v4_no_ADAS13.npz"
)

EXTERNAL_COHORT_REL = (
    Path("results")
    / "features"
    / "94d_nacc_external_raw_tokens_corrected_naccvnum"
    / "nacc_external_first_mci_cohort_corrected_naccvnum.csv"
)

OUT_REL = Path("results") / "reports" / "95d_nacc_external_frozen_v4_no_adas13_eval"

SCENARIO = "no_ADAS13"
HORIZON_NAMES = ["1y", "2y", "3y", "5y"]
HORIZON_YEARS = [1, 2, 3, 5]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_step89_module(path: Path):
    spec = importlib.util.spec_from_file_location("step89_v4_module", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Step89 script from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_npz(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing NPZ: {path}")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def make_scenario_mask(module_order: List[str], scenario: str) -> np.ndarray:
    keep = np.ones(len(module_order), dtype=np.float32)

    if scenario == "no_ADAS13":
        for i, m in enumerate(module_order):
            if m == "ADAS13":
                keep[i] = 0.0
    else:
        raise ValueError(f"Unsupported scenario in this script: {scenario}")

    return keep


def unpack_build_inputs_output(obj):
    """
    Step89 build_v4_module_inputs may return either:

        (seq_arrays, seq_masks, summary_arrays, availability)

    or a longer tuple where the first four objects are:

        seq_arrays, seq_masks, summary_arrays, availability

    We only need these four for V4Dataset.
    """

    if isinstance(obj, tuple):
        if len(obj) >= 4:
            seq_arrays = obj[0]
            seq_masks = obj[1]
            summary_arrays = obj[2]
            availability = obj[3]

            if not isinstance(seq_arrays, dict):
                raise RuntimeError(
                    "Parsed build_v4_module_inputs tuple, but first item is not a dict. "
                    f"type={type(seq_arrays)}, repr={repr(seq_arrays)[:500]}"
                )

            if not isinstance(seq_masks, dict):
                raise RuntimeError(
                    "Parsed build_v4_module_inputs tuple, but second item is not a dict. "
                    f"type={type(seq_masks)}, repr={repr(seq_masks)[:500]}"
                )

            if not isinstance(summary_arrays, dict):
                raise RuntimeError(
                    "Parsed build_v4_module_inputs tuple, but third item is not a dict. "
                    f"type={type(summary_arrays)}, repr={repr(summary_arrays)[:500]}"
                )

            return seq_arrays, seq_masks, summary_arrays, availability

    if isinstance(obj, dict):
        candidates = [
            ("seq_arrays", "seq_masks", "summary_arrays", "availability"),
            ("module_inputs", "module_masks", "module_summaries", "availability"),
        ]
        for keys in candidates:
            if all(k in obj for k in keys):
                return obj[keys[0]], obj[keys[1]], obj[keys[2]], obj[keys[3]]

    raise RuntimeError(
        "Could not parse output from build_v4_module_inputs. "
        f"Type={type(obj)}, repr={repr(obj)[:500]}"
    )

def unpack_predict_output(pred_out, n_expected: int, n_horizons: int = 4):
    """
    Try to robustly parse Step89 predict output.
    Expected common output:
        probs
        or (probs, logits)
        or (probs, logits, gates)
    """
    probs = None
    logits = None
    gates = None

    def as_array(x):
        try:
            a = np.asarray(x)
            if a.ndim == 2 and a.shape[0] == n_expected and a.shape[1] == n_horizons:
                return a.astype(np.float64)
        except Exception:
            return None
        return None

    if isinstance(pred_out, dict):
        for k in ["probs", "prob", "pred_probs", "predictions"]:
            if k in pred_out:
                probs = as_array(pred_out[k])
        for k in ["logits", "pred_logits"]:
            if k in pred_out:
                logits = as_array(pred_out[k])
        for k in ["gates", "gate", "gate_weights"]:
            if k in pred_out:
                gates = pred_out[k]

    elif isinstance(pred_out, (tuple, list)):
        arrays = []
        others = []

        for x in pred_out:
            arr = as_array(x)
            if arr is not None:
                arrays.append(arr)
            else:
                others.append(x)

        for arr in arrays:
            mn = np.nanmin(arr)
            mx = np.nanmax(arr)
            if probs is None and mn >= -1e-5 and mx <= 1 + 1e-5:
                probs = arr
            elif logits is None:
                logits = arr

        if probs is None and len(arrays) >= 1:
            probs = arrays[0]

        if logits is None and len(arrays) >= 2:
            logits = arrays[1]

        if others:
            gates = others[0]

    else:
        probs = as_array(pred_out)

    if probs is None:
        raise RuntimeError(
            "Could not parse prediction probabilities from Step89 predict output. "
            f"Type={type(pred_out)}, repr={repr(pred_out)[:500]}"
        )

    probs = np.clip(probs.astype(np.float64), 1e-6, 1 - 1e-6)

    if logits is None:
        logits = logit(probs)
    else:
        logits = logits.astype(np.float64)

    return probs, logits, gates


def metric_or_nan(fn, y_true, y_score) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(fn(y_true, y_score))
    except Exception:
        return np.nan


def compute_metrics(
    probs: np.ndarray,
    y: np.ndarray,
    y_obs: np.ndarray,
    risk_type: str,
) -> pd.DataFrame:
    rows = []

    for h_idx, h_name in enumerate(HORIZON_NAMES):
        mask = (y_obs[:, h_idx] > 0.5) & np.isfinite(y[:, h_idx]) & np.isfinite(probs[:, h_idx])
        yy = y[mask, h_idx].astype(int)
        pp = probs[mask, h_idx].astype(float)

        n = int(len(yy))
        n_event = int(np.sum(yy == 1))
        n_nonevent = int(np.sum(yy == 0))

        if n == 0 or len(np.unique(yy)) < 2:
            auroc = np.nan
            auprc = np.nan
            brier = np.nan
        else:
            auroc = metric_or_nan(roc_auc_score, yy, pp)
            auprc = metric_or_nan(average_precision_score, yy, pp)
            brier = float(brier_score_loss(yy, pp))

        rows.append(
            {
                "dataset": "NACC_external",
                "scenario": SCENARIO,
                "risk_type": risk_type,
                "horizon": h_name,
                "horizon_year": HORIZON_YEARS[h_idx],
                "n_known": n,
                "n_event": n_event,
                "n_nonevent": n_nonevent,
                "event_rate": n_event / n if n else np.nan,
                "AUROC": auroc,
                "AUPRC": auprc,
                "Brier": brier,
                "mean_predicted_risk": float(np.mean(pp)) if n else np.nan,
                "median_predicted_risk": float(np.median(pp)) if n else np.nan,
            }
        )

    return pd.DataFrame(rows)


def apply_platt_from_csv(logits: np.ndarray, platt_csv: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    if not platt_csv.exists():
        raise FileNotFoundError(f"Missing Platt calibration file: {platt_csv}")

    cal = pd.read_csv(platt_csv)
    out = np.zeros_like(logits, dtype=np.float64)

    for h_idx, h_year in enumerate(HORIZON_YEARS):
        sub = cal[
            (cal["scenario"].astype(str) == SCENARIO)
            & (pd.to_numeric(cal["horizon_year"], errors="coerce") == h_year)
        ]

        if len(sub) == 0:
            raise RuntimeError(f"No Platt row found for scenario={SCENARIO}, horizon={h_year}")

        row = sub.iloc[0]
        coef = float(row["coef"])
        intercept = float(row["intercept"])
        out[:, h_idx] = sigmoid(coef * logits[:, h_idx] + intercept)

    return np.clip(out, 1e-6, 1 - 1e-6), cal


def calibration_bins(
    probs: np.ndarray,
    y: np.ndarray,
    y_obs: np.ndarray,
    risk_type: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    rows = []

    for h_idx, h_name in enumerate(HORIZON_NAMES):
        mask = (y_obs[:, h_idx] > 0.5) & np.isfinite(y[:, h_idx]) & np.isfinite(probs[:, h_idx])
        yy = y[mask, h_idx].astype(int)
        pp = probs[mask, h_idx].astype(float)

        if len(yy) == 0:
            continue

        temp = pd.DataFrame({"y": yy, "p": pp})

        try:
            temp["bin"] = pd.qcut(temp["p"], q=n_bins, duplicates="drop")
        except Exception:
            temp["bin"] = pd.cut(temp["p"], bins=n_bins)

        grouped = temp.groupby("bin", observed=True)

        for b, g in grouped:
            rows.append(
                {
                    "dataset": "NACC_external",
                    "scenario": SCENARIO,
                    "risk_type": risk_type,
                    "horizon": h_name,
                    "horizon_year": HORIZON_YEARS[h_idx],
                    "bin": str(b),
                    "n": int(len(g)),
                    "mean_predicted_risk": float(g["p"].mean()),
                    "observed_event_rate": float(g["y"].mean()),
                    "min_predicted_risk": float(g["p"].min()),
                    "max_predicted_risk": float(g["p"].max()),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    root = project_root()
    out_dir = root / OUT_REL
    ensure_outdir(out_dir)

    step89_path = root / STEP89_SCRIPT_REL
    ckpt_path = root / CKPT_REL
    platt_path = root / PLATT_REL
    aligned_path = root / EXTERNAL_ALIGNED_REL
    cohort_path = root / EXTERNAL_COHORT_REL

    print("=" * 88)
    print("[STEP 95d] Frozen v4 external evaluation on NACC no_ADAS13")
    print(f"[SCRIPT VERSION] {SCRIPT_VERSION}")
    print(f"[STEP89 SCRIPT] {step89_path}")
    print(f"[CHECKPOINT] {ckpt_path}")
    print(f"[ALIGNED NPZ] {aligned_path}")
    print(f"[COHORT] {cohort_path}")
    print(f"[OUTPUT DIR] {out_dir}")
    print("=" * 88)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    step89 = load_step89_module(step89_path)

    # Make Step89 use the same device if it defines a global DEVICE.
    try:
        step89.DEVICE = device
    except Exception:
        pass

    try:
        step89.BATCH_SIZE = 512
    except Exception:
        pass

    print("[INFO] Loading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location=device)

    feature_names = list(ckpt["feature_names"])
    module_order = list(ckpt["module_order"])
    seq_input_dims = ckpt["seq_input_dims"]
    summary_input_dims = ckpt["summary_input_dims"]

    print(f"[INFO] feature_names={feature_names}")
    print(f"[INFO] module_order={module_order}")

    print("[INFO] Loading aligned external tokens...")
    data = load_npz(aligned_path)

    Xv = data["X_values_imputed_scaled"].astype(np.float32)
    Xm = data["X_feature_mask"].astype(np.float32)
    Xt = data["X_time_scaled"].astype(np.float32)
    Xd = data["X_delta_from_first"].astype(np.float32)
    Xs = data["X_slope_from_first"].astype(np.float32)
    visit_mask = data["X_visit_mask"].astype(np.float32)
    y_labels = data["y_labels"].astype(np.float32)
    y_obs = data["y_observed"].astype(np.float32)

    n = Xv.shape[0]
    indices = np.arange(n)

    print(f"[INFO] External N={n}")

    print("[INFO] Building v4 module inputs via Step89 contract...")
    built = step89.build_v4_module_inputs(
        Xv,
        Xm,
        Xt,
        Xd,
        Xs,
        visit_mask,
        feature_names,
    )
    seq_arrays, seq_masks, summary_arrays, availability = unpack_build_inputs_output(built)

    y_for_dataset = np.nan_to_num(y_labels, nan=0.0).astype(np.float32)

    print("[INFO] Creating dataset/loader...")
    ds = step89.V4Dataset(
        seq_arrays,
        seq_masks,
        summary_arrays,
        availability,
        y_for_dataset,
        y_obs,
        indices,
        module_order,
    )

    if hasattr(step89, "make_loader"):
        loader = step89.make_loader(ds, shuffle=False)
    else:
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=512, shuffle=False)

    print("[INFO] Loading frozen model state...")
    model = step89.SummaryAugmentedV4Model(
        seq_input_dims=seq_input_dims,
        summary_input_dims=summary_input_dims,
        module_order=module_order,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    scenario_mask = make_scenario_mask(module_order, SCENARIO)
    print(f"[INFO] scenario={SCENARIO}, scenario_mask={scenario_mask.tolist()}")

    print("[INFO] Predicting external NACC risks...")
    with torch.no_grad():
        pred_out = step89.predict(
            model,
            loader,
            module_order,
            scenario_mask,
            return_gate=True,
        )

    probs_raw, logits_raw, gates = unpack_predict_output(pred_out, n_expected=n, n_horizons=4)

    print("[INFO] Applying scenario-specific validation-set Platt calibration...")
    probs_platt, platt_df = apply_platt_from_csv(logits_raw, platt_path)

    print("[INFO] Computing metrics...")
    metrics_raw = compute_metrics(probs_raw, y_labels, y_obs, risk_type="raw_sigmoid")
    metrics_platt = compute_metrics(probs_platt, y_labels, y_obs, risk_type="scenario_platt")
    metrics = pd.concat([metrics_raw, metrics_platt], axis=0, ignore_index=True)

    bins_raw = calibration_bins(probs_raw, y_labels, y_obs, risk_type="raw_sigmoid")
    bins_platt = calibration_bins(probs_platt, y_labels, y_obs, risk_type="scenario_platt")
    bins = pd.concat([bins_raw, bins_platt], axis=0, ignore_index=True)

    print("[INFO] Building prediction table...")
    if cohort_path.exists():
        cohort = pd.read_csv(cohort_path)
    else:
        cohort = pd.DataFrame({"sample_index": np.arange(n)})

    if len(cohort) != n:
        print(f"[WARN] cohort rows={len(cohort)} != predictions={n}; using sample_index only.")
        cohort = pd.DataFrame({"sample_index": np.arange(n)})

    pred = cohort.copy()

    for h_idx, h_name in enumerate(HORIZON_NAMES):
        pred[f"y_{h_name}"] = y_labels[:, h_idx]
        pred[f"known_{h_name}"] = y_obs[:, h_idx].astype(int)
        pred[f"risk_raw_{h_name}"] = probs_raw[:, h_idx]
        pred[f"logit_raw_{h_name}"] = logits_raw[:, h_idx]
        pred[f"risk_platt_{h_name}"] = probs_platt[:, h_idx]

    # Output paths.
    metrics_path = out_dir / "external_metrics_raw_and_platt_no_ADAS13.csv"
    pred_path = out_dir / "external_predictions_no_ADAS13.csv"
    bins_path = out_dir / "external_calibration_bins_no_ADAS13.csv"
    platt_copy_path = out_dir / "scenario_platt_calibration_parameters_used.csv"
    readme_path = out_dir / "README_step95d_nacc_external_frozen_v4_no_ADAS13_eval.md"
    audit_path = out_dir / "step95d_audit.json"

    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pred.to_csv(pred_path, index=False, encoding="utf-8-sig")
    bins.to_csv(bins_path, index=False, encoding="utf-8-sig")
    platt_df.to_csv(platt_copy_path, index=False, encoding="utf-8-sig")

    gate_note = "Gate weights not exported."
    if gates is not None:
        try:
            if isinstance(gates, pd.DataFrame):
                gates.to_csv(out_dir / "external_gate_outputs_no_ADAS13.csv", index=False, encoding="utf-8-sig")
                gate_note = "Gate outputs exported as DataFrame."
            else:
                arr = np.asarray(gates)
                np.save(out_dir / "external_gate_outputs_no_ADAS13.npy", arr)
                gate_note = f"Gate outputs exported as npy, shape={arr.shape}."
        except Exception as e:
            gate_note = f"Gate output parsing/export failed: {repr(e)}"

    readme = f"""# Step95d NACC external frozen v4 evaluation: no_ADAS13

## Purpose

This step evaluates the frozen internal Step89 v4 model on the corrected NACC external first-MCI cohort.

No model retraining or NACC refitting is performed.

## Inputs

- Step89 script: `{step89_path}`
- Frozen checkpoint: `{ckpt_path}`
- External aligned NPZ: `{aligned_path}`
- External cohort CSV: `{cohort_path}`
- Scenario Platt calibration: `{platt_path}`

## External cohort

- Dataset: NACC investigator single table
- Landmark: first eligible MCI visit per patient
- Time axis: NACCVNUM
- Event: future `NACCUDSD == 4` and `NACCALZD == 1`
- Scenario: `no_ADAS13`
- Samples: {n}

## Metrics

{metrics.to_string(index=False)}

## Important interpretation

These are external validation results, not retraining results.

Because NACC `ADAS13` is unavailable, the main external scenario is `no_ADAS13`.

Because many NACC first-MCI patients have limited pre-landmark history, this external test partly evaluates low-history / first-landmark transportability.

Raw sigmoid and validation-set scenario-Platt calibrated risks are both reported. Scenario-Platt calibration was fit internally on ADNI validation data, not on NACC.

## Gate output

{gate_note}

## Output files

- `external_metrics_raw_and_platt_no_ADAS13.csv`
- `external_predictions_no_ADAS13.csv`
- `external_calibration_bins_no_ADAS13.csv`
- `scenario_platt_calibration_parameters_used.csv`
- `README_step95d_nacc_external_frozen_v4_no_ADAS13_eval.md`
- `step95d_audit.json`
"""

    readme_path.write_text(readme, encoding="utf-8")

    audit = {
        "script_version": SCRIPT_VERSION,
        "scenario": SCENARIO,
        "n_samples": int(n),
        "device": str(device),
        "step89_script": str(step89_path),
        "checkpoint": str(ckpt_path),
        "aligned_npz": str(aligned_path),
        "cohort": str(cohort_path),
        "metrics_csv": str(metrics_path),
        "predictions_csv": str(pred_path),
        "calibration_bins_csv": str(bins_path),
        "gate_note": gate_note,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print("=" * 88)
    print("[DONE] Step95d NACC external frozen v4 evaluation finished.")
    print("[EXTERNAL METRICS]")
    print(metrics.to_string(index=False))
    print("[OUTPUTS]")
    print(f"  metrics: {metrics_path}")
    print(f"  predictions: {pred_path}")
    print(f"  calibration bins: {bins_path}")
    print(f"  readme: {readme_path}")
    print(f"[GATE] {gate_note}")
    print("=" * 88)


if __name__ == "__main__":
    main()