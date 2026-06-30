# -*- coding: utf-8 -*-
"""
Step97: Finalize NACC external validation report package.

Purpose
-------
Collect the completed NACC external validation outputs into a clean, paper-ready
summary folder.

This step does NOT:
    - train
    - predict
    - recalibrate
    - delete files

It only:
    - consolidates final tables
    - marks deprecated diagnostic steps
    - writes claims audit
    - writes final README
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "v1_finalize_nacc_external_validation_report"


STEP95E_DIR_REL = Path("results") / "reports" / "95e_nacc_external_bootstrap_ci_sensitivity_calibration"
STEP96_DIR_REL = Path("results") / "reports" / "96_nacc_crossfit_local_recalibration_audit"
STEP95D_DIR_REL = Path("results") / "reports" / "95d_nacc_external_frozen_v4_no_adas13_eval"
STEP94D_DIR_REL = Path("results") / "features" / "94d_nacc_external_raw_tokens_corrected_naccvnum"
STEP95C_DIR_REL = Path("results") / "features" / "95c_nacc_external_aligned_tokens"

OUT_REL = Path("results") / "reports" / "97_nacc_external_validation_final_report"

HORIZON_ORDER = ["1y", "2y", "3y", "5y"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def format_ci(point: float, low: float, high: float, decimals: int = 3) -> str:
    if pd.isna(point) or pd.isna(low) or pd.isna(high):
        return ""
    return f"{point:.{decimals}f} ({low:.{decimals}f}–{high:.{decimals}f})"


def make_frozen_external_main_table(boot: pd.DataFrame) -> pd.DataFrame:
    """
    Paper-ready frozen external discrimination table from Step95e bootstrap CI.
    Main risk type: raw_sigmoid, seq_group all_seq_len_ge_1.
    """
    main = boot[
        (boot["seq_group"] == "all_seq_len_ge_1")
        & (boot["risk_type"] == "raw_sigmoid")
    ].copy()

    rows = []

    for h in HORIZON_ORDER:
        sub = main[main["horizon"].eq(h)]
        if len(sub) == 0:
            continue

        row_base = sub.iloc[0]

        out = {
            "horizon": h,
            "horizon_year": int(row_base["horizon_year"]),
            "n_known": int(row_base["n_known"]),
            "n_event": int(row_base["n_event"]),
            "n_nonevent": int(row_base["n_nonevent"]),
        }

        for metric in ["AUROC", "AUPRC", "Brier"]:
            r = sub[sub["metric"].eq(metric)].iloc[0]
            out[metric] = float(r["point"])
            out[f"{metric}_95CI"] = format_ci(r["point"], r["ci_low"], r["ci_high"])

        rows.append(out)

    return pd.DataFrame(rows)


def make_sequence_length_table(seq_dist: pd.DataFrame) -> pd.DataFrame:
    return seq_dist.copy()


def make_recalibration_paper_table(metrics96: pd.DataFrame) -> pd.DataFrame:
    """
    Compact table comparing raw frozen, ADNI Platt, and NACC Platt.
    """
    keep_methods = [
        "raw_sigmoid",
        "adni_scenario_platt",
        "nacc_platt_raw",
        "nacc_isotonic_raw",
    ]

    m = metrics96[metrics96["method"].isin(keep_methods)].copy()

    method_name = {
        "raw_sigmoid": "Frozen raw sigmoid",
        "adni_scenario_platt": "ADNI validation Platt",
        "nacc_platt_raw": "NACC cross-fitted Platt",
        "nacc_isotonic_raw": "NACC cross-fitted isotonic",
    }

    m["method_label"] = m["method"].map(method_name)

    cols = [
        "method_label",
        "horizon",
        "n_known",
        "event_rate",
        "AUROC",
        "AUPRC",
        "Brier",
        "mean_predicted_risk",
        "calibration_gap_mean_pred_minus_observed",
    ]

    out = m[cols].copy()
    out = out.sort_values(
        by=["horizon", "method_label"],
        key=lambda s: s.map({h: i for i, h in enumerate(HORIZON_ORDER)}) if s.name == "horizon" else s,
    )

    return out


def make_recalibration_delta_table(delta96: pd.DataFrame) -> pd.DataFrame:
    keep = delta96[delta96["method"].isin(["adni_scenario_platt", "nacc_platt_raw", "nacc_isotonic_raw"])].copy()
    return keep[
        [
            "horizon",
            "horizon_year",
            "method",
            "delta_AUROC_vs_raw",
            "delta_AUPRC_vs_raw",
            "delta_Brier_vs_raw",
            "delta_abs_calibration_gap_vs_raw",
            "raw_Brier",
            "method_Brier",
            "raw_gap",
            "method_gap",
        ]
    ].copy()


def make_claims_audit() -> pd.DataFrame:
    rows = [
        {
            "claim": "Frozen ADNI-trained v4 was externally validated on NACC.",
            "status": "supported",
            "wording": "Frozen v4 was externally evaluated on a NACC first-MCI cohort under the no_ADAS13 scenario.",
            "caution": "Do not call this full-module external validation because ADAS13 is unavailable in NACC.",
        },
        {
            "claim": "The model retained external discrimination.",
            "status": "supported",
            "wording": "The model retained moderate external discrimination, with AUROC approximately 0.733–0.749 across 1/2/3/5-year horizons.",
            "caution": "Report confidence intervals and external setting.",
        },
        {
            "claim": "The raw frozen risks are well calibrated on NACC.",
            "status": "not supported",
            "wording": "Raw frozen risks showed systematic underestimation of absolute conversion risk, especially for 3y/5y.",
            "caution": "Do not claim externally accurate absolute probabilities from raw frozen outputs.",
        },
        {
            "claim": "Local recalibration improves external calibration.",
            "status": "supported as post-hoc sensitivity",
            "wording": "Cross-fitted NACC local Platt recalibration improved Brier score and calibration-in-the-large without retraining the prediction model.",
            "caution": "Label this as local recalibration sensitivity, not untouched external validation.",
        },
        {
            "claim": "The model is clinically deployable.",
            "status": "not supported",
            "wording": "The model is suitable for retrospective external risk stratification analysis, but prospective validation and local calibration are required before clinical deployment.",
            "caution": "Avoid deployment claims.",
        },
        {
            "claim": "This is rich longitudinal external validation.",
            "status": "partially supported / limited",
            "wording": "External validation was performed in a first-MCI, low-history NACC setting.",
            "caution": "Most NACC samples have sequence_length=1, so avoid overstating longitudinal history.",
        },
    ]

    return pd.DataFrame(rows)


def make_step_manifest(root: Path) -> pd.DataFrame:
    rows = [
        {
            "step": "93b",
            "status": "keep",
            "role": "NACC investigator-only schema audit",
            "main_output": "results/reports/93b_nacc_investigator_only_schema_audit",
            "paper_use": "Methods / data mapping support",
        },
        {
            "step": "94 original",
            "status": "deprecated",
            "role": "Initial NACC raw token attempt using NACCDAYS",
            "main_output": "results/features/94_nacc_external_raw_tokens",
            "paper_use": "Do not use; all-positive label failure diagnostic only",
        },
        {
            "step": "94b",
            "status": "diagnostic",
            "role": "NACCDAYS label sanity audit",
            "main_output": "results/reports/94b_nacc_label_sanity_audit",
            "paper_use": "Internal audit only",
        },
        {
            "step": "94c / 94c-fast",
            "status": "diagnostic keep",
            "role": "Time-axis/event-definition sanity audit",
            "main_output": "results/reports/94c...",
            "paper_use": "Supports choosing NACCVNUM",
        },
        {
            "step": "94d",
            "status": "keep",
            "role": "Corrected NACC first-MCI raw token builder",
            "main_output": str(root / STEP94D_DIR_REL),
            "paper_use": "External cohort construction",
        },
        {
            "step": "95a",
            "status": "diagnostic keep",
            "role": "External eval preflight inspect",
            "main_output": "console/report",
            "paper_use": "Internal audit only",
        },
        {
            "step": "95b",
            "status": "diagnostic keep",
            "role": "v4 schema/preprocessing contract inspect",
            "main_output": "console/report",
            "paper_use": "Internal audit only",
        },
        {
            "step": "95c",
            "status": "keep",
            "role": "NACC aligned tokens for frozen v4",
            "main_output": str(root / STEP95C_DIR_REL),
            "paper_use": "External preprocessing alignment",
        },
        {
            "step": "95d",
            "status": "keep",
            "role": "Frozen v4 NACC no_ADAS13 evaluation",
            "main_output": str(root / STEP95D_DIR_REL),
            "paper_use": "Main external validation metrics",
        },
        {
            "step": "95e",
            "status": "keep",
            "role": "External bootstrap CI, calibration, sequence-length sensitivity",
            "main_output": str(root / STEP95E_DIR_REL),
            "paper_use": "Main external validation report",
        },
        {
            "step": "96",
            "status": "keep as post-hoc sensitivity",
            "role": "Cross-fitted NACC local recalibration audit",
            "main_output": str(root / STEP96_DIR_REL),
            "paper_use": "Calibration sensitivity / model updating by recalibration",
        },
        {
            "step": "97",
            "status": "keep",
            "role": "Final external validation report package",
            "main_output": str(root / OUT_REL),
            "paper_use": "Final paper-ready summary",
        },
    ]

    return pd.DataFrame(rows)


def main() -> None:
    root = project_root()
    out_dir = root / OUT_REL
    ensure_outdir(out_dir)

    step95e = root / STEP95E_DIR_REL
    step96 = root / STEP96_DIR_REL

    print("=" * 88)
    print("[STEP 97] Finalize NACC external validation report")
    print(f"[SCRIPT VERSION] {SCRIPT_VERSION}")
    print(f"[STEP95E DIR] {step95e}")
    print(f"[STEP96 DIR] {step96}")
    print(f"[OUTPUT DIR] {out_dir}")
    print("=" * 88)

    boot95e = read_csv_required(step95e / "external_bootstrap_ci_by_sequence_length.csv")
    seq_dist95e = read_csv_required(step95e / "external_sequence_length_distribution.csv")
    sens95e = read_csv_required(step95e / "external_sensitivity_metrics_by_sequence_length.csv")
    cal95e = read_csv_required(step95e / "external_calibration_summary_by_sequence_length.csv")

    metrics96 = read_csv_required(step96 / "nacc_crossfit_recalibration_metrics.csv")
    cal96 = read_csv_required(step96 / "nacc_crossfit_recalibration_calibration_summary.csv")
    delta96 = read_csv_required(step96 / "nacc_crossfit_recalibration_delta_vs_raw.csv")
    params96 = read_csv_required(step96 / "nacc_crossfit_recalibration_parameters.csv")

    frozen_table = make_frozen_external_main_table(boot95e)
    seq_table = make_sequence_length_table(seq_dist95e)
    recal_table = make_recalibration_paper_table(metrics96)
    recal_delta = make_recalibration_delta_table(delta96)
    claims = make_claims_audit()
    manifest = make_step_manifest(root)

    # Paper-ready filtered tables.
    sensitivity_main = sens95e[
        sens95e["risk_type"].eq("raw_sigmoid")
    ].copy()

    calibration_raw_main = cal95e[
        (cal95e["seq_group"] == "all_seq_len_ge_1")
        & (cal95e["risk_type"].eq("raw_sigmoid"))
    ].copy()

    calibration_recal_main = cal96[
        cal96["method"].isin(["raw_sigmoid", "adni_scenario_platt", "nacc_platt_raw", "nacc_isotonic_raw"])
    ].copy()

    # Output paths.
    frozen_path = out_dir / "paper_table_1_frozen_nacc_external_main_metrics.csv"
    seq_path = out_dir / "paper_table_2_nacc_sequence_length_distribution.csv"
    sens_path = out_dir / "paper_table_3_sequence_length_sensitivity.csv"
    cal_raw_path = out_dir / "paper_table_4_raw_frozen_calibration_audit.csv"
    recal_path = out_dir / "paper_table_5_crossfit_local_recalibration.csv"
    recal_delta_path = out_dir / "paper_table_6_recalibration_delta_vs_raw.csv"
    claims_path = out_dir / "external_validation_claims_audit.csv"
    manifest_path = out_dir / "external_validation_step_manifest.csv"
    params_path = out_dir / "crossfit_recalibration_parameters_copy.csv"
    readme_path = out_dir / "README_step97_nacc_external_validation_final_report.md"
    audit_path = out_dir / "step97_audit.json"

    frozen_table.to_csv(frozen_path, index=False, encoding="utf-8-sig")
    seq_table.to_csv(seq_path, index=False, encoding="utf-8-sig")
    sensitivity_main.to_csv(sens_path, index=False, encoding="utf-8-sig")
    calibration_raw_main.to_csv(cal_raw_path, index=False, encoding="utf-8-sig")
    recal_table.to_csv(recal_path, index=False, encoding="utf-8-sig")
    recal_delta.to_csv(recal_delta_path, index=False, encoding="utf-8-sig")
    claims.to_csv(claims_path, index=False, encoding="utf-8-sig")
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    params96.to_csv(params_path, index=False, encoding="utf-8-sig")

    # Extract compact text values for README.
    n_total = int(seq_table.loc[seq_table["sequence_length"].astype(str).eq("all_seq_len_ge_1"), "n"].iloc[0])
    n_len1 = int(seq_table.loc[seq_table["sequence_length"].astype(str).eq("1"), "n"].iloc[0])
    rate_len1 = float(seq_table.loc[seq_table["sequence_length"].astype(str).eq("1"), "rate"].iloc[0])

    readme = f"""# Step97 Final NACC External Validation Report

## Final status

The NACC external validation branch is now closed and ready for manuscript writing.

This folder consolidates the paper-ready external validation tables from:

- Step94d: corrected NACC first-MCI raw token construction
- Step95c: NACC aligned tokens for frozen v4 evaluation
- Step95d: frozen v4 NACC no_ADAS13 external evaluation
- Step95e: bootstrap CI, calibration audit, and sequence-length sensitivity
- Step96: cross-fitted NACC local recalibration audit

No training, prediction, recalibration, or deletion was performed in Step97.

## Main external validation setting

- Training cohort/model: ADNI-trained frozen v4
- External cohort: NACC investigator table
- Landmark: first eligible MCI visit per patient
- Event: future AD dementia
- Time axis: NACCVNUM
- Scenario: no_ADAS13
- External sample size: {n_total}
- sequence_length = 1: {n_len1} ({rate_len1:.1%})

## Main frozen external results

{frozen_table.to_string(index=False)}

## Sequence-length distribution

{seq_table.to_string(index=False)}

## Raw frozen calibration audit

{calibration_raw_main.to_string(index=False)}

## Cross-fitted local recalibration

{recal_table.to_string(index=False)}

## Recalibration delta vs raw

{recal_delta.to_string(index=False)}

## Final recommended manuscript wording

The frozen ADNI-trained v4 model retained moderate external discrimination on the NACC first-MCI cohort under the no_ADAS13 scenario, with AUROC approximately 0.733–0.749 across 1/2/3/5-year horizons. Raw frozen absolute risks were systematically underestimated on NACC, especially for longer horizons, indicating cross-cohort calibration shift. Cross-fitted NACC local Platt recalibration improved calibration and Brier score without retraining the prediction model, supporting the need for cohort-specific recalibration before clinical interpretation.

## Claims boundary

Use:
- external validation under no_ADAS13
- moderate external discrimination
- calibration shift under cross-cohort transport
- local recalibration improves absolute risk calibration

Avoid:
- full-module external validation
- clinically deployable model
- externally well-calibrated raw probabilities
- rich longitudinal external validation without caveat
- NACC-retrained model

## Output files

1. `paper_table_1_frozen_nacc_external_main_metrics.csv`
2. `paper_table_2_nacc_sequence_length_distribution.csv`
3. `paper_table_3_sequence_length_sensitivity.csv`
4. `paper_table_4_raw_frozen_calibration_audit.csv`
5. `paper_table_5_crossfit_local_recalibration.csv`
6. `paper_table_6_recalibration_delta_vs_raw.csv`
7. `external_validation_claims_audit.csv`
8. `external_validation_step_manifest.csv`
9. `crossfit_recalibration_parameters_copy.csv`
10. `README_step97_nacc_external_validation_final_report.md`
11. `step97_audit.json`
"""

    readme_path.write_text(readme, encoding="utf-8")

    audit = {
        "script_version": SCRIPT_VERSION,
        "output_dir": str(out_dir),
        "source_step95e": str(step95e),
        "source_step96": str(step96),
        "n_external": n_total,
        "sequence_length_1_n": n_len1,
        "sequence_length_1_rate": rate_len1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "outputs": {
            "frozen_main": str(frozen_path),
            "sequence_length_distribution": str(seq_path),
            "sequence_length_sensitivity": str(sens_path),
            "raw_calibration": str(cal_raw_path),
            "recalibration": str(recal_path),
            "recalibration_delta": str(recal_delta_path),
            "claims_audit": str(claims_path),
            "manifest": str(manifest_path),
            "readme": str(readme_path),
        },
    }

    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print("=" * 88)
    print("[DONE] Step97 final external validation report created.")
    print("[FROZEN EXTERNAL MAIN TABLE]")
    print(frozen_table.to_string(index=False))
    print("[RAW CALIBRATION AUDIT]")
    print(calibration_raw_main.to_string(index=False))
    print("[CROSSFIT LOCAL RECALIBRATION TABLE]")
    print(recal_table.to_string(index=False))
    print("[CLAIMS AUDIT]")
    print(claims.to_string(index=False))
    print("[OUTPUT DIR]")
    print(out_dir)
    print("=" * 88)


if __name__ == "__main__":
    main()