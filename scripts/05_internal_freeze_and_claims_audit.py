# Step 90: freeze summary-augmented modular v4 mainline and audit claims.
# Purpose: consolidate the Step89 v4 result as the new internal modular mainline.
# No training. No inference. No external data. No raw-data edits. No deletion.

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import numpy as np


SCRIPT_VERSION = "v1_internal_modular_v4_mainline_freeze_claims_audit"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEP89_MODEL_DIR = PROJECT_ROOT / "results" / "models" / "89_summary_augmented_scenario_dropout_modular_v4_internal"
STEP89_REPORT_DIR = PROJECT_ROOT / "results" / "reports" / "89_summary_augmented_scenario_dropout_modular_v4_internal"

STEP89_METRICS = STEP89_MODEL_DIR / "metrics_by_scenario_raw_and_scenario_platt.csv"
STEP89_GATES = STEP89_MODEL_DIR / "gate_summary_by_scenario.csv"
STEP89_CALIBRATION = STEP89_MODEL_DIR / "scenario_platt_calibration_parameters.csv"
STEP89_CONFIG = STEP89_MODEL_DIR / "config.json"
STEP89_SCHEMA = STEP89_MODEL_DIR / "module_schema.csv"
STEP89_TRAINING_HISTORY = STEP89_MODEL_DIR / "training_history.csv"
STEP89_CKPT = STEP89_MODEL_DIR / "best_model.pt"

STEP89_COMPARISON = STEP89_REPORT_DIR / "v4_vs_v3_and_best_baseline_by_scenario.csv"
STEP89_PAPER_TABLE = STEP89_REPORT_DIR / "paper_table_v4_vs_v3_baseline_stress.csv"
STEP89_README = STEP89_REPORT_DIR / "README_step89_summary_augmented_scenario_dropout_modular_v4.md"

STEP88_COMBINED = PROJECT_ROOT / "results" / "reports" / "88_internal_baseline_missing_feature_stress_comparison" / "combined_v3_and_baseline_stress_metrics.csv"
STEP88_COMPARISON = PROJECT_ROOT / "results" / "reports" / "88_internal_baseline_missing_feature_stress_comparison" / "v3_vs_best_baseline_by_scenario.csv"
STEP86_FREEZE = PROJECT_ROOT / "results" / "reports" / "86_internal_modular_v3_freeze_and_claims_audit" / "MAINLINE_FREEZE_INTERNAL_MODULAR_V3.md"

OUT_DIR = PROJECT_ROOT / "results" / "reports" / "90_internal_modular_v4_freeze_and_claims_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAIN_TITLE = "Summary-augmented feature-level modular longitudinal risk prediction for MCI-to-AD conversion under incomplete clinical assessment"
MAIN_MODEL = "step89_summary_augmented_modular_v4"
MAIN_RISK = "scenario-specific validation-set Platt-calibrated risk"
MODEL_FAMILY = "summary-augmented feature-level modular longitudinal GRU with scenario-dropout training"


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
    except Exception:
        return pd.DataFrame()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def fmt(x, digits=3) -> str:
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


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


def evidence_inventory() -> pd.DataFrame:
    items = [
        ("step89_checkpoint", STEP89_CKPT, "Locked v4 checkpoint."),
        ("step89_metrics", STEP89_METRICS, "V4 raw/scenario-Platt metrics across full and missing-feature scenarios."),
        ("step89_gates", STEP89_GATES, "V4 gate behavior across scenarios."),
        ("step89_calibration", STEP89_CALIBRATION, "Scenario-specific validation Platt calibration parameters."),
        ("step89_config", STEP89_CONFIG, "V4 model and training configuration."),
        ("step89_schema", STEP89_SCHEMA, "V4 module schema and summary components."),
        ("step89_training_history", STEP89_TRAINING_HISTORY, "V4 training history."),
        ("step89_comparison", STEP89_COMPARISON, "V4 versus v3 and best scenario-specific baseline."),
        ("step89_paper_table", STEP89_PAPER_TABLE, "Compact paper-style v4 comparison table."),
        ("step88_combined", STEP88_COMBINED, "Scenario-specific baseline stress metrics from Step88."),
        ("step86_v3_freeze", STEP86_FREEZE, "Previous v3 mainline freeze for lineage context."),
    ]
    rows = []
    for name, path, role in items:
        rows.append(
            {
                "evidence_item": name,
                "relative_path": rel(path),
                "exists": bool(path.exists()),
                "type": "dir" if path.exists() and path.is_dir() else ("file" if path.exists() else "missing"),
                "role": role,
            }
        )
    return pd.DataFrame(rows)


def full_module_metrics() -> pd.DataFrame:
    df = read_csv(STEP89_METRICS)
    if df.empty:
        return df
    d = df[
        (df["split"].astype(str).eq("test"))
        & (df["scenario"].astype(str).eq("full_all_modules"))
        & (df["risk_type"].astype(str).eq("scenario_platt"))
    ].copy()
    keep = ["horizon_year", "observed_n", "positive_n", "AUROC", "AUPRC", "Brier"]
    return d[[c for c in keep if c in d.columns]].sort_values("horizon_year")


def scenario_metrics() -> pd.DataFrame:
    df = read_csv(STEP89_METRICS)
    if df.empty:
        return df
    key_scenarios = [
        "full_all_modules",
        "no_ADAS13",
        "no_MMSE",
        "no_MMSE_ADAS13",
        "no_FAQTOTAL",
        "no_CDGLOBAL_CDRSB",
        "no_visit_process",
        "basic_plus_MMSE_CDRSB_FAQ",
        "basic_plus_MMSE_only",
        "basic_only",
        "visit_process_only",
    ]
    d = df[
        (df["split"].astype(str).eq("test"))
        & (df["risk_type"].astype(str).eq("scenario_platt"))
        & (df["scenario"].isin(key_scenarios))
    ].copy()
    keep = ["scenario", "horizon_year", "observed_n", "positive_n", "AUROC", "AUPRC", "Brier"]
    return d[[c for c in keep if c in d.columns]].sort_values(["scenario", "horizon_year"])


def gate_summary_full() -> pd.DataFrame:
    df = read_csv(STEP89_GATES)
    if df.empty:
        return df
    d = df[
        (df["split"].astype(str).eq("test"))
        & (df["scenario"].astype(str).eq("full_all_modules"))
    ].copy()
    keep = ["module", "kept_by_scenario", "mean_gate_weight", "median_gate_weight", "std_gate_weight"]
    d = d[[c for c in keep if c in d.columns]].copy()
    if "mean_gate_weight" in d.columns:
        d = d.sort_values("mean_gate_weight", ascending=False)
    return d


def comparison_table() -> pd.DataFrame:
    df = read_csv(STEP89_COMPARISON)
    if df.empty:
        return df
    key_scenarios = [
        "full_all_modules",
        "no_ADAS13",
        "no_MMSE_ADAS13",
        "no_FAQTOTAL",
        "basic_plus_MMSE_CDRSB_FAQ",
        "basic_plus_MMSE_only",
        "basic_only",
        "visit_process_only",
    ]
    d = df[df["scenario"].isin(key_scenarios)].copy()
    keep = [
        "scenario", "horizon_year",
        "v4_AUROC", "v3_AUROC", "v4_minus_v3_AUROC", "best_baseline_AUROC", "v4_minus_best_baseline_AUROC",
        "v4_AUPRC", "v3_AUPRC", "v4_minus_v3_AUPRC", "best_baseline_AUPRC", "v4_minus_best_baseline_AUPRC",
        "v4_Brier", "v3_Brier", "v4_minus_v3_Brier", "best_baseline_Brier", "v4_minus_best_baseline_Brier",
    ]
    return d[[c for c in keep if c in d.columns]].sort_values(["scenario", "horizon_year"])


def scenario_gain_summary() -> pd.DataFrame:
    df = read_csv(STEP89_COMPARISON)
    if df.empty:
        return df
    rows = []
    for scenario, g in df.groupby("scenario"):
        rows.append(
            {
                "scenario": scenario,
                "mean_v4_minus_v3_AUROC": float(g["v4_minus_v3_AUROC"].mean()),
                "mean_v4_minus_v3_AUPRC": float(g["v4_minus_v3_AUPRC"].mean()),
                "mean_v4_minus_v3_Brier": float(g["v4_minus_v3_Brier"].mean()),
                "mean_v4_minus_best_baseline_AUROC": float(g["v4_minus_best_baseline_AUROC"].mean()),
                "mean_v4_minus_best_baseline_AUPRC": float(g["v4_minus_best_baseline_AUPRC"].mean()),
                "mean_v4_minus_best_baseline_Brier": float(g["v4_minus_best_baseline_Brier"].mean()),
                "n_horizons": int(len(g)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_v4_minus_v3_AUROC", ascending=False)


def claims_audit() -> pd.DataFrame:
    rows = [
        {
            "claim": "The main internal model is a summary-augmented feature-level modular longitudinal model.",
            "status": "supported",
            "evidence": rel(STEP89_SCHEMA),
            "safe_wording": "The selected v4 model uses one module per clinical feature plus a visit-process module, and each module combines observed-visit temporal encoding with module-local trajectory summaries.",
            "unsafe_wording": "The model is a full reproduction of MoDN/MultiModN.",
        },
        {
            "claim": "Module-local trajectory summaries improved the modular model.",
            "status": "supported",
            "evidence": rel(STEP89_COMPARISON),
            "safe_wording": "Compared with the v3 modular model, v4 improved internal scenario-Platt AUROC/AUPRC/Brier across most full and missing-feature scenarios.",
            "unsafe_wording": "Hand-crafted summaries prove deep learning is unnecessary.",
        },
        {
            "claim": "Scenario-dropout training improved missing-feature behavior.",
            "status": "supported_as_internal_model_behavior",
            "evidence": rel(STEP89_COMPARISON),
            "safe_wording": "Internal scenario tests showed substantial gains over v3 under no_ADAS13, no_MMSE_ADAS13, and low-burden scenarios.",
            "unsafe_wording": "The model is guaranteed robust to all missingness patterns.",
        },
        {
            "claim": "V4 is competitive with scenario-specific strong baselines.",
            "status": "supported_with_boundary",
            "evidence": rel(STEP89_COMPARISON),
            "safe_wording": "V4 approached or exceeded the best scenario-specific baseline in selected scenario-horizon AUROC/Brier comparisons, while remaining a single modular model.",
            "unsafe_wording": "V4 universally outperforms all HGB/logistic baselines.",
        },
        {
            "claim": "Strong dynamic tabular baselines remain important.",
            "status": "supported",
            "evidence": rel(STEP88_COMBINED),
            "safe_wording": "Scenario-specific history-summary and flattened-dynamic baselines remained strong, motivating the v4 module-local summary augmentation.",
            "unsafe_wording": "Baselines are weak or irrelevant.",
        },
        {
            "claim": "The model can do availability-aware inference as a single model.",
            "status": "supported",
            "evidence": rel(STEP89_CONFIG),
            "safe_wording": "The v4 model is evaluated by masking feature modules at inference time without retraining the neural model for each scenario.",
            "unsafe_wording": "No calibration or adaptation is needed under any scenario.",
        },
        {
            "claim": "Scenario-specific Platt calibration supports probability reporting.",
            "status": "supported_with_boundary",
            "evidence": rel(STEP89_CALIBRATION),
            "safe_wording": "Scenario-specific Platt calibration was fitted on validation logits and used for internal test probability metrics.",
            "unsafe_wording": "Calibration is solved for deployment.",
        },
        {
            "claim": "Gate weights are clinically interpretable module importance.",
            "status": "supported_as_model_diagnostic_only",
            "evidence": rel(STEP89_GATES),
            "safe_wording": "Gate weights are reported as model-behavior diagnostics describing module usage.",
            "unsafe_wording": "Gate weights are causal clinical importance.",
        },
        {
            "claim": "The model is clinically deployable.",
            "status": "not_supported",
            "evidence": "No prospective study, deployment evaluation, safety assessment, or external validation in this freeze.",
            "safe_wording": "This is a retrospective internal research model.",
            "unsafe_wording": "The model can be used for patient-level clinical decisions.",
        },
        {
            "claim": "The model is externally validated.",
            "status": "not_supported_yet",
            "evidence": "External validation is intentionally deferred until after internal refinement.",
            "safe_wording": "External validation is a planned next-stage evaluation after internal freeze.",
            "unsafe_wording": "The model generalizes across cohorts.",
        },
    ]
    return pd.DataFrame(rows)


def mainline_freeze_md(metrics, gates, comparison, gain, claims) -> str:
    lines = []
    lines.append("# Step90 internal modular v4 mainline freeze\n\n")
    lines.append("## Frozen project title\n\n")
    lines.append(f"**{MAIN_TITLE}**\n\n")
    lines.append("## Frozen main model\n\n")
    lines.append(f"- Main model: **{MAIN_MODEL}**\n")
    lines.append(f"- Main reported risk: **{MAIN_RISK}**\n")
    lines.append(f"- Model family: **{MODEL_FAMILY}**\n")
    lines.append("- Internal-only freeze: all conclusions in this report are based on internal split analyses.\n\n")

    lines.append("## Why v4 replaces v3 as the mainline\n\n")
    lines.append(
        "Step88 showed that strong scenario-specific history-summary and flattened-dynamic baselines remained highly competitive. "
        "Instead of ignoring that result, v4 integrates the lesson into the modular architecture: each clinical feature module combines observed-visit GRU encoding with module-local longitudinal summaries. "
        "Training also uses scenario dropout, exposing the single modular model to clinically meaningful missing-feature patterns. "
        "Step89 then showed broad improvements over v3, especially under no_ADAS13, no_MMSE_ADAS13, and low-burden scenarios.\n\n"
    )

    lines.append("## Full-module internal test performance\n\n")
    lines.append(md_table(metrics))
    lines.append("\n## Full-module gate behavior\n\n")
    lines.append(md_table(gates))
    lines.append("\n## V4 versus v3 and best scenario-specific baselines\n\n")
    lines.append(md_table(comparison, max_rows=120))
    lines.append("\n## Average scenario gains\n\n")
    lines.append(md_table(gain))
    lines.append("\n## Frozen interpretation\n\n")
    lines.append(
        "The main contribution should now be framed as a summary-augmented, scenario-trained, feature-level modular framework. "
        "The strongest internal evidence is that v4 preserves single-model availability-aware inference while substantially improving v3 under missing-feature scenarios. "
        "V4 can be described as competitive with strong scenario-specific baselines, but not universally superior to all of them.\n\n"
    )
    lines.append("## Do not claim\n\n")
    lines.append("- Do not claim universal superiority over all baselines.\n")
    lines.append("- Do not claim clinical deployment readiness.\n")
    lines.append("- Do not claim external validation yet.\n")
    lines.append("- Do not claim causal module importance from gate weights or stress tests.\n")
    lines.append("- Do not claim full reproduction of PROMISE-AD, MoDN, or MultiModN.\n\n")
    lines.append("## Claims audit summary\n\n")
    lines.append(md_table(claims[["claim", "status", "safe_wording"]], max_rows=30))
    return "".join(lines)


def paper_outline_md() -> str:
    lines = []
    lines.append(f"# Paper outline: {MAIN_TITLE}\n\n")
    lines.append("## Core message\n\n")
    lines.append(
        "Strong dynamic tabular baselines showed that longitudinal summary information is highly prognostic. "
        "We therefore refined a feature-level modular GRU into a summary-augmented, scenario-dropout-trained architecture that retains availability-aware inference while improving robustness under missing-feature scenarios.\n\n"
    )
    lines.append("## 1. Introduction\n\n")
    lines.append("- MCI-to-AD conversion risk prediction requires longitudinal clinical information.\n")
    lines.append("- Incomplete clinical assessment is common and motivates availability-aware modeling.\n")
    lines.append("- Strong tabular baselines can be highly competitive, so modular deep models should be audited rather than simply assumed superior.\n")
    lines.append("- This work studies whether module-local trajectory summaries and scenario-dropout training can improve a single feature-level modular model.\n\n")

    lines.append("## 2. Methods\n\n")
    lines.append("### 2.1 Dynamic longitudinal token construction\n")
    lines.append("- Values, feature masks, timing features, delta-from-first, slope-like change, visit masks, and observed multi-horizon labels.\n\n")
    lines.append("### 2.2 Internal baseline and stress-test audits\n")
    lines.append("- Same-split baseline comparison.\n")
    lines.append("- Missing-feature stress tests.\n")
    lines.append("- Scenario-specific HGB/logistic baselines as strong oracle-style comparators.\n\n")
    lines.append("### 2.3 Summary-augmented feature-level modular v4\n")
    lines.append("- One module per clinical feature plus visit-process module.\n")
    lines.append("- Observed-visit GRU encoder per module.\n")
    lines.append("- Module-local summary embedding: last, mean, min, max, std, last-first change, last delta, last slope, observed count, observed rate.\n")
    lines.append("- Availability-aware gated fusion.\n")
    lines.append("- Scenario-dropout training with clinically meaningful feature-removal scenarios.\n")
    lines.append("- Scenario-specific validation Platt calibration.\n\n")
    lines.append("## 3. Results\n\n")
    lines.append("### 3.1 Strong baselines motivate summary augmentation\n")
    lines.append("- History-summary and flattened-dynamic baselines remain competitive.\n\n")
    lines.append("### 3.2 V4 improves full-module and missing-feature performance over v3\n")
    lines.append("- Report full-module 1/2/3/5-year AUROC/AUPRC/Brier.\n")
    lines.append("- Report v4 minus v3 changes across no_ADAS13, no_MMSE_ADAS13, low-burden scenarios.\n\n")
    lines.append("### 3.3 V4 becomes competitive with scenario-specific baselines\n")
    lines.append("- Highlight where v4 approaches or exceeds best baseline.\n")
    lines.append("- Clearly state where scenario-specific baselines remain better.\n\n")
    lines.append("### 3.4 Gate and stress-test behavior\n")
    lines.append("- Gate weights as model diagnostics.\n")
    lines.append("- Scenario behavior as internal audit, not deployment proof.\n\n")
    lines.append("## 4. Discussion\n\n")
    lines.append("- Baseline audit changed the architecture design.\n")
    lines.append("- Summary augmentation provides a practical inductive bias for small longitudinal clinical datasets.\n")
    lines.append("- Scenario dropout improves single-model missing-feature behavior.\n")
    lines.append("- The contribution is modular availability-aware modeling and auditability, not universal performance dominance.\n\n")
    lines.append("## 5. Limitations\n\n")
    lines.append("- Retrospective internal analysis.\n")
    lines.append("- Scenario-specific calibration is internal.\n")
    lines.append("- Gate weights are diagnostics, not causal effects.\n")
    lines.append("- External validation and prospective evaluation remain future work.\n")
    return "".join(lines)


def outreach_description_md(metrics, gain) -> str:
    # Create a compact description useful for emails.
    full5 = metrics[metrics["horizon_year"].astype(float).eq(5.0)].iloc[0] if not metrics.empty and any(metrics["horizon_year"].astype(float).eq(5.0)) else None
    lines = []
    lines.append("# Outreach project description: modular v4\n\n")
    lines.append("## Short paragraph\n\n")
    lines.append(
        "I am refining a summary-augmented feature-level modular longitudinal model for predicting MCI-to-Alzheimer’s disease conversion under incomplete clinical assessment. "
        "The project began with dynamic longitudinal visit-token construction and same-split baseline audits. After finding that history-summary and flattened-dynamic tabular baselines were highly competitive, I redesigned the modular GRU so that each clinical feature module combines observed-visit temporal encoding with its own trajectory summaries. "
        "I also introduced scenario-dropout training to expose the single modular model to clinically meaningful missing-feature patterns such as missing ADAS13 or low-burden input sets.\n\n"
    )
    lines.append("## Result sentence\n\n")
    if full5 is not None:
        lines.append(
            f"Internally, the current v4 model achieved 5-year AUROC {fmt(full5['AUROC'])}, AUPRC {fmt(full5['AUPRC'])}, and Brier {fmt(full5['Brier'])} under the full-module setting, "
            "while substantially improving over the previous v3 model in missing-feature scenarios.\n\n"
        )
    else:
        lines.append("Internally, the current v4 model improved over the previous v3 model in missing-feature scenarios.\n\n")
    lines.append("## Honest boundary\n\n")
    lines.append(
        "I frame the project as an availability-aware, auditable clinical-AI modeling study rather than as a claim of universal superiority over all tabular baselines. "
        "External validation is planned after this internal freeze.\n"
    )
    return "".join(lines)


def main():
    print("=" * 88)
    print("[STEP 90] Internal modular v4 mainline freeze and claims audit")
    print(f"[SCRIPT VERSION] {SCRIPT_VERSION}")
    print("=" * 88)
    print(f"[PROJECT ROOT] {PROJECT_ROOT}")
    print("[MODE] read-only consolidation; no training; no inference; no external data")
    print(f"[OUTPUT DIR] {OUT_DIR}")

    inventory = evidence_inventory()
    metrics = full_module_metrics()
    scenarios = scenario_metrics()
    gates = gate_summary_full()
    comp = comparison_table()
    gain = scenario_gain_summary()
    claims = claims_audit()

    inventory.to_csv(OUT_DIR / "evidence_inventory.csv", index=False)
    metrics.to_csv(OUT_DIR / "frozen_v4_full_module_test_metrics.csv", index=False)
    scenarios.to_csv(OUT_DIR / "frozen_v4_scenario_test_metrics.csv", index=False)
    gates.to_csv(OUT_DIR / "frozen_v4_full_module_gate_summary.csv", index=False)
    comp.to_csv(OUT_DIR / "v4_vs_v3_and_best_baseline_key_comparison.csv", index=False)
    gain.to_csv(OUT_DIR / "v4_average_scenario_gain_summary.csv", index=False)
    claims.to_csv(OUT_DIR / "claims_audit_internal_modular_v4.csv", index=False)

    (OUT_DIR / "MAINLINE_FREEZE_INTERNAL_MODULAR_V4.md").write_text(mainline_freeze_md(metrics, gates, comp, gain, claims), encoding="utf-8")
    (OUT_DIR / "PAPER_OUTLINE_INTERNAL_MODULAR_V4.md").write_text(paper_outline_md(), encoding="utf-8")
    (OUT_DIR / "OUTREACH_PROJECT_DESCRIPTION_MODULAR_V4.md").write_text(outreach_description_md(metrics, gain), encoding="utf-8")

    audit = {
        "script_version": SCRIPT_VERSION,
        "mode": "internal_modular_v4_mainline_freeze_claims_audit",
        "no_training": True,
        "no_inference": True,
        "no_external_data": True,
        "no_raw_data_edits": True,
        "project_root": str(PROJECT_ROOT),
        "out_dir": str(OUT_DIR),
        "frozen_title": MAIN_TITLE,
        "frozen_main_model": MAIN_MODEL,
        "frozen_main_risk": MAIN_RISK,
        "model_family": MODEL_FAMILY,
        "outputs": {
            "evidence_inventory": str(OUT_DIR / "evidence_inventory.csv"),
            "full_metrics": str(OUT_DIR / "frozen_v4_full_module_test_metrics.csv"),
            "scenario_metrics": str(OUT_DIR / "frozen_v4_scenario_test_metrics.csv"),
            "gate_summary": str(OUT_DIR / "frozen_v4_full_module_gate_summary.csv"),
            "key_comparison": str(OUT_DIR / "v4_vs_v3_and_best_baseline_key_comparison.csv"),
            "gain_summary": str(OUT_DIR / "v4_average_scenario_gain_summary.csv"),
            "claims_audit": str(OUT_DIR / "claims_audit_internal_modular_v4.csv"),
            "mainline_freeze": str(OUT_DIR / "MAINLINE_FREEZE_INTERNAL_MODULAR_V4.md"),
            "paper_outline": str(OUT_DIR / "PAPER_OUTLINE_INTERNAL_MODULAR_V4.md"),
            "outreach_description": str(OUT_DIR / "OUTREACH_PROJECT_DESCRIPTION_MODULAR_V4.md"),
        },
    }
    (OUT_DIR / "step90_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("\n[FROZEN V4 FULL-MODULE METRICS]")
    print(metrics.to_string(index=False))

    print("\n[V4 AVERAGE SCENARIO GAINS]")
    print(gain.to_string(index=False))

    print("\n[CLAIMS AUDIT]")
    print(claims[["claim", "status", "safe_wording"]].to_string(index=False))

    print("\n[OUTPUTS]")
    for _, path in audit["outputs"].items():
        print(f"  - {path}")
    print(f"  - {OUT_DIR / 'step90_audit.json'}")

    print("\n[DONE] Step90 internal modular v4 mainline freeze complete.")


if __name__ == "__main__":
    main()
