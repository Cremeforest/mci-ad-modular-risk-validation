# -*- coding: utf-8 -*-
"""
Step94d: Corrected NACC external raw token construction using NACCVNUM.

Purpose
-------
Build a non-degenerate NACC external validation cohort and raw dynamic tokens.

Reads ONLY:
    data/data_raw/NACC_investigator.csv

Main corrected design
---------------------
time axis = NACCVNUM, treated as approximate annual visit index
landmark = first eligible MCI visit per patient, NACCUDSD == 3
event = future AD dementia, NACCUDSD == 4 and NACCALZD == 1
ADAS13 = all missing
main external scenario = no_ADAS13

Does NOT:
    - read ADNI-style small tables
    - train/retrain model
    - fit scaler/imputer on NACC
    - evaluate frozen v4 checkpoint

Outputs
-------
results/features/94d_nacc_external_raw_tokens_corrected_naccvnum/
    nacc_external_dynamic_tokens_raw_k8_corrected_naccvnum.npz
    nacc_external_first_mci_cohort_corrected_naccvnum.csv
    nacc_external_label_summary_corrected_naccvnum.csv
    nacc_external_feature_missingness_corrected_naccvnum.csv
    nacc_external_event_time_summary_corrected_naccvnum.csv
    README_step94d_nacc_external_raw_tokens_corrected_naccvnum.md
    step94d_audit.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "v1_corrected_naccvnum_first_mci_external_raw_tokens"

RAW_REL = Path("data") / "data_raw" / "NACC_investigator.csv"
OUT_REL = Path("results") / "features" / "94d_nacc_external_raw_tokens_corrected_naccvnum"

K = 8

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

HORIZONS_YEARS = [1.0, 2.0, 3.0, 5.0]
HORIZON_NAMES = ["1y", "2y", "3y", "5y"]

FAQ_ITEMS = [
    "BILLS",
    "TAXES",
    "SHOPPING",
    "GAMES",
    "STOVE",
    "MEALPREP",
    "EVENTS",
    "PAYATTN",
    "REMDATES",
    "TRAVEL",
]

USECOLS = [
    "NACCID",
    "NACCVNUM",
    "VISITDATE",
    "NACCAGE",
    "NACCSEX",
    "EDUC",
    "NACCMMSE",
    "CDRGLOB",
    "CDRSUM",
    "NACCUDSD",
    "NACCALZD",
] + FAQ_ITEMS


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def clean_range(s: pd.Series, low: float, high: float) -> pd.Series:
    x = num(s).astype(float)
    x[(x < low) | (x > high)] = np.nan
    return x


def clean_allowed(s: pd.Series, allowed: List[float]) -> pd.Series:
    x = num(s).astype(float)
    x[~x.isin(allowed)] = np.nan
    return x


def add_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["NACCVNUM_num"] = num(df["NACCVNUM"])
    df["NACCUDSD_num"] = num(df["NACCUDSD"])
    df["NACCALZD_num"] = num(df["NACCALZD"])

    # Corrected time axis: visit number as approximate annual follow-up.
    df["time_years"] = df["NACCVNUM_num"].astype(float)
    df["visit_order"] = df["NACCVNUM_num"].astype(float)

    # Features.
    df["age_at_visit"] = clean_range(df["NACCAGE"], 18, 120)

    sex = num(df["NACCSEX"])
    df["sex_male"] = np.nan
    df.loc[sex == 1, "sex_male"] = 1.0
    df.loc[sex == 2, "sex_male"] = 0.0

    df["PTEDUCAT"] = clean_range(df["EDUC"], 0, 36)
    df["MMSE"] = clean_range(df["NACCMMSE"], 0, 30)

    # ADAS13 absent in NACC investigator file.
    df["ADAS13"] = np.nan

    df["CDGLOBAL"] = clean_allowed(df["CDRGLOB"], [0.0, 0.5, 1.0, 2.0, 3.0])
    df["CDRSB"] = clean_range(df["CDRSUM"], 0, 18)

    # Strict FAQTOTAL: all 10 items valid 0-3.
    faq_clean = pd.DataFrame(index=df.index)
    for c in FAQ_ITEMS:
        item = num(df[c]).astype(float)
        item[~item.isin([0.0, 1.0, 2.0, 3.0])] = np.nan
        faq_clean[c] = item

    valid_item_count = faq_clean.notna().sum(axis=1)
    faq_total = faq_clean.sum(axis=1, skipna=True)
    faq_total[valid_item_count < len(FAQ_ITEMS)] = np.nan

    df["FAQTOTAL"] = faq_total
    df["FAQTOTAL_valid_item_count"] = valid_item_count

    # Diagnosis flags.
    df["is_mci"] = df["NACCUDSD_num"].eq(3)
    df["is_dementia"] = df["NACCUDSD_num"].eq(4)
    df["is_ad_dementia_event"] = df["NACCUDSD_num"].eq(4) & df["NACCALZD_num"].eq(1)

    return df


def build_feature_missingness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)

    for f in FEATURE_NAMES:
        n_obs = int(df[f].notna().sum())
        rows.append(
            {
                "feature": f,
                "n_rows": total,
                "n_observed": n_obs,
                "observed_rate": n_obs / total if total else np.nan,
                "n_missing": total - n_obs,
                "missing_rate": 1 - n_obs / total if total else np.nan,
                "min": df[f].min(skipna=True),
                "max": df[f].max(skipna=True),
                "mean": df[f].mean(skipna=True),
            }
        )

    return pd.DataFrame(rows)


def compute_labels(g: pd.DataFrame, landmark_pos: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    landmark = g.iloc[landmark_pos]
    t0 = float(landmark["time_years"])

    # Strict future by NACCVNUM.
    future = g[g["time_years"] > t0].copy()

    if len(future) > 0:
        last_followup_years = float(future["time_years"].max() - t0)
    else:
        last_followup_years = 0.0

    event_future = future[future["is_ad_dementia_event"] == True].copy()

    if len(event_future) > 0:
        first_event = event_future.iloc[0]
        event_observed = True
        event_time_years = float(first_event["time_years"] - t0)
    else:
        first_event = None
        event_observed = False
        event_time_years = np.nan

    y = np.full(len(HORIZONS_YEARS), -1, dtype=np.int64)
    y_mask = np.zeros(len(HORIZONS_YEARS), dtype=np.int64)

    for i, h in enumerate(HORIZONS_YEARS):
        if event_observed and event_time_years <= h:
            y[i] = 1
            y_mask[i] = 1
        elif event_observed and event_time_years > h:
            y[i] = 0
            y_mask[i] = 1
        elif (not event_observed) and last_followup_years >= h:
            y[i] = 0
            y_mask[i] = 1
        else:
            y[i] = -1
            y_mask[i] = 0

    info = {
        "event_observed": bool(event_observed),
        "event_time_years": event_time_years,
        "last_followup_years": last_followup_years,
        "event_NACCVNUM": first_event["NACCVNUM"] if first_event is not None else "",
        "event_VISITDATE": first_event["VISITDATE"] if first_event is not None else "",
        "event_NACCAGE": first_event["NACCAGE"] if first_event is not None else "",
        "event_NACCUDSD": first_event["NACCUDSD"] if first_event is not None else "",
        "event_NACCALZD": first_event["NACCALZD"] if first_event is not None else "",
    }

    return y, y_mask, info


def fill_window_tokens(
    g: pd.DataFrame,
    landmark_pos: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build K-visit left-padded window ending at landmark visit.
    """
    start_pos = max(0, landmark_pos - K + 1)
    window = g.iloc[start_pos : landmark_pos + 1].copy()
    n = len(window)
    pad_start = K - n

    Xv = np.full((K, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    Xm = np.zeros((K, len(FEATURE_NAMES)), dtype=np.float32)
    Xt = np.zeros((K, 4), dtype=np.float32)
    Xd = np.full((K, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    Xs = np.full((K, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    visit_mask = np.zeros((K,), dtype=np.float32)

    landmark_time = float(g.iloc[landmark_pos]["time_years"])
    first_time = float(g["time_years"].min())

    full_times = g["time_years"].astype(float).to_numpy()
    full_prev_gap = np.zeros(len(g), dtype=float)
    if len(g) > 1:
        full_prev_gap[1:] = np.diff(full_times)

    for local_i, (_, row) in enumerate(window.iterrows()):
        out_i = pad_start + local_i
        original_pos = start_pos + local_i

        visit_mask[out_i] = 1.0

        vals = row[FEATURE_NAMES].astype(float).to_numpy()
        Xv[out_i, :] = vals
        Xm[out_i, :] = (~np.isnan(vals)).astype(np.float32)

        t = float(row["time_years"])
        Xt[out_i, 0] = t - first_time
        Xt[out_i, 1] = full_prev_gap[original_pos] if original_pos < len(full_prev_gap) else 0.0
        Xt[out_i, 2] = t - landmark_time
        Xt[out_i, 3] = float(row["visit_order"]) if pd.notna(row["visit_order"]) else float(original_pos + 1)

    # Feature deltas/slopes within observed window.
    times_window = window["time_years"].astype(float).to_numpy()

    for f_idx, f in enumerate(FEATURE_NAMES):
        values = window[f].astype(float).to_numpy()

        last_val = np.nan
        last_time = np.nan

        for local_i in range(n):
            out_i = pad_start + local_i
            v = values[local_i]
            t = times_window[local_i]

            if not np.isnan(v) and not np.isnan(last_val):
                delta = v - last_val
                gap = t - last_time
                Xd[out_i, f_idx] = delta
                if gap > 0:
                    Xs[out_i, f_idx] = delta / gap

            if not np.isnan(v):
                last_val = v
                last_time = t

    return Xv, Xm, Xt, Xd, Xs, visit_mask


def build_first_mci_tokens(df: pd.DataFrame):
    Xv_list = []
    Xm_list = []
    Xt_list = []
    Xd_list = []
    Xs_list = []
    visit_mask_list = []
    y_list = []
    y_mask_list = []
    cohort_rows = []

    df = df.dropna(subset=["NACCID", "NACCVNUM_num"]).copy()
    df = df.sort_values(["NACCID", "NACCVNUM_num"]).reset_index(drop=True)

    for naccid, g in df.groupby("NACCID", sort=False):
        g = g.sort_values("NACCVNUM_num").reset_index(drop=True)

        # Exclude landmarks after prior dementia.
        prior_dementia_before = g["is_dementia"].shift(1, fill_value=False).cummax()

        landmark_pos = None
        for pos in range(len(g)):
            if bool(g.iloc[pos]["is_mci"]) and not bool(prior_dementia_before.iloc[pos]):
                landmark_pos = pos
                break

        if landmark_pos is None:
            continue

        y, y_mask, label_info = compute_labels(g, landmark_pos)

        # Need at least one known horizon.
        if int(y_mask.sum()) == 0:
            continue

        Xv, Xm, Xt, Xd, Xs, visit_mask = fill_window_tokens(g, landmark_pos)

        landmark = g.iloc[landmark_pos]

        sample_index = len(cohort_rows)

        Xv_list.append(Xv)
        Xm_list.append(Xm)
        Xt_list.append(Xt)
        Xd_list.append(Xd)
        Xs_list.append(Xs)
        visit_mask_list.append(visit_mask)
        y_list.append(y)
        y_mask_list.append(y_mask)

        cohort_rows.append(
            {
                "sample_index": sample_index,
                "NACCID": naccid,
                "landmark_mode": "first_mci_per_patient",
                "time_axis": "NACCVNUM",
                "landmark_NACCVNUM": landmark["NACCVNUM"],
                "landmark_VISITDATE": landmark["VISITDATE"],
                "landmark_NACCAGE": landmark["NACCAGE"],
                "landmark_time_years_by_NACCVNUM": landmark["time_years"],
                "landmark_NACCUDSD": landmark["NACCUDSD"],
                "landmark_NACCALZD": landmark["NACCALZD"],
                "event_observed_future_ad_dementia": label_info["event_observed"],
                "event_time_years_by_NACCVNUM": label_info["event_time_years"],
                "last_followup_years_by_NACCVNUM": label_info["last_followup_years"],
                "event_NACCVNUM": label_info["event_NACCVNUM"],
                "event_VISITDATE": label_info["event_VISITDATE"],
                "event_NACCAGE": label_info["event_NACCAGE"],
                "event_NACCUDSD": label_info["event_NACCUDSD"],
                "event_NACCALZD": label_info["event_NACCALZD"],
                "n_visits_in_window": int(visit_mask.sum()),
                "known_1y": int(y_mask[0]),
                "known_2y": int(y_mask[1]),
                "known_3y": int(y_mask[2]),
                "known_5y": int(y_mask[3]),
                "y_1y": int(y[0]),
                "y_2y": int(y[1]),
                "y_3y": int(y[2]),
                "y_5y": int(y[3]),
                "ADAS13_available_in_external": False,
            }
        )

    if not Xv_list:
        raise RuntimeError("No eligible first-MCI NACC samples were constructed.")

    Xv = np.stack(Xv_list, axis=0)
    Xm = np.stack(Xm_list, axis=0)
    Xt = np.stack(Xt_list, axis=0)
    Xd = np.stack(Xd_list, axis=0)
    Xs = np.stack(Xs_list, axis=0)
    visit_mask = np.stack(visit_mask_list, axis=0)
    y = np.stack(y_list, axis=0)
    y_mask = np.stack(y_mask_list, axis=0)
    cohort = pd.DataFrame(cohort_rows)

    return Xv, Xm, Xt, Xd, Xs, visit_mask, y, y_mask, cohort


def build_label_summary(cohort: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for h in HORIZON_NAMES:
        y_col = f"y_{h}"
        k_col = f"known_{h}"

        known = cohort[cohort[k_col] == 1]
        n_known = len(known)
        n_event = int((known[y_col] == 1).sum())
        n_nonevent = int((known[y_col] == 0).sum())

        rows.append(
            {
                "horizon": h,
                "n_total_landmarks": len(cohort),
                "n_patients": int(cohort["NACCID"].nunique()),
                "n_known": n_known,
                "n_unknown": int(len(cohort) - n_known),
                "n_event": n_event,
                "n_nonevent": n_nonevent,
                "event_rate_among_known": n_event / n_known if n_known else np.nan,
            }
        )

    return pd.DataFrame(rows)


def build_event_time_summary(cohort: pd.DataFrame) -> pd.DataFrame:
    event_rows = cohort[cohort["event_observed_future_ad_dementia"] == True].copy()
    x = event_rows["event_time_years_by_NACCVNUM"].dropna()

    if len(x) == 0:
        return pd.DataFrame(
            [
                {
                    "n_future_events": 0,
                    "min_event_time": np.nan,
                    "p25_event_time": np.nan,
                    "median_event_time": np.nan,
                    "p75_event_time": np.nan,
                    "max_event_time": np.nan,
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                "n_future_events": int(len(x)),
                "min_event_time": float(x.min()),
                "p25_event_time": float(x.quantile(0.25)),
                "median_event_time": float(x.median()),
                "p75_event_time": float(x.quantile(0.75)),
                "max_event_time": float(x.max()),
            }
        ]
    )


def main() -> None:
    root = project_root()
    raw_path = root / RAW_REL
    out_dir = root / OUT_REL
    ensure_outdir(out_dir)

    print("=" * 88)
    print("[STEP 94d] Corrected NACC external raw token construction")
    print(f"[SCRIPT VERSION] {SCRIPT_VERSION}")
    print(f"[RAW FILE] {raw_path}")
    print(f"[OUTPUT DIR] {out_dir}")
    print("=" * 88)

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing required file: {raw_path}")

    print("[INFO] Reading selected NACC investigator columns only...")
    df = pd.read_csv(
        raw_path,
        usecols=USECOLS,
        encoding="utf-8-sig",
        low_memory=False,
    )

    print(f"[INFO] Loaded rows: {len(df)}")
    print(f"[INFO] Unique NACCID: {df['NACCID'].nunique()}")

    duplicate_pairs = int(df.duplicated(["NACCID", "NACCVNUM"]).sum())
    print(f"[CHECK] duplicate NACCID/NACCVNUM rows: {duplicate_pairs}")

    print("[INFO] Cleaning mapped features and diagnosis flags...")
    df = add_clean_columns(df)

    feature_missingness = build_feature_missingness(df)

    print("[INFO] Building first-MCI external tokens using NACCVNUM time axis...")
    Xv, Xm, Xt, Xd, Xs, visit_mask, y, y_mask, cohort = build_first_mci_tokens(df)

    label_summary = build_label_summary(cohort)
    event_summary = build_event_time_summary(cohort)

    npz_path = out_dir / "nacc_external_dynamic_tokens_raw_k8_corrected_naccvnum.npz"
    cohort_path = out_dir / "nacc_external_first_mci_cohort_corrected_naccvnum.csv"
    label_summary_path = out_dir / "nacc_external_label_summary_corrected_naccvnum.csv"
    feature_missing_path = out_dir / "nacc_external_feature_missingness_corrected_naccvnum.csv"
    event_summary_path = out_dir / "nacc_external_event_time_summary_corrected_naccvnum.csv"
    readme_path = out_dir / "README_step94d_nacc_external_raw_tokens_corrected_naccvnum.md"
    json_path = out_dir / "step94d_audit.json"

    print("[INFO] Writing outputs...")
    np.savez_compressed(
        npz_path,
        Xv=Xv,
        Xm=Xm,
        Xt=Xt,
        Xd=Xd,
        Xs=Xs,
        visit_mask=visit_mask,
        y=y,
        y_mask=y_mask,
        feature_names=np.array(FEATURE_NAMES, dtype=object),
        horizon_names=np.array(HORIZON_NAMES, dtype=object),
        horizons_years=np.array(HORIZONS_YEARS, dtype=np.float32),
        external_time_axis=np.array(["NACCVNUM"], dtype=object),
        landmark_mode=np.array(["first_mci_per_patient"], dtype=object),
        main_external_scenario=np.array(["no_ADAS13"], dtype=object),
        note=np.array(
            [
                "Corrected raw external NACC tokens. Raw/unscaled. "
                "Step95 must apply internal preprocessing/alignment before frozen model evaluation."
            ],
            dtype=object,
        ),
    )

    cohort.to_csv(cohort_path, index=False, encoding="utf-8-sig")
    label_summary.to_csv(label_summary_path, index=False, encoding="utf-8-sig")
    feature_missingness.to_csv(feature_missing_path, index=False, encoding="utf-8-sig")
    event_summary.to_csv(event_summary_path, index=False, encoding="utf-8-sig")

    readme = f"""# Step94d Corrected NACC external raw token construction

## Purpose

This step rebuilds the NACC external validation tensor after Step94/94b/94c-fast identified that `NACCDAYS` should not be used as the primary time axis.

## Input

`{raw_path}`

Only `NACC_investigator.csv` was read.

## Corrected external cohort definition

- Time axis: `NACCVNUM`, treated as approximate annual visit index.
- Landmark: first eligible MCI visit per patient.
- MCI definition: `NACCUDSD == 3`.
- Exclusion: MCI visits after prior dementia are not eligible.
- Event: future AD dementia, `NACCUDSD == 4` and `NACCALZD == 1`.
- Future means: strictly larger `NACCVNUM`.
- ADAS13: absent in NACC investigator file, therefore all missing.
- Main external scenario: `no_ADAS13`.
- Feature schema: `{FEATURE_NAMES}`.
- Visit window length K: `{K}`.
- Horizons: `{HORIZON_NAMES}`.

## Array shapes

- Xv: {Xv.shape}
- Xm: {Xm.shape}
- Xt: {Xt.shape}
- Xd: {Xd.shape}
- Xs: {Xs.shape}
- visit_mask: {visit_mask.shape}
- y: {y.shape}
- y_mask: {y_mask.shape}

## Label summary

{label_summary.to_string(index=False)}

## Event time summary

{event_summary.to_string(index=False)}

## Important caution

These arrays are raw and unscaled.

Do not feed them directly into the frozen Step89 v4 checkpoint until Step95 applies the internal preprocessing/alignment logic.

## Output files

- `nacc_external_dynamic_tokens_raw_k8_corrected_naccvnum.npz`
- `nacc_external_first_mci_cohort_corrected_naccvnum.csv`
- `nacc_external_label_summary_corrected_naccvnum.csv`
- `nacc_external_feature_missingness_corrected_naccvnum.csv`
- `nacc_external_event_time_summary_corrected_naccvnum.csv`
- `README_step94d_nacc_external_raw_tokens_corrected_naccvnum.md`
- `step94d_audit.json`
"""

    readme_path.write_text(readme, encoding="utf-8")

    audit: Dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "raw_file": str(raw_path),
        "output_dir": str(out_dir),
        "loaded_rows": int(len(df)),
        "unique_NACCID": int(df["NACCID"].nunique()),
        "duplicate_NACCID_NACCVNUM_rows": duplicate_pairs,
        "feature_names": FEATURE_NAMES,
        "K": K,
        "horizon_names": HORIZON_NAMES,
        "horizons_years": HORIZONS_YEARS,
        "time_axis": "NACCVNUM",
        "landmark_mode": "first_mci_per_patient",
        "main_external_scenario": "no_ADAS13",
        "event_definition": "future NACCUDSD == 4 and NACCALZD == 1",
        "array_shapes": {
            "Xv": list(Xv.shape),
            "Xm": list(Xm.shape),
            "Xt": list(Xt.shape),
            "Xd": list(Xd.shape),
            "Xs": list(Xs.shape),
            "visit_mask": list(visit_mask.shape),
            "y": list(y.shape),
            "y_mask": list(y_mask.shape),
        },
        "outputs": {
            "npz": str(npz_path),
            "cohort": str(cohort_path),
            "label_summary": str(label_summary_path),
            "feature_missingness": str(feature_missing_path),
            "event_summary": str(event_summary_path),
            "readme": str(readme_path),
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print("=" * 88)
    print("[DONE] Step94d corrected NACC external raw token construction finished.")
    print(f"[NPZ] {npz_path}")
    print(f"[COHORT] {cohort_path}")
    print("[SHAPES]")
    print(f"  Xv={Xv.shape}")
    print(f"  Xm={Xm.shape}")
    print(f"  Xt={Xt.shape}")
    print(f"  Xd={Xd.shape}")
    print(f"  Xs={Xs.shape}")
    print(f"  visit_mask={visit_mask.shape}")
    print(f"  y={y.shape}")
    print(f"  y_mask={y_mask.shape}")
    print("[LABEL SUMMARY]")
    print(label_summary.to_string(index=False))
    print("[EVENT SUMMARY]")
    print(event_summary.to_string(index=False))
    print("[NEXT] Step95: apply internal preprocessing/alignment and evaluate frozen Step89 v4.")
    print("=" * 88)


if __name__ == "__main__":
    main()