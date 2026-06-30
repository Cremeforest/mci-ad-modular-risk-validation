# -*- coding: utf-8 -*-
"""
Step93b: NACC_investigator-only schema audit

Purpose
-------
Read ONLY:
    data/data_raw/NACC_investigator.csv

Do NOT read ADNI-style small tables.

This script audits:
1. File shape and core ID/time columns
2. Candidate mappings for internal v4 features
3. Diagnosis-related candidate columns
4. Special/missing code distributions
5. Candidate value counts for review

Outputs
-------
results/reports/93b_nacc_investigator_only_schema_audit/
    nacc_investigator_file_summary.csv
    nacc_investigator_core_columns.csv
    nacc_investigator_feature_mapping_candidates.csv
    nacc_investigator_diagnosis_candidates.csv
    nacc_investigator_missing_code_summary.csv
    nacc_investigator_candidate_value_counts.csv
    README_step93b_nacc_investigator_only_schema_audit.md
    step93b_audit.json
"""

from __future__ import annotations

import json
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd


SCRIPT_VERSION = "v1_nacc_investigator_only_schema_audit"

RAW_REL = Path("data") / "data_raw" / "NACC_investigator.csv"
OUT_REL = Path("results") / "reports" / "93b_nacc_investigator_only_schema_audit"

MAX_COLUMNS_TO_LOAD = 260
VALUE_COUNTS_TOP_N = 30
VALUE_COUNTS_MAX_UNIQUE = 120

# Important:
# Some of these codes are variable-specific.
# For example, 8/9 may be real values for some continuous or score variables.
# This audit reports their presence; Step94 should decide cleaning rules per variable.
SPECIAL_CODES = [
    -4,
    8,
    9,
    88,
    95,
    96,
    97,
    98,
    99,
    888,
    999,
    8888,
    9999,
]

CORE_COLUMNS = [
    "NACCID",
    "NACCVNUM",
    "VISITDATE",
    "NACCDAYS",
    "NACCAGE",
]

FEATURE_SPECS: "OrderedDict[str, Dict[str, List[str]]]" = OrderedDict(
    [
        (
            "age_at_visit",
            {
                "exact": ["NACCAGE"],
                "patterns": [r"(^|_)NACCAGE($|_)", r"(^|_)AGE($|_)"],
            },
        ),
        (
            "sex_male",
            {
                "exact": ["NACCSEX", "SEX"],
                "patterns": [r"(^|_)NACCSEX($|_)", r"(^|_)SEX($|_)"],
            },
        ),
        (
            "PTEDUCAT",
            {
                "exact": ["EDUC", "PTEDUCAT"],
                "patterns": [r"(^|_)EDUC($|_)", r"EDUCAT"],
            },
        ),
        (
            "MMSE",
            {
                "exact": ["NACCMMSE", "MMSE"],
                "patterns": [r"MMSE"],
            },
        ),
        (
            "ADAS13",
            {
                "exact": ["ADAS13", "ADAS", "NACCADAS", "NACCADAS", "ADASCog", "ADASCog13"],
                "patterns": [r"ADAS", r"ADASC", r"COG13"],
            },
        ),
        (
            "CDGLOBAL",
            {
                "exact": ["CDRGLOB", "CDGLOBAL"],
                "patterns": [r"CDRGLOB", r"CDR.*GLOB", r"GLOBAL.*CDR"],
            },
        ),
        (
            "CDRSB",
            {
                "exact": ["CDRSUM", "CDRSB"],
                "patterns": [r"CDRSUM", r"CDR.*SUM", r"CDR.*SB"],
            },
        ),
        (
            "FAQTOTAL",
            {
                "exact": ["FAQTOTAL", "FAQSUM", "NACCFAQ", "FAQ"],
                "patterns": [
                    r"FAQ",
                    r"BILLS",
                    r"TAXES",
                    r"SHOPPING",
                    r"GAMES",
                    r"STOVE",
                    r"MEALPREP",
                    r"EVENTS",
                    r"PAYATTN",
                    r"REMDATES",
                    r"TRAVEL",
                ],
            },
        ),
    ]
)

DIAGNOSIS_EXACT = [
    "NACCUDSD",
    "DECSUB",
    "NACCALZD",
    "NACCADMD",
    "DEMENTED",
    "NORMCOG",
    "IMPNOMCI",
    "MCI",
    "COGSTAT",
    "NACCMCII",
    "NACCMCI",
]

DIAGNOSIS_PATTERNS = [
    r"NACCUDSD",
    r"DECSUB",
    r"NACCALZD",
    r"NACCADMD",
    r"DEMENT",
    r"ALZ",
    r"(^|_)MCI($|_)",
    r"MILD.*COG",
    r"COG.*STAT",
    r"NORM.*COG",
    r"IMP.*NOMCI",
    r"(^|_)DX($|_)",
    r"DIAG",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def count_data_rows_binary(path: Path) -> int:
    """
    Count data rows quickly without loading the full CSV.
    Returns total line count minus header.
    """
    line_count = 0
    last_byte = b""

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024 * 16)
            if not chunk:
                break
            line_count += chunk.count(b"\n")
            last_byte = chunk[-1:]

    if line_count == 0:
        return 0

    if last_byte not in [b"\n", b"\r"]:
        line_count += 1

    return max(line_count - 1, 0)


def read_header(path: Path) -> Tuple[List[str], str]:
    """
    Try common encodings and return CSV header.
    """
    encodings = ["utf-8-sig", "utf-8", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            df0 = pd.read_csv(path, nrows=0, encoding=enc)
            return list(df0.columns), enc
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not read CSV header. Last error: {last_error}")


def normalize_col_map(columns: List[str]) -> Dict[str, str]:
    return {c.upper(): c for c in columns}


def resolve_column(columns: List[str], name: str) -> str | None:
    lookup = normalize_col_map(columns)
    return lookup.get(name.upper())


def find_pattern_columns(columns: List[str], patterns: List[str]) -> List[Tuple[str, str]]:
    hits: List[Tuple[str, str]] = []
    seen = set()

    for pat in patterns:
        regex = re.compile(pat, flags=re.IGNORECASE)
        for col in columns:
            if regex.search(col):
                key = (col, pat)
                if key not in seen:
                    hits.append((col, pat))
                    seen.add(key)

    return hits


def add_unique(seq: List[str], item: str | None) -> None:
    if item is not None and item not in seq:
        seq.append(item)


def to_jsonable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def compute_column_stats(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}

    for col in df.columns:
        s = df[col]
        total = int(len(s))
        pandas_na = int(s.isna().sum())

        try:
            blank_count = int(s.astype("string").str.strip().eq("").fillna(False).sum())
        except Exception:
            blank_count = 0

        non_missing = int(total - pandas_na - blank_count)
        nunique = int(s.nunique(dropna=True))

        numeric = pd.to_numeric(s, errors="coerce")
        n_numeric = int(numeric.notna().sum())

        if n_numeric > 0:
            num_min = to_jsonable(numeric.min())
            num_max = to_jsonable(numeric.max())
            num_mean = to_jsonable(numeric.mean())
        else:
            num_min = None
            num_max = None
            num_mean = None

        stats[col] = {
            "dtype": str(s.dtype),
            "total_rows": total,
            "pandas_na_count": pandas_na,
            "blank_count": blank_count,
            "non_missing_count": non_missing,
            "non_missing_rate": non_missing / total if total else None,
            "n_unique_non_missing": nunique,
            "n_numeric": n_numeric,
            "numeric_min": num_min,
            "numeric_max": num_max,
            "numeric_mean": num_mean,
        }

    return stats


def empty_stats() -> Dict[str, Any]:
    return {
        "dtype": "",
        "total_rows": "",
        "pandas_na_count": "",
        "blank_count": "",
        "non_missing_count": "",
        "non_missing_rate": "",
        "n_unique_non_missing": "",
        "n_numeric": "",
        "numeric_min": "",
        "numeric_max": "",
        "numeric_mean": "",
    }


def collect_feature_candidates(columns: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    load_cols: List[str] = []

    for target, spec in FEATURE_SPECS.items():
        seen_for_target = set()
        existing_count = 0

        for name in spec["exact"]:
            actual = resolve_column(columns, name)
            exists = actual is not None
            if exists:
                existing_count += 1
                add_unique(load_cols, actual)

            candidate_col = actual if actual is not None else name
            key = (target, candidate_col, "exact")
            if key not in seen_for_target:
                rows.append(
                    {
                        "target_internal_feature": target,
                        "candidate_column": candidate_col,
                        "exists": exists,
                        "match_type": "exact",
                        "matched_pattern": "",
                        "comment": "Exact candidate name.",
                    }
                )
                seen_for_target.add(key)

        pattern_hits = find_pattern_columns(columns, spec["patterns"])
        for col, pat in pattern_hits:
            existing_count += 1
            add_unique(load_cols, col)
            key = (target, col, "pattern")
            if key not in seen_for_target:
                rows.append(
                    {
                        "target_internal_feature": target,
                        "candidate_column": col,
                        "exists": True,
                        "match_type": "pattern",
                        "matched_pattern": pat,
                        "comment": "Column name matched feature-related search pattern.",
                    }
                )
                seen_for_target.add(key)

        if existing_count == 0:
            rows.append(
                {
                    "target_internal_feature": target,
                    "candidate_column": "",
                    "exists": False,
                    "match_type": "none",
                    "matched_pattern": "",
                    "comment": "No existing column found by exact names or search patterns.",
                }
            )

    return rows, load_cols


def collect_diagnosis_candidates(columns: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    load_cols: List[str] = []
    seen = set()

    for name in DIAGNOSIS_EXACT:
        actual = resolve_column(columns, name)
        exists = actual is not None
        if exists:
            add_unique(load_cols, actual)

        candidate_col = actual if actual is not None else name
        key = (candidate_col, "exact")
        if key not in seen:
            rows.append(
                {
                    "candidate_column": candidate_col,
                    "exists": exists,
                    "match_type": "exact",
                    "matched_pattern": "",
                    "comment": "Exact diagnosis-related candidate name.",
                }
            )
            seen.add(key)

    pattern_hits = find_pattern_columns(columns, DIAGNOSIS_PATTERNS)
    for col, pat in pattern_hits:
        add_unique(load_cols, col)
        key = (col, "pattern", pat)
        if key not in seen:
            rows.append(
                {
                    "candidate_column": col,
                    "exists": True,
                    "match_type": "pattern",
                    "matched_pattern": pat,
                    "comment": "Column name matched diagnosis-related search pattern.",
                }
            )
            seen.add(key)

    return rows, load_cols


def attach_stats(rows: List[Dict[str, Any]], stats: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        row2 = dict(row)
        col = row2.get("candidate_column", "")
        if row2.get("exists") is True and col in stats:
            row2.update(stats[col])
        else:
            row2.update(empty_stats())
        out.append(row2)
    return out


def make_core_columns_table(columns: List[str], stats: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for col in CORE_COLUMNS:
        actual = resolve_column(columns, col)
        exists = actual is not None
        row = {
            "required_core_column": col,
            "actual_column": actual if actual else "",
            "exists": exists,
        }
        if exists and actual in stats:
            row.update(stats[actual])
        else:
            row.update(empty_stats())
        rows.append(row)
    return pd.DataFrame(rows)


def make_missing_code_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in df.columns:
        s = df[col]
        total = int(len(s))
        numeric = pd.to_numeric(s, errors="coerce")

        try:
            blank_count = int(s.astype("string").str.strip().eq("").fillna(False).sum())
        except Exception:
            blank_count = 0

        row: Dict[str, Any] = {
            "column": col,
            "dtype": str(s.dtype),
            "total_rows": total,
            "pandas_na_count": int(s.isna().sum()),
            "blank_count": blank_count,
        }

        special_total_mask = numeric.isin(SPECIAL_CODES)
        row["any_special_code_count"] = int(special_total_mask.sum())
        row["any_special_code_rate"] = (
            row["any_special_code_count"] / total if total else None
        )

        for code in SPECIAL_CODES:
            cnt = int((numeric == code).sum())
            row[f"code_{code}_count"] = cnt
            row[f"code_{code}_rate"] = cnt / total if total else None

        rows.append(row)

    return pd.DataFrame(rows)


def should_value_count(col: str, s: pd.Series) -> bool:
    col_upper = col.upper()

    if col_upper in {"NACCID", "VISITDATE"}:
        return False

    nunique = int(s.nunique(dropna=False))
    if nunique <= VALUE_COUNTS_MAX_UNIQUE:
        return True

    important_tokens = [
        "NACCUDSD",
        "DECSUB",
        "NACCALZD",
        "NACCADMD",
        "DEMENT",
        "ALZ",
        "MCI",
        "CDRGLOB",
        "CDRSUM",
        "NACCMMSE",
        "FAQ",
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
    return any(tok in col_upper for tok in important_tokens)


def make_candidate_value_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in df.columns:
        s = df[col]
        if not should_value_count(col, s):
            continue

        nunique = int(s.nunique(dropna=False))
        vc = s.value_counts(dropna=False).head(VALUE_COUNTS_TOP_N)

        for value, count in vc.items():
            rows.append(
                {
                    "column": col,
                    "n_unique_including_na": nunique,
                    "value": "<NA>" if pd.isna(value) else str(value),
                    "count": int(count),
                    "rate": int(count) / len(s) if len(s) else None,
                }
            )

    return pd.DataFrame(rows)


def preview_list(items: List[str], max_n: int = 30) -> str:
    if not items:
        return "None"
    shown = items[:max_n]
    suffix = "" if len(items) <= max_n else f"\n... truncated, total={len(items)}"
    return "\n".join([f"- {x}" for x in shown]) + suffix


def main() -> None:
    root = project_root()
    raw_path = root / RAW_REL
    out_dir = root / OUT_REL
    ensure_outdir(out_dir)

    print("=" * 88)
    print("[STEP 93b] NACC_investigator-only schema audit")
    print(f"[SCRIPT VERSION] {SCRIPT_VERSION}")
    print(f"[PROJECT ROOT] {root}")
    print(f"[RAW FILE] {raw_path}")
    print(f"[OUTPUT DIR] {out_dir}")
    print("=" * 88)

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing required NACC file: {raw_path}")

    file_size_mb = raw_path.stat().st_size / (1024 * 1024)
    columns, encoding = read_header(raw_path)
    n_cols = len(columns)

    print(f"[INFO] File size: {file_size_mb:.2f} MB")
    print(f"[INFO] Header columns: {n_cols}")
    print("[INFO] Counting rows without loading full file...")
    n_rows = count_data_rows_binary(raw_path)
    print(f"[INFO] Data rows: {n_rows}")

    feature_rows_raw, feature_load_cols = collect_feature_candidates(columns)
    diagnosis_rows_raw, diagnosis_load_cols = collect_diagnosis_candidates(columns)

    load_cols: List[str] = []

    # Highest priority: core columns.
    for c in CORE_COLUMNS:
        add_unique(load_cols, resolve_column(columns, c))

    # Then exact feature candidates.
    for spec in FEATURE_SPECS.values():
        for c in spec["exact"]:
            add_unique(load_cols, resolve_column(columns, c))

    # Then exact diagnosis candidates.
    for c in DIAGNOSIS_EXACT:
        add_unique(load_cols, resolve_column(columns, c))

    # Then all discovered pattern candidates.
    for c in feature_load_cols:
        add_unique(load_cols, c)
    for c in diagnosis_load_cols:
        add_unique(load_cols, c)

    skipped_cols = []
    if len(load_cols) > MAX_COLUMNS_TO_LOAD:
        skipped_cols = load_cols[MAX_COLUMNS_TO_LOAD:]
        load_cols = load_cols[:MAX_COLUMNS_TO_LOAD]

    print(f"[INFO] Candidate columns selected for detailed audit: {len(load_cols)}")
    if skipped_cols:
        print(
            f"[WARN] Candidate columns exceeded cap={MAX_COLUMNS_TO_LOAD}; "
            f"skipped detailed loading for {len(skipped_cols)} columns."
        )

    print("[INFO] Loading selected candidate columns only...")
    df = pd.read_csv(
        raw_path,
        usecols=load_cols,
        encoding=encoding,
        low_memory=False,
    )

    stats = compute_column_stats(df)

    file_summary = pd.DataFrame(
        [
            {
                "script_version": SCRIPT_VERSION,
                "raw_file": str(raw_path),
                "file_exists": raw_path.exists(),
                "file_size_mb": file_size_mb,
                "n_rows_counted": n_rows,
                "n_columns_header": n_cols,
                "encoding_used": encoding,
                "loaded_candidate_columns": len(load_cols),
                "skipped_candidate_columns_due_to_cap": len(skipped_cols),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "note": "Only NACC_investigator.csv was read. ADNI-style small tables were not used.",
            }
        ]
    )

    core_df = make_core_columns_table(columns, stats)
    feature_df = pd.DataFrame(attach_stats(feature_rows_raw, stats))
    diagnosis_df = pd.DataFrame(attach_stats(diagnosis_rows_raw, stats))
    missing_df = make_missing_code_summary(df)
    value_counts_df = make_candidate_value_counts(df)

    output_paths = {
        "file_summary": out_dir / "nacc_investigator_file_summary.csv",
        "core_columns": out_dir / "nacc_investigator_core_columns.csv",
        "feature_mapping_candidates": out_dir / "nacc_investigator_feature_mapping_candidates.csv",
        "diagnosis_candidates": out_dir / "nacc_investigator_diagnosis_candidates.csv",
        "missing_code_summary": out_dir / "nacc_investigator_missing_code_summary.csv",
        "candidate_value_counts": out_dir / "nacc_investigator_candidate_value_counts.csv",
        "readme": out_dir / "README_step93b_nacc_investigator_only_schema_audit.md",
        "json": out_dir / "step93b_audit.json",
    }

    file_summary.to_csv(output_paths["file_summary"], index=False, encoding="utf-8-sig")
    core_df.to_csv(output_paths["core_columns"], index=False, encoding="utf-8-sig")
    feature_df.to_csv(output_paths["feature_mapping_candidates"], index=False, encoding="utf-8-sig")
    diagnosis_df.to_csv(output_paths["diagnosis_candidates"], index=False, encoding="utf-8-sig")
    missing_df.to_csv(output_paths["missing_code_summary"], index=False, encoding="utf-8-sig")
    value_counts_df.to_csv(output_paths["candidate_value_counts"], index=False, encoding="utf-8-sig")

    existing_feature_map = {}
    for target in FEATURE_SPECS.keys():
        sub = feature_df[
            (feature_df["target_internal_feature"] == target)
            & (feature_df["exists"] == True)
        ]
        existing_feature_map[target] = list(dict.fromkeys(sub["candidate_column"].astype(str).tolist()))

    existing_diag_cols = list(
        dict.fromkeys(
            diagnosis_df[diagnosis_df["exists"] == True]["candidate_column"]
            .astype(str)
            .tolist()
        )
    )

    faq_cols = existing_feature_map.get("FAQTOTAL", [])
    adas_cols = existing_feature_map.get("ADAS13", [])

    readme = f"""# Step93b NACC investigator-only schema audit

## Purpose

This audit prepares NACC external validation for the frozen internal v4 model.

It reads **only**:

`{raw_path}`

It does **not** read or merge ADNI-style small tables such as ADAS.csv, DXSUM.csv, FAQ.csv, MMSE.csv, PTDEMOG.csv, or VISITS.csv.

## File summary

- Script version: `{SCRIPT_VERSION}`
- File size: `{file_size_mb:.2f} MB`
- Counted data rows: `{n_rows}`
- Header columns: `{n_cols}`
- Encoding used by pandas: `{encoding}`
- Candidate columns loaded for detailed audit: `{len(load_cols)}`
- Candidate columns skipped due to safety cap: `{len(skipped_cols)}`

## Required core ID/time columns

See:

`nacc_investigator_core_columns.csv`

Expected core columns:

{preview_list(CORE_COLUMNS)}

## Internal v4 feature candidate mapping

See:

`nacc_investigator_feature_mapping_candidates.csv`

Current feature candidate hits:

### age_at_visit

{preview_list(existing_feature_map.get("age_at_visit", []))}

### sex_male

{preview_list(existing_feature_map.get("sex_male", []))}

### PTEDUCAT

{preview_list(existing_feature_map.get("PTEDUCAT", []))}

### MMSE

{preview_list(existing_feature_map.get("MMSE", []))}

### ADAS13

{preview_list(adas_cols)}

### CDGLOBAL

{preview_list(existing_feature_map.get("CDGLOBAL", []))}

### CDRSB

{preview_list(existing_feature_map.get("CDRSB", []))}

### FAQTOTAL

{preview_list(faq_cols)}

## Diagnosis candidate columns

See:

`nacc_investigator_diagnosis_candidates.csv`

Diagnosis-related candidate hits include:

{preview_list(existing_diag_cols, max_n=60)}

## Missing/special code audit

See:

`nacc_investigator_missing_code_summary.csv`

Important caution:

This script reports candidate special-code frequencies for:

`{SPECIAL_CODES}`

These codes are **not automatically converted to missing here**. Some values such as `8` or `9` can be real values for some score variables. Step94 must define variable-specific cleaning rules after reviewing these outputs.

## Candidate value counts

See:

`nacc_investigator_candidate_value_counts.csv`

This is especially useful for reviewing:

- NACCUDSD / DECSUB / NACCALZD / NACCADMD diagnosis-related coding
- CDRGLOB / CDRSUM
- NACCMMSE
- FAQ-related item or total columns
- possible ADAS-related columns, if any

## Interpretation for next step

Step93b does not build external tensors and does not define final labels.

Before Step94, manually confirm:

1. Which column defines baseline MCI.
2. Which column or combination defines AD dementia conversion.
3. Whether ADAS13 exists in NACC. If not, external validation should use `no_ADAS13` or low-burden scenarios.
4. Whether FAQTOTAL is a direct total column or must be reconstructed from FAQ item columns.
5. Which special codes should be cleaned for each selected variable.

## Output files

- `nacc_investigator_file_summary.csv`
- `nacc_investigator_core_columns.csv`
- `nacc_investigator_feature_mapping_candidates.csv`
- `nacc_investigator_diagnosis_candidates.csv`
- `nacc_investigator_missing_code_summary.csv`
- `nacc_investigator_candidate_value_counts.csv`
- `README_step93b_nacc_investigator_only_schema_audit.md`
- `step93b_audit.json`
"""

    output_paths["readme"].write_text(readme, encoding="utf-8")

    json_payload = {
        "script_version": SCRIPT_VERSION,
        "raw_file": str(raw_path),
        "output_dir": str(out_dir),
        "file_size_mb": file_size_mb,
        "n_rows_counted": n_rows,
        "n_columns_header": n_cols,
        "encoding_used": encoding,
        "core_columns": CORE_COLUMNS,
        "existing_feature_map": existing_feature_map,
        "existing_diagnosis_candidate_columns": existing_diag_cols,
        "loaded_candidate_columns": load_cols,
        "skipped_candidate_columns_due_to_cap": skipped_cols,
        "special_codes_reported_not_automatically_cleaned": SPECIAL_CODES,
        "outputs": {k: str(v) for k, v in output_paths.items()},
        "important_note": (
            "This is schema audit only. It does not use ADNI-style small tables, "
            "does not build labels, does not build tensors, and does not refit "
            "scalers/imputers."
        ),
    }

    with open(output_paths["json"], "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    print("=" * 88)
    print("[DONE] Step93b NACC investigator-only schema audit finished.")
    print(f"[OUT] {out_dir}")
    print("[KEY CHECK] Existing feature candidate map:")
    for k, v in existing_feature_map.items():
        print(f"  - {k}: {v[:12]}{' ...' if len(v) > 12 else ''}")
    print(f"[KEY CHECK] Diagnosis candidate columns found: {len(existing_diag_cols)}")
    print("[NEXT] Review README and candidate CSVs before Step94 tensor construction.")
    print("=" * 88)


if __name__ == "__main__":
    main()