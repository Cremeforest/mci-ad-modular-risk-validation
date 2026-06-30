# Step 89: train summary-augmented scenario-dropout feature-level modular v4.
# Purpose:
#   1) Add module-local trajectory summaries so the GRU does not have to rediscover HGB-style last/mean/delta features.
#   2) Train with scenario dropout so a single modular model learns clinically meaningful missing-feature scenarios.
#   3) Evaluate full and missing-feature scenarios with scenario-specific validation Platt calibration.
#
# Internal only. No external data. No raw-data edits. No deletion.

from __future__ import annotations

from pathlib import Path
import json
import random
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


SCRIPT_VERSION = "v4_summary_augmented_scenario_dropout_feature_modular_gru"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_NPZ = PROJECT_ROOT / "results" / "features" / "12_promise_dynamic_tokens_cleaned" / "primary_promise_dynamic_tokens_k8.npz"
TOKEN_METADATA_JSON = PROJECT_ROOT / "results" / "features" / "12_promise_dynamic_tokens_cleaned" / "primary_promise_dynamic_tokens_metadata.json"

STEP85_SCENARIOS = PROJECT_ROOT / "results" / "reports" / "85_internal_missing_feature_stress_test_v3" / "scenario_definitions.csv"
STEP85_METRICS = PROJECT_ROOT / "results" / "reports" / "85_internal_missing_feature_stress_test_v3" / "metrics_by_scenario_raw_and_platt.csv"
STEP88_COMPARISON = PROJECT_ROOT / "results" / "reports" / "88_internal_baseline_missing_feature_stress_comparison" / "v3_vs_best_baseline_by_scenario.csv"
STEP88_COMBINED = PROJECT_ROOT / "results" / "reports" / "88_internal_baseline_missing_feature_stress_comparison" / "combined_v3_and_baseline_stress_metrics.csv"

OUT_DIR = PROJECT_ROOT / "results" / "models" / "89_summary_augmented_scenario_dropout_modular_v4_internal"
REPORT_DIR = PROJECT_ROOT / "results" / "reports" / "89_summary_augmented_scenario_dropout_modular_v4_internal"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 2, 3, 5]
SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 180
PATIENCE = 32
LR = 5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 5.0

D_MODEL = 48
SUMMARY_HIDDEN = 48
HIDDEN = 96
DROPOUT = 0.12
INPUT_DROPOUT = 0.04

POS_WEIGHT_CAP = 5.0
BRIER_LOSS_WEIGHT = 0.08

# Selection score is averaged across validation missing-feature scenarios.
SELECTION_AUPRC_WEIGHT = 0.18
SELECTION_BRIER_WEIGHT = 0.45

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_FEATURE_NAMES = ["age_at_visit", "sex_male", "PTEDUCAT", "MMSE", "ADAS13", "CDGLOBAL", "CDRSB", "FAQTOTAL"]
TIME_FEATURE_NAMES = ["years_from_first_mci", "visit_interval_years", "time_from_visit_to_landmark_years", "visit_index_after_mci"]

EVAL_SCENARIOS = [
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

# Training scenario probabilities. Intentionally emphasizes full setting while repeatedly exposing the model
# to severe and clinically meaningful missing-feature patterns.
TRAIN_SCENARIO_WEIGHTS = {
    "full_all_modules": 0.42,
    "no_ADAS13": 0.14,
    "no_MMSE": 0.05,
    "no_MMSE_ADAS13": 0.10,
    "no_FAQTOTAL": 0.08,
    "no_CDGLOBAL_CDRSB": 0.06,
    "basic_plus_MMSE_CDRSB_FAQ": 0.09,
    "basic_plus_MMSE_only": 0.04,
    "basic_only": 0.02,
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def finite_guard(name: str, arr: np.ndarray, replace: bool = True) -> np.ndarray:
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.sum(~np.isfinite(arr)))
        msg = f"{name} contains {n_bad} non-finite values."
        if replace:
            warnings.warn(msg + " Applying np.nan_to_num.")
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            raise ValueError(msg)
    return arr


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def require_key(data, candidates: List[str], semantic_name: str) -> str:
    for key in candidates:
        if key in data.files:
            return key
    raise KeyError(f"Could not find {semantic_name}. Tried keys: {candidates}. Available keys: {data.files}")


def get_feature_names(meta: dict, n_features: int) -> List[str]:
    for key in ["primary_feature_names", "feature_names", "clinical_feature_names", "features"]:
        value = meta.get(key)
        if isinstance(value, list) and len(value) == n_features:
            return [str(x) for x in value]
    if n_features == len(DEFAULT_FEATURE_NAMES):
        return DEFAULT_FEATURE_NAMES.copy()
    return [f"feature_{i}" for i in range(n_features)]


def prepare_labels(y_raw: np.ndarray, y_obs_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_obs = finite_guard("y_observed", y_obs_raw.astype(np.float32), True)
    y_obs = (y_obs > 0.5).astype(np.float32)
    y_raw = y_raw.astype(np.float32)
    observed_values = y_raw[y_obs > 0.5]
    if not np.all(np.isfinite(observed_values)):
        raise ValueError("Observed labels contain non-finite values.")
    y = np.where(y_obs > 0.5, y_raw, 0.0).astype(np.float32)
    bad = [float(v) for v in np.unique(y[y_obs > 0.5]) if v not in [0.0, 1.0]]
    if bad:
        raise ValueError(f"Observed labels contain non-binary values: {bad}")
    return y, y_obs


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    probs = 1.0 / (1.0 + np.exp(-logits))
    return np.clip(np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float64)


def logit_np(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def last_valid_index(mask_2d: np.ndarray) -> np.ndarray:
    return np.maximum((mask_2d > 0.5).sum(axis=1).astype(int) - 1, 0)


def nan_summary_for_feature(
    values: np.ndarray,
    observed_mask: np.ndarray,
    delta: np.ndarray,
    slope: np.ndarray,
    visit_mask: np.ndarray,
) -> np.ndarray:
    # shapes: values/masks/delta/slope/visit_mask = (N, T)
    n, t = values.shape
    obs = (observed_mask > 0.5) & (visit_mask > 0.5)
    arr = np.where(obs, values, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(arr, axis=1)
        minv = np.nanmin(arr, axis=1)
        maxv = np.nanmax(arr, axis=1)
        stdv = np.nanstd(arr, axis=1)

    count = obs.sum(axis=1).astype(np.float32)
    rate = count / float(max(t, 1))

    first_idx = np.zeros(n, dtype=int)
    last_idx = np.zeros(n, dtype=int)
    for i in range(n):
        idx = np.where(obs[i])[0]
        if len(idx) > 0:
            first_idx[i] = int(idx[0])
            last_idx[i] = int(idx[-1])
        else:
            first_idx[i] = 0
            last_idx[i] = 0

    first = values[np.arange(n), first_idx]
    last = values[np.arange(n), last_idx]
    last_delta = delta[np.arange(n), last_idx]
    last_slope = slope[np.arange(n), last_idx]

    summary = np.stack(
        [
            last,
            mean,
            minv,
            maxv,
            stdv,
            last - first,
            last_delta,
            last_slope,
            count,
            rate,
        ],
        axis=1,
    ).astype(np.float32)
    summary = np.where(np.isfinite(summary), summary, 0.0).astype(np.float32)
    return summary


def visit_process_summary(Xt: np.ndarray, Xm: np.ndarray, visit_mask: np.ndarray) -> np.ndarray:
    n, t, kt = Xt.shape
    real_visit = visit_mask > 0.5
    xt_masked = np.where(real_visit[:, :, None], Xt, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        time_mean = np.nanmean(xt_masked, axis=1)
        time_std = np.nanstd(xt_masked, axis=1)

    last_idx = last_valid_index(visit_mask)
    time_last = Xt[np.arange(n), last_idx, :]

    n_visits = real_visit.sum(axis=1).astype(np.float32)
    visit_rate = n_visits / float(max(t, 1))

    # Missingness summaries use all feature masks, matching the visit-process design.
    xm_masked = np.where(real_visit[:, :, None], Xm, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_observed_rate = np.nanmean(xm_masked, axis=(1, 2))
        mean_missing_rate = 1.0 - mean_observed_rate

    summary = np.concatenate(
        [
            time_last,
            np.nan_to_num(time_mean, nan=0.0),
            np.nan_to_num(time_std, nan=0.0),
            n_visits[:, None],
            visit_rate[:, None],
            np.nan_to_num(mean_observed_rate, nan=0.0)[:, None],
            np.nan_to_num(mean_missing_rate, nan=0.0)[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    return finite_guard("visit_process_summary", summary, True).astype(np.float32)


def build_v4_module_inputs(
    Xv: np.ndarray,
    Xm: np.ndarray,
    Xt: np.ndarray,
    Xd: np.ndarray,
    Xs: np.ndarray,
    visit_mask: np.ndarray,
    feature_names: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, Dict[str, dict]]:
    module_order = feature_names + ["visit_process"]

    seq_arrays: Dict[str, np.ndarray] = {}
    seq_masks: Dict[str, np.ndarray] = {}
    summary_arrays: Dict[str, np.ndarray] = {}
    schema: Dict[str, dict] = {}

    for j, feature in enumerate(feature_names):
        arr = np.concatenate(
            [
                Xv[:, :, [j]],
                Xm[:, :, [j]],
                Xd[:, :, [j]],
                Xs[:, :, [j]],
            ],
            axis=-1,
        ).astype(np.float32)

        local_mask = ((visit_mask > 0.5) & (Xm[:, :, j] > 0.5)).astype(np.float32)
        arr = finite_guard(f"v4_feature_seq_{feature}", arr * local_mask[:, :, None], True).astype(np.float32)

        summary = nan_summary_for_feature(
            values=Xv[:, :, j],
            observed_mask=Xm[:, :, j],
            delta=Xd[:, :, j],
            slope=Xs[:, :, j],
            visit_mask=visit_mask,
        )

        seq_arrays[feature] = arr
        seq_masks[feature] = local_mask
        summary_arrays[feature] = summary

        schema[feature] = {
            "module_type": "summary_augmented_clinical_feature_module",
            "feature": feature,
            "feature_index": int(j),
            "sequence_input_components": ["value", "feature_mask", "delta_from_first", "slope_like"],
            "summary_components": [
                "last_observed_value",
                "mean_observed_value",
                "min_observed_value",
                "max_observed_value",
                "std_observed_value",
                "last_minus_first_observed_value",
                "last_delta_from_first",
                "last_slope_like",
                "observed_count",
                "observed_rate",
            ],
            "sequence_input_dim": int(arr.shape[-1]),
            "summary_input_dim": int(summary.shape[-1]),
            "module_visit_mask": "visit_mask AND feature_observed_mask",
            "availability_rule": "this feature observed at least once across retained history",
            "v4_refinement": "module-local summary embedding is fused with observed-visit GRU state",
        }

    process_seq = np.concatenate([Xt, Xm], axis=-1).astype(np.float32)
    process_mask = (visit_mask > 0.5).astype(np.float32)
    process_seq = finite_guard("v4_visit_process_seq", process_seq * process_mask[:, :, None], True).astype(np.float32)
    process_summary = visit_process_summary(Xt, Xm, visit_mask)

    seq_arrays["visit_process"] = process_seq
    seq_masks["visit_process"] = process_mask
    summary_arrays["visit_process"] = process_summary
    schema["visit_process"] = {
        "module_type": "summary_augmented_visit_process_module",
        "feature": "X_time_scaled + all_feature_masks",
        "sequence_input_components": ["time_features", "all_feature_masks"],
        "summary_components": [
            "last_time_features",
            "mean_time_features",
            "std_time_features",
            "n_visits",
            "visit_rate",
            "mean_observed_rate",
            "mean_missing_rate",
        ],
        "sequence_input_dim": int(process_seq.shape[-1]),
        "summary_input_dim": int(process_summary.shape[-1]),
        "module_visit_mask": "visit_mask",
        "availability_rule": "any valid retained visit",
        "v4_refinement": "visit-process summary embedding is fused with temporal process encoder",
    }

    avails = []
    for module in module_order:
        if module == "visit_process":
            avail = (np.sum(process_mask > 0.5, axis=1) > 0).astype(np.float32)
        else:
            avail = (np.sum(seq_masks[module] > 0.5, axis=1) > 0).astype(np.float32)
        avails.append(avail)
    availability = np.stack(avails, axis=1).astype(np.float32)

    return seq_arrays, seq_masks, summary_arrays, availability, schema


def default_scenarios(module_order: List[str]) -> pd.DataFrame:
    feature_names = [m for m in module_order if m != "visit_process"]

    def keep_mask(keep_modules: List[str]) -> List[int]:
        keep = set(keep_modules)
        return [1 if m in keep else 0 for m in module_order]

    def drop_mask(drop_modules: List[str]) -> List[int]:
        drop = set(drop_modules)
        return [0 if m in drop else 1 for m in module_order]

    scenarios = [
        ("full_all_modules", "All feature modules and visit_process are available.", drop_mask([])),
        ("no_ADAS13", "ADAS13 feature module removed.", drop_mask(["ADAS13"])),
        ("no_MMSE", "MMSE feature module removed.", drop_mask(["MMSE"])),
        ("no_MMSE_ADAS13", "Both cognitive feature modules removed.", drop_mask(["MMSE", "ADAS13"])),
        ("no_FAQTOTAL", "Functional assessment module removed.", drop_mask(["FAQTOTAL"])),
        ("no_CDGLOBAL_CDRSB", "Global severity modules removed.", drop_mask(["CDGLOBAL", "CDRSB"])),
        ("no_visit_process", "Visit timing/missingness process module removed.", drop_mask(["visit_process"])),
        (
            "basic_plus_MMSE_CDRSB_FAQ",
            "Low-burden clinical set.",
            keep_mask(["age_at_visit", "sex_male", "PTEDUCAT", "MMSE", "CDRSB", "FAQTOTAL", "visit_process"]),
        ),
        (
            "basic_plus_MMSE_only",
            "Very low-burden set.",
            keep_mask(["age_at_visit", "sex_male", "PTEDUCAT", "MMSE", "visit_process"]),
        ),
        (
            "basic_only",
            "Demographic/basic variables only.",
            keep_mask(["age_at_visit", "sex_male", "PTEDUCAT", "visit_process"]),
        ),
        ("visit_process_only", "Only visit-process module.", keep_mask(["visit_process"])),
    ]

    rows = []
    for name, desc, mask in scenarios:
        row = {"scenario": name, "description": desc}
        for m, v in zip(module_order, mask):
            row[f"keep_{m}"] = int(v)
        rows.append(row)
    return pd.DataFrame(rows)


def load_scenarios(module_order: List[str]) -> pd.DataFrame:
    df = read_csv(STEP85_SCENARIOS)
    if df.empty:
        df = default_scenarios(module_order)

    rows = []
    for _, r in df.iterrows():
        if str(r.get("scenario")) not in EVAL_SCENARIOS:
            continue
        row = {"scenario": str(r.get("scenario")), "description": str(r.get("description", ""))}
        for m in module_order:
            col = f"keep_{m}"
            row[col] = int(r.get(col, 0))
        rows.append(row)
    out = pd.DataFrame(rows)

    # Ensure full scenario is present.
    if out.empty or "full_all_modules" not in set(out["scenario"]):
        fallback = default_scenarios(module_order)
        out = fallback[fallback["scenario"].isin(EVAL_SCENARIOS)].copy()

    return out.reset_index(drop=True)


class V4Dataset(Dataset):
    def __init__(self, seq_arrays, seq_masks, summary_arrays, availability, y, y_obs, indices, module_order):
        self.seq_arrays = {k: torch.tensor(v[indices], dtype=torch.float32) for k, v in seq_arrays.items()}
        self.seq_masks = {k: torch.tensor(v[indices], dtype=torch.float32) for k, v in seq_masks.items()}
        self.summary_arrays = {k: torch.tensor(v[indices], dtype=torch.float32) for k, v in summary_arrays.items()}
        self.availability = torch.tensor(availability[indices], dtype=torch.float32)
        self.y = torch.tensor(y[indices], dtype=torch.float32)
        self.y_obs = torch.tensor(y_obs[indices], dtype=torch.float32)
        self.indices = torch.tensor(indices, dtype=torch.long)
        self.module_order = module_order

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, i):
        item = {
            "availability": self.availability[i],
            "y": self.y[i],
            "y_obs": self.y_obs[i],
            "index": self.indices[i],
        }
        for module in self.module_order:
            item[f"X_{module}"] = self.seq_arrays[module][i]
            item[f"M_{module}"] = self.seq_masks[module][i]
            item[f"S_{module}"] = self.summary_arrays[module][i]
        return item


class SummaryAugmentedModuleEncoder(nn.Module):
    def __init__(self, seq_input_dim: int, summary_input_dim: int, d_model: int = D_MODEL):
        super().__init__()
        self.seq_norm = nn.LayerNorm(seq_input_dim)
        self.seq_dropout = nn.Dropout(INPUT_DROPOUT)
        self.seq_proj = nn.Sequential(
            nn.Linear(seq_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.attn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

        self.summary_encoder = nn.Sequential(
            nn.LayerNorm(summary_input_dim),
            nn.Linear(summary_input_dim, SUMMARY_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(SUMMARY_HIDDEN, d_model),
            nn.GELU(),
        )

        self.fuse = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.LayerNorm(d_model),
        )

    def forward(self, X: torch.Tensor, mask: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        mask = (mask > 0.5).float()
        z = self.seq_norm(X * mask[:, :, None])
        z = self.seq_dropout(z)
        z = self.seq_proj(z)
        z, _ = self.gru(z)

        has_any = mask.sum(dim=1, keepdim=True) > 0.5
        scores = self.attn(z).squeeze(-1).masked_fill(mask < 0.5, -1e9)
        scores = torch.where(has_any, scores, torch.zeros_like(scores))

        weights = torch.softmax(scores, dim=1) * mask
        denom = torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-6)
        weights = weights / denom
        seq_state = torch.sum(z * weights[:, :, None], dim=1)
        seq_state = torch.where(has_any, seq_state, torch.zeros_like(seq_state))

        summary_state = self.summary_encoder(summary)
        # If no sequence observation exists, summary should not create hidden availability;
        # availability gate masks the module, but zeroing here makes representation safe.
        summary_state = torch.where(has_any, summary_state, torch.zeros_like(summary_state))

        return self.fuse(torch.cat([seq_state, summary_state], dim=-1))


class SummaryAugmentedV4Model(nn.Module):
    def __init__(self, seq_input_dims: Dict[str, int], summary_input_dims: Dict[str, int], module_order: List[str]):
        super().__init__()
        self.module_order = module_order
        self.encoders = nn.ModuleDict(
            {
                m: SummaryAugmentedModuleEncoder(seq_input_dims[m], summary_input_dims[m])
                for m in module_order
            }
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, len(HORIZONS)),
        )

    def forward(self, module_inputs, module_masks, module_summaries, availability, scenario_mask=None, return_gate=False):
        reps = [
            self.encoders[m](module_inputs[m], module_masks[m], module_summaries[m])
            for m in self.module_order
        ]
        z = torch.stack(reps, dim=1)

        avail = (availability > 0.5).float()
        if scenario_mask is not None:
            if scenario_mask.dim() == 1:
                scenario_mask = scenario_mask[None, :].expand_as(avail)
            avail = avail * (scenario_mask > 0.5).float()

        any_avail = avail.sum(dim=1, keepdim=True) > 0.5
        if not torch.all(any_avail):
            fallback = torch.zeros_like(avail)
            fallback[:, self.module_order.index("visit_process")] = 1.0
            avail = torch.where(any_avail, avail, fallback)

        gate_logits = self.gate(z).squeeze(-1).masked_fill(avail < 0.5, -1e9)
        weights = torch.softmax(gate_logits, dim=1)
        fused = torch.sum(z * weights[:, :, None], dim=1)
        out = self.head(fused)
        return (out, weights) if return_gate else out


def batch_inputs(batch, module_order):
    module_inputs = {m: batch[f"X_{m}"].to(DEVICE) for m in module_order}
    module_masks = {m: batch[f"M_{m}"].to(DEVICE) for m in module_order}
    module_summaries = {m: batch[f"S_{m}"].to(DEVICE) for m in module_order}
    availability = batch["availability"].to(DEVICE)
    return module_inputs, module_masks, module_summaries, availability


def make_loader(ds, shuffle: bool = False):
    gen = torch.Generator()
    gen.manual_seed(SEED)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False, generator=gen if shuffle else None)


def compute_pos_weight(y: np.ndarray, y_obs: np.ndarray, train_idx: np.ndarray) -> torch.Tensor:
    weights = []
    yt = y[train_idx]
    ot = y_obs[train_idx]
    for j in range(len(HORIZONS)):
        obs = ot[:, j] > 0.5
        yy = yt[obs, j]
        pos = float(np.sum(yy == 1))
        neg = float(np.sum(yy == 0))
        if pos <= 0:
            w = 1.0
        else:
            w = np.sqrt(neg / pos)
            w = np.clip(w, 1.0, POS_WEIGHT_CAP)
        weights.append(float(w))
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def masked_loss(logits, y, y_obs, pos_weight):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
    probs = torch.sigmoid(logits)
    brier = (probs - y).pow(2)
    raw = bce + BRIER_LOSS_WEIGHT * brier
    masked = raw * y_obs
    return masked.sum() / torch.clamp(y_obs.sum(), min=1.0)


def scenario_mask_table(scenarios: pd.DataFrame, module_order: List[str]) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[str]]:
    masks = {}
    for _, row in scenarios.iterrows():
        s = str(row["scenario"])
        mask = np.array([int(row.get(f"keep_{m}", 0)) for m in module_order], dtype=np.float32)
        if mask.sum() <= 0:
            mask[module_order.index("visit_process")] = 1.0
        masks[s] = mask

    train_names = [s for s in TRAIN_SCENARIO_WEIGHTS if s in masks]
    if "full_all_modules" not in train_names:
        train_names = ["full_all_modules"] + train_names

    probs = np.array([TRAIN_SCENARIO_WEIGHTS.get(s, 0.01) for s in train_names], dtype=np.float64)
    probs = probs / probs.sum()
    mask_array = np.stack([masks[s] for s in train_names], axis=0).astype(np.float32)
    return masks, mask_array, probs, train_names


def sample_training_scenario_mask(batch_size: int, scenario_mask_array: np.ndarray, scenario_probs: np.ndarray) -> torch.Tensor:
    idx = np.random.choice(np.arange(len(scenario_probs)), size=batch_size, p=scenario_probs)
    mask = scenario_mask_array[idx]
    return torch.tensor(mask, dtype=torch.float32, device=DEVICE)


def train_one_epoch(model, loader, optimizer, pos_weight, module_order, scenario_mask_array, scenario_probs):
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        module_inputs, module_masks, module_summaries, availability = batch_inputs(batch, module_order)
        y = batch["y"].to(DEVICE)
        y_obs = batch["y_obs"].to(DEVICE)
        scenario_mask = sample_training_scenario_mask(y.shape[0], scenario_mask_array, scenario_probs)

        optimizer.zero_grad(set_to_none=True)
        logits = model(module_inputs, module_masks, module_summaries, availability, scenario_mask=scenario_mask)
        loss = masked_loss(logits, y, y_obs, pos_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        bs = int(y.shape[0])
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict(model, loader, module_order, scenario_mask_np=None, return_gate=False):
    model.eval()
    all_idx, all_logits, all_y, all_obs, all_avail, all_gates = [], [], [], [], [], []
    scenario_mask = None
    if scenario_mask_np is not None:
        scenario_mask = torch.tensor(scenario_mask_np, dtype=torch.float32, device=DEVICE)

    for batch in loader:
        module_inputs, module_masks, module_summaries, availability = batch_inputs(batch, module_order)
        if return_gate:
            logits, gates = model(
                module_inputs,
                module_masks,
                module_summaries,
                availability,
                scenario_mask=scenario_mask,
                return_gate=True,
            )
            all_gates.append(gates.detach().cpu().numpy())
        else:
            logits = model(module_inputs, module_masks, module_summaries, availability, scenario_mask=scenario_mask)

        all_idx.append(batch["index"].cpu().numpy())
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(batch["y"].cpu().numpy())
        all_obs.append(batch["y_obs"].cpu().numpy())
        all_avail.append(batch["availability"].cpu().numpy())

    idx = np.concatenate(all_idx)
    logits = np.concatenate(all_logits)
    y = np.concatenate(all_y)
    obs = np.concatenate(all_obs)
    avail = np.concatenate(all_avail)
    probs = sigmoid_np(logits)
    gates = np.concatenate(all_gates) if return_gate else None
    return idx, logits, probs, y, obs, avail, gates


def eval_metrics(model_name: str, split: str, scenario: str, probs: np.ndarray, y: np.ndarray, y_obs: np.ndarray, risk_type: str):
    rows = []
    for j, h in enumerate(HORIZONS):
        obs = y_obs[:, j] > 0.5
        yy = y[obs, j]
        pp = probs[obs, j]
        observed_n = int(obs.sum())
        pos = int(np.sum(yy == 1))
        neg = int(np.sum(yy == 0))
        if observed_n == 0 or len(np.unique(yy)) < 2:
            auroc = auprc = brier = np.nan
        else:
            auroc = float(roc_auc_score(yy, pp))
            auprc = float(average_precision_score(yy, pp))
            brier = float(brier_score_loss(yy, pp))
        rows.append(
            {
                "model": model_name,
                "family": "summary_augmented_feature_modular_v4",
                "split": split,
                "scenario": scenario,
                "risk_type": risk_type,
                "horizon_year": h,
                "observed_n": observed_n,
                "positive_n": pos,
                "negative_n": neg,
                "AUROC": auroc,
                "AUPRC": auprc,
                "Brier": brier,
            }
        )
    return pd.DataFrame(rows)


def mean_metric(df: pd.DataFrame, metric: str) -> float:
    vals = df[df[metric].notna()][metric]
    return float(vals.mean()) if len(vals) else np.nan


def selection_score(metrics: pd.DataFrame) -> float:
    auroc = mean_metric(metrics, "AUROC")
    auprc = mean_metric(metrics, "AUPRC")
    brier = mean_metric(metrics, "Brier")
    if not np.isfinite(auroc):
        auroc = 0.0
    if not np.isfinite(auprc):
        auprc = 0.0
    if not np.isfinite(brier):
        brier = 1.0
    return float(auroc + SELECTION_AUPRC_WEIGHT * auprc - SELECTION_BRIER_WEIGHT * brier)


@torch.no_grad()
def validate_across_scenarios(model, val_loader, module_order, scenario_masks: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, float]:
    frames = []
    # Use a focused validation set for checkpointing to avoid over-optimizing every possible scenario.
    selection_scenarios = [
        "full_all_modules",
        "no_ADAS13",
        "no_MMSE_ADAS13",
        "no_FAQTOTAL",
        "basic_plus_MMSE_CDRSB_FAQ",
        "basic_plus_MMSE_only",
    ]
    for scenario in selection_scenarios:
        if scenario not in scenario_masks:
            continue
        _, _, probs, yy, obs, _, _ = predict(model, val_loader, module_order, scenario_masks[scenario], return_gate=False)
        frames.append(eval_metrics("step89_summary_augmented_modular_v4", "val", scenario, probs, yy, obs, risk_type="raw"))

    if not frames:
        return pd.DataFrame(), -np.inf
    metrics = pd.concat(frames, ignore_index=True)
    # Macro over scenario-horizon rows.
    score = selection_score(metrics)
    return metrics, score


def fit_platt_calibrators_by_scenario(model, val_loader, module_order, scenario_masks: Dict[str, np.ndarray]):
    calibrators: Dict[Tuple[str, int], LogisticRegression | None] = {}
    rows = []
    val_cache = {}

    for scenario, mask in scenario_masks.items():
        idx, logits, raw_probs, yy, obs, avail, gates = predict(model, val_loader, module_order, mask, return_gate=False)
        val_cache[scenario] = {"logits": logits, "raw_probs": raw_probs, "y": yy, "obs": obs}
        for j, h in enumerate(HORIZONS):
            o = obs[:, j] > 0.5
            yj = yy[o, j].astype(int)
            xj = logits[o, j].reshape(-1, 1)
            if o.sum() < 20 or len(np.unique(yj)) < 2:
                calibrators[(scenario, h)] = None
                rows.append({"scenario": scenario, "horizon_year": h, "status": "skipped", "coef": np.nan, "intercept": np.nan, "n": int(o.sum())})
                continue
            clf = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
            clf.fit(xj, yj)
            calibrators[(scenario, h)] = clf
            rows.append(
                {
                    "scenario": scenario,
                    "horizon_year": h,
                    "status": "fit",
                    "coef": float(clf.coef_[0, 0]),
                    "intercept": float(clf.intercept_[0]),
                    "n": int(o.sum()),
                }
            )
    return calibrators, pd.DataFrame(rows), val_cache


def apply_platt_for_scenario(calibrators, scenario: str, logits: np.ndarray) -> np.ndarray:
    raw = sigmoid_np(logits)
    probs = np.zeros_like(raw, dtype=np.float64)
    for j, h in enumerate(HORIZONS):
        clf = calibrators.get((scenario, h))
        if clf is None:
            probs[:, j] = raw[:, j]
        else:
            probs[:, j] = clf.predict_proba(logits[:, j].reshape(-1, 1))[:, 1]
    return np.clip(probs, 0.0, 1.0)


def evaluate_all_scenarios(model, loaders, module_order, scenario_masks):
    raw_frames = []
    platt_frames = []
    pred_frames = []
    gate_frames = []

    calibrators, calibrator_df, _ = fit_platt_calibrators_by_scenario(model, loaders["val"], module_order, scenario_masks)

    for split_name, loader in loaders.items():
        for scenario, mask in scenario_masks.items():
            idx, logits, raw_probs, yy, obs, avail, gates = predict(model, loader, module_order, mask, return_gate=True)
            platt_probs = apply_platt_for_scenario(calibrators, scenario, logits)

            raw_frames.append(eval_metrics("step89_summary_augmented_modular_v4", split_name, scenario, raw_probs, yy, obs, "raw"))
            platt_frames.append(eval_metrics("step89_summary_augmented_modular_v4", split_name, scenario, platt_probs, yy, obs, "scenario_platt"))

            for i in range(len(idx)):
                row = {"split": split_name, "scenario": scenario, "sample_index": int(idx[i])}
                for j, h in enumerate(HORIZONS):
                    row[f"logit_{h}y"] = float(logits[i, j])
                    row[f"raw_risk_{h}y"] = float(raw_probs[i, j])
                    row[f"scenario_platt_risk_{h}y"] = float(platt_probs[i, j])
                    row[f"label_{h}y"] = float(yy[i, j])
                    row[f"observed_{h}y"] = float(obs[i, j])
                pred_frames.append(row)

            for m_i, m in enumerate(module_order):
                gate_frames.append(
                    {
                        "split": split_name,
                        "scenario": scenario,
                        "module": m,
                        "kept_by_scenario": int(mask[m_i]),
                        "mean_gate_weight": float(gates[:, m_i].mean()),
                        "median_gate_weight": float(np.median(gates[:, m_i])),
                        "std_gate_weight": float(gates[:, m_i].std()),
                    }
                )

    metrics = pd.concat(raw_frames + platt_frames, ignore_index=True)
    predictions = pd.DataFrame(pred_frames)
    gate_summary = pd.DataFrame(gate_frames)
    return metrics, predictions, gate_summary, calibrator_df


def compare_to_v3_and_baselines(v4_metrics: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_v4 = v4_metrics[
        (v4_metrics["split"].eq("test")) & (v4_metrics["risk_type"].eq("scenario_platt"))
    ].copy()

    v3 = read_csv(STEP85_METRICS)
    if not v3.empty:
        v3 = v3[v3["risk_type"].eq("platt")].copy()
        v3["model"] = "step84_feature_modular_v3"
        v3["family"] = "feature_modular_v3"
        v3["split"] = "test"
        v3["risk_type"] = "platt"
        keep = ["model", "family", "split", "scenario", "risk_type", "horizon_year", "observed_n", "positive_n", "negative_n", "AUROC", "AUPRC", "Brier"]
        v3 = v3[[c for c in keep if c in v3.columns]]
    else:
        v3 = pd.DataFrame()

    step88 = read_csv(STEP88_COMBINED)
    if not step88.empty:
        step88 = step88[step88["risk_type"].eq("platt")].copy()
        step88["split"] = "test"
        keep = ["model", "family", "split", "scenario", "risk_type", "horizon_year", "observed_n", "positive_n", "negative_n", "AUROC", "AUPRC", "Brier"]
        step88 = step88[[c for c in keep if c in step88.columns]]
    else:
        step88 = pd.DataFrame()

    combined = pd.concat([v3, step88, test_v4], ignore_index=True)
    combined = combined.drop_duplicates(subset=["model", "scenario", "risk_type", "horizon_year"], keep="last")

    rows = []
    for scenario in sorted(test_v4["scenario"].unique()):
        for h in HORIZONS:
            row4 = test_v4[(test_v4["scenario"].eq(scenario)) & (test_v4["horizon_year"].eq(h))]
            if row4.empty:
                continue
            row4 = row4.iloc[0]

            row3 = v3[(v3["scenario"].eq(scenario)) & (v3["horizon_year"].eq(h))]
            if not row3.empty:
                row3 = row3.iloc[0]
            else:
                row3 = None

            baselines = step88[
                (~step88["model"].eq("step84_feature_modular_v3"))
                & (step88["scenario"].eq(scenario))
                & (step88["horizon_year"].eq(h))
            ].copy()

            best_auc = baselines.sort_values("AUROC", ascending=False).iloc[0] if not baselines.empty else None
            best_ap = baselines.sort_values("AUPRC", ascending=False).iloc[0] if not baselines.empty else None
            best_brier = baselines.sort_values("Brier", ascending=True).iloc[0] if not baselines.empty else None

            rows.append(
                {
                    "scenario": scenario,
                    "horizon_year": h,
                    "v4_AUROC": row4["AUROC"],
                    "v3_AUROC": np.nan if row3 is None else row3["AUROC"],
                    "v4_minus_v3_AUROC": np.nan if row3 is None else row4["AUROC"] - row3["AUROC"],
                    "best_baseline_AUROC": np.nan if best_auc is None else best_auc["AUROC"],
                    "best_AUROC_baseline": "" if best_auc is None else best_auc["model"],
                    "v4_minus_best_baseline_AUROC": np.nan if best_auc is None else row4["AUROC"] - best_auc["AUROC"],
                    "v4_AUPRC": row4["AUPRC"],
                    "v3_AUPRC": np.nan if row3 is None else row3["AUPRC"],
                    "v4_minus_v3_AUPRC": np.nan if row3 is None else row4["AUPRC"] - row3["AUPRC"],
                    "best_baseline_AUPRC": np.nan if best_ap is None else best_ap["AUPRC"],
                    "best_AUPRC_baseline": "" if best_ap is None else best_ap["model"],
                    "v4_minus_best_baseline_AUPRC": np.nan if best_ap is None else row4["AUPRC"] - best_ap["AUPRC"],
                    "v4_Brier": row4["Brier"],
                    "v3_Brier": np.nan if row3 is None else row3["Brier"],
                    "v4_minus_v3_Brier": np.nan if row3 is None else row4["Brier"] - row3["Brier"],
                    "best_baseline_Brier": np.nan if best_brier is None else best_brier["Brier"],
                    "best_Brier_baseline": "" if best_brier is None else best_brier["model"],
                    "v4_minus_best_baseline_Brier": np.nan if best_brier is None else row4["Brier"] - best_brier["Brier"],
                }
            )

    return combined, pd.DataFrame(rows)


def paper_table(v4_metrics: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    key_scenarios = [
        "full_all_modules",
        "no_ADAS13",
        "no_MMSE_ADAS13",
        "no_FAQTOTAL",
        "basic_plus_MMSE_CDRSB_FAQ",
        "basic_plus_MMSE_only",
        "basic_only",
    ]
    d = comparison[comparison["scenario"].isin(key_scenarios)].copy()
    return d.sort_values(["scenario", "horizon_year"])


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
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
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Only first {max_rows} rows shown; full table has {len(df)} rows._")
    return "\n".join(lines) + "\n"


def write_readme(metrics, comparison, paper, gate_summary, training_history):
    test = metrics[(metrics["split"].eq("test")) & (metrics["risk_type"].eq("scenario_platt"))].copy()
    full = test[test["scenario"].eq("full_all_modules")].copy()

    lines = []
    lines.append("# Step89 summary-augmented scenario-dropout modular v4\n\n")
    lines.append("## Purpose\n\n")
    lines.append(
        "This model refines the feature-level modular route after Step88 showed that scenario-specific HGB/history-summary baselines remained strong. "
        "The v4 model adds module-local trajectory summaries to each feature encoder and trains with scenario dropout, so a single modular model learns to operate under clinically meaningful missing-feature patterns.\n\n"
    )

    lines.append("## Key design changes\n\n")
    lines.append("- One clinical feature per module plus visit_process.\n")
    lines.append("- Each feature module combines observed-visit GRU encoding with module-local trajectory summary embedding.\n")
    lines.append("- Summary components include last, mean, min, max, std, last-first change, last delta, last slope, observed count, and observed rate.\n")
    lines.append("- Training uses scenario dropout rather than only light random module dropout.\n")
    lines.append("- Scenario-specific Platt calibration is fitted on validation logits and reported separately.\n\n")

    lines.append("## Full-module internal test performance\n\n")
    lines.append(md_table(full[["horizon_year", "observed_n", "positive_n", "AUROC", "AUPRC", "Brier"]]))
    lines.append("\n## V4 versus V3 and best scenario-specific baseline\n\n")
    lines.append(md_table(paper, max_rows=120))
    lines.append("\n## Gate summary\n\n")
    gs = gate_summary[(gate_summary["split"].eq("test")) & (gate_summary["scenario"].eq("full_all_modules"))].copy()
    if not gs.empty:
        gs = gs.sort_values("mean_gate_weight", ascending=False)
    lines.append(md_table(gs[["module", "kept_by_scenario", "mean_gate_weight", "median_gate_weight", "std_gate_weight"]]))
    lines.append("\n## Training history tail\n\n")
    lines.append(md_table(training_history.tail(20), max_rows=20))
    lines.append("\n## Interpretation boundary\n\n")
    lines.append(
        "Step89 should be interpreted as an internal refinement. If it improves v3 in missing-feature scenarios, the claim should be limited to a single summary-augmented modular model with better internal scenario behavior. "
        "It should still not be described as clinically deployed or universally superior to oracle scenario-specific baselines unless the results support that claim.\n"
    )

    (REPORT_DIR / "README_step89_summary_augmented_scenario_dropout_modular_v4.md").write_text("".join(lines), encoding="utf-8")


def main():
    print("=" * 88)
    print("[STEP 89] Train summary-augmented scenario-dropout modular v4")
    print(f"[SCRIPT VERSION] {SCRIPT_VERSION}")
    print("=" * 88)
    print(f"[PROJECT ROOT] {PROJECT_ROOT}")
    print("[MODE] internal modular refinement only; no external data; no raw data edits")
    print(f"[DEVICE] {DEVICE}")
    print(f"[INPUT NPZ] {INPUT_NPZ}")
    print(f"[OUT DIR] {OUT_DIR}")
    print(f"[REPORT DIR] {REPORT_DIR}")

    set_seed(SEED)

    if not INPUT_NPZ.exists():
        raise FileNotFoundError(f"Missing input NPZ: {INPUT_NPZ}")

    data = np.load(INPUT_NPZ, allow_pickle=True)
    value_key = require_key(data, ["X_values_imputed_scaled", "X_values_scaled", "X_values", "Xv"], "scaled/imputed values")
    mask_key = require_key(data, ["X_feature_mask", "Xm", "X_missing_mask"], "feature masks")
    time_key = require_key(data, ["X_time_scaled", "X_time", "Xt"], "time features")
    delta_key = require_key(data, ["X_delta_from_first", "X_delta", "Xd"], "delta features")
    slope_key = require_key(data, ["X_slope_from_first", "X_slope", "Xs"], "slope features")
    visit_mask_key = require_key(data, ["X_visit_mask", "visit_mask"], "visit mask")
    y_key = require_key(data, ["y_labels", "y"], "labels")
    yobs_key = require_key(data, ["y_observed", "y_obs"], "observed labels")

    Xv = finite_guard(value_key, data[value_key].astype(np.float32), True)
    Xm = finite_guard(mask_key, data[mask_key].astype(np.float32), True)
    Xt = finite_guard(time_key, data[time_key].astype(np.float32), True)
    Xd = finite_guard(delta_key, data[delta_key].astype(np.float32), True)
    Xs = finite_guard(slope_key, data[slope_key].astype(np.float32), True)
    visit_mask = finite_guard(visit_mask_key, data[visit_mask_key].astype(np.float32), True)
    y, y_obs = prepare_labels(data[y_key], data[yobs_key])

    train_idx = data["train_idx"].astype(np.int64)
    val_idx = data["val_idx"].astype(np.int64)
    test_idx = data["test_idx"].astype(np.int64)

    meta = read_json(TOKEN_METADATA_JSON)
    feature_names = get_feature_names(meta, Xv.shape[-1])
    module_order = feature_names + ["visit_process"]

    seq_arrays, seq_masks, summary_arrays, availability, schema = build_v4_module_inputs(
        Xv, Xm, Xt, Xd, Xs, visit_mask, feature_names
    )
    seq_input_dims = {m: int(seq_arrays[m].shape[-1]) for m in module_order}
    summary_input_dims = {m: int(summary_arrays[m].shape[-1]) for m in module_order}

    scenarios = load_scenarios(module_order)
    scenario_masks, train_scenario_mask_array, train_scenario_probs, train_scenario_names = scenario_mask_table(scenarios, module_order)

    scenarios.to_csv(REPORT_DIR / "scenario_definitions_used.csv", index=False)
    pd.DataFrame(
        {
            "scenario": train_scenario_names,
            "training_probability": train_scenario_probs,
        }
    ).to_csv(OUT_DIR / "training_scenario_probabilities.csv", index=False)

    print("\n[DATA]")
    print(f"Xv={Xv.shape}, Xm={Xm.shape}, Xt={Xt.shape}, Xd={Xd.shape}, Xs={Xs.shape}, visit_mask={visit_mask.shape}")
    print(f"y={y.shape}, y_obs={y_obs.shape}")
    print(f"train/val/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print(f"modules={module_order}")
    print(f"seq_input_dims={seq_input_dims}")
    print(f"summary_input_dims={summary_input_dims}")
    print(f"train_scenarios={list(zip(train_scenario_names, train_scenario_probs.round(4).tolist()))}")

    train_ds = V4Dataset(seq_arrays, seq_masks, summary_arrays, availability, y, y_obs, train_idx, module_order)
    val_ds = V4Dataset(seq_arrays, seq_masks, summary_arrays, availability, y, y_obs, val_idx, module_order)
    test_ds = V4Dataset(seq_arrays, seq_masks, summary_arrays, availability, y, y_obs, test_idx, module_order)

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)
    test_loader = make_loader(test_ds, shuffle=False)

    model = SummaryAugmentedV4Model(seq_input_dims, summary_input_dims, module_order).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    pos_weight = compute_pos_weight(y, y_obs, train_idx)

    print(f"[POS WEIGHT] {pos_weight.detach().cpu().numpy().tolist()}")

    best_score = -np.inf
    best_epoch = -1
    patience_count = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            pos_weight,
            module_order,
            train_scenario_mask_array,
            train_scenario_probs,
        )
        val_metrics, val_score = validate_across_scenarios(model, val_loader, module_order, scenario_masks)

        val_mean_auroc = mean_metric(val_metrics, "AUROC")
        val_mean_auprc = mean_metric(val_metrics, "AUPRC")
        val_mean_brier = mean_metric(val_metrics, "Brier")

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_selection_score": val_score,
            "val_mean_AUROC_across_scenarios": val_mean_auroc,
            "val_mean_AUPRC_across_scenarios": val_mean_auprc,
            "val_mean_Brier_across_scenarios": val_mean_brier,
        }
        history.append(row)

        improved = val_score > best_score + 1e-5
        if improved:
            best_score = val_score
            best_epoch = epoch
            patience_count = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "script_version": SCRIPT_VERSION,
                    "feature_names": feature_names,
                    "module_order": module_order,
                    "horizons": HORIZONS,
                    "seq_input_dims": seq_input_dims,
                    "summary_input_dims": summary_input_dims,
                    "schema": schema,
                    "model_config": {
                        "d_model": D_MODEL,
                        "summary_hidden": SUMMARY_HIDDEN,
                        "hidden": HIDDEN,
                        "dropout": DROPOUT,
                        "input_dropout": INPUT_DROPOUT,
                        "brier_loss_weight": BRIER_LOSS_WEIGHT,
                        "pos_weight_cap": POS_WEIGHT_CAP,
                        "scenario_dropout": True,
                        "train_scenario_weights": TRAIN_SCENARIO_WEIGHTS,
                    },
                    "best_epoch": best_epoch,
                    "best_selection_score": float(best_score),
                    "best_val_mean_AUROC": float(val_mean_auroc) if np.isfinite(val_mean_auroc) else None,
                    "best_val_mean_AUPRC": float(val_mean_auprc) if np.isfinite(val_mean_auprc) else None,
                    "best_val_mean_Brier": float(val_mean_brier) if np.isfinite(val_mean_brier) else None,
                },
                OUT_DIR / "best_model.pt",
            )
        else:
            patience_count += 1

        print(
            f"[EPOCH {epoch:03d}] train_loss={train_loss:.5f} "
            f"val_AUROC={val_mean_auroc:.4f} val_AUPRC={val_mean_auprc:.4f} "
            f"val_Brier={val_mean_brier:.4f} score={val_score:.5f} {'*' if improved else ''}"
        )

        if patience_count >= PATIENCE:
            print(f"[EARLY STOP] No selection-score improvement for {PATIENCE} epochs.")
            break

    checkpoint = torch.load(OUT_DIR / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    metrics, predictions, gate_summary, calibrator_df = evaluate_all_scenarios(model, loaders, module_order, scenario_masks)

    training_history = pd.DataFrame(history)
    schema_df = pd.DataFrame([{"module": k, **v} for k, v in schema.items()])

    combined, comparison = compare_to_v3_and_baselines(metrics)
    paper = paper_table(metrics, comparison)

    metrics.to_csv(OUT_DIR / "metrics_by_scenario_raw_and_scenario_platt.csv", index=False)
    predictions.to_csv(OUT_DIR / "predictions_by_scenario.csv", index=False)
    gate_summary.to_csv(OUT_DIR / "gate_summary_by_scenario.csv", index=False)
    calibrator_df.to_csv(OUT_DIR / "scenario_platt_calibration_parameters.csv", index=False)
    training_history.to_csv(OUT_DIR / "training_history.csv", index=False)
    schema_df.to_csv(OUT_DIR / "module_schema.csv", index=False)
    combined.to_csv(REPORT_DIR / "combined_v3_v4_baseline_stress_metrics.csv", index=False)
    comparison.to_csv(REPORT_DIR / "v4_vs_v3_and_best_baseline_by_scenario.csv", index=False)
    paper.to_csv(REPORT_DIR / "paper_table_v4_vs_v3_baseline_stress.csv", index=False)

    config = {
        "script_version": SCRIPT_VERSION,
        "input_npz": str(INPUT_NPZ),
        "feature_names": feature_names,
        "module_order": module_order,
        "horizons": HORIZONS,
        "seq_input_dims": seq_input_dims,
        "summary_input_dims": summary_input_dims,
        "model_config": {
            "d_model": D_MODEL,
            "summary_hidden": SUMMARY_HIDDEN,
            "hidden": HIDDEN,
            "dropout": DROPOUT,
            "input_dropout": INPUT_DROPOUT,
            "brier_loss_weight": BRIER_LOSS_WEIGHT,
            "pos_weight_cap": POS_WEIGHT_CAP,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "seed": SEED,
            "selection_auprc_weight": SELECTION_AUPRC_WEIGHT,
            "selection_brier_weight": SELECTION_BRIER_WEIGHT,
            "scenario_dropout": True,
            "train_scenario_weights": TRAIN_SCENARIO_WEIGHTS,
        },
        "best_epoch": int(best_epoch),
        "best_selection_score": float(best_score),
        "device": DEVICE,
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    write_readme(metrics, comparison, paper, gate_summary, training_history)

    audit = {
        "script_version": SCRIPT_VERSION,
        "mode": "internal_summary_augmented_scenario_dropout_modular_v4_training",
        "no_external_data": True,
        "no_raw_data_edits": True,
        "project_root": str(PROJECT_ROOT),
        "input_npz": str(INPUT_NPZ),
        "out_dir": str(OUT_DIR),
        "report_dir": str(REPORT_DIR),
        "outputs": {
            "checkpoint": str(OUT_DIR / "best_model.pt"),
            "metrics": str(OUT_DIR / "metrics_by_scenario_raw_and_scenario_platt.csv"),
            "predictions": str(OUT_DIR / "predictions_by_scenario.csv"),
            "gates": str(OUT_DIR / "gate_summary_by_scenario.csv"),
            "calibration": str(OUT_DIR / "scenario_platt_calibration_parameters.csv"),
            "training_history": str(OUT_DIR / "training_history.csv"),
            "schema": str(OUT_DIR / "module_schema.csv"),
            "config": str(OUT_DIR / "config.json"),
            "comparison": str(REPORT_DIR / "v4_vs_v3_and_best_baseline_by_scenario.csv"),
            "paper_table": str(REPORT_DIR / "paper_table_v4_vs_v3_baseline_stress.csv"),
            "readme": str(REPORT_DIR / "README_step89_summary_augmented_scenario_dropout_modular_v4.md"),
        },
    }
    (REPORT_DIR / "step89_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("\n[FINAL TEST METRICS: V4 SCENARIO-PLATT]")
    print(
        metrics[
            (metrics["split"].eq("test")) & (metrics["risk_type"].eq("scenario_platt"))
        ].sort_values(["scenario", "horizon_year"]).to_string(index=False)
    )

    print("\n[V4 VS V3 AND BEST BASELINE]")
    print(comparison.to_string(index=False))

    print("\n[OUTPUTS]")
    for _, path in audit["outputs"].items():
        print(f"  - {path}")
    print(f"  - {REPORT_DIR / 'step89_audit.json'}")

    print("\n[DONE] Step89 summary-augmented scenario-dropout modular v4 trained.")


if __name__ == "__main__":
    main()
