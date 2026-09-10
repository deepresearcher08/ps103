"""Pace / schedule-slippage features for MoSPI prediction models.

Two features that extend the dominant expenditure_ratio signal with TIME:

  elapsed_share = elapsed_years / planned_duration_years
    (#2 schedule slippage ratio). 1.0 = project has consumed its entire
    originally sanctioned calendar window (contractually "late" in elapsed
    terms). Different axis from expenditure_ratio: financial vs calendar pace.

  spend_pace = expenditure_ratio / elapsed_share   (clipped)
    (#1 velocity, single-vintage substitute). The budget fraction spent PER
    fraction of the planned schedule elapsed. spend_pace == 1.0 means spend is
    exactly on the linear schedule line; > 1 burns budget faster than time;
    < 1 lags. expenditure_ratio alone is timing-blind (0.6 spent could be year
    1 of a 10-year plan OR year 7) — pace resolves that.

HONESTY NOTE (why this is NOT the multi-vintage velocity):
  True velocity = delta(expenditure_ratio) BETWEEN monthly Flash Report
  vintages. Only ONE vintage exists in this repo (data/raw/FR_May2025.pdf plus
  the May-2024 annexure-derived tables), so delta-between-vintages is not
  extractable. spend_pace is the within-snapshot substitute and must be
  described as such in the demo, not sold as accelerator data.

Point-in-time discipline:
  Features and labels are both measured as of the SAME snapshot:
    annexure rows -> 2024.42 (May 2024 Flash Report)
    snapshot rows -> 2025.42 (May 2025 Flash Report)
  elapsed = snap_year - approval_year. No feature uses the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SNAP_YEAR_ANNEXURE = 2024.42   # May 2024 Flash Report (source of annexure tables)
SNAP_YEAR_2025 = 2025.42       # May 2025 Flash Report (data/raw/FR_May2025.pdf)
MIN_SHARE = 0.10               # floor for elapsed_share before division
MAX_PACE = 12.0                # clip for spend_pace (12x budget-burn per unit time)
MAX_SHARE = 6.0                # clip for elapsed_share


# --------------------------------------------------------------------------- #
# helpers: approval_year attachment for the COST table (which has no dates)
# --------------------------------------------------------------------------- #
def join_approval(df: pd.DataFrame, time_csv: str | None = None) -> pd.DataFrame:
    """Attach approval_year(+snap_year) to the COST table from the TIME table.

    The cost CSV carries no dates; the time table does (same project key,
    sector|project_name). Rows that do not resolve keep NaN and get imputed by
    the feature builders.
    """
    df = df.copy()
    if "approval_year" in df.columns and df["approval_year"].notna().any():
        return df
    key_col = df.get("key")
    if key_col is None or key_col.notna().sum() == 0:
        if "sector" not in df.columns or "project_name" not in df.columns:
            return df
        key = df["sector"].astype(str) + "|" + df["project_name"].astype(str)
    else:
        key = key_col
    try:
        time = pd.read_csv(time_csv or "data/real_mospi_time.csv", usecols=["sector", "project_name", "approval_year"])
    except Exception:
        return df
    tk = time["sector"].astype(str) + "|" + time["project_name"].astype(str)
    amap = dict(zip(tk, time["approval_year"]))
    df["approval_year"] = key.map(amap)
    return df


def attach_duration(df: pd.DataFrame, time_csv: str | None = None) -> pd.DataFrame:
    """Join planned_duration_years from the TIME table onto a date-less frame.

    The COST annexure table has no approval/DOC columns, so pace/slippage
    features need the planning horizon from the time table (same key,
    sector|project_name). Unmatched rows stay NaN (imputed downstream).
    """
    df = df.copy()
    if "planned_duration_years" in df.columns and df["planned_duration_years"].notna().any():
        return df
    if "sector" not in df.columns or "project_name" not in df.columns:
        return df
    key = df["sector"].astype(str) + "|" + df["project_name"].astype(str)
    try:
        time = pd.read_csv(time_csv or "data/real_mospi_time.csv",
                           usecols=["sector", "project_name", "planned_duration_years"])
    except Exception:
        return df
    tk = time["sector"].astype(str) + "|" + time["project_name"].astype(str)
    dmap = dict(zip(tk, time["planned_duration_years"]))
    df["planned_duration_years"] = key.map(dmap)
    return df


def add_snap_year(df: pd.DataFrame, snap_year: float | None = None) -> pd.DataFrame:
    """Mark each row's measurement vintage (defaults annexure 2024.42)."""
    df = df.copy()
    if "snap_year" in df.columns and df["snap_year"].notna().any():
        return df
    df["snap_year"] = float(snap_year if snap_year is not None else SNAP_YEAR_ANNEXURE)
    return df


def detect_snap_year(df: pd.DataFrame) -> float:
    """Snapshot year of a frame: 2025.42 if it carries the May-2025 Flash
    Report column layout (original_doc_year), else 2024.42 (annexure tables)."""
    if "original_doc_year" in df.columns and df["original_doc_year"].notna().any():
        return SNAP_YEAR_2025
    return SNAP_YEAR_ANNEXURE


# --------------------------------------------------------------------------- #
# feature builders
# --------------------------------------------------------------------------- #
def pace_features(df: pd.DataFrame, snap_year: float | None = None,
                  ensure_cols: bool = True) -> pd.DataFrame:
    """Add elapsed_share and spend_pace columns to df (in place).

    Requires df['approval_year'] and a planned-duration column:
      'planned_duration_years' preferred, else computed from
      'planned_doc_year'/'original_doc_year' - approval_year.
    Missing approval_year / duration -> NaN (impute at the caller/learner).

    snap_year=None -> per-row measurement vintage: rows carrying an
    original_doc_year (May-2025 Flash-Report snapshot) are measured at 2025.42,
    all others (annexure tables) at 2024.42. Pass an explicit float to force one
    vintage for the whole frame.

    Honest modelling note: with one vintage (May-2024 annexure + May-2025
    snapshot) TRUE cross-month velocity cannot be measured; spend_pace is the
    within-snapshot substitute. elapsed_share collides with the label-reveal
    observation window (young projects cannot be flagged delayed yet), so the
    grouped-CV / external-holdout A/B is the defensible figure, NOT the
    ungrouped RepeatedStratifiedKFold AUC reported by predict.py.
    """
    if "approval_year" not in df.columns:
        if ensure_cols:
            df["elapsed_share"] = np.nan
            df["spend_pace"] = np.nan
        return df
    if snap_year is None:
        if "original_doc_year" in df.columns:
            o = pd.to_numeric(df["original_doc_year"], errors="coerce").to_numpy(dtype=float)
            df = df.copy()
            df["snap_year"] = np.where(np.isfinite(o), SNAP_YEAR_2025, SNAP_YEAR_ANNEXURE)
        else:
            df = add_snap_year(df, SNAP_YEAR_ANNEXURE)
    else:
        df = add_snap_year(df, snap_year)
    sy = pd.to_numeric(df["snap_year"], errors="coerce").to_numpy(dtype=float)
    ap = pd.to_numeric(df["approval_year"], errors="coerce").to_numpy(dtype=float)
    dur = None
    if "planned_duration_years" in df.columns:
        dur = pd.to_numeric(df["planned_duration_years"], errors="coerce").to_numpy(dtype=float)
    if dur is None or not np.isfinite(dur).any():
        for c in ("planned_doc_year", "original_doc_year"):
            if c in df.columns:
                d2 = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
                dur = d2 - ap
                break
        if dur is None or not np.isfinite(dur).any():
            df["elapsed_share"] = np.nan
            df["spend_pace"] = np.nan
            return df
    elapsed = sy - ap
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(np.isfinite(dur) & (dur > 0) & np.isfinite(elapsed),
                         np.clip(elapsed / np.maximum(dur, 0.5), 0.0, MAX_SHARE), np.nan)
    orig = df["original_cost_cr"].clip(lower=0).to_numpy(dtype=float)
    exp = df["expenditure_cum_cr"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    er = exp / np.maximum(orig, 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        pace = np.where(np.isfinite(share),
                        np.clip(er / np.maximum(share, MIN_SHARE), 0.0, MAX_PACE), np.nan)
    df["elapsed_share"] = share
    df["spend_pace"] = pace
    return df


def impute_pace(df: pd.DataFrame, median: float | None = None,
                pace_median: float | None = None) -> pd.DataFrame:
    """Fill NaN elapsed_share / spend_pace with training medians (serving/split use)."""
    df = df.copy()
    es = pd.to_numeric(df["elapsed_share"], errors="coerce")
    sp = pd.to_numeric(df["spend_pace"], errors="coerce")
    es_med = median if median is not None else float(es.median()) if np.isfinite(es).any() else 0.5
    sp_med = pace_median if pace_median is not None else float(sp.median()) if np.isfinite(sp).any() else 1.0
    df["elapsed_share"] = es.fillna(es_med)
    df["spend_pace"] = sp.fillna(sp_med)
    return df