"""Persistent, versioned, LEAN prediction service for the MoSPI dashboard.

Provides practical estimates an official can act on:
  - cost (₹ overrun):  P(overrun) + estimated overrun_ratio  ->  ₹ over original
  - time (delay):       P(delayed) + estimated tor_months (delay magnitude)

LEAK-FREE rule (consistent with time_model.py / real_model_v2.py):
  * cost features: log original cost, expenditure ratio, log expenditure,
    out-of-fold sector ratio  (all available at decision time).
  * time features: log original cost, expenditure ratio, planned duration,
    out-of-fold sector delay rate. The CSV `cost_overrun_pct` column is NOT
    used (it is target-collinear); observation-window (age) features are NOT
    used (survivorship artifact).
  * schedule features (elapsed_share, spend_pace) are used only when a row's
    approval year AND planned duration are known. Rows scored without those
    dates use the SCHEDULE-BLIND model (pace columns dropped, fit alongside),
    never a silently-assumed mid-flight stage - predictions expose
    `schedule_used` so the dashboard can say which estimator answered.
  * schedule_risk (cost model) = OOF P(delayed) from time models trained on
    OTHER states ONLY (GroupKFold-by-state); never the realized delayed/tor
    outcome, which is future relative to a live prediction. The realized time
    labels are never cost features.

Models are versioned and persisted to data/models/ so the dashboard can serve
the last-trained model and a `retrain()` can update them in place (continual
learning). Components are fit with OOF so predictions are honest.

Public API
----------
  get_service()          -> cached singleton with .train(seed), .predict_project(row), ...
  train_all(force=False) -> fit + persist cost/C-time estimators, return report dict
  retrain_with(new_pdf?) -> ingest new data (if given) then train_all()  [continual]
  predict_project(row)   -> {'cost': {...}, 'time': {...}} estimates for one project
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, GroupKFold

from src.pace import pace_features, join_approval, attach_duration, SNAP_YEAR_ANNEXURE, SNAP_YEAR_2025
from src.geo import extract_state, STATE_COORDINATES
from src.price_index import attach_price_index, escalations

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False

try:
    import joblib
    HAVE_JOBLIB = True
except Exception:
    HAVE_JOBLIB = False

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS_DIR = DATA / "models"
COST_CSV = DATA / "real_mospi_projects.csv"
TIME_CSV = DATA / "real_mospi_time.csv"
SNAPSHOT_CSV = DATA / "mospi_may2025_projects.csv"

COST_OVERUN_THRESHOLD = 0.05
CAP_TOR = 240.0
CURRENT_YEAR = 2026

COST_FEATURES = ["log_original_cost", "expenditure_ratio", "log_expenditure", "sector_rate",
                  "geo_latitude", "geo_longitude", "elapsed_share", "spend_pace",
                  "revise_log", "price_esc", "price_level_approval", "schedule_risk"]
TIME_FEATURES = ["log_original_cost", "expenditure_ratio", "planned_duration_yrs", "sector_delay_rate",
                 "geo_latitude", "geo_longitude", "elapsed_share", "spend_pace", "agency_delay_rate"]
# PACE_FEATURES = date/approval-dependent features. They need approval_year (and
# for pace, planned_duration_years), which schedule-blind rows do not have, so the
# schedule-blind model excludes exactly this set. price_esc/price_level_approval are
# scheduled-dependent too: they are functions of approval_year alone.
PACE_FEATURES = ["elapsed_share", "spend_pace", "price_esc", "price_level_approval"]

# NOTE ON GEO HONESTY (src/grouped_cv_leak_test.py):
# The geo columns are real town/district coords for ~1/3 of rows (from city
# tokens in project names, see src/district_geo.py) and state centroids for the
# rest (data/*.statecentroid.bak hold the pre-district versions). DO NOT quote
# the ungrouped RepeatedStratifiedKFold AUCs in this module as the geo story:
# those figures come from a SPLIT THAT SHARES STATES BETWEEN FOLDS, so a
# state-centroid geo feature memorises state identity (group leakage). Under a
# GroupKFold-by-state split the geo features add ~0 (COST +0.01, TIME -0.02),
# and the external temporal holdout agrees (~0). Quoting grouped-CV or the
# external holdout is the defensible number; geo is kept because it is REAL
# geography for the map, not because it wins a CV letter.

# NOTE ON PACE / SLIPPAGE HONESTY (src/grouped_cv_leak_test.py + temporal_eval):
# elapsed_share (elapsed/planned duration) and spend_pace (expenditure
# velocity) are the user-requested #1/#2 features. They measured +0.04..+0.14
# AUC on the external temporal holdout and +0.08..+0.16 under GroupKFold-by-state,
# so they are REAL signal -- but identical to geo they collide with the
# label-reveal observation window, so the defensible A/B is grouped-CV /
# external holdout, not the repeated-stratified figures on the model card.

# NOTE ON schedule_risk (feature #1: time risk -> cost):
# schedule_risk = OOF P(delayed) from time models trained on other states only.
# Honest A/Bs: grouped-by-state LR +0.008 / RF +0.000 / LGBM -0.007 (reg MAE
# -0.004, R2 +0.013); external re-measure RF -0.069..+0.015 (reg ~flat).
# Verdict: NEUTRAL for the cost classifier (flexible models already extract the
# delay signal from expenditure_ratio + pace), slight reg benefit. Kept for the
# ministry's stated causal chain (delay -> cost) but NOT claimed as a cost lift.

# NOTE ON price escalation (feature #2: input-price index at sanction):
# price_esc = log(index(2024)/index(approval_year)) and price_level_approval =
# log(index(approval_year)), sector-mapped (steel+cement WPI for civil-works
# sectors, USD/INR for import-equipment sectors - see src/price_index.py).
# Fixed reference year (2024) for BOTH training and serving, so CV and live
# predictions are scale-consistent; only public macro series, known in advance
# -> leak-free. HONESTY: with a single data vintage and approvals clustered
# 2018-2022 the index moves are a smooth function of year, so in-fold it is
# ~collinear with approval_year and expected to add ~0 (like schedule_risk); its
# real payoff is future out-of-sample vintages (a project sanctioned in a
# high/low-price regime). Measured A/Bs: grouped-by-state clf LR +0.008 / RF
# +0.010 / LGBM +0.010, reg MAE -0.003 / R2 +0.008; EXTERNAL clf RF
# -0.035..+0.019 (wash, as predicted) but reg MAE -0.017..-0.018 on ALL cutoffs.
# Verdict: KEEP - the classifier gain washes (~collinear with year), but the
# overrun-size estimate (₹ exposure) improves consistently out-of-sample.

# NOTE ON revise_log (COST feature #3: sanction revision so far):
# revise_log = log1p((revised_cost_cr - original_cost_cr)_+ / original_cost_cr),
# the sanction growth the ministry has ALREADY taken at decision time (a proxy
# for revision count: only one vintage exists, no revision-history column).
# Leak-free: revised_cost_cr is current sanctioned cost, observable now; the
# target is the ANTICIPATED-vs-ORIGINAL ratio (future gap), which is not fed in.
# It is a base feature (no dates needed) so the schedule-blind model uses it
# too. Rows never revised = 0 (revised==original). Measured in grouped CV as a
# none -> none+rev delta; expect a small positive (already-revised projects
# tend to keep overrunning) but smaller than pace.

# NOTE ON agency_delay_rate (TIME feature #4: implementing agency):
# agency parsed from the "[...]ORG,STATE" suffix in project names; encoded as an
# OOF smoothed delay rate per agency, same discipline as sector_delay_rate
# (RepeatedStratifiedKFold by row - the honest grouped-CV harness recomputes it
# per train fold). Rows without a suffix fall back to the global delay rate.
# Expected: management/execution signal orthogonal to sector/pacing.


# --------------------------------------------------------------------------- #
# leak-free OOF encoders
# --------------------------------------------------------------------------- #
def _oof_ratio_kf(y, groups, m=20.0, seed=42):
    """Out-of-fold m-estimate-smoothed group rate (KFold: for continuous targets)."""
    enc = np.zeros(len(y))
    g = pd.Series(groups)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in kf.split(g):
        ytr, gtr, gva = y[tr], g.iloc[tr], g.iloc[va]
        gm = pd.Series(ytr).groupby(gtr.values).mean()
        cnt = pd.Series(ytr).groupby(gtr.values).size()
        enc[va] = gva.map((cnt * gm + m * ytr.mean()) / (cnt + m)).fillna(ytr.mean()).to_numpy()
    return enc


def _oof_rate_strat(y, groups, m=20.0, seed=42):
    """Out-of-fold group rate (stratified: for binary targets)."""
    enc = np.zeros(len(y))
    g = pd.Series(groups)
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
    for tr, va in skf.split(g, y):
        ytr, gtr, gva = y[tr], g.iloc[tr], g.iloc[va]
        gm = pd.Series(ytr).groupby(gtr.values).mean()
        cnt = pd.Series(ytr).groupby(gtr.values).size()
        enc[va] = gva.map((cnt * gm + m * ytr.mean()) / (cnt + m)).fillna(ytr.mean()).to_numpy()
    return enc


# --------------------------------------------------------------------------- #
# OOF schedule-slippage risk (feature #1: time risk -> cost model)
# --------------------------------------------------------------------------- #
_AGENCY_RE = re.compile(r"\]\s*([^,\]]+?)\s*,")


def _agency_from_name(name):
    """Implementing agency from the '[...]ORG,STATE' suffix, else NaN.

    e.g. '... - [020100044]BHAVNI,TAMIL NADU' -> 'BHAVNI'."""
    if not isinstance(name, str):
        return np.nan
    m = _AGENCY_RE.search(name)
    return m.group(1).strip().upper() if m else np.nan


def _states_for(df, time_state_map=None):
    """Per-row state: name-suffix -> (er) time-key join -> (er) geo-nearest.

    Mirrors src/grouped_cv_leak_test._attach_state so state grouping (and thus
    the OOF schedule-risk folds) is identical to the leak-test harness."""
    states = df["project_name"].apply(extract_state)
    if time_state_map is not None:
        key = df["sector"].astype(str) + "|" + df["project_name"].astype(str)
        states = states.where(states.notna(), key.map(time_state_map))
    lat = pd.to_numeric(df.get("geo_latitude"), errors="coerce")
    lon = pd.to_numeric(df.get("geo_longitude"), errors="coerce")
    needs = states.isna()
    for i in df.index[needs]:
        if not (np.isfinite(lat.at[i]) and np.isfinite(lon.at[i])):
            continue
        best, bd = "UNKNOWN", np.inf
        for st, c in STATE_COORDINATES.items():
            if st == "MULTI STATE":
                continue
            d = (float(lat.at[i]) - c["lat"]) ** 2 + (float(lon.at[i]) - c["lon"]) ** 2
            if d < bd:
                bd, best = d, st
        states.at[i] = best
    return states.fillna("UNKNOWN").astype(str).to_numpy()


def _prepare_time_table():
    """Annexure TIME table with pace features and a per-row state (for grouping)."""
    t = _load_time()
    t = pace_features(t, SNAP_YEAR_ANNEXURE)
    t["state"] = _states_for(t)
    return t


def _time_feature_matrix(df, rate_map=None, overall=None, geo_means=None, agency_map=None):
    """Time-model feature matrix for arbitrary rows (pace-inclusive).

    rate_map/agency_map: sector/agency -> smoothed P(delayed) built on TRAIN
    states; when None they are fitted on df itself (for the train-side time
    model). Returns (matrix, rate_map, overall, geo_means, agency_map) so the
    caller can reuse the maps. Column order == TIME_FEATURES."""
    df2, gm = _impute_geo(df, geo_means)
    orig = df2["original_cost_cr"].clip(lower=0).to_numpy(dtype=float)
    exp = df2["expenditure_cum_cr"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    er = exp / np.maximum(orig, 1e-9)
    pdur = pd.to_numeric(df2["planned_duration_years"], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(pdur).any():
        pdur = np.nan_to_num(pdur, nan=np.nanmedian(pdur))
    else:
        pdur = np.full(len(pdur), 5.0)
    sec = df2["sector"].astype(str)
    if rate_map is None:
        y = (df2["delayed"] > 0).astype(int).to_numpy()
        g = pd.Series(y).groupby(sec.values).mean()
        cnt = pd.Series(y).groupby(sec.values).size()
        overall = float(y.mean())
        rate_map = ((cnt * g + 20.0 * overall) / (cnt + 20.0)).to_dict()
    rate = np.array([rate_map.get(x, overall) for x in sec.values], dtype=float)
    ag = df2["project_name"].apply(_agency_from_name).fillna("UNKNOWN").astype(str)
    if agency_map is None:
        y = (df2["delayed"] > 0).astype(int).to_numpy()
        g = pd.Series(y).groupby(ag.values).mean()
        cnt = pd.Series(y).groupby(ag.values).size()
        agency_map = ((cnt * g + 20.0 * overall) / (cnt + 20.0)).to_dict()
    agency_rate = np.array([agency_map.get(x, overall) for x in ag.values], dtype=float)
    lat = pd.to_numeric(df2.get("geo_latitude"), errors="coerce").fillna(21.1458).to_numpy(dtype=float)
    lon = pd.to_numeric(df2.get("geo_longitude"), errors="coerce").fillna(79.0882).to_numpy(dtype=float)
    es = pd.to_numeric(df2.get("elapsed_share"), errors="coerce")
    sp = pd.to_numeric(df2.get("spend_pace"), errors="coerce")
    es_med = float(es.median()) if np.isfinite(es.median()) else 0.5
    sp_med = float(sp.median()) if np.isfinite(sp.median()) else 1.0
    es = es.fillna(es_med).to_numpy(dtype=float)
    sp = sp.fillna(sp_med).to_numpy(dtype=float)
    return (np.column_stack([np.log1p(orig), er, pdur, rate, lat, lon, es, sp, agency_rate]),
            rate_map, overall, gm, agency_map)


def _oof_schedule_risk(cost_df, time_prep, states, seed: int = 7):
    """OOF P(delayed) per COST training row - the leak-safe schedule-risk feature.

    GroupKFold(5) grouped by STATE. Each held-out fold's risk comes from a time
    classifier fit on TIME rows of TRAINING states ONLY, so the feature never
    touches the row's own state - and never the realized delay / tor outcome,
    which is future relative to a live prediction. This is what the cost model
    consumes; the raw delayed/tor columns are never features."""
    risk = np.full(len(cost_df), np.nan)
    base = float((time_prep["delayed"] > 0).astype(int).mean()) if len(time_prep) else 0.5
    if len(time_prep) < 100 or len(cost_df) < 20:
        return np.full(len(cost_df), base)
    gkf = GroupKFold(n_splits=5)
    rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=seed)
    for tr, va in gkf.split(cost_df, groups=states):
        tr_states = set(states[tr])
        tt = time_prep[time_prep["state"].isin(tr_states)]
        if len(tt) < 100:
            risk[va] = base
            continue
        ytt = (tt["delayed"] > 0).astype(int).to_numpy()
        Xtt, rate, overall, gm, agency_map = _time_feature_matrix(tt)
        Xva, _, _, _, _ = _time_feature_matrix(cost_df.iloc[va], rate, overall, gm, agency_map)
        best_k, best_auc = None, -1.0
        for k, mh in _classifiers().items():
            aucs = []
            for i, j in rskf.split(Xtt, ytt):
                aucs.append(roc_auc_score(ytt[j], clone(mh).fit(Xtt[i], ytt[i]).predict_proba(Xtt[j])[:, 1]))
            a = float(np.mean(aucs))
            if a > best_auc:
                best_auc, best_k = a, k
        clf = clone(_classifiers()[best_k]).fit(Xtt, ytt)
        risk[va] = clf.predict_proba(Xva)[:, 1]
    return np.nan_to_num(risk, nan=base)


# --------------------------------------------------------------------------- #
# feature builders
# --------------------------------------------------------------------------- #
def _ensure_pace(df, snap_year: float | None = None):
    """Guarantee elapsed_share / spend_pace columns exist on a training frame.

    Attaches approval_year + planned_duration_years from the TIME table to the
    date-less COST annexure rows, then computes the pace features at each row's
    own snapshot year (row-wise when snap_year=None: snapshot rows 2025.42,
    annexure rows 2024.42). Already-populated columns are left untouched."""
    if "elapsed_share" in df.columns and df["elapsed_share"].notna().any():
        return df
    df = df.copy()
    if "approval_year" not in df.columns or df["approval_year"].notna().mean() < 0.2:
        df = join_approval(df, str(TIME_CSV))
    if "planned_duration_years" not in df.columns or df["planned_duration_years"].notna().mean() < 0.2:
        df = attach_duration(df, str(TIME_CSV))
    df = pace_features(df, snap_year)
    return df


def build_cost_frame(df, geo_means=None, with_pace: bool = True):
    """Leak-free cost features + (y_binary overrun_flag, y_reg overrun_ratio).

    Returns (frame, yb, yr, sector, geo_means) where geo_means maps sector ->
    (lat, lon) used to impute missing coordinates at serving time.
    with_pace=False drops elapsed_share/spend_pace (schedule-blind variant used
    for serving projects that have no approval/planned-duration dates)."""
    df = _ensure_pace(df)
    df = attach_price_index(df)
    df, geo_means = _impute_geo(df, geo_means)
    orig = df["original_cost_cr"].clip(lower=0).to_numpy(dtype=float)
    exp = df["expenditure_cum_cr"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    er = exp / np.maximum(orig, 1e-9)
    yb = (df["overrun_flag"] > 0).astype(int).to_numpy()
    yr = df["overrun_ratio"].to_numpy(dtype=float)
    sector = df["sector"].astype(str)
    # OOF target encodings (leak-free) - binary rate for clf, continuous for reg
    sector_rate_clf = _oof_rate_strat(yb, sector)
    sector_rate_reg = _oof_ratio_kf(yr, sector)
    geo_lat = pd.to_numeric(df.get("geo_latitude"), errors="coerce").fillna(0).to_numpy()
    geo_lon = pd.to_numeric(df.get("geo_longitude"), errors="coerce").fillna(0).to_numpy()
    frame = pd.DataFrame({
        "log_original_cost": np.log1p(orig),
        "expenditure_ratio": er,
        "log_expenditure": np.log1p(exp),
        "sector_rate": sector_rate_clf,
        "geo_latitude": geo_lat,
        "geo_longitude": geo_lon,
    })
    revised = pd.to_numeric(df.get("revised_cost_cr"), errors="coerce").to_numpy(dtype=float)
    rev = np.where(np.isnan(revised), orig, revised)
    gain = np.clip(rev - orig, 0.0, None) / np.maximum(orig, 1e-9)
    frame["revise_log"] = np.log1p(gain)
    if with_pace:
        es_med = float(pd.to_numeric(df["elapsed_share"], errors="coerce").median())
        sp_med = float(pd.to_numeric(df["spend_pace"], errors="coerce").median())
        frame["elapsed_share"] = pd.to_numeric(df["elapsed_share"], errors="coerce").fillna(es_med).to_numpy(dtype=float)
        frame["spend_pace"] = pd.to_numeric(df["spend_pace"], errors="coerce").fillna(sp_med).to_numpy(dtype=float)
        pe_med = float(pd.to_numeric(df["price_esc"], errors="coerce").median())
        pl_med = float(pd.to_numeric(df["price_level_approval"], errors="coerce").median())
        frame["price_esc"] = pd.to_numeric(df["price_esc"], errors="coerce").fillna(pe_med).to_numpy(dtype=float)
        frame["price_level_approval"] = pd.to_numeric(df["price_level_approval"], errors="coerce").fillna(pl_med).to_numpy(dtype=float)
    frame_reg = frame.copy()
    frame_reg["sector_rate"] = sector_rate_reg
    return frame, yb, yr, sector, geo_means


def build_time_frame(df, geo_means=None, with_pace: bool = True):
    """Leak-free time features + (y_binary delayed).

    Returns (frame, yb, sector, geo_means) where geo_means maps sector -> (lat, lon)
    used to impute missing coordinates at serving time.
    with_pace=False drops elapsed_share/spend_pace (schedule-blind variant)."""
    df = _ensure_pace(df)
    df, geo_means = _impute_geo(df, geo_means)
    orig = df["original_cost_cr"].clip(lower=0).to_numpy(dtype=float)
    exp = df["expenditure_cum_cr"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    er = exp / np.maximum(orig, 1e-9)
    yb = (df["delayed"] > 0).astype(int).to_numpy()
    pdur = pd.to_numeric(df["planned_duration_years"], errors="coerce").to_numpy(dtype=float)
    pdur = np.nan_to_num(pdur, nan=np.nanmedian(pdur) if np.isfinite(pdur).any() else 5.0)
    sector = df["sector"].astype(str)
    sector_rate = _oof_rate_strat(yb, sector)
    agency = df["project_name"].apply(_agency_from_name)
    agency_rate = _oof_rate_strat(yb, agency)
    geo_lat = pd.to_numeric(df.get("geo_latitude"), errors="coerce").fillna(0).to_numpy()
    geo_lon = pd.to_numeric(df.get("geo_longitude"), errors="coerce").fillna(0).to_numpy()
    frame = pd.DataFrame({
        "log_original_cost": np.log1p(orig),
        "expenditure_ratio": er,
        "planned_duration_yrs": pdur,
        "sector_delay_rate": sector_rate,
        "geo_latitude": geo_lat,
        "geo_longitude": geo_lon,
        "agency_delay_rate": agency_rate,
    })
    if with_pace:
        es_med = float(pd.to_numeric(df["elapsed_share"], errors="coerce").median())
        sp_med = float(pd.to_numeric(df["spend_pace"], errors="coerce").median())
        frame["elapsed_share"] = pd.to_numeric(df["elapsed_share"], errors="coerce").fillna(es_med).to_numpy(dtype=float)
        frame["spend_pace"] = pd.to_numeric(df["spend_pace"], errors="coerce").fillna(sp_med).to_numpy(dtype=float)
    return frame, yb, sector, geo_means


# --------------------------------------------------------------------------- #
# regression targets - keep shift so log1p is safe on signed overrun_ratio
# --------------------------------------------------------------------------- #
def _serve_pace_row(row, orig_cr, exp_cr, es_med=0.5, sp_med=1.0):
    """Serving-time elapsed_share / spend_pace for a single new-project row.

    Uses row[:approval_year] + row[:planned_duration_years] against the row's
    snap_year (default = current measurement vintage) when both resolve, else
    falls back to the training medians. Mirrors src/pace.pace_features."""
    ap = row.get("approval_year")
    dur = row.get("planned_duration_years")
    try:
        ap = None if pd.isna(ap) else float(ap)
    except (TypeError, ValueError):
        ap = None
    try:
        dur = None if pd.isna(dur) else float(dur)
    except (TypeError, ValueError):
        dur = None
    if ap is not None and dur is not None and dur > 0:
        try:
            sy = row.get("snap_year")
            sy = CURRENT_YEAR + 0.42 if pd.isna(sy) else float(sy)
        except (TypeError, ValueError):
            sy = CURRENT_YEAR + 0.42
        elapsed = sy - ap
        if elapsed >= 0:
            es = float(np.clip(elapsed / dur, 0.0, 6.0))
            er = exp_cr / max(orig_cr, 1e-9)
            sp = float(np.clip(er / max(es, 0.10), 0.0, 12.0))
            return es, sp
    return float(es_med), float(sp_med)


def _serve_price_row(row, esc_med=0.0, lev_med=0.0):
    """Serving-time price_esc / price_level_approval for one new-project row.

    Falls back to the training medians when the row has no sector/approval_year
    resolution (mirrors training median-imputation in build_cost_frame)."""
    esc, lev = escalations(row.get("sector"), row.get("approval_year"))
    if not np.isfinite(esc):
        esc = esc_med
    if not np.isfinite(lev):
        lev = lev_med
    return float(esc), float(lev)


def _row_revise_log(row, orig_cr):
    """Serving-time revise_log = log1p(sanction growth taken so far), 0 if none."""
    rev = row.get("revised_cost_cr")
    try:
        rev = float(rev) if rev is not None and not pd.isna(rev) else float(orig_cr or 0)
    except (TypeError, ValueError):
        rev = float(orig_cr or 0)
    gain = max(float(rev) - float(orig_cr or 0), 0.0) / max(float(orig_cr or 0), 1e-9)
    return float(np.log1p(gain))


def _impute_geo(df, means=None, fallback=(21.1458, 79.0882)):
    """Impute missing geo_latitude/longitude.

    Uses per-sector means when provided (built from geo'd rows), otherwise
    computes them on the fly; any unresolvable cell lands on the India centroid.
    Returns (df, sector->(lat, lon) map).
    """
    df = df.copy()
    lat = pd.to_numeric(df.get("geo_latitude"), errors="coerce")
    lon = pd.to_numeric(df.get("geo_longitude"), errors="coerce")
    sector = df["sector"].astype(str)
    ok = lat.notna() & lon.notna()

    if means is None:
        means = {}
        if ok.any():
            lm = lat[ok].groupby(sector[ok]).mean()
            lonm = lon[ok].groupby(sector[ok]).mean()
            means = {s: (lm[s], lonm[s]) for s in lm.index}
    overall_geo = (float(lat[ok].mean()), float(lon[ok].mean())) if ok.any() else fallback

    def _cell(i):
        if np.isfinite(lat.at[i]) and np.isfinite(lon.at[i]):
            return lat.at[i], lon.at[i]
        return means.get(sector.at[i], overall_geo)

    res = [_cell(i) for i in df.index]
    df["geo_latitude"] = [c[0] for c in res]
    df["geo_longitude"] = [c[1] for c in res]
    return df, means


def _reg_targets(yr):
    ymin = float(yr.min())
    shift = max(0.0, -ymin) + 1.0
    return np.log1p(yr + shift), shift, ymin


def _inv_reg(pred, shift):
    return np.expm1(pred) - shift


# --------------------------------------------------------------------------- #
# model configs (consistent, regularized, honest)
# --------------------------------------------------------------------------- #
def _classifiers():
    m = {
        "Logistic Regression": LogisticRegression(max_iter=3000, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=8,
                                                max_depth=8, random_state=42, n_jobs=-1),
    }
    if HAVE_LGBM:
        m["LightGBM"] = LGBMClassifier(n_estimators=400, num_leaves=8, learning_rate=0.05,
                                       random_state=42, verbose=-1, n_jobs=-1)
    return m


def _regressors():
    m = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=8,
                                               max_depth=8, random_state=42, n_jobs=-1),
    }
    if HAVE_LGBM:
        m["LightGBM"] = LGBMRegressor(n_estimators=400, num_leaves=8, learning_rate=0.05,
                                      random_state=42, verbose=-1, n_jobs=-1)
    return m


# --------------------------------------------------------------------------- #
# estimators (fit + honest CV metrics)
# --------------------------------------------------------------------------- #
@dataclass
class CostEstimator:
    """Binary overrun classifier + overrun_ratio regressor (fit on COST data)."""
    clf: object = None
    reg: object = None
    clf_auc: float = 0.0
    reg_mae: float = 0.0
    reg_r2: float = 0.0
    base_rate: float = 0.0
    n_rows: int = 0
    features: list = field(default_factory=lambda: list(COST_FEATURES))
    features_ns: list = field(default_factory=lambda:
                              [f for f in COST_FEATURES if f not in PACE_FEATURES and f != "schedule_risk"])
    shift: float = 1.0
    sector_rates: dict = field(default_factory=dict)
    geo_means: dict = field(default_factory=dict)
    pace_med_es: float = 0.5
    pace_med_sp: float = 1.0
    schedule_risk_med: float = 0.5
    price_esc_med: float = 0.0
    price_level_med: float = 0.0
    clf_ml_ns: object = None
    reg_ml_ns: object = None
    clf_auc_ns: float = 0.0
    reg_mae_ns: float = 0.0

    def _schedule_known(self, row: dict) -> bool:
        """True only when the row can be scored with schedule features."""
        try:
            ap = row.get("approval_year")
            dur = row.get("planned_duration_years")
            return (pd.notna(ap) and float(ap) >= 1960
                    and pd.notna(dur) and float(dur) > 0)
        except (TypeError, ValueError):
            return False

    def fit(self, df: pd.DataFrame, seed: int = 7):
        # OOF schedule-slippage risk: P(delayed) from time models trained on
        # OTHER states (see _oof_schedule_risk). Aligned to df row order.
        time_prep = _prepare_time_table()
        tkey = time_prep["sector"].astype(str) + "|" + time_prep["project_name"].astype(str)
        t_state_map = dict(zip(tkey, time_prep["state"]))
        states_c = _states_for(df, t_state_map)
        risk = _oof_schedule_risk(df, time_prep, states_c, seed)
        self.schedule_risk_med = float(np.nanmedian(risk)) if np.isfinite(risk).any() else 0.5

        frame, yb, yr, sector, geo_means = build_cost_frame(df)
        frame["schedule_risk"] = risk
        X = frame[self.features].to_numpy(dtype=float)
        self.geo_means = geo_means
        self.pace_med_es = float(frame["elapsed_share"].median())
        self.pace_med_sp = float(frame["spend_pace"].median())
        self.price_esc_med = float(frame["price_esc"].median())
        self.price_level_med = float(frame["price_level_approval"].median())
        self.base_rate = float(yb.mean())
        self.n_rows = int(len(yb))

        # Store smoothed sector rate map for serving new project rows
        m = 20.0
        sec_series = df["sector"].astype(str)
        gm = pd.Series(yb).groupby(sec_series.values).mean()
        cnt = pd.Series(yb).groupby(sec_series.values).size()
        self.sector_rates = ((cnt * gm + m * self.base_rate) / (cnt + m)).to_dict()

        # --- classification: repeated stratified CV (never uses reg target) ---
        clfs = _classifiers()
        auc = {k: [] for k in clfs}
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
        for tr, va in rskf.split(X, yb):
            for k, mh in clfs.items():
                cl = clone(mh).fit(X[tr], yb[tr])
                auc[k].append(roc_auc_score(yb[va], cl.predict_proba(X[va])[:, 1]))
        best_k = max(auc, key=lambda k: np.mean(auc[k]))
        self.clf = {"type": "classifier", "name": best_k,
                    "auc": float(np.mean(auc[best_k])), "features": self.features}
        self.clf_auc = float(np.mean(auc[best_k]))
        # fit final classifier on all data for serving
        self.clf_ml = clone(clfs[best_k]).fit(X, yb)

        # --- regression: KFold on overrun_ratio (leak-free) ---
        regs = _regressors()
        t, shift, ymin = _reg_targets(yr)
        self.shift = shift
        mae, r2 = {k: [] for k in regs}, {k: [] for k in regs}
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, va in kf.split(X):
            for k, mh in regs.items():
                r = clone(mh).fit(X[tr], t[tr])
                pp = _inv_reg(r.predict(X[va]), shift)
                mae[k].append(mean_absolute_error(yr[va], pp))
                r2[k].append(r2_score(t[va], r.predict(X[va])))
        best_r = min(mae, key=lambda k: np.mean(mae[k]))
        self.reg_mae = float(np.mean(mae[best_r]))
        self.reg_r2 = float(np.mean(r2[best_r]))
        self.reg = {"type": "regressor", "name": best_r,
                    "mae": self.reg_mae, "r2": self.reg_r2, "shift": shift,
                    "features": self.features}
        self.reg_ml = clone(regs[best_r]).fit(X, t)

        # --- schedule-blind variant (serving rows with no approval/duration) ---
        self._fit_schedule_blind(df, yb, yr, geo_means, seed)
        return self

    def _fit_schedule_blind(self, df, yb, yr, geo_means, seed: int):
        frame, _, _, _, _ = build_cost_frame(df, geo_means, with_pace=False)
        self.features_ns = [f for f in self.features if f not in PACE_FEATURES and f != "schedule_risk"]
        X = frame[self.features_ns].to_numpy(dtype=float)
        if len(X) < 100:
            self.clf_ml_ns = self.reg_ml_ns = None
            self.clf_auc_ns = self.reg_mae_ns = 0.0
            return
        clfs = _classifiers()
        auc = {k: [] for k in clfs}
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
        for tr, va in rskf.split(X, yb):
            for k, mh in clfs.items():
                cl = clone(mh).fit(X[tr], yb[tr])
                auc[k].append(roc_auc_score(yb[va], cl.predict_proba(X[va])[:, 1]))
        best_k = max(auc, key=lambda k: np.mean(auc[k]))
        self.clf_auc_ns = float(np.mean(auc[best_k]))
        self.clf_ml_ns = clone(clfs[best_k]).fit(X, yb)
        regs = _regressors()
        t, shift, _ = _reg_targets(yr)
        mae = {k: [] for k in regs}
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, va in kf.split(X):
            for k, mh in regs.items():
                r = clone(mh).fit(X[tr], t[tr])
                pp = _inv_reg(r.predict(X[va]), shift)
                mae[k].append(mean_absolute_error(yr[va], pp))
        best_r = min(mae, key=lambda k: np.mean(mae[k]))
        self.reg_mae_ns = float(np.mean(mae[best_r]))
        self.reg_ml_ns = clone(regs[best_r]).fit(X, t)

    def predict(self, row: dict, schedule_risk: float | None = None) -> dict:
        """Return {'overrun_prob': float, 'overrun_ratio': float, 'overrun_cr': float,
        'fiscal_exposure_cr': float, 'original_cost_cr': float, 'anticipated_cr': float,
        'mae_ratio': float, 'schedule_used': bool}.

        schedule_risk = expected P(delayed) from the time estimator (service wires
        it in). Rows without approval/planned-duration dates use the schedule-blind
        model, which does not contain the schedule features at all."""
        with_pace = self._schedule_known(row) and self.clf_ml_ns is not None
        f = self._row_frame(row, with_pace=with_pace, schedule_risk=schedule_risk)
        cols = self.features if with_pace else self.features_ns
        X = f[cols].to_numpy(dtype=float)
        if with_pace:
            p = self.clf_ml.predict_proba(X)[0, 1]
            r = _inv_reg(self.reg_ml.predict(X)[0], self.shift)
            mae_ratio = self.reg_mae
        else:
            p = self.clf_ml_ns.predict_proba(X)[0, 1]
            r = _inv_reg(self.reg_ml_ns.predict(X)[0], self.shift)
            mae_ratio = self.reg_mae_ns
        orig = float(row.get("original_cost_cr") or 0)
        est_overrun_cr = float(max(0.0, r * orig))
        fiscal_exposure_cr = float(p * est_overrun_cr)
        return {
            "overrun_prob": float(p),
            "overrun_ratio": float(r),
            "overrun_cr": est_overrun_cr,
            "fiscal_exposure_cr": fiscal_exposure_cr,
            "original_cost_cr": orig,
            "anticipated_cr": float(orig * (1 + r)),
            "mae_ratio": mae_ratio,
            "schedule_used": bool(with_pace),
        }

    def _row_frame(self, row: dict, with_pace: bool = True, schedule_risk: float | None = None) -> pd.DataFrame:
        orig = float(row.get("original_cost_cr") or 0)
        exp = float(row.get("expenditure_cum_cr") or 0)
        er = exp / max(orig, 1e-9)
        sec = str(row.get("sector") or "").strip()
        sec_rates = getattr(self, "sector_rates", {}) or {}
        sec_rate = sec_rates.get(sec, self.base_rate)
        geo_lat = row.get("geo_latitude")
        geo_lon = row.get("geo_longitude")
        try:
            geo_lat = None if pd.isna(geo_lat) else float(geo_lat)
            geo_lon = None if pd.isna(geo_lon) else float(geo_lon)
        except (TypeError, ValueError):
            geo_lat = geo_lon = None
        if geo_lat is None or geo_lon is None:
            geo_means = getattr(self, "geo_means", {}) or {}
            geo_lat, geo_lon = geo_means.get(sec, (21.1458, 79.0882))
        frame = pd.DataFrame([{
            "log_original_cost": np.log1p(max(orig, 0)),
            "expenditure_ratio": er,
            "log_expenditure": np.log1p(max(exp, 0)),
            "sector_rate": sec_rate,
            "geo_latitude": geo_lat,
            "geo_longitude": geo_lon,
            "revise_log": _row_revise_log(row, orig),
        }])
        if with_pace:
            es, sp = _serve_pace_row(row, orig, exp,
                                     getattr(self, "pace_med_es", 0.5),
                                     getattr(self, "pace_med_sp", 1.0))
            frame["elapsed_share"] = es
            frame["spend_pace"] = sp
            esc, lev = _serve_price_row(row,
                                        getattr(self, "price_esc_med", 0.0),
                                        getattr(self, "price_level_med", 0.0))
            frame["price_esc"] = esc
            frame["price_level_approval"] = lev
            if schedule_risk is None:
                schedule_risk = getattr(self, "schedule_risk_med", 0.5)
            frame["schedule_risk"] = float(schedule_risk)
        return frame


@dataclass
class TimeEstimator:
    """Delay classifier + tor_months regressor (fit on TIME data)."""
    clf: object = None
    reg: object = None
    clf_auc: float = 0.0
    reg_mae: float = 0.0
    reg_r2: float = 0.0
    base_rate: float = 0.0
    n_rows: int = 0
    features: list = field(default_factory=lambda: list(TIME_FEATURES))
    features_ns: list = field(default_factory=lambda:
                              [f for f in TIME_FEATURES if f not in PACE_FEATURES])
    sector_rates: dict = field(default_factory=dict)
    agency_rates: dict = field(default_factory=dict)
    geo_means: dict = field(default_factory=dict)
    pace_med_es: float = 0.5
    pace_med_sp: float = 1.0
    clf_ml_ns: object = None
    reg_ml_ns: object = None
    clf_auc_ns: float = 0.0
    reg_mae_ns: float = 0.0

    def _schedule_known(self, row: dict) -> bool:
        """True only when the row can be scored with schedule features."""
        try:
            ap = row.get("approval_year")
            dur = row.get("planned_duration_years")
            return (pd.notna(ap) and float(ap) >= 1960
                    and pd.notna(dur) and float(dur) > 0)
        except (TypeError, ValueError):
            return False

    def fit(self, df: pd.DataFrame, seed: int = 7):
        frame, yb, sector, geo_means = build_time_frame(df)
        X = frame[self.features].to_numpy(dtype=float)
        self.geo_means = geo_means
        self.pace_med_es = float(frame["elapsed_share"].median())
        self.pace_med_sp = float(frame["spend_pace"].median())
        self.base_rate = float(yb.mean())
        self.n_rows = int(len(yb))

        # Store smoothed sector rate map for serving new project rows
        m = 20.0
        sec_series = df["sector"].astype(str)
        gm = pd.Series(yb).groupby(sec_series.values).mean()
        cnt = pd.Series(yb).groupby(sec_series.values).size()
        self.sector_rates = ((cnt * gm + m * self.base_rate) / (cnt + m)).to_dict()
        ag_series = df["project_name"].apply(_agency_from_name)
        agm = pd.Series(yb).groupby(ag_series.astype(str).values).mean()
        acnt = pd.Series(yb).groupby(ag_series.astype(str).values).size()
        self.agency_rates = ((acnt * agm + m * self.base_rate) / (acnt + m)).to_dict()

        # --- classification: honest content-only, repeated stratified CV ---
        clfs = _classifiers()
        auc = {k: [] for k in clfs}
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
        for tr, va in rskf.split(X, yb):
            for k, mh in clfs.items():
                cl = clone(mh).fit(X[tr], yb[tr])
                auc[k].append(roc_auc_score(yb[va], cl.predict_proba(X[va])[:, 1]))
        best_k = max(auc, key=lambda k: np.mean(auc[k]))
        self.clf_auc = float(np.mean(auc[best_k]))
        self.clf = {"type": "classifier", "name": best_k,
                    "auc": self.clf_auc, "features": self.features}
        self.clf_ml = clone(clfs[best_k]).fit(X, yb)

        # --- regression: tor_months on delayed projects, leak-free ---
        d = df[df["delayed"] > 0].dropna(subset=["tor_months"]).copy()
        if len(d) < 20:
            self.reg_mae = self.reg_r2 = 0.0
            self.reg = self.reg_ml = None
        else:
            frame_d, _, _, _ = build_time_frame(d, self.geo_means)
            Xd = frame_d[self.features].to_numpy(dtype=float)
            yt = d["tor_months"].astype(float).clip(upper=CAP_TOR).to_numpy(dtype=float)
            tgt = np.log1p(yt)
            regs = _regressors()
            mae, r2 = {k: [] for k in regs}, {k: [] for k in regs}
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            for tr, va in kf.split(Xd):
                for k, mh in regs.items():
                    r = clone(mh).fit(Xd[tr], tgt[tr])
                    pp = np.expm1(r.predict(Xd[va]))
                    mae[k].append(mean_absolute_error(yt[va], pp))
                    r2[k].append(r2_score(tgt[va], r.predict(Xd[va])))
            best_r = min(mae, key=lambda k: np.mean(mae[k]))
            self.reg_mae = float(np.mean(mae[best_r]))
            self.reg_r2 = float(np.mean(r2[best_r]))
            self.reg = {"type": "regressor", "name": best_r,
                        "mae": self.reg_mae, "r2": self.reg_r2, "features": self.features}
            self.reg_ml = clone(regs[best_r]).fit(Xd, tgt)
        self._fit_schedule_blind(df, yb, geo_means, seed)
        return self

    def _fit_schedule_blind(self, df, yb, geo_means, seed: int):
        """Schedule-blind variant: clf + delay-regression on features without pace."""
        self.features_ns = [f for f in self.features if f not in PACE_FEATURES]
        frame, _, _, _ = build_time_frame(df, geo_means, with_pace=False)
        X = frame[self.features_ns].to_numpy(dtype=float)
        if len(X) < 100:
            self.clf_ml_ns = self.reg_ml_ns = None
            self.clf_auc_ns = self.reg_mae_ns = 0.0
            return
        clfs = _classifiers()
        auc = {k: [] for k in clfs}
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
        for tr, va in rskf.split(X, yb):
            for k, mh in clfs.items():
                cl = clone(mh).fit(X[tr], yb[tr])
                auc[k].append(roc_auc_score(yb[va], cl.predict_proba(X[va])[:, 1]))
        self.clf_auc_ns = float(np.mean(auc[max(auc, key=lambda k: np.mean(auc[k]))]))
        self.clf_ml_ns = clone(clfs[max(auc, key=lambda k: np.mean(auc[k]))]).fit(X, yb)
        d = df[df["delayed"] > 0].dropna(subset=["tor_months"]).copy()
        if len(d) < 20:
            self.reg_ml_ns = None
            self.reg_mae_ns = 0.0
            return
        frame_d, _, _, _ = build_time_frame(d, geo_means, with_pace=False)
        Xd = frame_d[self.features_ns].to_numpy(dtype=float)
        yt = d["tor_months"].astype(float).clip(upper=CAP_TOR).to_numpy(dtype=float)
        tgt = np.log1p(yt)
        regs = _regressors()
        mae = {k: [] for k in regs}
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, va in kf.split(Xd):
            for k, mh in regs.items():
                r = clone(mh).fit(Xd[tr], tgt[tr])
                pp = np.expm1(r.predict(Xd[va]))
                mae[k].append(mean_absolute_error(yt[va], pp))
        best_r = min(mae, key=lambda k: np.mean(mae[k]))
        self.reg_mae_ns = float(np.mean(mae[best_r]))
        self.reg_ml_ns = clone(regs[best_r]).fit(Xd, tgt)

    def predict(self, row: dict) -> dict:
        """Return {'delay_prob', 'delay_months' (regression), 'mae_months',
        'schedule_used'}."""
        with_pace = self._schedule_known(row) and self.clf_ml_ns is not None
        f = self._row_frame(row, with_pace=with_pace)
        cols = self.features if with_pace else self.features_ns
        X = f[cols].to_numpy(dtype=float)
        if with_pace:
            p = self.clf_ml.predict_proba(X)[0, 1]
            months = None
            if self.reg_ml is not None:
                months = float(np.expm1(self.reg_ml.predict(X)[0]))
            mae = self.reg_mae
        else:
            p = self.clf_ml_ns.predict_proba(X)[0, 1]
            months = None
            if self.reg_ml_ns is not None:
                months = float(np.expm1(self.reg_ml_ns.predict(X)[0]))
            mae = self.reg_mae_ns
        return {"delay_prob": float(p), "delay_months": months, "mae_months": mae,
                "schedule_used": bool(with_pace)}

    def _row_frame(self, row: dict, with_pace: bool = True) -> pd.DataFrame:
        orig = float(row.get("original_cost_cr") or 0)
        exp = float(row.get("expenditure_cum_cr") or 0)
        er = exp / max(orig, 1e-9)
        pdur = float(row.get("planned_duration_years") or 5)
        sec = str(row.get("sector") or "").strip()
        sec_rates = getattr(self, "sector_rates", {}) or {}
        sec_rate = sec_rates.get(sec, self.base_rate)
        geo_lat = row.get("geo_latitude")
        geo_lon = row.get("geo_longitude")
        try:
            geo_lat = None if pd.isna(geo_lat) else float(geo_lat)
            geo_lon = None if pd.isna(geo_lon) else float(geo_lon)
        except (TypeError, ValueError):
            geo_lat = geo_lon = None
        if geo_lat is None or geo_lon is None:
            geo_means = getattr(self, "geo_means", {}) or {}
            geo_lat, geo_lon = geo_means.get(sec, (21.1458, 79.0882))
        frame = pd.DataFrame([{
            "log_original_cost": np.log1p(max(orig, 0)),
            "expenditure_ratio": er,
            "planned_duration_yrs": pdur if pdur > 0 else 5.0,
            "sector_delay_rate": sec_rate,
            "geo_latitude": geo_lat,
            "geo_longitude": geo_lon,
            "agency_delay_rate": self._agency_rate(row),
        }])
        if with_pace:
            es, sp = _serve_pace_row(row, orig, exp,
                                     getattr(self, "pace_med_es", 0.5),
                                     getattr(self, "pace_med_sp", 1.0))
            frame["elapsed_share"] = es
            frame["spend_pace"] = sp
        return frame

    def _agency_rate(self, row: dict) -> float:
        """Smoothed P(delayed|agency) for a row; global rate when no suffix."""
        ag = _agency_from_name(row.get("project_name"))
        ag = "UNKNOWN" if (isinstance(ag, float) or ag is None) else ag
        rates = getattr(self, "agency_rates", {}) or {}
        return float(rates.get(ag, self.base_rate))


class ModelService:
    """Versioned store + helper to train/load/serve cost & time estimators."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cost = None
        self.time = None
        self.version = "unset"

    # ---- persistence ----
    def _save(self):
        if not HAVE_JOBLIB:
            return
        for name, obj in (("cost", self.cost), ("time", self.time)):
            if obj is not None:
                joblib.dump(obj, self.models_dir / f"{name}.joblib")
        (self.models_dir / "version.json").write_text(json.dumps({
            "version": self.version,
            "cost_auc": self.cost.clf_auc if self.cost else None,
            "time_auc": self.time.clf_auc if self.time else None,
            "cost_auc_metric": ("RepeatedStratifiedKFold 5x2 CV on the training table "
                                "(annexure + May-2025 snapshot). In-fold holdout, "
                                "random-project split that SHARES STATES across folds - "
                                "optimistic; quotes the grouped-by-state figure"),
            "time_auc_metric": ("RepeatedStratifiedKFold 5x2 CV on the annexure-only "
                                "time table (same shared-state caveat as cost)"),
            "external_holdout_reference": {"cost": "0.617-0.688 AUC (test = May-2025 Flash Report "
                                            "cohorts 2018-2021, labels resolved, genuinely external "
                                            "document; RF best). revise_log (sanction growth to date): "
                                            "external clf +0.007..+0.011 on ALL cutoffs, reg MAE flat "
                                            "(0.261/0.210/0.172) + grouped reg -0.012/-0.014 - KEEP. "
                                            "schedule_risk re-measured external: -0.069..+0.015 (reg MAE "
                                            "~flat) - NEUTRAL, not oversold. price_esc (sector-mapped "
                                            "WPI/fx, ref 2024): clf wash, reg MAE -0.017..-0.018 on ALL "
                                            "cutoffs + grouped R2 +0.008 - KEEP (magnitude-estimation), "
                                            "classifier not oversold",
                                            "time": "0.807-0.875 AUC (same external cohorts; pace delta "
                                                    "+0.056..+0.111; MAE 14.7/12.9/10.8 mo). "
                                                    "agency_delay_rate: grouped +0.039..+0.042 (real, "
                                                    "annexure rows carry the agency suffix) but external "
                                                    "~0 because May-2025 snapshot names lack the suffix - "
                                                    "keep, inert-when-absent, not oversold"},
        }))

    def _load(self):
        if not HAVE_JOBLIB:
            return self
        cf = self.models_dir / "cost.joblib"
        tf = self.models_dir / "time.joblib"
        if cf.exists():
            try:
                self.cost = joblib.load(cf)
            except Exception as e:  # noqa: BLE001 - surfaced so retrains are never silent
                warnings.warn(f"cost.joblib failed to load ({e!r}); will retrain", RuntimeWarning, stacklevel=2)
                self.cost = None
        if tf.exists():
            try:
                self.time = joblib.load(tf)
            except Exception as e:  # noqa: BLE001 - surfaced so retrains are never silent
                warnings.warn(f"time.joblib failed to load ({e!r}); will retrain", RuntimeWarning, stacklevel=2)
                self.time = None
        vf = self.models_dir / "version.json"
        if vf.exists():
            try:
                self.version = json.loads(vf.read_text()).get("version", "unset")
            except Exception:
                pass
        return self

    # ---- training (continual) ----
    def train(self, cost_df=None, time_df=None, version: str | None = None, force=False,
              augment_cost: bool = True, augment_time: bool = False):
        """Fit (or retrain on accumulated) estimators. Always leak-free OOF.

        augment_cost=True (default): COST train set = annexure + the RESOLVED May-2025
        Flash-Report rows. Measured improvement (data/augmented_train_holdout_2025.json):
        cost AUC +~0.03 and cost reg MAE 28-35% lower out-of-sample. augment_time stays
        False ON PURPOSE: the augment A/B showed new time rows HURT (the snapshot's delay
        label is defined differently from the annexure's — anticipated DOC vs reported
        TOR — so merging injects label noise). Time keeps the annexure-only train set.
        """
        if cost_df is None:
            cost_df = _augmented_cost_df() if augment_cost else pd.read_csv(COST_CSV)
        if time_df is None:
            time_df = (_augmented_time_df() if augment_time else _load_time())
        self.cost = CostEstimator().fit(cost_df)
        self.time = TimeEstimator().fit(time_df)
        self.version = version or _bump_version(self.version)
        self._save()
        return self

    def predict_project(self, row: dict) -> dict:
        """Full estimate for one project row.

        The cost model consumes the TIME model's expected P(delayed) as the
        `schedule_risk` feature (only when the row has approval/planned dates,
        i.e. the schedule-aware path) - an OOF-trained feature, never the
        realized delay outcome."""
        out = {"cost": None, "time": None}
        if self.time is not None:
            out["time"] = self.time.predict(row)
        if self.cost is not None:
            schedule_risk = None
            if out["time"] is not None and out["time"].get("schedule_used"):
                schedule_risk = out["time"].get("delay_prob")
            out["cost"] = self.cost.predict(row, schedule_risk=schedule_risk)
        return out


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load_time():
    df = pd.read_csv(TIME_CSV)
    df["original_cost_cr"] = df["original_cost_cr"].fillna(df["original_cost_cr"].median())
    df["expenditure_cum_cr"] = df["expenditure_cum_cr"].fillna(0)
    df["tor_months"] = pd.to_numeric(df["tor_months"], errors="coerce")
    df["planned_duration_years"] = pd.to_numeric(df["planned_duration_years"], errors="coerce")
    return df


_COST_KEEP = ["sector", "project_name", "original_cost_cr", "revised_cost_cr", "expenditure_cum_cr",
              "overrun_flag", "overrun_ratio", "geo_latitude", "geo_longitude"]
_SNAP_COST_KEEP = _COST_KEEP + ["approval_year", "planned_duration_years"]


def _augmented_cost_df():
    """COST train set = annexure + resolved May-2025 snapshot rows (evidence-backed).

    The 1,559 snapshot rows all carry overrun flags/ratios (resolved as of May-2025)
    and share the same label semantics (anticipated vs original cost) as the annexure,
    so they add genuine signal: external-holdout cost AUC +~0.03 and reg MAE 28-35%
    lower (data/augmented_train_holdout_2025.json)."""
    cost = pd.read_csv(COST_CSV)[_COST_KEEP]
    cost = join_approval(cost, str(TIME_CSV))
    cost = attach_duration(cost, str(TIME_CSV))
    cost = pace_features(cost, SNAP_YEAR_ANNEXURE)
    snap = pd.read_csv(SNAPSHOT_CSV)[_SNAP_COST_KEEP].copy()
    snap = snap.dropna(subset=["overrun_flag", "overrun_ratio"])
    snap["expenditure_cum_cr"] = snap["expenditure_cum_cr"].fillna(0)
    snap["original_cost_cr"] = snap["original_cost_cr"].fillna(snap["original_cost_cr"].median())
    snap = pace_features(snap, SNAP_YEAR_2025)
    out = pd.concat([cost, snap], ignore_index=True)
    out["expenditure_cum_cr"] = out["expenditure_cum_cr"].fillna(0)
    return out


_TIME_KEEP = ["sector", "project_name", "original_cost_cr", "expenditure_cum_cr",
              "delayed", "tor_months", "planned_duration_years", "geo_latitude", "geo_longitude"]


def _augmented_time_df():
    """TIME train set = annexure + May-2025 rows (experimental; NOT used by default).

    Keep this around because the snapshot's delay label semantics differ from the
    annexure's (anticipated-DOC vs reported-TOR), which measurably hurts time AUC and
    MAE (see data/augmented_train_holdout_2025.json). Only use after normalizing the
    delay definition to one rule."""
    time = _load_time()
    snap = pd.read_csv(SNAPSHOT_CSV)[_TIME_KEEP].copy()
    snap = snap.dropna(subset=["delayed"])
    snap["expenditure_cum_cr"] = snap["expenditure_cum_cr"].fillna(0)
    snap["original_cost_cr"] = snap["original_cost_cr"].fillna(snap["original_cost_cr"].median())
    snap["tor_months"] = pd.to_numeric(snap["tor_months"], errors="coerce").fillna(0)
    snap["planned_duration_years"] = pd.to_numeric(snap["planned_duration_years"], errors="coerce")
    return pd.concat([time, snap], ignore_index=True)


_VERSION_CTR = None


def _bump_version(cur: str) -> str:
    global _VERSION_CTR
    # prefer the PERSISTED version.json so each retrain has a stable, monotonically
    # increasing version across app restarts (v1 -> v2 -> ...).
    try:
        vf = MODELS_DIR / "version.json"
        if vf.exists():
            cur = json.loads(vf.read_text()).get("version", cur)
        if str(cur).startswith("v") and str(cur)[1:].isdigit():
            return f"v{int(str(cur)[1:]) + 1}"
    except Exception:
        pass
    if _VERSION_CTR is None:
        _VERSION_CTR = 0
    _VERSION_CTR += 1
    return f"v{_VERSION_CTR}"


_service = None


def get_service(force_train: bool = True):
    """Return a cached ModelService; train once if no persisted model exists."""
    global _service
    if _service is None:
        _service = ModelService()._load()
        if _service.cost is None or _service.time is None:
            _service.train()
    return _service


def train_all(force=False, augment_cost=True, augment_time=False):
    """Retrain both estimators from current data and persist. Returns report.

    Default: COST trains on annexure + resolved May-2025 rows (measured +0.03 AUC,
    -28-35% reg MAE); TIME stays annexure-only (new time rows hurt until the delay
    definition is normalized). Pass augment_time=True to override at your own risk."""
    svc = ModelService()
    svc.train(force=force, augment_cost=augment_cost, augment_time=augment_time)
    return {
        "version": svc.version,
        "cost": {"clf_auc": svc.cost.clf_auc, "reg_mae": svc.cost.reg_mae,
                 "reg_r2": svc.cost.reg_r2, "base_rate": svc.cost.base_rate,
                 "n": svc.cost.n_rows,
                 "clf_auc_metric": "RepeatedStratifiedKFold 5x2 CV (training table, shared-state)"},
        "time": {"clf_auc": svc.time.clf_auc, "reg_mae": svc.time.reg_mae,
                 "reg_r2": svc.time.reg_r2, "base_rate": svc.time.base_rate,
                 "n": svc.time.n_rows,
                 "clf_auc_metric": "RepeatedStratifiedKFold 5x2 CV (annexure time table, shared-state)"},
"external_holdout_reference": {"cost": "0.617-0.688 AUC (test = May-2025 Flash Report "
                                                "cohorts 2018-2021, labels resolved, genuinely external "
                                                "document; RF best). revise_log external clf +0.007..+0.011 "
                                                "on ALL cutoffs, reg MAE flat, grouped reg -0.012/-0.014 - "
                                                "KEEP. schedule_risk re-measured external: -0.069..+0.015 "
                                                "(reg MAE ~flat) - NEUTRAL, not oversold. price_esc "
                                                "(sector-mapped WPI/fx, ref 2024): clf wash, reg MAE "
                                                "-0.017..-0.018 on ALL cutoffs + grouped R2 +0.008 - KEEP "
                                                "(magnitude-estimation), classifier not oversold",
                                       "time": "0.807-0.875 AUC (same external cohorts; pace delta "
                                               "+0.056..+0.111; MAE 14.7/12.9/10.8 mo). agency_delay_rate: "
                                               "grouped +0.039..+0.042 (real) but external ~0 because the "
                                               "May-2025 snapshot lacks the agency suffix - keep, "
                                               "inert-when-absent, not oversold"},
        "training_set": {"cost": "annexure + May-2025 snapshot",
                         "time": "annexure only (label semantics differ)"},
    }


def predict_project(row: dict) -> dict:
    return get_service().predict_project(row)


CostEstimator.__module__ = "src.predict"
TimeEstimator.__module__ = "src.predict"


if __name__ == "__main__":
    # Make estimators pickle/load as "src.predict.CostEstimator" even when this
    # file runs via `python -m` (i.e. as __main__). Without the alias, pickles
    # reference "__main__.CostEstimator", which no other process can import,
    # and every server restart silently retrains (the old v11->v12 bump).
    sys.modules.setdefault("src.predict", sys.modules["__main__"])
    augment_cost = "--no-augment" not in sys.argv
    report = train_all(augment_cost=augment_cost)
    print(json.dumps(report, indent=2))
    print("Saved models ->", MODELS_DIR)
