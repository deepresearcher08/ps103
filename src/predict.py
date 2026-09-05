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
from sklearn.model_selection import RepeatedStratifiedKFold, KFold

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

COST_FEATURES = ["log_original_cost", "expenditure_ratio", "log_expenditure", "sector_rate"]
TIME_FEATURES = ["log_original_cost", "expenditure_ratio", "planned_duration_yrs", "sector_delay_rate"]


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
# feature builders
# --------------------------------------------------------------------------- #
def build_cost_frame(df):
    """Leak-free cost features + (y_binary overrun_flag, y_reg overrun_ratio)."""
    orig = df["original_cost_cr"].clip(lower=0).to_numpy(dtype=float)
    exp = df["expenditure_cum_cr"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    er = exp / np.maximum(orig, 1e-9)
    yb = (df["overrun_flag"] > 0).astype(int).to_numpy()
    yr = df["overrun_ratio"].to_numpy(dtype=float)
    sector = df["sector"].astype(str)
    # OOF target encodings (leak-free) - binary rate for clf, continuous for reg
    sector_rate_clf = _oof_rate_strat(yb, sector)
    sector_rate_reg = _oof_ratio_kf(yr, sector)
    frame = pd.DataFrame({
        "log_original_cost": np.log1p(orig),
        "expenditure_ratio": er,
        "log_expenditure": np.log1p(exp),
        "sector_rate": sector_rate_clf,
    })
    frame_reg = frame.copy()
    frame_reg["sector_rate"] = sector_rate_reg
    return frame, yb, yr, sector


def build_time_frame(df):
    """Leak-free time features + (y_binary delayed)."""
    orig = df["original_cost_cr"].clip(lower=0).to_numpy(dtype=float)
    exp = df["expenditure_cum_cr"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    er = exp / np.maximum(orig, 1e-9)
    yb = (df["delayed"] > 0).astype(int).to_numpy()
    pdur = pd.to_numeric(df["planned_duration_years"], errors="coerce").to_numpy(dtype=float)
    pdur = np.nan_to_num(pdur, nan=np.nanmedian(pdur) if np.isfinite(pdur).any() else 5.0)
    sector = df["sector"].astype(str)
    sector_rate = _oof_rate_strat(yb, sector)
    frame = pd.DataFrame({
        "log_original_cost": np.log1p(orig),
        "expenditure_ratio": er,
        "planned_duration_yrs": pdur,
        "sector_delay_rate": sector_rate,
    })
    return frame, yb, sector


# --------------------------------------------------------------------------- #
# regression targets - keep shift so log1p is safe on signed overrun_ratio
# --------------------------------------------------------------------------- #
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
    shift: float = 1.0
    sector_rates: dict = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, seed: int = 7):
        frame, yb, yr, sector = build_cost_frame(df)
        X = frame[self.features].to_numpy(dtype=float)
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
            for k, m in clfs.items():
                cl = clone(m).fit(X[tr], yb[tr])
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
            for k, m in regs.items():
                r = clone(m).fit(X[tr], t[tr])
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
        return self

    def predict(self, row: dict) -> dict:
        """Return {'overrun_prob': float, 'overrun_ratio': float, 'overrun_cr': float,
        'fiscal_exposure_cr': float, 'original_cost_cr': float, 'anticipated_cr': float, 'mae_ratio': float}."""
        f = self._row_frame(row)
        X = f[self.features].to_numpy(dtype=float)
        p = self.clf_ml.predict_proba(X)[0, 1]
        r = _inv_reg(self.reg_ml.predict(X)[0], self.shift)
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
            "mae_ratio": self.reg_mae,
        }

    def _row_frame(self, row: dict) -> pd.DataFrame:
        orig = float(row.get("original_cost_cr") or 0)
        exp = float(row.get("expenditure_cum_cr") or 0)
        er = exp / max(orig, 1e-9)
        sec = str(row.get("sector") or "").strip()
        sec_rates = getattr(self, "sector_rates", {}) or {}
        sec_rate = sec_rates.get(sec, self.base_rate)
        return pd.DataFrame([{
            "log_original_cost": np.log1p(max(orig, 0)),
            "expenditure_ratio": er,
            "log_expenditure": np.log1p(max(exp, 0)),
            "sector_rate": sec_rate,
        }])


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
    sector_rates: dict = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, seed: int = 7):
        frame, yb, sector = build_time_frame(df)
        X = frame[self.features].to_numpy(dtype=float)
        self.base_rate = float(yb.mean())
        self.n_rows = int(len(yb))

        # Store smoothed sector rate map for serving new project rows
        m = 20.0
        sec_series = df["sector"].astype(str)
        gm = pd.Series(yb).groupby(sec_series.values).mean()
        cnt = pd.Series(yb).groupby(sec_series.values).size()
        self.sector_rates = ((cnt * gm + m * self.base_rate) / (cnt + m)).to_dict()

        # --- classification: honest content-only, repeated stratified CV ---
        clfs = _classifiers()
        auc = {k: [] for k in clfs}
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
        for tr, va in rskf.split(X, yb):
            for k, m in clfs.items():
                cl = clone(m).fit(X[tr], yb[tr])
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
            frame_d, _, _ = build_time_frame(d)
            Xd = frame_d[self.features].to_numpy(dtype=float)
            yt = d["tor_months"].astype(float).clip(upper=CAP_TOR).to_numpy(dtype=float)
            tgt = np.log1p(yt)
            regs = _regressors()
            mae, r2 = {k: [] for k in regs}, {k: [] for k in regs}
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            for tr, va in kf.split(Xd):
                for k, m in regs.items():
                    r = clone(m).fit(Xd[tr], tgt[tr])
                    pp = np.expm1(r.predict(Xd[va]))
                    mae[k].append(mean_absolute_error(yt[va], pp))
                    r2[k].append(r2_score(tgt[va], r.predict(Xd[va])))
            best_r = min(mae, key=lambda k: np.mean(mae[k]))
            self.reg_mae = float(np.mean(mae[best_r]))
            self.reg_r2 = float(np.mean(r2[best_r]))
            self.reg = {"type": "regressor", "name": best_r,
                        "mae": self.reg_mae, "r2": self.reg_r2, "features": self.features}
            self.reg_ml = clone(regs[best_r]).fit(Xd, tgt)
        return self

    def predict(self, row: dict) -> dict:
        """Return {'delay_prob', 'delay_months' (regression), 'mae_months'}."""
        f = self._row_frame(row)
        X = f[self.features].to_numpy(dtype=float)
        p = self.clf_ml.predict_proba(X)[0, 1]
        months = None
        if self.reg_ml is not None:
            months = float(np.expm1(self.reg_ml.predict(X)[0]))
        return {"delay_prob": float(p), "delay_months": months, "mae_months": self.reg_mae}

    def _row_frame(self, row: dict) -> pd.DataFrame:
        orig = float(row.get("original_cost_cr") or 0)
        exp = float(row.get("expenditure_cum_cr") or 0)
        er = exp / max(orig, 1e-9)
        pdur = float(row.get("planned_duration_years") or 5)
        sec = str(row.get("sector") or "").strip()
        sec_rates = getattr(self, "sector_rates", {}) or {}
        sec_rate = sec_rates.get(sec, self.base_rate)
        return pd.DataFrame([{
            "log_original_cost": np.log1p(max(orig, 0)),
            "expenditure_ratio": er,
            "planned_duration_yrs": pdur if pdur > 0 else 5.0,
            "sector_delay_rate": sec_rate,
        }])


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
        (self.models_dir / "version.json").write_text(
            json.dumps({"version": self.version, "cost_auc": self.cost.clf_auc if self.cost else None,
                        "time_auc": self.time.clf_auc if self.time else None}))

    def _load(self):
        if not HAVE_JOBLIB:
            return self
        cf = self.models_dir / "cost.joblib"
        tf = self.models_dir / "time.joblib"
        if cf.exists():
            try:
                self.cost = joblib.load(cf)
            except Exception:
                self.cost = None
        if tf.exists():
            try:
                self.time = joblib.load(tf)
            except Exception:
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
        """Full estimate for one project row."""
        out = {"cost": None, "time": None}
        if self.cost is not None:
            out["cost"] = self.cost.predict(row)
        if self.time is not None:
            out["time"] = self.time.predict(row)
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


_COST_KEEP = ["sector", "project_name", "original_cost_cr", "expenditure_cum_cr",
              "overrun_flag", "overrun_ratio"]


def _augmented_cost_df():
    """COST train set = annexure + resolved May-2025 snapshot rows (evidence-backed).

    The 1,559 snapshot rows all carry overrun flags/ratios (resolved as of May-2025)
    and share the same label semantics (anticipated vs original cost) as the annexure,
    so they add genuine signal: external-holdout cost AUC +~0.03 and reg MAE 28-35%
    lower (data/augmented_train_holdout_2025.json)."""
    cost = pd.read_csv(COST_CSV)[_COST_KEEP]
    snap = pd.read_csv(SNAPSHOT_CSV)[_COST_KEEP].copy()
    snap = snap.dropna(subset=["overrun_flag", "overrun_ratio"])
    snap["expenditure_cum_cr"] = snap["expenditure_cum_cr"].fillna(0)
    snap["original_cost_cr"] = snap["original_cost_cr"].fillna(snap["original_cost_cr"].median())
    out = pd.concat([cost, snap], ignore_index=True)
    out["expenditure_cum_cr"] = out["expenditure_cum_cr"].fillna(0)
    return out


_TIME_KEEP = ["sector", "project_name", "original_cost_cr", "expenditure_cum_cr",
              "delayed", "tor_months", "planned_duration_years"]


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
                 "n": svc.cost.n_rows},
        "time": {"clf_auc": svc.time.clf_auc, "reg_mae": svc.time.reg_mae,
                 "reg_r2": svc.time.reg_r2, "base_rate": svc.time.base_rate,
                 "n": svc.time.n_rows},
        "training_set": {"cost": "annexure + May-2025 snapshot",
                         "time": "annexure only (label semantics differ)"},
    }


def predict_project(row: dict) -> dict:
    return get_service().predict_project(row)


if __name__ == "__main__":
    augment_cost = "--no-augment" not in sys.argv
    report = train_all(augment_cost=augment_cost)
    print(json.dumps(report, indent=2))
    print("Saved models ->", MODELS_DIR)
