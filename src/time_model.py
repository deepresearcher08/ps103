"""Time-Overrun prediction model on REAL MoSPI data (PS SIH26103 outcome b).

Two tasks (SPEC/PS required):
  1. Classification  -> P(delayed) from project attributes + cost overrun.
  2. Regression      -> tor_months (magnitude of delay) on delayed projects.

Features (leak-free vs the TIME target):
  - log original cost, expenditure ratio, planned duration,
    sector (out-of-fold rate), and an explicit elapsed-years
    (observation-window / survivorship effect made visible).
ML-vs-stats baseline comparison is reported (outcome b of the PS), as is
per-sector breakdown, mirroring the cost module.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold, KFold

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "real_mospi_time.csv"

CAP_TOR = 240.0  # months; >20yr treated as PDF cell-shift artifact (note in docs)
CURRENT_YEAR = 2026


def load_time() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["original_cost_cr"] = df["original_cost_cr"].fillna(df["original_cost_cr"].median())
    df["expenditure_cum_cr"] = df["expenditure_cum_cr"].fillna(0)
    df["cost_overrun_pct"] = df["cost_overrun_pct"].fillna(0)
    df["tor_months"] = pd.to_numeric(df["tor_months"], errors="coerce")
    df.loc[df["delayed"] == 0, "tor_months"] = 0
    df.loc[df["delayed"] == 1, "tor_months"] = df.loc[df["delayed"] == 1, "tor_months"].clip(upper=CAP_TOR)
    return df


def oof_rate(y, groups, m=20.0, n_splits=5, seed=42):
    enc = np.zeros(len(y))
    g = pd.Series(groups)
    skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=seed)
    for tr, va in skf.split(g, y):
        gtr, gva = g.iloc[tr], g.iloc[va]
        gm = pd.Series(y[tr]).groupby(gtr.values).mean()
        cnt = pd.Series(y[tr]).groupby(gtr.values).size()
        shrunk = (cnt * gm + m * y[tr].mean()) / (cnt + m)
        enc[va] = gva.map(shrunk).fillna(y[tr].mean()).to_numpy()
    return enc


def build_features(df, include_temporal: bool = True):
    """Leak-free feature matrix for P(delayed).

    `include_temporal=True` uses the age-conditional task and names the
    survivorship effect explicitly via `elapsed_years` (time since approval)
    rather than burying it inside approval_year. `include_temporal=False`
    reports the CONTENT-ONLY signal (cost, expenditure, plan, sector) — the
    defensible signal that does not ride on the observation window.
    """
    y = df["delayed"].to_numpy()
    sector = df["sector"].astype(str)
    orig = df["original_cost_cr"].clip(lower=0)
    exp = df["expenditure_cum_cr"].clip(lower=0)
    F = pd.DataFrame()
    F["log_original_cost"] = np.log1p(orig)
    F["expenditure_ratio"] = exp / orig.clip(lower=1e-9)
    F["planned_duration_yrs"] = df["planned_duration_years"].fillna(df["planned_duration_years"].median())
    F["sector_delay_rate"] = oof_rate(y, sector)
    F["cost_overrun_pct"] = df["cost_overrun_pct"].fillna(0)
    feats = ["log_original_cost", "expenditure_ratio",
             "planned_duration_yrs", "sector_delay_rate", "cost_overrun_pct"]
    if include_temporal:
        appr = df["approval_year"]
        F["elapsed_years"] = (CURRENT_YEAR - appr).fillna(appr.median() if pd.notna(appr.median()) else 2010)
        feats.append("elapsed_years")
    X = F[feats].to_numpy(dtype=float)
    return X, y, F[feats], sector.to_numpy()


def top_sectors(sect, n=4):
    return pd.Series(sect).value_counts().head(n).index


def _cv_delay_class(X, y, classifiers, rskf):
    oof_cls = {k: np.zeros(len(y)) for k in classifiers}
    auc = {k: [] for k in classifiers}
    for tr, va in rskf.split(X, y):
        for k, m in classifiers.items():
            m.fit(X[tr], y[tr])
            oof_cls[k][va] = m.predict_proba(X[va])[:, 1]
            auc[k].append(roc_auc_score(y[va], oof_cls[k][va]))
    return {k: float(np.mean(auc[k])) for k in auc}, oof_cls


def run() -> dict:
    df = load_time()
    X_age, y, F, sect = build_features(df, include_temporal=True)
    X_content, _, _, _ = build_features(df, include_temporal=False)
    print(f"Time dataset: {df.shape[0]} rows, delayed rate={y.mean():.3f}")

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=7)
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=8,
                                                max_depth=8, random_state=42, n_jobs=-1),
    }
    if HAVE_LGBM:
        classifiers["LightGBM"] = LGBMClassifier(n_estimators=400, num_leaves=8, learning_rate=0.05,
                                                 subsample=0.8, colsample_bytree=0.8, random_state=42,
                                                 verbose=-1, n_jobs=-1)

    # Age-conditional model (elapsed-years named explicitly)
    auc, oof_age = _cv_delay_class(X_age, y, classifiers, rskf)
    # Content-only model (no temporal features) - the defensible signal
    auc_content, oof_content = _cv_delay_class(X_content, y, classifiers, rskf)

    print("\nClassification: delay probability (real data):")
    print("  AGE-CONDITIONAL (incl. elapsed-years):")
    for k, v in auc.items():
        print(f"    {k:28s} ROC-AUC={v:.3f}")
    print("  CONTENT-ONLY (no temporal) - defensible:")
    for k, v in auc_content.items():
        print(f"    {k:28s} ROC-AUC={v:.3f}")

    # per-sector on content-only (the fair comparison)
    best_cls = max(auc_content, key=lambda k: auc_content[k])
    per_sec = {}
    for s in top_sectors(sect):
        mask = sect == s
        if np.unique(y[mask]).size == 2 and mask.sum() >= 10:
            a = roc_auc_score(y[mask], oof_content[best_cls][mask])
            per_sec[s] = round(a, 3)
            print(f"  {s:30s} n={mask.sum():4d} rate={y[mask].mean():.2f} ROC-AUC={a:.3f}")

    # ---- Regression: tor_months on delayed projects ----
    d = df[df["delayed"] == 1].dropna(subset=["tor_months"]).copy()
    Xr, _, _, _ = build_features(d)
    yr = d["tor_months"].to_numpy(dtype=float)
    regressors = {"Linear Regression": LinearRegression()}
    regressors["Random Forest"] = RandomForestRegressor(n_estimators=400, min_samples_leaf=8,
                                                        max_depth=8, random_state=42, n_jobs=-1)
    if HAVE_LGBM:
        regressors["LightGBM"] = LGBMRegressor(n_estimators=400, num_leaves=8, learning_rate=0.05,
                                               subsample=0.8, colsample_bytree=0.8, random_state=42,
                                               verbose=-1, n_jobs=-1)
    target = np.log1p(yr)
    mae = {k: [] for k in regressors}
    r2 = {k: [] for k in regressors}
    kf = KFold(n_splits=5, shuffle=True, random_state=7)
    for tr, va in kf.split(Xr):
        for k, m in regressors.items():
            m.fit(Xr[tr], target[tr])
            pred = np.expm1(m.predict(Xr[va]))
            mae[k].append(float(mean_absolute_error(yr[va], pred)))
            r2[k].append(float(r2_score(target[va], m.predict(Xr[va]))))
    print("\nRegression: log(tor_months) on delayed projects, MAE (months, back-transformed):")
    for k in regressors:
        print(f"  {k:28s} MAE={np.mean(mae[k]):.1f} mo  R2(log)={np.mean(r2[k]):.3f}")

    return {
        "n_rows": int(len(y)),
        "delayed_rate": float(y.mean()),
        "delayed_class_auc": auc,
        "delayed_class_auc_content_only": auc_content,
        "per_sector_delayed_auc": per_sec,
        "tor_reg_mae_months": {k: float(np.mean(v)) for k, v in mae.items()},
        "tor_reg_r2_log": {k: float(np.mean(v)) for k, v in r2.items()},
    }


if __name__ == "__main__":
    results = run()
    out = ROOT / "data" / "time_model_results.json"
    out.write_text(json.dumps(results, indent=2))
    print("\nSaved results ->", out)
