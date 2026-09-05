"""Time-Overrun prediction model on REAL MoSPI data (PS SIH26103 outcome b).

Two tasks (SPEC/PS required):
  1. Classification  -> P(delayed) from leak-free project-content attributes.
  2. Regression      -> tor_months (magnitude of delay) on delayed projects.

IMPORTANT — two data-leakage traps are explicitly avoided here:

  * `cost_overrun_pct` in real_mospi_time.csv is TARGET-COLLINEAR
    (`cost_overrun_pct > 0  <=>  delayed == 1` is true for every row). It is a
    construction artifact of how the dataset was assembled (pre-split on the
    "time AND cost overrun" annexure), not a legitimate predictor, and using it
    inflates AUC by ~+0.08 (0.83 -> 0.91 for RF). It is therefore NOT used as a
    feature for P(delayed).

  * Approval/observation-window features (`elapsed_years`, `approval_year`,
    `planned_doc_year`) encode the survivorship effect (a project observed for
    longer is more likely to already be 'delayed'), producing a misleading
    ~0.95 AUC. This is NOT a real risk signal and is NOT used for the headline.
    It is only reported once, as a labelled "observation-window artifact"
    diagnostic to show evaluators why it must be discarded.

The honest, defensible CONTENT-ONLY signal (~0.82-0.83 AUC) uses only attributes
available at a point in time without knowing the outcome: log original cost,
expenditure ratio, planned duration, and out-of-fold sector delay rate.
This directly answers PS outcome (b) (does ML beat a conventional baseline?)
and outcome (c) (what do current CUF content fields buy us?).
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

    Content-only (the defensible signal): log original cost, expenditure ratio,
    planned duration, and out-of-fold sector delay rate. These are all available
    at a point in time without knowing the outcome.

    NOTE: `cost_overrun_pct` from the CSV is deliberately NOT used — it is
    target-collinear (`cost_overrun_pct > 0  <=>  delayed == 1`), i.e. a
    construction artifact, not a predictor. Using it inflates AUC by ~+0.08.

    `include_temporal=True` additionally computes the observation-window proxy
    `elapsed_years` (time since approval). We keep this ONLY as a labelled
    artefact diagnostic: age-conditional ~0.95 AUC comes from survivorship, not
    real risk, and must never be used as the headline number.
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
    feats = ["log_original_cost", "expenditure_ratio",
             "planned_duration_yrs", "sector_delay_rate"]
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
    # CONTENT-ONLY is the honest, defensible task. include_temporal computes the
    # observation-window (survivorship) variant ONLY as a labelled artefact.
    X_content, y, F, sect = build_features(df, include_temporal=False)
    X_age, _, _, _ = build_features(df, include_temporal=True)
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

    # HONEST result: content-only (no outcome-leak, no survivorship)
    auc_content, oof_content = _cv_delay_class(X_content, y, classifiers, rskf)
    print("\nClassification: delay probability (real data, LEAK-FREE content-only):")
    print("  (features: log cost, expenditure ratio, planned duration, OOF sector rate)")
    for k, v in auc_content.items():
        print(f"    {k:28s} ROC-AUC={v:.3f}")

    # ARTEFACT DIAGNOSTIC (for documentation only, NEVER a headline): adding the
    # observation-window 'elapsed_years' makes AUC jump to ~0.95 purely via
    # survivorship (projects approved long ago have already had time to become
    # 'delayed'). This is NOT real risk signal and users are told to discard it.
    auc_age, _ = _cv_delay_class(X_age, y, classifiers, rskf)
    print("\n  [ARTIFACT DIAGNOSTIC - do not use] observation-window 'elapsed_years' added:")
    for k, v in auc_age.items():
        print(f"    {k:28s} ROC-AUC={v:.3f}  (survivorship, NOT a real signal)")

    # Honest ML-vs-stats: is a linear model with the same features enough?
    stats_baseline = auc_content["Logistic Regression"]
    ml_best = max(auc_content.values())
    print(f"\n  ML-vs-stats (outcome b): conventional LR={stats_baseline:.3f}; "
          f"best ML={ml_best:.3f}; ML gain={ml_best - stats_baseline:+.3f}")

    # per-sector on content-only (the fair comparison)
    best_cls = max(auc_content, key=lambda k: auc_content[k])
    per_sec = {}
    for s in top_sectors(sect):
        mask = sect == s
        if np.unique(y[mask]).size == 2 and mask.sum() >= 10:
            a = roc_auc_score(y[mask], oof_content[best_cls][mask])
            per_sec[s] = round(a, 3)
            print(f"  {s:30s} n={mask.sum():4d} rate={y[mask].mean():.2f} ROC-AUC={a:.3f}")

    # ---- Regression: tor_months on delayed projects (leak-free content features) ----
    d = df[df["delayed"] == 1].dropna(subset=["tor_months"]).copy()
    Xr, _, _, _ = build_features(d, include_temporal=False)
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
        "delayed_class_auc_content_only": auc_content,
        "observation_window_auc_artifact_diagnostic": auc_age,
        "per_sector_delayed_auc": per_sec,
        "tor_reg_mae_months": {k: float(np.mean(v)) for k, v in mae.items()},
        "tor_reg_r2_log": {k: float(np.mean(v)) for k, v in r2.items()},
    }


if __name__ == "__main__":
    results = run()
    out = ROOT / "data" / "time_model_results.json"
    out.write_text(json.dumps(results, indent=2))
    print("\nSaved results ->", out)
