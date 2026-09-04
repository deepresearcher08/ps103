"""Improved cost-overrun model on the REAL MoSPI dataset (v2).

Same dataset as v1 — ONLY model/feature/code changes. No CSV edits, no new columns
added to the data files.

Changes vs v1:
  - Leak-free feature additions from existing columns only (original_cost,
    expenditure, sector). Drops the target column `cost_overrun_pct` and avoids
    `anticipated_cost` (which reveals the annexure label).
  - m-estimate-smoothed out-of-fold sector rate (fixes low-count sector overfit).
  - Expenditure progress buckets (one-hot) + interactions (exp_ratio x sector_rate).
  - Feature scaling for the LR baseline.
  - Regularized trees (min_samples_leaf / max_depth / max_features) to stop the
    overfitting that made RF/LightGBM lose to LR in v1.
  - Probability calibration (isotonic) on all models.
  - Stacked meta-learner on OOF preds + raw features, plus a soft-voting average,
    evaluated with repeated stratified CV.
"""
from __future__ import annotations

import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "real_mospi_projects.csv"

OVERUN_THRESHOLD = 0.05


def load_real() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    # Dataset is not modified on disk. overrun_flag comes from the annexure split;
    # re-derive only if absent (fallback, non-mutating beyond in-memory copy).
    if "overrun_flag" not in df.columns:
        df["overrun_flag"] = (df["overrun_ratio"] > OVERUN_THRESHOLD).astype(int)
    # We do NOT use cost_overrun_pct / overrun_ratio / anticipated_cost as features
    # (target leakage). Only original_cost, expenditure, sector.
    return df


def _m_estimate_mean(y: np.ndarray, groups: pd.Series, m: float = 20.0) -> np.ndarray:
    global_mean = y.mean()
    gb_mean = pd.Series(y).groupby(groups).mean()
    count = pd.Series(y).groupby(groups).size()
    shrunk = (count * gb_mean + m * global_mean) / (count + m)
    return groups.map(shrunk).fillna(global_mean).to_numpy()


def oof_encode(y: np.ndarray, groups: pd.Series, n_splits: int = 5, m: float = 20.0) -> np.ndarray:
    enc = np.zeros(len(y))
    skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=42)
    for tr, va in skf.split(groups, y):
        ytr = y[tr]
        gtr = groups.iloc[tr]
        gva = groups.iloc[va]
        global_mean = ytr.mean()
        gb_mean = pd.Series(ytr).groupby(gtr.values).mean()
        count = pd.Series(ytr).groupby(gtr.values).size()
        relaxed = (count * gb_mean + m * global_mean) / (count + m)
        enc[va] = gva.map(relaxed).fillna(global_mean).to_numpy()
    return enc


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """Leak-free feature matrix. y = overrun_flag."""
    y = df["overrun_flag"].to_numpy()

    sector = df["sector"].astype(str)
    orig = df["original_cost_cr"].clip(lower=0)
    exp = df["expenditure_cum_cr"].fillna(0).clip(lower=0)
    exp_ratio = exp / orig.clip(lower=1e-9)

    F = pd.DataFrame()
    F["sector"] = sector
    F["log_original_cost"] = np.log1p(orig)
    F["exp_ratio"] = exp_ratio
    F["log_exp"] = np.log1p(exp)
    F["sector_rate"] = oof_encode(y, sector)
    F["exp_x_sector"] = exp_ratio * F["sector_rate"]
    F["exp_x_logcost"] = exp_ratio * F["log_original_cost"]

    # expenditure progress buckets (weak ordinal from existing expenditure data)
    exp_bucket = pd.cut(
        exp_ratio,
        bins=[-np.inf, 0.2, 0.5, 1.0, 2.0, np.inf],
        labels=["lt20pct", "20to50", "50to100", "100to200", "gt200"],
    )
    dummies = pd.get_dummies(exp_bucket, prefix="expb", dtype=float)
    F = pd.concat([F, dummies], axis=1)

    feats = [
        "log_original_cost", "exp_ratio", "log_exp", "sector_rate",
        "exp_x_sector", "exp_x_logcost",
        *dummies.columns.tolist(),
    ]
    X = F[feats].to_numpy(dtype=float)
    return X, y, F[feats], sector.to_numpy()


def top_sectors(sect: np.ndarray, n: int = 4):
    return pd.Series(sect).value_counts().head(n).index


def run() -> dict:
    df = load_real()
    X, y, F, sect = build_features(df)
    names = list(F.columns)
    print(f"Real dataset: {df.shape[0]} rows, base rate={y.mean():.3f}, "
          f"features={len(names)} (leak-free)")
    print("Features:", names)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=7)

    # base models (regularized for trees)
    base = {
        "Logistic Regression": LogisticRegression(max_iter=3000, C=1.0),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=8, max_depth=8, max_features="sqrt",
            random_state=42, n_jobs=-1),
        "LightGBM": LGBMClassifier(
            n_estimators=500, num_leaves=8, learning_rate=0.05, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1),
    } if HAVE_LGBM else {
        "Logistic Regression": LogisticRegression(max_iter=3000, C=1.0),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=8, max_depth=8, max_features="sqrt",
            random_state=42, n_jobs=-1),
    }

    names_order = list(base.keys())
    oof = {k: np.zeros((len(y),)) for k in names_order}
    auc = {k: [] for k in names_order}

    for tr, va in rskf.split(X, y):
        for k, m in base.items():
            if k == "Logistic Regression":
                sc = StandardScaler().fit(X[tr])
                m_clone = LogisticRegression(max_iter=3000, C=1.0)
                cal = CalibratedClassifierCV(m_clone, cv=3, method="isotonic")
                cal.fit(sc.transform(X[tr]), y[tr])
                p = cal.predict_proba(sc.transform(X[va]))[:, 1]
            else:
                m.fit(X[tr], y[tr])
                p = CalibratedClassifierCV(m, cv=3, method="isotonic").fit(X[tr], y[tr])\
                    .predict_proba(X[va])[:, 1]
            oof[k][va] = p
            auc[k].append(roc_auc_score(y[va], p))

    print("\nBase models (repeated 5x2 CV, calibrated):")
    for k in names_order:
        print(f"  {k:28s} ROC-AUC = {np.mean(auc[k]):.3f} ± {np.std(auc[k]):.3f}")

    # --- Stacked meta-learner on OOF preds + raw features ---
    stack_X = np.column_stack([oof[k] for k in names_order] + [X])
    meta_auc = []
    for tr, va in rskf.split(X, y):
        sc = StandardScaler().fit(stack_X[tr])
        meta = LogisticRegression(max_iter=3000, C=1.0)
        meta.fit(sc.transform(stack_X[tr]), y[tr])
        meta_auc.append(roc_auc_score(y[va], meta.predict_proba(sc.transform(stack_X[va]))[:, 1]))
    print(f"  {'Stacked (OOF + raw)':28s} ROC-AUC = {np.mean(meta_auc):.3f} ± {np.std(meta_auc):.3f}")

    # --- Soft-voting average of base models ---
    avg = np.mean([oof[k] for k in names_order], axis=0)
    vote_auc = []
    for tr, va in rskf.split(X, y):
        vote_auc.append(roc_auc_score(y[va], avg[va]))
    print(f"  {'Soft-voting average':28s} ROC-AUC = {np.mean(vote_auc):.3f} ± {np.std(vote_auc):.3f}")

    # --- Per-sector breakdown for the best base model ---
    best = names_order[-1]  # LightGBM if available else RF
    print(f"\nPer-sector ROC-AUC for '{best}':")
    per_sector = {}
    for s in top_sectors(sect):
        mask = sect == s
        if np.unique(y[mask]).size == 2 and mask.sum() >= 10:
            a = roc_auc_score(y[mask], oof[best][mask])
            per_sector[s] = round(a, 3)
            print(f"  {s:30s} n={mask.sum():4d} ROC-AUC={a:.3f}")

    return {
        "n_rows": int(len(y)),
        "base_rate": float(y.mean()),
        "model_auc": {k: float(np.mean(auc[k])) for k in names_order},
        "stack_auc": float(np.mean(meta_auc)),
        "vote_auc": float(np.mean(vote_auc)),
        "per_sector": per_sector,
    }


if __name__ == "__main__":
    results = run()
    out = ROOT / "data" / "real_model_results_v2.json"
    out.write_text(json.dumps(results, indent=2))
    print("\nSaved results ->", out)