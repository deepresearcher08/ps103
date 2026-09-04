"""Run the SPEC.md cost-overrun model architecture on the REAL MoSPI dataset.

Implements: leak-free features (log cost, expenditure ratio, out-of-fold sector &
ministry overrun rate), LR baseline, Random Forest, LightGBM, stacked meta-learner,
per-sector ROC-AUC breakdown. Mirrors SPEC.md's mandatory comparison deliverable
(does ML beat the conventional statistical baseline?).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "real_mospi_projects.csv"

OVERRUN_THRESHOLD = 0.05  # SPEC.md config constant: ratio > 5% => flag=1


def load_real() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    # SPEC.md binary target: mimic derived overrun_flag from the annexure split, but also
    # re-derive from overrun_ratio for robustness when a unified ratio exists.
    if "overrun_flag" not in df.columns:
        df["overrun_flag"] = (df["overrun_ratio"] > OVERRUN_THRESHOLD).astype(int)
    return df


def out_of_fold_target_encode(y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """Leave-one-type-out-ish encoding via strat KFold, leak-free (SPEC.md)."""
    enc = np.zeros(len(y))
    g = pd.Series(groups)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr, va in skf.split(groups, y):
        means = pd.Series(y[tr]).groupby(g[tr].values).mean()
        enc[va] = g[va].map(means).fillna(y[tr].mean()).values
    return enc


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    y = df["overrun_flag"].to_numpy()
    y_reg = df["overrun_ratio"].to_numpy()

    F = pd.DataFrame()
    F["sector_raw"] = df["sector"].astype(str)
    F["log_original_cost"] = np.log1p(df["original_cost_cr"].fillna(1).clip(lower=0))
    F["expenditure_ratio"] = (df["expenditure_cum_cr"].fillna(0) / df["original_cost_cr"].clip(lower=1e-9))
    # leak-free out-of-fold sector overrun rate + ministry proxy (= sector here)
    F["sector_rate"] = out_of_fold_target_encode(y, F["sector_raw"].values)
    F["ministry_rate"] = F["sector_rate"]

    feats = ["log_original_cost", "expenditure_ratio", "sector_rate", "ministry_rate"]
    X = F[feats].to_numpy()
    sect = F["sector_raw"].to_numpy()
    return X, y, F[feats], sect, y_reg


def top_sectors(sect: np.ndarray, n: int = 4) -> pd.Index:
    s = pd.Series(sect)
    return s.value_counts().head(n).index


def run() -> dict:
    df = load_real()
    X, y, F, sect, y_reg = build_features(df)

    print(f"Real dataset: {df.shape[0]} rows, base rate overrun_flag={y.mean():.3f}, "
          f"sectors={df['sector'].nunique()}")
    print(f"Features: {list(F.columns)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Logistic Regression (baseline)": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }
    if HAVE_LGBM:
        models["LightGBM"] = LGBMClassifier(n_estimators=500, random_state=42, n_jobs=-1, verbose=-1)

    oof = {name: np.zeros(len(y)) for name in models}
    auc = {name: [] for name in models}
    for tr, va in skf.split(X, y):
        for name, m in models.items():
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[va])[:, 1]
            oof[name][va] = p
            auc[name].append(roc_auc_score(y[va], p))
    for name in models:
        print(f"  {name:38s} 5-fold ROC-AUC = {np.mean(auc[name]):.3f} ± {np.std(auc[name]):.3f}")

    # Stacked meta-learner on OOF preds
    stack_X = np.column_stack([oof[n] for n in models])
    meta = LogisticRegression(max_iter=2000)
    stack_auc = []
    for tr, va in skf.split(stack_X, y):
        meta.fit(stack_X[tr], y[tr])
        stack_auc.append(roc_auc_score(y[va], meta.predict_proba(stack_X[va])[:, 1]))
    print(f"  {'Stacked meta-learner':38s} 5-fold ROC-AUC = {np.mean(stack_auc):.3f} ± {np.std(stack_auc):.3f}")

    # Per-sector breakdown on the best single model (LightGBM if present else RF)
    best = "LightGBM" if HAVE_LGBM else "Random Forest"
    sect_index = top_sectors(sect)
    print(f"\nPer-sector ROC-AUC for '{best}' (top sectors):")
    per_sector = {}
    for s in sect_index:
        mask = sect == s
        if np.unique(y[mask]).size == 2 and mask.sum() >= 10:
            a = roc_auc_score(y[mask], oof[best][mask])
            per_sector[s] = round(a, 3)
            print(f"  {s:32s} n={mask.sum():4d}  base={y[mask].mean():.2f}  ROC-AUC={a:.3f}")

    return {
        "n_rows": int(len(y)),
        "base_rate": float(y.mean()),
        "model_auc": {k: float(np.mean(v)) for k, v in auc.items()},
        "stack_auc": float(np.mean(stack_auc)),
        "per_sector": per_sector,
    }


if __name__ == "__main__":
    results = run()
    out = ROOT / "data" / "real_model_results.json"
    out.write_text(json.dumps(results, indent=2))
    print("\nSaved results ->", out)