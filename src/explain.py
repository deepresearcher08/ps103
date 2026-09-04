"""SHAP driver analysis on the real MoSPI cost-overrun dataset (SPEC.md deliverable f).

Trains a LightGBM on leak-free features and reports global SHAP importance + a per-feature
dependence profile, giving the "cost escalation driver analysis" outcome.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from real_model import build_features, load_real  # noqa: E402


def main():
    df = load_real()
    X, y, F, sect, _ = build_features(df)
    names = list(F.columns)

    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    model = LGBMClassifier(n_estimators=500, random_state=42, n_jobs=-1, verbose=-1)
    model.fit(Xtr, ytr)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xte)

    print("Mean |SHAP| driver ranking (global):")
    order = np.argsort(-np.abs(sv).mean(axis=0))
    for i in order:
        print(f"  {names[i]:22s} {np.abs(sv).mean(axis=0)[i]:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    shap.summary_plot(sv, Xte, feature_names=names, show=False)
    plt.title("SHAP summary — real MoSPI cost overrun")
    plt.tight_layout()
    out = ROOT / "data" / "shap_summary.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print("\nSaved SHAP summary ->", out)


if __name__ == "__main__":
    main()