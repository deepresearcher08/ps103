# PAIMANA Cost-Overrun Prediction & Risk Scoring — Implementation Spec

## Context

This implements the "AI for Infrastructure Monitoring" hackathon use-case (Project 103) using
the actual public PAIMANA portal export as the dataset — not synthetic or scraped data.

**Critical scope decision (already made, do not revisit):** the available export has no
date/timeline/status fields. Time-overrun prediction and any trajectory/sequence modeling
are explicitly OUT of scope. This is a cost-overrun-only system. State this limitation in
all generated documentation/README output rather than silently omitting it.

## Input Data

File: `SectorWiseProjects_Report.csv`
- Format: CSV with 2 junk header lines before the real header row (title line, then blank line)
- Real header (row 3): `Sr.No., Sector, Line Ministry, ProjectID, Project Name, Original Cost (in Cr), Latest Revised Cost (in Cr), Expenditure (Cumm.) (in Cr)`
- ~1,775 rows, 22 sectors, 17 ministries
- No duplicate ProjectIDs, no missing/zero values in this file (verified) — but code should not
  assume this holds for future re-exports; validate on load and fail loudly on unexpected nulls
  rather than silently coercing.
- Known distribution facts to design around:
  - `Roads & Highways` sector = ~56% of all rows → severe class imbalance by sector
  - Overrun ratio `(revised - original) / original` ranges roughly -0.9996 to +19.2, mean ~0.10,
    heavily right-skewed
  - ~17.8% of rows have revised cost below original cost (legitimate — descoping/re-costing,
    not a data error) — these must resolve to "no overrun" under a threshold-based binary
    target, not be dropped or treated as ambiguous.

## Target Definition

Compute both, do not pick only one:
- `overrun_ratio` (float) = `(latest_revised_cost - original_cost) / original_cost`
  - Winsorize or log1p-transform before use as a regression target (raw values are dominated
    by the ~19.2 outlier)
- `overrun_flag` (bool) = `overrun_ratio > 0.05` (5% threshold — expose as a config constant,
  not a magic number, so it can be tuned)

## Feature Engineering

From the available columns only:
- `sector` (categorical, one-hot or target-encoded)
- `line_ministry` (categorical, one-hot or target-encoded)
- `log_original_cost` = `log1p(original_cost)`
- `expenditure_ratio` = `expenditure_cumm / original_cost` (proxy for how far along the project is,
  since no dates exist)
- `sector_historical_overrun_rate` — mean `overrun_flag` for that sector, computed **out-of-fold**
  (i.e. via k-fold target encoding or leave-one-out during training) to avoid leakage
- `ministry_historical_overrun_rate` — same, out-of-fold, per ministry
- Optional/stretch: simple keyword flags extracted from `Project Name` (e.g. contains "Phase",
  "Linking", "Multipurpose", "Package") as weak complexity signals — treat as a nice-to-have,
  not a blocker

## Model Architecture — Stacked Ensemble

Train and report each of these individually before the stack, so the comparison itself is
part of the deliverable (the problem statement explicitly asks whether ML beats conventional
statistical methods):

1. **Logistic Regression** (classification) / **Linear Regression** (regression) — the
   "conventional statistical method" baseline. Must be reported as a named baseline, not
   silently folded into the ensemble.
2. **Random Forest** (both tasks)
3. **LightGBM or XGBoost** (both tasks) — primary workhorse model; run SHAP on this one for
   the driver-analysis deliverable.
4. **Stacked meta-learner**: train a shallow model (logistic regression for classification /
   linear regression for regression) on out-of-fold predictions of models 1–3.

Report metrics for all four (three base + stack) side by side. Do not just report the final
stacked number — the individual baseline-vs-ensemble comparison is a required output, not
an intermediate step to discard.

### Evaluation

- Split: stratified k-fold (k=5) on `overrun_flag`, stratified additionally by sector if
  feasible, to avoid a fold that's accidentally all-Roads&Highways.
- Classification metrics: ROC-AUC, F1, precision/recall (imbalance-aware — check base rate
  of `overrun_flag` first and report it).
- Regression metrics: MAE and RMSE on the (transformed) `overrun_ratio`, plus reported
  separately on the untransformed scale for interpretability.
- Report metrics broken out **by sector** (at minimum for the top 3–4 sectors by row count),
  not just in aggregate — this directly surfaces whether the model is really just a
  Roads & Highways model wearing a general-purpose costume.

## Deliverables to produce (map to hackathon outcome list)

| # | Outcome | Status |
|---|---|---|
| a | Cost Overrun Prediction Model | Full — the core model above |
| c | Project Risk Scoring Framework | Reframed as "Cost Risk Score" — output a 0–1 risk score per project from the stacked model's predicted probability |
| e | Benchmarking / Comparative Analytics | Sector-vs-sector and ministry-vs-ministry overrun-rate tables/charts |
| f | Cost Escalation Driver Analysis | SHAP summary + dependence plots from the LightGBM model |
| g | AI-powered Monitoring Dashboard | Simple dashboard (see below) surfacing risk scores, driver analysis, benchmarking |
| h | LLM-enabled Project Intelligence Assistant | Lightweight: an open-source LLM generates a natural-language explanation per project from its SHAP values + historical rates — NOT a RAG system (no free-text field exists to RAG over) |
| b | Time Overrun Prediction Model | Explicitly out of scope — document why (no date fields in public export) |
| d | Early Warning Alert System | Reframed as static "Cost Risk Alert" (threshold on risk score), not a temporal/trajectory-based early-warning system — no time-series data exists to support drift detection |

## Tech Stack (open-source only, per hackathon requirement)

- Python: pandas, scikit-learn, lightgbm (or xgboost), shap
- Dashboard: Streamlit (fastest path to a working demo) or a simple Plotly Dash app
- LLM explanation layer: any locally-runnable open-source model (e.g. via Ollama) or a
  templated-prompt call to whatever open-source model is available in the environment —
  do not hard-code a paid API as a requirement

## File Structure

```
ps103/
├── data/
│   └── SectorWiseProjects_Report.csv
├── src/
│   ├── data_loader.py       # load + validate CSV, fail loudly on schema drift
│   ├── features.py          # feature engineering incl. out-of-fold target encoding
│   ├── models.py            # baseline, RF, LightGBM, stacking ensemble
│   ├── evaluate.py          # cross-val, metrics incl. per-sector breakdown
│   ├── explain.py           # SHAP + LLM-generated natural-language explanations
│   └── dashboard.py         # Streamlit app
├── notebooks/
│   └── eda.ipynb            # sanity checks matching the stats already verified above
├── README.md                # must state the time-overrun scope limitation explicitly
└── requirements.txt
```

## Non-negotiable constraints

- Do not fabricate or synthesize date/timeline data to "complete" the time-overrun deliverable.
  If tempted to interpolate fake schedule data, stop — this was explicitly rejected in favor
  of an honest, scoped submission.
- Do not silently drop the 316 negative-overrun rows — they are real data.
- Every reported metric must also be reported per-sector or explicitly flagged as
  aggregate-only with a note on the Roads & Highways imbalance.
