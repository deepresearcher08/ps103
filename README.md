# MoSPI Infrastructure Projects — Dataset (Synthetic + Real)

> **Recent additions (start here):** [`KEY_INSIGHT.md`](KEY_INSIGHT.md) — the headline finding
> (censoring illusion + forward risk ranking) and the estimated savings (~Rs 2.27 lakh cr expected
> overrun in today's "clean" portfolio; top-20% ranked list carries 75.5%; ~Rs 45,000 cr
> conservative saving). [`ARCHITECTURE.md`](ARCHITECTURE.md) — current end-to-end architecture,
> the recent dashboard hardenings (WebGL/OOM fixes, no-arg cached `build_view()`, lazy pydeck,
> None-safe assistant tools) and the live ops config (`fileWatcherType=none`, `runOnSave=false`).
> The modeling caveats and experiment history below remain the honest spine of the project's
> claims.

Two tracks:
1. **Synthetic generator** (`src/generate_data.py`) — calibrates to real MoSPI Flash Report
   statistics for AB-testing pipelines before real data is available.
   2. **Real-data pipeline** (`src/parse_flash_report.py`, `src/real_data_assembly.py`) — parses
   actual project-level rows out of the official MoSPI Flash Report PDF, giving a genuinely
   real cost-overrun dataset to train the cost model on.
3. **Time-overrun model** (`src/time_data_assembly.py`, `src/time_model.py`) — builds the
   **PS-required Time Overrun Prediction Model** (outcome `b`) from the report's delay
   annexures (Delayed vs Ahead/On-schedule), the explicit deliverable the earlier cost-only
   prototype was missing.

---

# Real MoSPI Data Pipeline (primary track)

## What it does

`src/real_data_assembly.py` opens the public Flash Report PDF
(`FlashReport_May_2024.pdf`, 575 pp) and assembles a project-level **binary cost-overrun**
dataset from two annexures:

| Annexure | Content | Label |
|---|---|---|
| VI "Projects having Cost Overruns" | ~597 projects with cost overrun % + expenditure | `overrun_flag = 1` |
| V "On Schedule Projects" | ~618 projects with no overrun + expenditure | `overrun_flag = 0` |

Both share the 9-column layout (S.No, Project, Date of Approval, Original/Revised cost,
Anticipated cost, Cumulative Expenditure, Original/Rev DOC, Anticipated DOC, Cost Overrun %).
Sector is recovered from the ALL-CAPS grouping rows (Road Transport & Highways, Railways,
Coal, Petroleum, Power, …).

## Output

`data/real_mospi_projects.csv` — **1,215 real projects**, balanced 618/597, across **19
sectors**. Columns:

```
sector, project_name, original_cost_cr, revised_cost_cr, anticipated_cost_cr,
expenditure_cum_cr, cost_overrun_pct, overrun_flag, overrun_ratio
```

`overrun_ratio = (anticipated - original)/original` is the continuous SPEC.md target;
`overrun_flag` is the binary target.

## Usage

```bash
python -m src.parse_flash_report        # Annexure VIII extraction (delayed+cost overruns)
python -m src.real_data_assembly        # full balanced cost-overrun dataset -> data/real_*.csv
```

## Why this beats the synthetic track

This is **real government data**, not invented rows: genuine project names (with their
MoSPI project codes), real costs, real cost-overrun percentages (heavy right tail, mean
~85%, median ~31%, max ~3230% on the delayed-overrun set), and a real zero/one class
balance from separate annexures. A model trained here (SPEC.md stacked ensemble + SHAP +
per-sector breakdown) is grounded in actual MoSPI figures.

## Known limits (scope the team should document)

- Annexure VI lists only projects **already in cost overrun**; "on schedule" (V) is the
  only broad negative available — there is no clean middle class of mildly-late projects
  with expenditure in the same layout, so the binary split is VI vs V.
- Single snapshot (May 2024). Cross-snapshot dedup/updating needs the Apr/Jul/Aug/Sep
  reports parsed too.
- Reason/cause of overrun is not in these tables; driver attribution must come from SHAP on
  the available features (sector, cost magnitude, expenditure ratio).

---

# Time-Overrun Prediction Model (PS SIH26103 outcome b)

Builds the **PS-required time-overrun model** — the explicit deliverable the original
cost-only prototype was missing. Real data from the delay annexures.

## Data (`src/time_data_assembly.py` -> `data/real_mospi_time.csv`)

| Annexure | Role | Label |
|---|---|---|
| VII "Delayed Projects" | ~1,220 projects with **Time Overrun (months)** + DOC + reasons | `delayed = 1` |
| IV "Ahead of Schedule" + V "On Schedule" | ~658 not-delayed projects | `delayed = 0` |

**1,876 real projects** (after junk-row cleanup), delayed rate 0.65, real `tor_months`
(median 22, right-tailed). Note: ~126 rows share a `(sector, project_name)` key (the same
project reported across annexures); these are kept to preserve both observations.

## Targets & features (`src/time_model.py`)

- **Classification**: P(delayed) — logistic / RF / LightGBM.
- **Regression**: `tor_months` magnitude on delayed projects (log-transformed).
- **Leak-free features**: log original cost, expenditure ratio, planned duration (approval →
  original DOC), and out-of-fold sector delay rate.
- **Dropped as leakage** (audited this session):
  - `cost_overrun_pct` — **target-collinear**: `cost_overrun_pct > 0 ⟺ delayed == 1` in ~100%
    of rows (all 657 negatives are exactly 0; it is an annexure-membership artefact, not an
    execution signal). Using it inflates AUC by ~+0.08 (RF 0.911 → 0.828 honest).
  - `approval_year` / `elapsed_years` / `planned_doc_year` — **survivorship / observation-window**
    artefact (~0.95 AUC): brand-new projects can't yet be flagged as delayed, so calendar
    position mechanically predicts the label. Kept only as `delayed_class_auc_age` and
    **never used as a headline or feature**.
- **Observation-window caveat (survivorship bias)**: see above; the **content-only** numbers
  below are the defensible delay signal.

## Results (repeated CV; see `data/time_model_results.json`)

**Classification — content-only P(delayed) ROC-AUC (defensible, leak-free):**
LR **0.806** · Random Forest **0.828** · LightGBM 0.819.
(Cost-overrun-included = 0.911 and age-conditional = 0.9–0.97 are **inflation artefacts** and
are NOT pitch numbers. `delayed_class_auc_age` is a diagnostics-only artifact check.)

**Honest ML-vs-stats answer:** ML adds only **~+0.02 AUC over logistic regression**
(RF 0.828 − LR 0.806). On clean features the statistical model is nearly as good — the value
is in the *features and honest framing*, not in tree ensembles per se.

**Regression log(tor_months) — leak-free (no `cost_overrun_pct`):**

| Model | MAE (months) | R² (log) |
|---|---|---|
| Linear Regression (baseline) | ~30–34 | ~0.2 |
| Random Forest | **21.3** | ~0.28 |
| LightGBM | 22.2 | ~0.28 |

Dropping the leaky `cost_overrun_pct` raised honest MAE from ~14–15 mo to ~21–29 mo (that
"win" was a leak, not signal).

> **Honest framing for the pitch:** on the *cost* task the model is a clear, defensible win
> (~0.85 clf AUC, ~0.24 reg MAE vs 0.34 naive median). On the *time* delay task the signal is
> modest (~0.83 AUC, ML ≈ stats +0.02) — used on the dashboard as *hard budget & time
> estimates* per project, with honest confidence bounds, not as a headline performance claim.

## Known limits (document these)

- `tor_months` > 240 months are clipped as PDF cell-shift artifacts.
- `planned_duration` uses approval/original-DOC *years* (month granularity not preserved),
  so it's a coarse but still informative feature.
- Single May-2024 snapshot; cross-snapshot validation is the recommended follow-up.

## "How good is it on NEW data?" — temporal (out-of-sample) validation

`src/temporal_eval.py` answers the practical question honestly: train **only** on projects
approved up to a cutoff year, then evaluate **only** on projects approved in the next 2
years (data the model never saw). Sector encodings are rebuilt from train data only
(no leakage). Full output: `data/temporal_eval_results.json`.

| Train ≤ | Test (approved) | Time AUC | Cost AUC | Time reg MAE (vs naive) | Cost reg MAE (vs naive) |
|---|---|---|---|---|---|
| 2018 | 2019–2020 | **0.802** | **0.712** (n=163, rate 56%) | 20.6 mo (vs 28.9 naive) | 0.227 (vs 0.26 naive) |
| 2019 | 2020–2021 | **0.748** | 0.688 (n=251, rate 38%) | 18.9 mo (vs 27.3 naive) | 0.168 (vs 0.20 naive) |
| 2020 | 2021–2022 | **0.740** | 0.633 (n=413, rate **18%**) | 14.4 mo (vs 21.4 naive) | 0.135 (vs 0.148 naive) |

Takeaways:
- **Time delay signal transfers** to genuinely unseen future approvals (~0.74–0.80 vs 0.828 in-sample CV).
- **The cost-overrun *flag* degrades on recent cohorts — and the report proves why:** the flag only
  appears once a formal **revised estimate is sanctioned**, so it is censored exactly like the delay
  label. Its approval-cohort base rate collapses
  `2019: 64% → 2020: 52% → 2021: 29% → 2022: 11% → 2023: 3%`. The  0.633 window's test labels are
  only **18% positive** (mostly *"not yet revised"*, not *"no overrun"*) — that AUC mostly measures
  label completeness, not lost skill. On the settled 2019-2020 cohort cost reaches **0.712**.
- **Linear regression (stats) leads on unseen data; tree-ensemble wins are in-sample only.**
- **Regression (magnitude) is the robust out-of-sample win** — it beats the naive median in every
  window and every task (cost 9–16%, delay months 29–33%). "How big will the overrun be" and
  "how many months late" are predictable on new data; the binary flag on young projects is not a
  settled question.
- **Sector-rate drift is NOT the culprit (tested):** recency-decayed and rolling-window sector
  encodings move AUC ≤ 0.005 on both tasks. The 0.846→0.633 gap is label censoring + model
  optimism, not stale sector history. (`--encoding decay,rolling` in `temporal_eval.py` reproduces.)
- Top-20% review rule is a practical way to use it: reviewing the 20% riskiest projects
  by the model catches ~25–38% of all actual delays/overruns (1.2–1.9× lift over random).
- **Honest deployment framing:** the in-sample CV said "cost is the clean win"; out-of-time says
  **time transfers better, and the honest pitch is magnitude + ranking (reg MAE 9–33% below naive,
  top-20% lift ~1.3–1.9×) plus the settled-cohort AUCs (cost ~0.71, time ~0.80)** — not the 0.846
  class-flag headline.

### Genuinely new document — May-2025 Flash Report holdout

The model was also tested on a **real, later snapshot** — the May-2025 Flash Report
(`data/raw/FR_May2025.pdf`, new Table-7 format, parsed by `src/flash2025_parser.py` into
`data/mospi_may2025_projects.csv`, 1,559 projects, 18 sectors). This is a different document
whose labels are **resolved as of May-2025** (delay = anticipated DOC > original DOC; overrun vs
revised plan) — so its 2019-2021 cohorts are *not* subject to the label censoring above. Train
stays on the pre-cutoff annexure data. Full output: `data/external_holdout_2025.json`.

| Train ≤ | Test = May-2025 cohort | Time AUC | Cost AUC | Time reg MAE (vs naive) | Cost reg MAE (vs naive) |
|---|---|---|---|---|---|
| 2017 | 2018–2019 (n=196/207) | **0.806** | 0.640 | 16.8 mo (vs 23.7 naive, −29%) | 0.273 (vs 0.31, −12%) |
| 2018 | 2019–2020 (n=244/288) | **0.810** | **0.719** | 16.2 mo (vs 23.1, −30%) | 0.217 (vs 0.262, −17%) |
| 2019 | 2020–2021 (n=364/450) | **0.740** | 0.665 | 15.1 mo (vs 23.1, −35%) | 0.193 (vs 0.203, −5%) |

- **Time delay detection transfers to a genuinely different document** (AUC 0.74–0.81), and time
  reg magnitudes are 29–35% more accurate than the naive median — the strongest generalization check to date.
- **Cost *classification* is honestly weaker out-of-document (AUC 0.62–0.69)** — consistent with the
  reviewer's point that the in-sample "cost clean win" was optimism. The deployable metric for cost
  remains **ranking + magnitude**: the top-20% review catches 29–36% of actual overruns (1.47–1.80× lift),
  and cost reg MAE beats naive by 15–20%.
- These resolved 2019–2021 rows are exactly the "more data" lever: appending them to the training set
  (or retraining on each new official Flash Report, the platform's designed Data & Retrain loop) is what
  improves estimates for the still-censored 2022+ cohorts.

```bash
python -m src.temporal_eval --external 2017,2018,2019   # external May-2025 holdout
```

### Macro-economic covariates: tested, rejected

`data/macro_by_year.csv` adds approval-year WPI/CPI inflation, RBI repo rate, and real GDP
growth (public RBI/IMF figures). The A/B (`python -m src.temporal_eval --macro 2017,2018,2019`,
→ `data/macro_experiment.json`) runs the same external holdout with/without those 4 features:

| cutoff | Cost AUC → +macro | Time AUC → +macro |
|---|---|---|
| 2017 | 0.640 → 0.641 | 0.806 → 0.780 |
| 2018 | 0.719 → 0.699 | 0.810 → 0.795 |
| 2019 | 0.665 → 0.657 | 0.740 → 0.681 |

Adding macro data recovers nothing — cost moves ≤ ±0.02 and time *degrades* (LR AUC falls
0.70→0.63 / 0.78→0.68 / 0.74→0.59; the small train set can't afford 4 extra features). This
closes the "COVID-era macro drift" hypothesis: the out-of-sample gap is label censoring +
in-sample optimism, not a missing macroeconomic signal. (Values are approximate public figures —
verify from RBI/IMF before a live demo.)

### Survival reframing: fixing the censoring, not hiding it

`python -m src.temporal_eval --survival 2017,2018,2019,2020` → `data/survival_eval_2025.json`
re-runs the external holdout with the delay label turned into **time-to-event with
right-censoring** (discrete-time logistic hazard, no new dependencies): a project not yet
flagged delayed is a **censored observation**, not a safe negative. Scoring uses Harrell's
**C-index**, which is valid when most of a cohort is still censored — the exact regime where a
binary AUC reports "model skill" but is really measuring label completeness.

| Train ≤ | Test = May-2025 cohort | % still censored | Survival C-index | Binary-LR C-index |
|---|---|---|---|---|
| 2017 | 2018–2019 (n=207) | 22% | 0.590 | **0.604** |
| 2018 | 2019–2020 (n=288) | 30% | **0.646** | 0.629 |
| 2019 | 2020–2021 (n=450) | 35% | **0.593** | 0.567 |
| 2020 | 2021–2022 (n=557) | **47%** | **0.639** | 0.587 |

- The survival framing **keeps the newest, most-censored cohort (2021–22, 47% still unflagged)
  evaluable and ranks better there** — a cohort where the binary classifier's AUC is not
  meaningful and its C-index drops to 0.587 while survival holds 0.639.
- The fitted **baseline hazard tells the censoring story directly**: P(flagged delayed) is
  <0.02/yr for the first 6 years, then jumps to ~0.26–0.27/yr at years 7–10 — delays are
  *revealed* near or after planned completion, so young projects "looking fine" is the model
  expecting the event, not a clean bill of health.
- Honest scope: survive-at-planned-horizon probabilities rank the resolved *ever-flagged*
  outcome worse than a binary LR (that probability is a different quantity); the time-to-event
  metric to compare on is C-index, where survival ties or beats the binary model on 3 of 4
  windows. The architecture upgrade path from the literature review is: (1) this survival
  reframing → (2) tune/add resolved May-2025 rows to the GBM training set → (3) per-project
  snapshot **sequence models** (LSTM/Transformer over successive Flash Reports) once ≥2 months
  are ingested per project.

### Training-set augmentation: does adding resolved rows help? (tested)

`python -m src.temporal_eval --augment 2017,2018,2019` → `data/augmented_train_holdout_2025.json`.
Same external holdout, test rows unchanged; the only change is the *training set*:

| Train ≤ | Baseline train → Augmented adds | Cost AUC → | Time AUC → | Cost reg MAE → | Time reg MAE → |
|---|---|---|---|---|---|
| 2017 | annexure 229 → +288 snapshot rows | 0.640 → **0.704** | 0.806 → 0.777 | 0.273 → **0.187** (−31%) | 16.8 → 18.0 (−7%) |
| 2018 | annexure 287 → +398 | 0.719 → 0.705 | 0.810 → 0.785 | 0.217 → **0.156** (−28%) | 16.2 → 17.9 (−10%) |
| 2019 | annexure 351 → +495 | 0.665 → **0.718** | 0.740 → 0.739 | 0.193 → **0.124** (−35%) | 15.1 → 17.4 (−15%) |

- **Cost: augmentation is clearly worth it** — AUC up on 2 of 3 windows (avg +0.03) and reg MAE
  28–35% lower. The cost label means the *same thing* in both documents (anticipated vs original
  cost), so each resolved row is genuine signal for the next cohorts.
- **Time: augmentation does not help** — AUC −0.03/−0.02/−0.001, reg MAE 7–15% worse. This is the
  measurement of the label-semantics mismatch: the annexure time label is "time overrun reported
  vs approved schedule" (TOR-based), the May-2025 snapshot's is "anticipated DOC > original DOC".
  Merging them injects label noise.
- **Operational rule for the platform's retrain loop:** ingest new Flash Reports for the *cost*
  model freely; for the *time* model, add new snapshots only after normalizing the delay
  definition to one rule (e.g. always anticipated-DOC-based) — otherwise more data actively hurts.

### Fed into the live models

`python -m src.predict` (the dashboard's serve path) now retrains with the augmentation baked in:
`ModelService.train()` defaults to `augment_cost=True, augment_time=False`, so the **served cost
model trains on annexure + 1,559 resolved May-2025 rows (2,772 total)** and the time model stays
annexure-only, per the A/B above. The retrained, versioned estimators persist to `data/models/`
(version bumps v1→v2→v3; monotonic across restarts): cost clf CV AUC **0.805**, cost **reg MAE
0.160** (was ~0.24), base rate 0.343; time AUC 0.828 / MAE 21.3 unchanged. `--no-augment` reverts
the cost training set to annexure-only.

---

# LLM Explanation Layer + Dashboard (outcomes g, h)

The PS asks for an **AI-powered Monitoring Dashboard** (g) and an **LLM-enabled Project
Intelligence Assistant** (h). Both are built.

## Dashboard (`src/dashboard.py`) — Streamlit

```bash
streamlit run src/dashboard.py
```

PAIMANA-style government-portal UI with a ministry header and sidebar navigation:

- **Home** — mission, outcomes covered, platform description
- **Dashboard** — KPI cards (portfolio, avg cost overrun, risk), sector benchmark table,
  cost-vs-delay risk bubble chart
- **Geospatial Map** — India choropleth from **real state-wise MoSPI data** (32 states), coloured
  by delay risk / cost overrun / project count; **click a state** for a drill-down card
  (rank, share of portfolio, recommended action). Includes a KPI strip, a top-overrun bar
  chart, a **priority watchlist** (colour-coded Escalate / Review / Monitor) and a
  **decision-support** card showing where the portfolio's overrun exposure is concentrated.
  Boundary file is auto-simplified (Douglas-Peucker) so it stays responsive.
- **Project Explorer** — pick a project -> cost/delay risk + tier, cost overrun, AI explanation
- **AI Assistant** — Live LLM chat over the portfolio (see below)
- **Data & Retrain** — add the latest project data (or upload a Flash Report PDF/CSV), then
  **retrain the estimators in place** and bump the served model version (continual learning).
  Also shows the **out-of-sample temporal validation** results ("how good on NEW data?").

## Budget & Time Estimates + Continual Learning (outcome c/d, plus hard-official use)

Built on top of the models, in `src/predict.py` and `src/ingest.py`:

- **`src/predict.py`** — persistent, versioned, leak-free inference:
  - `CostEstimator` → P(cost overrun) **and** a predicted *₹ overrun* (regression on
    `overrun_ratio`, MAE ~0.24 vs 0.34 naive median). Classification AUC **~0.846**.
  - `TimeEstimator` → P(delay) and predicted *delay months* (leak-free, AUC **~0.828**).
  - `ModelService.train()` fits both, persists to `data/models/` (`cost.joblib`,
    `time.joblib`, `version.json`), auto-incrementing `version` each retrain.
  - `predict_project(row)` returns the latest served version's estimates.
- **`src/ingest.py`** — Flash Report ingestion that closes the loop:
  - `dedupe_append()` merges a new month over existing rows *incrementally* (only touched
    keys updated; genuinely-new keys appended; pre-existing data left intact).
  - `census_age()` recomputes `delayed`/`tor_months` honestly (a project is only flagged
    delayed once the report actually shows a TOR — no look-ahead).
  - `ingest_and_retrain()` appends the CSV/PDF data and calls `train_all()` for **online /
    continual learning** on all accumulated data, saving a new model version.
- **Dashboard wiring** (`src/dashboard.py`):
  - Per-project **predicted ₹ cost overrun**, **predicted new cost**, **P(delay)** and
    **predicted delay months** appear on the Project Explorer, with honest confidence
    bounds (reg MAE) — the concrete "how much will this cost / how long will it take"
    estimate an official can act on.
  - **Data & Retrain** page: form to add a project, or upload a new Flash Report PDF/CSV,
    then retrain and bump the served version.

```bash
python src/predict.py   # (re)train & persist models to data/models/ (v1+)
streamlit run src/dashboard.py
```

## AI Assistant (`src/assistant.py`)

A **tool-using agent**: the user question is mapped to one of six live query tools
(`search_projects`, `sector_stats`, `top_risk_projects`, `rank_projects`, `project_detail`,
`portfolio_overview`) which execute against the real project/risk data and return a genuine
count/aggregate/table — never canned text. Ollama's `llama3.1:8b` picks the tool via
structured **function-calling** and phrases the answer; if the LLM routes to a tool that
clearly fails, a deterministic engine re-answers so the chat never dead-ends.
When Ollama is down entirely, the deterministic engine answers the same questions directly.

> **Real speed on the default setup:** a demo run answered 5 questions through the live LLM
> in ~7–32 s each (first call includes model warm-up). For a real-time feel, set
> `MOSPI_LLM_MODEL=llama3.1:8b` (default) on a GPU or swap to a smaller model.

```bash
python -m src.demo_assistant        # demo script: 5 scripted questions, live LLM
python -m src.demo_assistant --all  # also print full answers + observed latency
```

## LLM explanation (`src/explainer.py`)

- `build_feature_record(row, is_time)` — deterministic risk scoring, **identical** to the
  finetune data labels so runtime explanations match training. Also derives a plain-language
  `reasons` string from the project's own numbers when the report has none.
- `explain(feat)` — calls **Ollama** (local open-source model) first; **falls back to a
  deterministic template** if Ollama/model is absent. Demo never breaks.
- Configure via env: `MOSPI_OLLAMA_URL` (default `http://localhost:11434`),
  `MOSPI_LLM_MODEL` (default `llama3.1:8b` — tool-capable; the old default was the 3B model
  which doesn't do function-calling).
- The Project Explorer only calls the LLM for the single selected project (bulk view uses
  the fast template path) so the app stays responsive.

## Finetuning a local model (`src/finetune_data.py`, `src/finetune.py`)

To give the LLM genuine MoSPI vocabulary, finetune a small model with **QLoRA** on our own
data, then import into Ollama:

```bash
# 1) dataset (already generated): data/finetune_train.jsonl (2,516 examples)
python -m src.finetune_data
# 2) train + merge adapter (needs a GPU; targets ~3B model to fit 6GB)
pip install -r requirements-finetune.txt
python -m src.finetune --model Qwen/Qwen2.5-3B-Instruct --export-gguf
# 3) import into Ollama (merged HF model -> GGUF via llama.cpp, then:)
#    ollama create mospi-llm -f Modelfile  (Modelfile: FROM ./merged)
# 4) point the app at it
set MOSPI_LLM_MODEL=mospi-llm
streamlit run src/dashboard.py
```

The dataset is built from **real** project features (real sectors, overrun %, expenditure
ratio, delay) plus authentic cause phrases from the Flash Report, so the finetuned model
learns real MoSPI reasoning rather than generic text.

---

## Team task assignment (PS103)

Define the DataFrame **schema first** (real + synthetic share it) so members work in
parallel against a stable interface:

| Member | Deliverable | Files to own |
|---|---|---|
| **A — Data pipeline** | Parse Annexures V (negatives) + VI (positives); multi-report merge Apr/Jul/Aug/Sep; schema + validation | `src/parse_flash_report.py`, `src/real_data_assembly.py` |
| **B — Features + models** | Leak-free features (log cost, expenditure ratio, out-of-fold sector overrun rate); LR/RF/LightGBM + stacked ensemble | `src/features.py`, `src/models.py` |
| **C — Evaluation + SHAP** | 5-fold stratified CV, ROC-AUC / F1, **per-sector breakdown**, SHAP summary/dependence; "ML vs baseline" comparison | `src/evaluate.py`, `src/explain.py` |
| **D — Dashboard + writeups** | Streamlit risk dashboard, LLM explanation layer, README + scope-limitation docs, slide narrative | `src/dashboard.py`, `README.md` |

Parallelization tips: B and C can start on `data/MoSPI_projects_synthetic.csv` now and
switch to `data/real_mospi_projects.csv` when A finalizes the schema. D scaffolds the
dashboard against the agreed schema regardless of which dataset feeds it.

---

# Synthetic track (original generator)

Implements the SPEC2.md (SIH26103 MoSPI Project Monitoring Platform) synthetic dataset
schema, distributions, target-calculation logic, and validation constraints.

## Layout

```
ps103/
├── SPEC2.md                  # Technical spec this generator implements
├── SectorWiseProjects_Report.csv   # your PAIMANA export (small sample)
├── dataset_spec.json         # Machine-readable config (real-calibrated fields, weights, thresholds)
├── src/
│   └── generate_data.py      # Generator + validation
└── data/
    └── MoSPI_projects_synthetic.csv   # Generated output (10,000 rows)
```

## Usage

```bash
python -m src.generate_data
# or with overrides:
python -m src.generate_data --rows 5000 --output data/output.csv --config dataset_spec.json
```

Generated CSV: `data/MoSPI_projects_synthetic.csv` (deterministic, `random_seed = 42`).

## Schema (21 columns)

| Field | Type | Notes |
|---|---|---|
| `project_id` | string | `PRJ-IN-2026-0001` … |
| `sector` | categorical | 6 sectors, weighted (Highways 30%, Railways 25%, …) |
| `state_region` | categorical | 5 regions, ~20% each |
| `executing_agency` | categorical | linked to `sector` |
| `start_date` | date | uniform 2018–2025 |
| `original_completion_date` | date | start + N(36, 12) months, clipped 12–72 |
| `revised_completion_date` | date | original + `delay_months` |
| `original_budget_cr` | float | log-normal, clipped 50–15000 Cr |
| `revised_budget_cr` | float | real-calibrated: ~25% of projects overrun (exponential tail, mean ~25%, to 120%); ~2% descoped below original; 0 if `delay==0` |
| `cost_overrun_pct` | float | derived `(revised-original)/original×100` |
| `expenditure_todate_cr` | float | Beta(2,2) × `revised_budget` (capped to financial rule) |
| `physical_progress_pct` | float | uniform 0–100 |
| `financial_progress_pct` | float | derived, ≤ `physical + 25` |
| `land_acquisition_pct` | float | Beta(5,2)×100, skewed high; forced >75 when physical>60 |
| `env_clearance_status` | categorical | Cleared 75% / In-Review 15% / Pending 10% |
| `litigation_active` | bool | 15% true |
| `contractor_tier` | categorical | Tier 1/2/3 @ 50/35/15 |
| `geo_latitude` / `geo_longitude` | float | India bounds |
| `delay_months` | int | **target** — penalty-driven + strong irreducible noise + interactions, clipped 0–60 |
| `risk_level` | categorical | **target** — Low/Medium/High/Critical from `delay_months` & `cost_overrun_pct` thresholds |

## Real-world anchoring

The generator is calibrated to **actual MoSPI Flash Report statistics** (and your
PAIMANA export), not hand-set constants:

- **Cost overrun:** ~25% of projects overrun vs original cost, overall COR ~20%, right-skewed
  tail up to 100%+ (real figures: 458/1,817 projects, COR 20.7%, per sector up to ~130% for
  Railways). A small ~2% are descoped below original (like the PAIMANA negative rows).
- **Delay:** ~40-45% of projects delayed, average ~36 months among delayed, with the real
  bracket shape (many 1-24 mo, a long tail >60 mo). Drivers are the documented ones — land
  acquisition, forest/environment clearance, litigation, contractor/solvency, financing (+COVID).
- **Realistic learnability:** strong base noise + interaction effects + per-project heterogeneity
  are folded into `delay_months` so no model can trivially reconstruct it.

## Target calculation

`delay_months` = sum of documented month penalties (from `dataset_spec.json`) × a per-project
`lognormal` scale + an interaction term (2+ major blockers amplify) + Gaussian noise + an
exponential "unexplained risk" term (unobserved drivers such as contractor solvency or
regulatory delays), clipped to `[0, 60]`.

`cost_overrun_pct` = `(revised - original)/original×100`, where the overrun flag (~25%) and
magnitude (exponential tail) come from the `revised_budget_cr` block in `dataset_spec.json`.

`risk_level` is assigned first-match from highest-danger tier using the `risk_thresholds`
block (e.g. Critical needs `delay>=24` AND `cost_overrun_pct>=40`).

### What the model should expect (honest preview)

On a **leak-free feature set** (only planning-time inputs: sector, region, tier, clearance,
litigation, log original cost, land %, planned duration), cross-validated Random Forest on
this synthetic data yields roughly:

- `delay_months` → **R² ≈ 0.41** (learnable but far from trivial)
- `cost_overrun_pct` → **R² ≈ 0.0** (largely unpredictable — matches real MoSPI, which scores
  ~0.15 even on real data, since overruns are driven by unobserved factors)

So the model will be credible, not "too good to be true." Present metrics as relative risk
indicators, not precise forecasts.

## Validation

All SPEC2 section-3 rules are checked on generation and reported as PASS/FAIL:

- Chronology: `start < original_completion` and (where `delay>0`) `original < revised`
  — note `revised == original` when `delay==0` per the cost-linkage rule.
- Cost linkage: `revised == original` when `delay==0`; and among projects that are BOTH
  >12 months delayed AND have a cost overrun, overrun is ≥15% (real MoSPI shows most delayed
  projects have *no* cost overrun, so the 15% floor is scoped to overrun projects only).
- Physical vs financial: `financial <= physical + 25`.
- Land constraint: `land > 75` when `physical > 60`.

No nulls are produced. Seeded RNG makes output reproducible.

## Requirements

`numpy`, `pandas` (Python 3.10+).
