# MoSPI Infrastructure Projects — Dataset (Synthetic + Real)

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

**1,878 real projects**, delayed rate 0.65, real `tor_months` (median 22, right-tailed).

## Targets & features (`src/time_model.py`)

- **Classification**: P(delayed) — logistic / RF / LightGBM.
- **Regression**: `tor_months` magnitude on delayed projects (log-transformed).
- **Leak-free features**: log original cost, expenditure ratio, planned duration (approval →
  original DOC), out-of-fold sector delay rate, and cost-overrun % (`cost_overrun_pct` — a
  legitimate execution signal for the *time* target, knowable at evaluation time; verified
  by ablation, it is a genuine driver, not an annexure-membership leak).
- **Observation-window caveat (survivorship bias)**: approval cohort year / elapsed time
  since approval mechanically predicts "delayed" (brand-new projects can't yet be flagged).
  We therefore report the model two ways (see `feedback.md`):
  - **Age-conditional** (with `elapsed_years` named explicitly) — dominated by the observation
    window; **not** the pitch number.
  - **Content-only** (no temporal features) — the defensible delay signal.

## Results (repeated 5x2 CV; see `data/time_model_results.json`)

**Classification — content-only P(delayed) ROC-AUC (defensible):**
LR **0.87** · Random Forest **0.91** · LightGBM 0.90.
(Age-conditional framing inflates these to ~0.90–0.97 via the observation window; per
`feedback.md` that top layer is **not** used on the pitch.)

**Per-sector (content-only, sectors with ≥10 rows & both classes):**
Railways **0.97** / Petroleum 0.95 / Road 0.89 / Coal 0.76.

**Regression log(tor_months) — ML clearly beats the linear baseline:**

| Model | MAE (months) | R² (log) |
|---|---|---|
| Linear Regression (baseline) | 34.4 | 0.20 |
| Random Forest | **14.9** | **0.59** |
| LightGBM | 14.0 | 0.62 |

> **Key story for the pitch:** on the *time-overrun magnitude* task, ML gives a real gain
> over the conventional statistical baseline (~10 months lower MAE). This is a concrete,
> positive answer to PS dimension (b) — the same comparison that was a wash on the cost
> task is a clear ML win on the time task.

## Known limits (document these)

- `tor_months` > 240 months are clipped as PDF cell-shift artifacts.
- `planned_duration` uses approval/original-DOC *years* (month granularity not preserved),
  so it's a coarse but still informative feature.
- Single May-2024 snapshot; cross-snapshot validation is the recommended follow-up.

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
  by delay risk / cost overrun / project count; **click a state** for a drill-down card.
  Boundary file is auto-simplified (Douglas-Peucker) so it stays responsive.
- **Project Explorer** — pick a project -> cost/delay risk + tier, cost overrun, AI explanation
- **AI Assistant** — interactive chat that answers free questions over the real data
  (e.g. *"how many airport-related projects?"*, *"top 10 at-risk projects"*,
  *"coal sector summary"*, *"overall portfolio status"*)

## AI Assistant (`src/assistant.py`)

A **tool-using agent**: the user question is mapped to one of five live query tools
(`search_projects`, `sector_stats`, `top_risk_projects`, `project_detail`,
`portfolio_overview`) which execute against the real project/risk data and return a genuine
count/aggregate/table — never canned text. When Ollama is available it chooses the tool
(structured function-calling) and phrases the answer; when it isn't, a deterministic query
engine answers the same questions so the demo still works.

> For the LLM tool-calling path use a tool-capable model, e.g. `ollama pull llama3.1:8b`
> or `qwen2.5:7b-instruct` (`set MOSPI_LLM_MODEL=...`). A 3B base model may not support
> function calling cleanly; the deterministic engine covers all demos regardless.

## LLM explanation (`src/explainer.py`)

- `build_feature_record(row, is_time)` — deterministic risk scoring, **identical** to the
  finetune data labels so runtime explanations match training.
- `explain(feat)` — calls **Ollama** (local open-source model) first; **falls back to a
  deterministic template** if Ollama/model is absent. Demo never breaks.
- Configure via env: `MOSPI_OLLAMA_URL` (default `http://localhost:11434`),
  `MOSPI_LLM_MODEL` (default `qwen2.5:3b-instruct`).
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
