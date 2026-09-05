# Architecture — MoSPI Project Monitoring Platform (PS103)

_Current-state, end-to-end technical architecture. Pairs with `KEY_INSIGHT.md` (the headline
finding + savings estimate) and `README.md` (full experiment history)._

---

## 1. System at a glance

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                      DATA / MODELS                     │
                    │  data/real_mospi_projects.csv   (cost,     n=1,213)     │
                    │  data/real_mospi_time.csv       (time,     n=1,876)     │
                    │  data/mospi_may2025_projects.csv (snapshot, n=1,559)    │
                    │  data/models/cost.joblib · time.joblib · version.json   │
                    │  data/india_state_light.geojson (local boundary, 182 KB)│
                    └────────────┬──────────────────────────┬────────────────┘
                                 │ joblib load              │ predict (leak-free OOF)
                    ┌────────────▼──────────┐   ┌────────────▼────────────────┐
                    │   src/predict.py      │   │  src/ingest.py              │
                    │   ModelService        │   │  dedupe_append · census_age │
                    │   CostEstimator       │   │  ingest_and_retrain (loop)  │
                    │   TimeEstimator       │   └─────────────────────────────┘
                    └────────────┬──────────┘
                                 │ predictions + confidence bounds
                 ┌───────────────▼────────────────────────────────┐
                 │          src/dashboard.py  (Streamlit)          │
                 │  Home · Dashboard · Geospatial Map · Explorer   │
                 │  AI Assistant · Data & Retrain                  │
                 └───────────────┬────────────────┬────────────────┘
                                 │                │ live LLM
                 ┌───────────────▼───────┐  ┌─────▼──────────────────────────┐
                 │ 6 query tools (hard)  │  │ src/assistant.py (tool agent)  │
                 │ deterministic engine  │  │ Ollama llama3.1:8b (function call)│
                 └───────────────────────┘  └────────────────────────────────┘
```

**Stack:** Python 3.14 · pandas / numpy · scikit-learn · LightGBM · joblib · Streamlit ·
pydeck (lazy) · plotly · altair · Ollama (`llama3.1:8b` at `localhost:11434`).

---

## 2. Data layer

| Dataset | Source | Rows | Role |
|---|---|---|---|
| `real_mospi_projects.csv` | Annexure V/VI of the official Flash Report | 1,213 (19 sectors) | **Cost** model train set |
| `real_mospi_time.csv` | Delay annexures VII + IV/V | 1,876 (delay base rate 0.65) | **Time** model train set |
| `mospi_may2025_projects.csv` | May-2025 Flash Report (`flash2025_parser.py`) | 1,559 | External holdout **and** cost-model augmentation; the early-warning screen |
| `india_state_light.geojson` | Auto-simplified India boundary (Douglas-Peucker) | — | Local map shapes (no online tiles) |
| `macro_by_year.csv` | RBI/IMF public figures | — | Tested as features → **rejected** (adds ≤ ±0.02 AUC, time degrades) |

**Augmentation rule (baked into serving):** the served **cost** model trains on annexure +
resolved May-2025 rows (`augment_cost=True`, 2,772 total; CV AUC 0.805, reg MAE 0.160); the
**time** model stays annexure-only (`augment_time=False`) because the two documents define the
delay label differently (reported TOR vs anticipated DOC), and the A/B showed merging hurts AUC
by ~0.03. Evidence: `data/augmented_train_holdout_2025.json`, `data/external_holdout_2025.json`.

**Two models, two roles (keep them labeled):**

1. **Research / screening model** — annexure-only fit (n=1,213), scored on the never-seen
   May-2025 rows. This produced the *headline* early-warning result (348 of 1,203 "clean"
   projects at risk) and all rupee exposure numbers in `KEY_INSIGHT.md`. A clean out-of-sample
   result — but it is **not** what the dashboard displays.
2. **Live dashboard model** — the refreshed production model (`ModelService`, augmented,
   n=2,772, cost AUC 0.805, reg MAE 0.160), what the Explorer/AI Assistant actually show, and
   what `Data & Retrain` refits in place.

Per-project risk scores on the running app therefore **differ by design** from the 348-list; the
dashboard serves the more-informed model (augmentation A/B: cost +0.03 AUC, −28–35% reg MAE).
Never present the two models' outputs as the same number.

---

## 3. Model layer (`src/predict.py`)

**Leak-free rule** — no target-collinear or observation-window features:

- Cost features: `log_original_cost`, `expenditure_ratio`, `log_expenditure`, OOF `sector_rate`.
- Time features: `log_original_cost`, `expenditure_ratio`, `planned_duration_yrs`, OOF `sector_delay_rate`.
- **Dropped as leakage (audited):** `cost_overrun_pct` (target-collinear), calendar/age features
  (survivorship artifact — inflates AUC to ~0.95).

**Estimators** (fit via `CostEstimator.fit` / `TimeEstimator.fit`):

- **Classification:** Logistic Regression / Random Forest / LightGBM, model selection by
  RepeatedStratifiedKFold (5×2) AUC; final classifier fit on all data for serving.
- **Regression** (magnitude): `overrun_ratio` and `tor_months` — log-target, KFold CV,
  best-by-MAE; capture shift for the signed `overrun_ratio`.
- **Serving:** `predict_project(row)` → cost `{overrun_prob, overrun_ratio, overrun_cr,
  fiscal_exposure_cr, anticipated_cr, mae_ratio}` and time `{delay_prob, delay_months, mae_months}` —
  with honest confidence bounds (reg MAE).
- **Persistence:** `joblib` + `version.json`; `_bump_version()` is monotonic across restarts
  (reads persisted version, then +1).

**Performance summary (honest, all out-of-sample):**

| Claim | Value | Evidence |
|---|---|---|
| Cost classification, annexure CV | AUC 0.846 | `real_model_results_v2.json` |
| Cost classification, served (augmented) | AUC 0.805 | `data/models/version.json` |
| Cost classification, external holdout (May-2025 doc, never seen) | **AUC 0.64–0.72** (LR best) — degrades; don't pitch the flag | `external_holdout_2025.json` |
| Cost reg MAE | 0.16 (vs ~0.34 naive) | `external_holdout_2025.json`, `augmented_train_holdout_2025.json` |
| Time classification (content-only) | AUC 0.828 (RF); LR 0.806 | `time_model_results.json` |
| Time delay reg MAE | 21.3 months (vs ~30-34 naive) | `time_model_results.json` |
| External holdout (May-2025 doc, never seen) | time AUC 0.74–0.81; cost **recall-lift** 1.85–2.23× at top-20% (share of actual overrun flags caught) | `external_holdout_2025.json` |
| Survival reframing (censoring-robust) | C-index 0.59–0.65 — **temporal external holdout** (train ≤ cutoff → May-2025 cohort), **not random CV**; modest (near 0.5 floor) but beats binary C-index on the most-censored cohort (0.639 vs 0.587) | `survival_eval_2025.json` |
| Macro covariates | **Rejected** — no gain, time degrades | `macro_experiment.json` |

**Key caveats to keep honest:** cost *classification* degrades out-of-document (0.64–0.72 vs
0.846 in-sample CV) — the strong pitch is **magnitude + ranking**, not the class-flag headline.
Regression log-scale R² ≈ 0.28 → rupee magnitudes are order-of-magnitude (rounded to 1–2 s.f.
everywhere), ranks are solid (AUC-driven). Two distinct lift numbers exist and must stay labeled:
**recall-lift** (holdout, share of actual flags) vs **rupee-concentration factor** (≈3.8×, in
`KEY_INSIGHT.md` §6) — never merge them into "3.8× lift".

---

## 4. Serving & continual learning (`src/ingest.py`)

- `dedupe_append()` — incremental merge over existing keys (update touched, append new, leave
  the rest intact).
- `census_age()` — recomputes `delayed`/`tor_months` honestly with no look-ahead (a project is
  delayed only when a report actually shows a TOR).
- `ingest_and_retrain()` — append new data → `train_all()` → persist + bump `version.json`
  (online / continual learning loop). Dashboard's **Data & Retrain** page drives this: add a
  row or upload a PDF/CSV, retrain in place.

---

## 5. Dashboard (`src/dashboard.py`) — Streamlit

Six pages: **Home · Dashboard · Geospatial Map · Project Explorer · AI Assistant ·
Data & Retrain.**

### 5.1 Geospatial Map — WebGL hardening (recent changes)

Original design shipped online basemap tiles (mapbox/pydeck tile layer) to a `pydeck` canvas —
heavy GPU/VRAM churn; in a browser it produced **"Oh snap" / out-of-memory** tabs.

Changes (per user choice — *"Kill WebGL basemap tiles"*):

- `map_style=None` — no online basemap; canvas renders only local vector layers.
- Local **India boundary GeoJsonLayer** from `load_geo_light()` (~182 KB auto-simplified
  boundary) with local theme fills — no network, no tile stacking.
- Selectbox relabeled **"Map Canvas Theme (Local, No Online Tiles)"**.
- **Arcs / borders toggles default OFF**; GPU vertex count reduced.
- `render_3d_india_map(...)` gained a `show_borders` param.
- `build_view()` became a **no-arg `@st.cache_data`** function → no more hashing of the
  1,876-row DataFrame on every render (big win for warm re-renders).
- Removed top-level `import pydeck as pdk`; **lazy `import pydeck as pdk` inside**
  `render_3d_india_map` (cut heavy import from cold start and from non-map pages).
- Call sites updated to no-arg `build_view()`: `page_explorer` (~line 1373), `page_assistant`
  (~line 1471), `demo_assistant.py:37`.

### 5.2 Performance characteristics

- **Cold start ~4–6 s** — dominated by importing altair / pydeck / plotly / pandas, **not**
  Ollama (dashboard pages never call the LLM; only the AI Assistant does).
- **Warm renders < 1 s** (0.14–0.15 s typical). All 6 pages verified via Streamlit `AppTest`
  with **0 exceptions**, including simulated sidebar navigation and a chat submit.
- `@st.cache_data` on `build_view()` means cached map frames serve instantly on re-render.

### 5.3 AI Assistant (`src/assistant.py`) + LLM explanation (`src/explainer.py`)

- **Tool-using agent:** 6 live query tools (`search_projects`, `sector_stats`,
  `top_risk_projects`, `rank_projects`, `project_detail`, `portfolio_overview`) execute
  against the real data and return genuine aggregates/tables — never canned text.
- Ollama `llama3.1:8b` picks the tool via structured function-calling and phrases the reply; a
  **deterministic engine re-answers** when routing to a failing tool or when Ollama is down —
  the chat never dead-ends.
- `explainer.py`: deterministic risk scoring (identical logic to finetune labels) + plain
  `reasons` string; LLM-first with deterministic-template fallback. Explorer calls the LLM only
  for the single selected project → stays responsive.
- **Recent robustness fix (live):** the LLM sometimes emitted `"limit": null`, blowing up
  `int(None)` at `assistant.py:115`. Fixed with None-safe clamping
  `min(int(x or default), 50)` in `tool_search` (114), `tool_top` (141), `tool_rank` (159).
  Verified end-to-end with a real LLM question ("list me airport projects") → 10-row table.
- **Latency profile:** warm ~3.7–7.9 s/answer; first call after idle ~35 s (model loads into RAM).

---

## 6. Ops & deployment (current live state)

- **Run command:** `streamlit run src/dashboard.py --server.fileWatcherType none
  --server.runOnSave false --server.headless true`
- **`.streamlit/config.toml`:** `fileWatcherType = "none"`, `runOnSave = false`, `headless = true`
  — kills the per-save file-watcher churn that contributed to device lag.
- **Network:** server binds `::` (all interfaces). Reachable at `http://localhost:8501` and on
  LAN at `http://172.24.59.144:8501` (Ethernet). Wi-Fi shows `169.254.x.x` (no DHCP — broken).
- **Ollama:** `localhost:11434`, `_ollama()` timeout 60 s, `MOSPI_LLM_MODEL=llama3.1:8b`.
- Device was previously sluggish largely due to OneDrive folder sync + Streamlit watcher churn;
  the flags + config above plus closing background apps resolved it.

---

## 7. Insight & estimated savings (the "winnable" result)

Full write-up in `KEY_INSIGHT.md`; headline numbers:

- **Censoring illusion:** delay base rate by cohort 100% (1980–95) → 18% (2022+) and cost
  overrun 42% → 9%. Recent cohorts *look* healthy only because they're too young to fail.
- **Early-warning screen (research model — annexure-only, scored on never-seen May-2025 rows;**
  **not the live dashboard model):** of 1,203 currently-clean projects, **348 (p≥0.6) are already
  on the overrun trajectory** → ~Rs 8 lakh cr sanctioned cost, ~Rs 2 lakh cr prob-weighted
  exposure.
- **Total expected future overrun** of the clean portfolio: **~Rs 2.3 lakh cr**; top-20% ranked
  list (241 projects) carries **~75%** of it → **≈3.8× rupee concentration** over random.
- **Savings scenarios:** preventing 10/20/30/50% of that expected overrun saves
  ~Rs 23,000 / **45,000** / 68,000 / ~1.1 lakh cr respectively (20% = conservative headline).
- The estimate is a real output of the model, **unreachable by prior static dashboards** — they
  react only after a revised cost/TOR is declared; we rank before it appears.
- Judges-caveat: rupee magnitudes inherit regressor uncertainty (R²≈0.28) and are rounded to
  order-of-magnitude; the solid claims are **counts + ranking + AUC-driven triage**.

---

## 8. What changed recently (session log)

| Change | Where | Why |
|---|---|---|
| Kill WebGL online basemap; local India-boundary GeoJsonLayer | `dashboard.py` `render_3d_india_map` | Browser OOM / "Oh snap" |
| `build_view()` → no-arg `@st.cache_data` | `dashboard.py` | Stop hashing 1,876-row df per render |
| Lazy `import pydeck as pdk` (drop top-level) | `dashboard.py` | Cut cold-start, keep non-map pages light |
| Arcs/borders default OFF, reduced GPU vertices | `dashboard.py` | VRAM budget |
| None-safe `limit`/`n` clamping | `assistant.py` 114/141/159 | LLM `null` arg crash |
| `fileWatcherType=none`, `runOnSave=false` | `.streamlit/config.toml` + CLI | Device lag from file-watch churn |
| Restored `_ollama()` timeout 60 s; fixed `_ollama_available()` after `OLLAMA_URL` defn | `assistant.py` | NameError on cold boot |
| 6 pages verified 0 exceptions | AppTest suite run | Regression gate |

- Prior work (documented in README.md): `flash2025_parser.py`, macro rejection, survival
  reframing, augmentation A/B, `ingest.py`, QLoRA finetune path (`finetune_data.py` /
  `finetune.py`), synthetic generator (`generate_data.py`).