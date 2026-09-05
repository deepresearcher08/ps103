# Key Insight — What the Models Reveal That the Flash Report Cannot

_One pagination of the strongest finding mined from the model results. All numbers are reproducible
from the repo's own data (`data/real_mospi_projects.csv`, `data/real_mospi_time.csv`,
`data/mospi_may2025_projects.csv`) using the annexure-trained cost estimator in `src/predict.py`._

> **Read this first — two models, two roles.** This doc's headline numbers (the "348 projects",
> all rupee exposures) were produced by the **research / screening model**: fit on the 1,213
> annexure rows **only**, then scored on the 1,559 May-2025 rows it had **never seen** — a genuine
> out-of-sample screen. The **live dashboard** instead serves a *refreshed* model (annexure +
> resolved May-2025 rows, n=2,772; cost AUC 0.805, reg MAE 0.160), which the augmentation A/B
> showed is the *better-informed* one for cost. So **per-project numbers on the running app will
> differ from the 348-list** — that is intentional. Label the headline as *research-model output*
> and the dashboard as *the refreshed production model*; never present them as the same number.
>
> **Reading convention:** rupee figures are order-of-magnitude (the regressor explains only ~28%
> of variance on a log scale). Counts and percentages are exact; rupees are rounded to 1–2
> significant figures.

---

## 1. The censoring illusion (the setup)

Look at delay and cost-overrun rates by **approval cohort**:

| Cohort    | Delayed (annexure, n=1,876) | Cost overrun (May-2025 snapshot, n=1,559) |
|-----------|------------------------------|-------------------------------------------|
| 1980–1995 | 100%                         | —                                          |
| 1996–2005 | 96%                          | —                                          |
| 2006–2015 | 95.5%                        | 42%                                       |
| 2016–2019 | 88%                          | 37%                                       |
| 2020–2021 | 69%                          | 25%                                       |
| 2022+     | **18%**                      | **9%**                                    |

A naive reading of the raw data says *"project performance is improving every year."* It is not.
The 2022+ rows aren't better — **they're just too young to have failed yet.** A 2022 project hasn't
had enough time to be flagged delayed (delay is declared only after the planned date passes), and on
the cost side an overrun is only *observed* the moment a revised (anticipated) cost gets declared —
early big projects still look "on budget" simply because nobody has revised their estimate.

Every cohort that has been observed long enough ends up badly: **~95–100% of central-sector
projects sanctioned before 2016 are delayed; ~40%+ overrun their sanctioned cost.** The 18% / 9% of
recent cohorts are statistical censorship, not a success story.

## 2. What only the model can see (the payoff)

The cost model was fit on the **1,213 annexure rows only** (out-of-fold AUC **0.846**, base rate
0.491) and then scored the **1,559 May-2025 Flash-report rows it had never seen** — a genuinely
out-of-sample early-warning screen (the "research model" above).

Of **1,203 projects that look "clean" today** (no cost overrun declared as of May 2025):

| Risk threshold | Clean projects flagged | Share of "green" portfolio | Sanctioned cost* | Prob-weighted exposure* |
|----------------|------------------------|----------------------------|------------------|--------------------------|
| p ≥ 0.5        | 441                    | 36.7%                      | ~Rs 8.9 lakh cr  | ~Rs 2.1 lakh cr          |
| p ≥ 0.6        | **348**                | **28.9%**                  | **~Rs 8 lakh cr** | **~Rs 2 lakh cr**      |
| p ≥ 0.7        | 241                    | 20.0%                      | ~Rs 6.3 lakh cr  | ~Rs 1.7 lakh cr          |

So **~one in four of today's "green" projects (348 of 1,203) is already on the classic overrun
trajectory** — roughly **Rs 8 lakh crore** of sanctioned cost and **~Rs 2 lakh crore** of expected
overrun exposure hiding in plain sight. The report shows green; the model reads the spend-vs-cost
trajectory and the sector's history and says these will turn red.

_*Exposure = predicted P(overrun) × predicted overrun ratio × original cost. The regressor's point
estimates carry wide error bars (log-scale R^2 ≈ 0.28) — treat rupee exposure as an
order-of-magnitude figure (rounded accordingly above); the **counts** and the **AUC-based ranking**
are the solid numbers._

## 3. The flagship surprises

Top currently-clean projects by predicted fiscal exposure (annexure-trained, scored out-of-sample):

| Overrun prob | Project | Sector | Approved | Original cost* | Exposure*,† |
|--------------|---------|--------|----------|----------------|-------------|
| 0.88         | HIGH SPEED RAIL | RAILWAYS | 2015 | Rs 108,000 cr | ~Rs 30,000 cr |
| 0.95         | FREIGHT | RAILWAYS | 2007 | Rs 51,000 cr | ~Rs 18,000 cr |
| 0.68         | KARNAPRAYAG NEW LINE | RAILWAYS | 2011 | Rs 39,000 cr | ~Rs 12,000 cr |
| 0.97         | TO IMPHAL | RAILWAYS | 2003 | Rs 15,000 cr | ~Rs 7,600 cr |
| 0.96         | CLUSTER II DEVELOPMENT | PETROLEUM | 2016 | Rs 34,000 cr | ~Rs 4,500 cr |
| 0.82         | MEERUT RRTS | URBAN DEVELOPMENT | 2019 | Rs 30,000 cr | ~Rs 4,200 cr |

The sector-level verdict (clean projects → flagged at p ≥ 0.6 → exposure*):

| Sector | Clean projects | Flagged at-risk (p≥0.6) | Exposure* |
|--------|---------------|--------------------------|-----------|
| RAILWAYS | 95 | 60 (**63%**) | ~Rs 1.2 lakh cr |
| POWER | 71 | 22 | ~Rs 29,000 cr |
| ROAD TRANSPORT & HIGHWAYS | 800 | 168 | ~Rs 18,000 cr |
| PETROLEUM | 86 | 25 | ~Rs 17,000 cr |
| URBAN DEVELOPMENT | 7 | 4 | ~Rs 8,400 cr |

Railways is the loudest signal: **63% of its "clean" projects are already flagged** — slippage in
flagship corridors is systematic and predictable. Road projects, by contrast, are mostly small and
idiosyncratic (only 21% flagged).

_†Single-project exposures inherit the regressor's wide error bars (log-scale R² ≈ 0.28) — quote
them as "the model ranks these first", not as invoiced amounts. One screened project (LOWER H.E.P.,
a 2003 hydro) produced an implausible ~3× predicted ratio and is deliberately **not** shown as a
headline example; it remains in the full reproducible output if asked._

## 4. What the model adds (vs. the raw report)

| | Flash-report numbers | Model |
|---|---|---|
| "Are recent projects healthy?" | Yes (18%/9%) | No — they're just censored (young) |
| "Which clean projects will turn red?" | Silent | Ranked at-risk list, OOF AUC 0.846 |
| "Where is the money at risk?" | Only after revision is declared | Before revision — from spend trajectory + sector history |
| Scaling | React after failure | Predict before failure |

**Honesty about the binary flag:** on the truly out-of-document May-2025 holdout, cost
*classification* AUC is **0.64–0.72** (vs 0.846 in-sample CV) — this is exactly why the project
leads with **magnitude + ranking**, not the class-flag headline. That reframe is used everywhere,
including the savings logic below.

## 5. How to say it in the final pitch

> "Every central-sector project that has been observed long enough eventually overruns. Newer
> cohorts only *look* healthy because they are too young to have failed yet. Our *research model* —
> trained purely on historical annexure data and validated on the never-seen May-2025 report —
> flags **348 of today's 1,203 'green' projects as already on the overrun trajectory**: **~Rs 8 lakh
> crore** of sanctioned cost and **~Rs 2 lakh crore** of expected exposure, dominated by flagship
> rail and power projects. Screen the portfolio with this ranking and you catch overruns *before* a
> revised estimate ever appears."

## 6. Theoretical savings estimate

The model's real output isn't "revenue" — for a government exchequer it's **avoided/early-prevented
expenditure**. The value comes in two layers: (a) the total slippage the model can see coming among
today's "clean" portfolio, and (b) the triage efficiency of catching nearly all of it by acting on
only a small, model-ranked slice.

**Total exposure the models surface (cost side, research model):**

| Metric | Value |
|---|---|
| Expected future overrun across all 1,203 currently-clean projects (Σ p × ratio × orig) | **~Rs 2.3 lakh cr** |
| Expected overrun of the p ≥ 0.6 slice (348 projects, 28.9% of clean) | ~Rs 2 lakh cr (~90% of total) |
| Top 20% of the model-ranked clean list (241 projects) | ~Rs 1.7 lakh cr (**~75%** of total) |
| Top 30% (361 projects) | ~Rs 2 lakh cr (~90% of total) |
| What a *random* 20% would capture | ~Rs 45,000 cr (20%) → **model ≈ 3.8× rupee concentration** |

**Savings scenarios** (share of the ~Rs 2.3 lakh cr expected overrun actually prevented by acting
on the ranked list):

| Prevented fraction | Saved | | Prevented fraction | Saved |
|---|---|---|---|---|
| 10% | ~Rs 23,000 cr | | 30% | ~Rs 68,000 cr |
| 20% | ~Rs 45,000 cr | | 50% | ~Rs 1.1 lakh cr |

**Delay side (rough back-of-envelope, 8% opportunity cost of stuck capital):**

| Action | Value |
|---|---|
| Cut 6 months of delay on flagged projects | ~Rs 84,000 cr |
| Cut 12 months of delay on flagged projects | ~Rs 1.7 lakh cr |

**Why these are defensible (and likely conservative):**
- On the already-overrun projects the model *under-predicts* magnitude (mean predicted ratio 0.28
  vs. observed 0.63 on flagged rows) — suggesting the exposure figures are **not inflated**
  estimates. Note this is measured on already-flagged projects, not on the (by construction
  unlabeled) clean set being screened, so it supports *not inflated*, not a guarantee.
- "Prevent" is deliberately conservative: it assumes realistic process gains (earlier procurement,
  de-scoping, bottleneck removal on the 241–361 flagged projects), not 100% avoidance.
- The rupee magnitudes inherit the regressor's uncertainty (log-scale R² ≈ 0.28). Keep the solid
  anchors for the judges: **~Rs 2.3 lakh cr expected overrun, ~75% captured by acting on the top
  20%, ≈3.8× rupee concentration, plus exact counts (348 of 1,203) and the AUC-based ranking.**

**Two "lift" numbers exist — never merge them:**
1. **1.85–2.23× recall-lift** (external holdout): the share of *actual overrun flags* caught by
   the top-20% ranked list vs a random 20% of the same size.
2. **≈3.8× rupee-concentration factor** (this doc): the *expected rupee overrun* in the top-20%
   ranked list (~75% of the total) vs the ~20% a random slice holds.
They measure different things and must be labeled as such ("recall-lift" vs "rupee concentration").

**Suggested savings line for the pitch:**

> "The portfolio expects **~Rs 2.3 lakh crore** of cost overrun from projects that currently look
> healthy. Acting on just the top-20% model-ranked list — 241 projects — addresses **~75%** of that
> (~Rs 1.7 lakh crore of expected exposure). Even preventing a tenth of it saves **~Rs 23,000
> crore** of exchequer money, at ≈3.8× the rupee concentration of acting on any 20% you'd pick by
> eye."

## 7. Validation-scheme notes (so the numbers survive scrutiny)

- **Everything external is temporal, not random-CV.** Every out-of-sample number (annexure CV,
  temporal windows, May-2025 holdout, survival C-index) trains only on rows **approved on/before a
  cutoff year** and tests only on *later* rows. No random-split AUC is quoted as a headline.
- **Survival C-index (0.59–0.65)** uses exactly the same external holdout as above —
  train ≤ cutoff on annexure, test = May-2025 cohort — with right-censored (survival) scoring
  (`survival_eval_2025.json`, cutoffs 2017–2020). It is NOT random-CV. Honest framing: 0.59–0.65
  is modest and near the non-informative 0.5 floor; its one real claim is that it ranks better than
  the binary model exactly on the *most-censored* cohort — 0.639 vs 0.587 at 47% still unflagged —
  and it keeps the newest cohort evaluable at all.
- **Cost classification on the May-2025 document is 0.64–0.72, not 0.85.** The 0.846 is
  in-sample (annexure CV). The deployable claim is ranking + magnitude (top-20% recall-lift
  1.85–2.23×; cost reg MAE beats naive by 5–17%).

## Reproducibility

```
# research / screening model (the numbers in this doc)
# NOTE: reproduces the doc WITHOUT touching data/models/ (no persistence side-effects)
python -c "from src.predict import CostEstimator; import pandas as pd; \
c=CostEstimator(); c.fit(pd.read_csv('data/real_mospi_projects.csv')); \
print('AUC', round(c.clf_auc,3), 'n', c.n_rows, 'base', round(c.base_rate,3))"

data source:      data/real_mospi_projects.csv  (train, n=1213)
                  data/mospi_may2025_projects.csv (score, n=1559)
honest setup:     model never sees a May-2025 row during fitting

# verified CLI (runs cleanly, BUT beware it RETRAINS AND PERSISTS -> overwrites the
# served models in data/models/ and bumps version.json; restore the augmented model
# afterwards with `python -m src.predict` (no flag) or keep a backup):
python -m src.predict --no-augment   # cost trains on annexure-only (n=1213, AUC 0.846)
                                     # default (no flag) = augmented (n=2,772, AUC 0.805)

# out-of-sample evidence for the claims above
python -m src.temporal_eval --external 2017,2018,2019        # cost/time AUC + recall-lift
python -m src.temporal_eval --survival 2017,2018,2019,2020   # C-index (censored-aware)
```