This is real progress — the `cost_overrun_pct` catch (100% collinear with the delay label on negatives) is a serious leak you found and killed, and the temporal validation is exactly the right next move. But the temporal numbers themselves undercut something you concluded two messages ago, and that needs to be dealt with before it goes in the pitch.

**The problem: your "honest headline" just flipped**

Two messages back, the conclusion was "ML-vs-baseline win lives on the COST task" because it looked clean and non-leaky while time-overrun only added ~0.02 AUC over LR. Now look at how each task degrades out-of-sample:

- Time: 0.828 in-sample → 0.740–0.802 out-of-sample (relative drop ~3–11%)
- Cost: 0.846 in-sample → **0.633–0.712** out-of-sample (relative drop ~16–25%)

The cost model degrades *more* under temporal validation than the time model does, and 0.633 is barely above chance. That's not a minor caveat to footnote — it directly contradicts the "cost is the clean win" framing you're currently carrying into the pitch. In-sample cross-validation was telling you cost was the stronger, cleaner task; out-of-time validation is telling you the opposite. The second number is the one that matters more, because it's the one that resembles how the model would actually behave in deployment on next year's projects.

**Before calling this "improved," I'd want to know:**

1. Is the 0.633 window (2020→2021-22) a one-off, or does it hold if you check the year boundary differently? One bad fold in 3 could be a small-sample fluke rather than a real generalization failure — worth checking the actual n in that test window.
2. Is the sector/ministry historical-rate encoding the culprit? Those encodings are trained only on past data — if overrun rates genuinely shifted between cohorts (COVID-era supply chain and steel/cement price shocks would plausibly do exactly this to a cost model, less so to a schedule model), the encoding becomes stale exactly when it matters. That's a testable, fixable story: try a recency-weighted or rolling-window version of the sector rate instead of using the full training history equally, and see if it recovers some of the temporal AUC.
3. Is the same degradation pattern showing up in the cost *regression* MAE, or only the classification AUC? You reported cost reg improves "9–16% over naive" out-of-sample — that's much less alarming than the classification number. If regression holds up better than classification, the honest story might be "cost overrun *magnitude* is predictable, cost overrun *flag* is not," which is a real and usable distinction for the pitch rather than a failure.

**Answering your actual question — should you keep improving the model:**
Not by tuning further on in-sample splits — you've already shown tree models overfit relative to LR, and squeezing another point out of in-sample CV will likely just widen the temporal gap further. The next real improvement is making the sector/ministry rate features robust to drift (rolling-window or exponentially-decayed history) and re-running `temporal_eval.py` to see if that closes any of the 0.846→0.633 gap. That's a more valuable use of remaining time than another hyperparameter pass.

Want me to sketch the rolling-window/decayed-rate feature change, or check the 2020→2021-22 window's sample size first to rule out a fluke?

---

## Response (Follow-up #5): the "honest headline flip" — investigated, reframed

Your catch is correct: in-sample CV said "cost is the clean win" but out-of-time says cost
degrades to 0.633 while time holds ~0.74-0.80. We did not defend the old framing — we ran the
three checks and let the data re-frame the pitch. The full working log of this exchange is in
`SESSION_CONTEXT.md`; numbers live in `data/temporal_eval_results.json`.

**1) Is 0.633 (2020→2021-22) a one-off? No — but it is largely label censoring, and we proved it.**
Test n is large (163 / 251 / 413) and the degradation is monotonic (0.712 → 0.688 → 0.633), so it
is not a fluke. But the report now prints approval-cohort base rates for **both** labels:

| approval | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|
| cost overrun flag rate | 64% | 52% | 29% | 11% | 3% | 0% |
| delayed rate | 80% | 77% | 62% | 24% | 7% | 0% |

The cost flag only appears once a **formal revised estimate is sanctioned**, so it is censored
exactly like the delay label. The 0.633 window's test labels are only **18% positive** — mostly
"not yet revised", not "no overrun". That AUC is measuring label completeness, not model skill.
On the settled 2019-2020 cohort cost reaches **0.712** (time **0.802**).

**2) Is the stale sector-rate the culprit? No — tested and rejected.** Recency-decayed (halflife
4y) and rolling (last 5y) train-only sector encodings move AUC by ≤0.005 on both tasks in every
window (`temporal_eval.py --encoding decay,rolling`; verdict saved in the JSON). Sector history
drift is not the cause, and changing it does not recover the 0.846→0.633 gap.

**3) Does cost regression hold up better than classification? Yes.** Regression beats the naive
median in every window and every task — cost MAE improvement 12.9% / 16.3% / 8.9%, delay-months
improvement 28.8% / 30.6% / 32.9%. Honest caveat: the cost-reg gains are partly the same
censoring (recent test labels lean 0), so the durable deployment story is the **delay regression
(~30% better than naive everywhere)** plus ranking.

**Reframed honest pitch (replaces "cost is the clean win"):**
- Deployment metric = **magnitude + ranking**, not the class flag: per-project predicted ₹
  overrun / delay months (reg MAE 9-33% below the naive median) and a **top-20% review list**
  (lift ~1.3-1.9×; catches 25-38% of actual overruns/delays).
- Binary cost-flag AUC on young cohorts is not a fair skill test (the label is censored); the
  fair settled-cohort numbers are cost ~**0.71**, time ~**0.80**.
- Out-of-time, **logistic/linear regression leads on both tasks** — the tree-ensemble "wins"
  were in-sample only (reconfirms follow-up #3).
- The dashboard's "How good on NEW data?" section now prints per-window **test n + label rate**
  and the two-label censoring table, so 0.633 is openly explained rather than hidden.

**Follow-up #5½ — Genuinely new-document validation (May-2025 Flash Report):**
The May-2025 Flash Report (`FR_May2025.pdf`, 1,559 projects, new Table-7 format, parsed by
`src/flash2025_parser.py`) provides a **genuinely different document** whose labels are
**resolved as of May-2025** (delay = anticipated DOC > original DOC; cost overrun = revised vs
original) — so the 2019-2021 approval cohorts there are *not* affected by the censoring above.
Train stays on the pre-cutoff annexure data. Results (from `data/external_holdout_2025.json`):

| Train ≤ | May-2025 test cohort | Time AUC | Cost AUC | Time reg MAE (vs naive) | Cost reg MAE (vs naive) |
|---|---|---|---|---|---|
| 2017 | 2018–2019 (n=196/207) | **0.806** | 0.640 | 16.8 mo (vs 23.7, −29%) | 0.273 (vs 0.31, −12%) |
| 2018 | 2019–2020 (n=244/288) | **0.810** | **0.719** | 16.2 mo (vs 23.1, −30%) | 0.217 (vs 0.262, −17%) |
| 2019 | 2020–2021 (n=364/450) | **0.740** | 0.665 | 15.1 mo (vs 23.1, −35%) | 0.193 (vs 0.203, −5%) |

What this adds:
- **Time delay detection generalises to a different document** (AUC 0.74–0.81 on resolved labels,
  29–35% reg improvement). This is the strongest "does it generalise?" check available.
- **Cost classification is honestly weaker out-of-document (0.64–0.72)** — confirming the
  reviewer's point that the in-sample 0.846 was optimistic. The deployable story for cost remains
  **ranking + magnitude** (top-20% catches 37–45% of actual overruns; reg beats naive 5–17%).
- The 2019–2021 rows in this snapshot are exactly the "more data" lever: appending them (or
  retraining on each new official Flash Report) improves the censored 2022+ cohorts. This is the
  platform's designed Data & Retrain loop working as intended.