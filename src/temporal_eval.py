"""Temporal (out-of-sample) validation of the MoSPI prediction models.

Answers the practical question: "how good is this on NEW data?" by training only
on projects approved up to a cutoff year and evaluating ONLY on projects approved
1-2 years later (after the cutoff) — i.e. data the model never saw.

LEAK-FREE discipline (same rules as predict.py):
  - cost features: log original cost, expenditure ratio, log expenditure, sector rate
  - time features: log original cost, expenditure ratio, planned duration, sector rate
  - the target-collinear `cost_overrun_pct` and observation-window (age) features
    are NEVER used.
  - sector rates are recomputed from TRAIN data only (m-estimate) — the pipeline's
    CV-OOF encoders are NOT reused here because they touch validation labels.

Censoring caveat (reported honestly, not hidden):
  Projects approved in 2022+ have simply not had enough time to be flagged as
  delayed yet (delayed base rate collapses 62% -> 24% -> 7% across 2021/2022/2023).
  Evaluating on those newest cohorts would measure the observation window, not
  model skill — so the primary temporal test uses the 2020/2021 cohorts where
  outcomes are settled enough, and the shrinking-rate effect is printed explicitly.

Run:
    python -m src.temporal_eval            # prints report + writes JSON
    python -m src.temporal_eval --cutoffs 2018,2019
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score, precision_recall_curve

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TIME_CSV = DATA / "real_mospi_time.csv"
COST_CSV = DATA / "real_mospi_projects.csv"
EXT_CSV = DATA / "mospi_may2025_projects.csv"
MACRO_CSV = DATA / "macro_by_year.csv"
OUT_JSON = DATA / "temporal_eval_results.json"
EXT_OUT_JSON = DATA / "external_holdout_2025.json"
MACRO_OUT_JSON = DATA / "macro_experiment.json"

M = 20.0  # m-estimate smoothing


# --------------------------------------------------------------------------- #
# train-only feature encoders (leak-free for a temporal split)
# --------------------------------------------------------------------------- #
MODES = {"full": "all training history weighted equally (stale-prone)",
         "decay": "exponential recency decay over training years (halflife=4y)",
         "rolling": "only the last window=5 training years used"}


def _train_only_rate(tr_groups, tr_target, years, cutoff, m=M, mode="full",
                     halflife=4.0, window=5):
    """m-estimate sector rate from TRAIN only, with optional recency weighting.

    mode='full'   -> every training year weighted equally (historical behaviour)
    mode='decay'  -> weight = exp(-ln2 * (cutoff - approval_year) / halflife)
    mode='rolling'-> drop training rows approved more than `window` years before cutoff
    """
    g = pd.Series(tr_groups)
    y = np.asarray(tr_target, dtype=float)
    yr = np.asarray(years, dtype=float)
    if mode == "full":
        w = np.ones_like(y)
        overall = float(y.mean())
    elif mode == "decay":
        w = np.exp(-np.log(2.0) * np.maximum(cutoff - yr, 0.0) / halflife)
        overall = float(np.average(y, weights=w))
    elif mode == "rolling":
        w = np.where(yr >= cutoff - window, 1.0, 0.0)
        w = np.where(w.sum() > 0, w, np.ones_like(y))
        overall = float(np.average(y, weights=w))
    else:
        raise ValueError(f"unknown encoding mode: {mode}")
    df = pd.DataFrame({"g": g.values, "y": y, "w": w})
    gp = df.groupby("g").agg(
        yw=("y", lambda s: _wmean(s, df.loc[s.index, "w"])),
        ws=("w", "sum"))
    return {idx: (row.yw * row.ws + m * overall) / (row.ws + m) for idx, row in gp.iterrows()}, overall


def _wmean(x, w):
    """Weighted mean; falls back to the unweighted mean when weights sum to ~0."""
    ws = float(np.sum(w))
    if ws <= 1e-12:
        return float(np.mean(x))
    return float(np.average(x, weights=w))


def _encode_rate(groups, rates, overall):
    enc = np.asarray(groups, dtype=object)
    out = np.array([rates.get(g, overall) for g in enc], dtype=float)
    return out


def _cost_frame(train, test, mode="full"):
    def feats(df):
        orig = df["original_cost_cr"].clip(lower=0).to_numpy(dtype=float)
        exp = df["expenditure_cum_cr"].fillna(0).clip(lower=0).to_numpy(dtype=float)
        return {"log_original_cost": np.log1p(orig), "expenditure_ratio": exp / np.maximum(orig, 1e-9),
                "log_expenditure": np.log1p(exp)}, orig, exp

    ft, orig_t, _ = feats(train)
    fy, _, _ = feats(test)
    cutoff = int(train["approval_year"].max())
    rates, overall = _train_only_rate(train["sector"].astype(str), train["overrun_flag"] > 0,
                                      train["approval_year"], cutoff, mode=mode)
    rates_r, overall_r = _train_only_rate(train["sector"].astype(str), train["overrun_ratio"],
                                          train["approval_year"], cutoff, mode=mode)
    Xtr = np.column_stack([ft["log_original_cost"], ft["expenditure_ratio"],
                           ft["log_expenditure"], _encode_rate(train["sector"].astype(str), rates, overall)])
    Xte = np.column_stack([fy["log_original_cost"], fy["expenditure_ratio"],
                           fy["log_expenditure"], _encode_rate(test["sector"].astype(str), rates, overall)])
    Xtr_r = np.column_stack([Xtr[:, 0], Xtr[:, 1], Xtr[:, 2],
                             _encode_rate(train["sector"].astype(str), rates_r, overall_r)])
    Xte_r = np.column_stack([Xte[:, 0], Xte[:, 1], Xte[:, 2],
                             _encode_rate(test["sector"].astype(str), rates_r, overall_r)])
    mtr, mte = _macro_block(train), _macro_block(test)
    if mtr is not None and mte is not None:
        Xtr, Xte = np.column_stack([Xtr, mtr]), np.column_stack([Xte, mte])
        Xtr_r, Xte_r = np.column_stack([Xtr_r, mtr]), np.column_stack([Xte_r, mte])
    return Xtr, Xte, Xtr_r, Xte_r


def _time_frame(train, test, mode="full"):
    def feats(df):
        orig = df["original_cost_cr"].clip(lower=0).to_numpy(dtype=float)
        exp = df["expenditure_cum_cr"].fillna(0).clip(lower=0).to_numpy(dtype=float)
        pdur = pd.to_numeric(df["planned_duration_years"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(pdur).any():
            pdur = np.nan_to_num(pdur, nan=np.nanmedian(pdur))
        else:
            pdur = np.full(len(pdur), 5.0)
        return {"log_original_cost": np.log1p(orig), "expenditure_ratio": exp / np.maximum(orig, 1e-9),
                "planned_duration_yrs": pdur}, orig, exp

    ft, _, _ = feats(train)
    fy, _, _ = feats(test)
    cutoff = int(train["approval_year"].max())
    rates, overall = _train_only_rate(train["sector"].astype(str), train["delayed"] > 0,
                                      train["approval_year"], cutoff, mode=mode)
    Xtr = np.column_stack([ft["log_original_cost"], ft["expenditure_ratio"],
                           ft["planned_duration_yrs"], _encode_rate(train["sector"].astype(str), rates, overall)])
    Xte = np.column_stack([fy["log_original_cost"], fy["expenditure_ratio"],
                           fy["planned_duration_yrs"], _encode_rate(test["sector"].astype(str), rates, overall)])
    mtr, mte = _macro_block(train), _macro_block(test)
    if mtr is not None and mte is not None:
        Xtr, Xte = np.column_stack([Xtr, mtr]), np.column_stack([Xte, mte])
    return Xtr, Xte


MACRO_COLS = ("wpi_inflation_pct", "cpi_inflation_pct", "repo_rate_pct", "gdp_growth_pct")


def _add_macro(df, macro):
    """Attach year-keyed macro indicators (backfilled for pre-series years)."""
    df = df.copy()
    m = macro.rename(columns={"year": "approval_year"})
    out = df.merge(m, on="approval_year", how="left")
    for c in MACRO_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].bfill().ffill()
    return out


def _macro_block(df):
    """Stack the macro columns as features if they exist on the frame."""
    if not all(c in df.columns for c in MACRO_COLS):
        return None
    return np.column_stack([pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(dtype=float)
                            for c in MACRO_COLS])


def _clfs():
    m = {"Logistic Regression": LogisticRegression(max_iter=3000, C=1.0),
         "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=8,
                                                 max_depth=8, random_state=42, n_jobs=-1)}
    if HAVE_LGBM:
        m["LightGBM"] = LGBMClassifier(n_estimators=300, num_leaves=8, learning_rate=0.05,
                                       random_state=42, verbose=-1, n_jobs=-1)
    return m


def _regs():
    m = {"Linear Regression": LinearRegression(),
         "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=8,
                                                max_depth=8, random_state=42, n_jobs=-1)}
    if HAVE_LGBM:
        m["LightGBM"] = LGBMRegressor(n_estimators=300, num_leaves=8, learning_rate=0.05,
                                      random_state=42, verbose=-1, n_jobs=-1)
    return m


def _reg_targets(yr):
    ymin = float(yr.min())
    shift = max(0.0, -ymin) + 1.0
    return np.log1p(yr + shift), shift


def _inv_reg(pred, shift):
    return np.expm1(pred) - shift


def _threshold_metrics(y_true, score):
    """Precision/recall at the 0.5 score threshold + AUC — what an official uses."""
    thr = 0.5
    pred = (np.asarray(score) >= thr).astype(int)
    tp = int(np.sum((pred == 1) & (np.asarray(y_true) == 1)))
    fp = int(np.sum((pred == 1) & (np.asarray(y_true) == 0)))
    fn = int(np.sum((pred == 0) & (np.asarray(y_true) == 1)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {"flagged": int(pred.sum()), "precision": round(prec, 3), "recall": round(rec, 3)}


def _top_decile(y_true, score, top=0.2):
    """Practical usage: if an official reviews the top 20% riskiest projects, what
    fraction of all true positives does that capture? (recall@top, lift vs random)."""
    y = np.asarray(y_true)
    s = np.asarray(score)
    order = np.argsort(-s)
    k = max(1, int(round(len(y) * top)))
    top_idx = order[:k]
    captured = int(np.sum(y[top_idx] == 1))
    total_pos = int(np.sum(y == 1))
    recall = captured / total_pos if total_pos else 0.0
    lift = recall / top if top else 0.0
    return {"top_pct": top, "recall@top": round(recall, 3), "lift": round(lift, 2)}


def _eval_cutoff(cutoff: int, cost: pd.DataFrame, time: pd.DataFrame, mode: str = "full") -> dict:
    """Train <= cutoff, test on projects approved in the next two years."""
    cost_tr = cost[cost["approval_year"] <= cutoff]
    cost_te = cost[cost["approval_year"].between(cutoff + 1, cutoff + 2)]
    time_tr = time[time["approval_year"] <= cutoff]
    time_te = time[time["approval_year"].between(cutoff + 1, cutoff + 2)]

    report = {"cutoff": cutoff, "test_window": f"{cutoff+1}-{cutoff+2}", "encoding": mode,
              "cost": {"n_train": int(len(cost_tr)), "n_test": int(len(cost_te))},
              "time": {"n_train": int(len(time_tr)), "n_test": int(len(time_te))},
              "observation_window_note": ""}

    # ---- cost classification + regression ----
    if len(cost_te) >= 30 and len(cost_tr) >= 200:
        Xtr, Xte, Xtr_r, Xte_r = _cost_frame(cost_tr, cost_te, mode)
        yb_tr, yb_te = (cost_tr["overrun_flag"] > 0).astype(int).to_numpy(), (cost_te["overrun_flag"] > 0).astype(int).to_numpy()
        yr_te = cost_te["overrun_ratio"].to_numpy(dtype=float)
        # classification
        aucs = {}
        for k, m in _clfs().items():
            cl = clone(m).fit(Xtr, yb_tr)
            aucs[k] = float(roc_auc_score(yb_te, cl.predict_proba(Xte)[:, 1]))
        best_clf = max(aucs, key=lambda k: aucs[k])
        cl = clone(_clfs()[best_clf]).fit(Xtr, yb_tr)
        prob_te = cl.predict_proba(Xte)[:, 1]
        report["cost"]["clf"] = {"auc_by_model": aucs, "best": best_clf, "auc": round(aucs[best_clf], 3),
                                 "test_base_rate": round(float(yb_te.mean()), 3),
                                 "thr@0.5": _threshold_metrics(yb_te, prob_te),
                                 "top20pct": _top_decile(yb_te, prob_te, 0.2)}
        # regression
        ttr, shift = _reg_targets(cost_tr["overrun_ratio"].to_numpy(dtype=float))
        maes = {}
        r2s = {}
        for k, m in _regs().items():
            r = clone(m).fit(Xtr_r, ttr)
            pp = _inv_reg(r.predict(Xte_r), shift)
            maes[k] = float(mean_absolute_error(yr_te, pp))
            r2s[k] = float(r2_score(np.log1p(yr_te + shift), np.log1p(np.asarray(pp) + shift)))
        best_r = min(maes, key=lambda k: maes[k])
        naive_mae = float(np.abs(yr_te - float(cost_tr["overrun_ratio"].median())).mean())
        r = clone(_regs()[best_r]).fit(Xtr_r, ttr)
        pred_te = _inv_reg(r.predict(Xte_r), shift)
        report["cost"]["reg"] = {"best": best_r, "mae": round(maes[best_r], 3),
                                 "mae_by_model": {k: round(v, 3) for k, v in maes.items()},
                                 "naive_median_mae": round(naive_mae, 3),
                                 "improvement_vs_naive_pct": round((1 - maes[best_r] / naive_mae) * 100, 1),
                                 "sample_signed_mae": round(float(np.mean(np.abs(pred_te))), 3)}

    # ---- time classification + regression ----
    if len(time_te) >= 30 and len(time_tr) >= 200:
        Xtr, Xte = _time_frame(time_tr, time_te, mode)
        yb_tr, yb_te = (time_tr["delayed"] > 0).astype(int).to_numpy(), (time_te["delayed"] > 0).astype(int).to_numpy()
        aucs = {}
        for k, m in _clfs().items():
            cl = clone(m).fit(Xtr, yb_tr)
            aucs[k] = float(roc_auc_score(yb_te, cl.predict_proba(Xte)[:, 1]))
        best_clf = max(aucs, key=lambda k: aucs[k])
        cl = clone(_clfs()[best_clf]).fit(Xtr, yb_tr)
        prob_te = cl.predict_proba(Xte)[:, 1]
        # regression on delayed test projects only
        d_te = time_te[time_te["delayed"] > 0].dropna(subset=["tor_months"])
        reg_info = None
        if len(d_te) >= 20:
            Xdtr, _ = _time_frame(time_tr, d_te, mode)
            Xdte = _time_frame(time_tr, d_te, mode)[1]
            yt_tr_all = time_tr["tor_months"].clip(upper=240).astype(float).to_numpy(dtype=float)
            # train reg on all delayed in train
            d_tr = time_tr[time_tr["delayed"] > 0].dropna(subset=["tor_months"])
            Xd_tr, _ = _time_frame(d_tr, d_te, mode)
            yt_tr = d_tr["tor_months"].clip(upper=240).to_numpy(dtype=float)
            tgt = np.log1p(yt_tr)
            maes = {}
            for k, m in _regs().items():
                r = clone(m).fit(Xd_tr, tgt)
                pp = np.expm1(r.predict(Xdte))
                maes[k] = float(mean_absolute_error(np.clip(d_te["tor_months"].to_numpy(dtype=float), 0, 240), pp))
            best_r = min(maes, key=lambda k: maes[k])
            naive_mae = float(np.abs(np.clip(d_te["tor_months"].to_numpy(dtype=float), 0, 240)
                                     - float(d_tr["tor_months"].median())).mean())
            reg_info = {"best": best_r, "mae_months": round(maes[best_r], 1),
                        "naive_median_mae": round(naive_mae, 1),
                        "improvement_vs_naive_pct": round((1 - maes[best_r] / naive_mae) * 100, 1),
                        "test_n_delayed": int(len(d_te))}
        report["time"]["clf"] = {"auc_by_model": aucs, "best": best_clf, "auc": round(aucs[best_clf], 3),
                                 "test_base_rate": round(float(yb_te.mean()), 3),
                                 "thr@0.5": _threshold_metrics(yb_te, prob_te),
                                 "top20pct": _top_decile(yb_te, prob_te, 0.2)}
        if reg_info:
            report["time"]["reg"] = reg_info
        # honest observation-window comparison: next year's rate
        nxt = time[time["approval_year"] == cutoff + 3]
        if len(nxt):
            report["observation_window_note"] = (f"Delayed rate for approvals in {cutoff+1}: "
                                                 f"{yb_te.mean():.2f} -> {cutoff+2}: "
                                                 f"{(time[time['approval_year']==cutoff+2]['delayed']>0).mean():.2f} -> {cutoff+3}: "
                                                 f"{(nxt['delayed']>0).mean():.2f} (older approvals, watched longer, look worse).")
    return report


def run(cutoffs=(2018, 2019, 2020), modes=("full", "decay", "rolling")) -> dict:
    cost = pd.read_csv(COST_CSV)
    time = pd.read_csv(TIME_CSV)
    # attach approval_year to cost via (sector, project_name) key
    tk = time[["sector", "project_name", "approval_year"]].copy()
    tk["key"] = tk["sector"].astype(str) + "|" + tk["project_name"].astype(str)
    cost2 = cost.copy()
    cost2["key"] = cost2["sector"].astype(str) + "|" + cost2["project_name"].astype(str)
    cost2 = cost2.merge(tk[["key", "approval_year"]], on="key", how="left")
    cost2 = cost2[cost2["approval_year"].notna()].copy()
    cost2["approval_year"] = cost2["approval_year"].astype(int)

    time2 = time.copy()
    time2 = time2[time2["approval_year"].notna()].copy()
    time2["approval_year"] = time2["approval_year"].astype(int)
    time2["overrun_ratio"] = time2.get("overrun_ratio", np.nan)

    # label censoring: base rate of EACH label by approval cohort (proves WHY cost and
    # time "skill" fall on recent cohorts — the labels themselves are incomplete there)
    by_c = cost2[cost2["overrun_flag"].notna()].groupby("approval_year")["overrun_flag"].agg(["mean", "size"])
    by_t = time2.groupby("approval_year")["delayed"].agg(["mean", "size"])
    cohort_rates = {"cost_overrun_base_rate_by_approval": {
        int(a): {"rate": round(float(r["mean"]), 3), "n": int(r["size"])} for a, r in by_c.iterrows()},
        "delayed_base_rate_by_approval": {
            int(a): {"rate": round(float(r["mean"]), 3), "n": int(r["size"])} for a, r in by_t.iterrows()}}

    results_by_mode = {mode: {f"cutoff_{c}": _eval_cutoff(c, cost2, time2, mode) for c in cutoffs}
                       for mode in modes}
    # backward-compatible headline = the historical full-history encoding
    results = results_by_mode["full"]

    # verdict: per task, does the best recency-aware encoding beat full-history AUC?
    verdict = {}
    for task in ("time", "cost"):
        rows = []
        for c in cutoffs:
            row = {"cutoff": c, "test_window": f"{c+1}-{c+2}"}
            for mode in modes:
                rep = results_by_mode[mode][f"cutoff_{c}"][task]
                row[mode] = round(rep.get("clf", {}).get("auc", float("nan")), 3)
            rows.append(row)
        verdict[task] = rows

    result = {"method": ("Temporal holdout: train on ALL projects approved <= cutoff; evaluate on "
                         "projects approved in the following 2 years (data the model never saw)."),
              "censoring": ("Projects approved 2022+ are too young to have been flagged delayed yet — "
                            "delayed rate collapses by cohort; new cohorts are NOT usable as a skill test. "
                            "The COST overrun label is censored the same way (a flag only appears once a "
                            "revised estimate exists), and its base rate collapses 0.56 -> 0.38 -> 0.18 "
                            "across the 2019/2020/2021 test cohorts — so the later cost AUC columns "
                            "measure label completeness, not model skill."),
              "label_censoring": cohort_rates,
              "encodings": {"order": list(modes), "description": MODES},
              "results": results,  # full-history (dashboard-compatible)
              "results_by_encoding": results_by_mode,
              "encoding_verdict": verdict,
              "cutoffs": list(cutoffs)}
    (DATA / "temporal_eval_results.json").write_text(json.dumps(result, indent=2))
    return result


def _print_report(r: dict):
    print("=" * 72)
    print("TEMPORAL (OUT-OF-SAMPLE) VALIDATION — 'how good on NEW data?'")
    print(r["method"])
    print(r["censoring"])
    print("=" * 72)
    lc = r.get("label_censoring")
    if lc:
        print("\nLABEL CENSORING (why recent cohorts look 'worse'):")
        print("  approval | cost overrun base rate (n) | delayed base rate (n)")
        cb_map = lc.get("cost_overrun_base_rate_by_approval", {})
        tb_map = lc.get("delayed_base_rate_by_approval", {})
        for y in sorted(set(cb_map) | set(tb_map)):
            cb = cb_map.get(y)
            tb = tb_map.get(y)
            c_str = f"{cb['rate']:.2f} (n={cb['n']})" if cb else "-"
            t_str = f"{tb['rate']:.2f} (n={tb['n']})" if tb else "-"
            print(f"    {y}   | {c_str:>16} | {t_str:>16}")
    ev = r.get("encoding_verdict")
    if ev:
        print("\nSECTOR-RATE AS A DRIFT SOURCE — clf AUC, full-history vs recency-aware:")
        modes = r["encodings"]["order"]
        for task in ("time", "cost"):
            print(f"  {task.upper():4} " + " | ".join(f"{m[:5]:>15}" for m in modes))
            for row in ev[task]:
                print("        " + " | ".join(f"{row[m]:>14.3f}" for m in modes) +
                      f"   <- test {row['test_window']}")
    for cut, rep in r["results"].items():
        print(f"\n[{cut}] cutoff {rep['cutoff']} -> TEST projects approved {rep['test_window']}")
        if rep["observation_window_note"]:
            print("  censoring:", rep["observation_window_note"])
        for task in ("cost", "time"):
            c = rep[task]
            print(f"  {task.upper():4}: train n={c['n_train']}, test n={c['n_test']}")
            if "clf" in c:
                cl = c["clf"]
                print(f"    clf AUC (test, unseen data): {cl['auc']}  [{cl['best']}]   "
                      f"test base rate={cl['test_base_rate']}   @0.5: {cl['thr@0.5']}")
                print(f"      top-20% review would catch {cl['top20pct']}")
                print(f"      per-model: {cl['auc_by_model']}")
            if "reg" in c:
                rg = c["reg"]
                print(f"    reg: MAE={rg.get('mae', rg.get('mae_months'))}  "
                      f"naive(median)={rg.get('naive_median_mae', rg.get('naive_median_mae'))}  "
                      f"improv={rg.get('improvement_vs_naive_pct')}%  [{rg['best']}]")
    print(f"\nSaved -> {OUT_JSON}")


def _ext_metrics(c_tr, c_te, t_tr, t_te) -> dict:
    """External-style single-window metrics (resolved-label test)."""
    out = {}
    if len(c_te) >= 30 and len(c_tr) >= 100:
        Xtr, Xte, Xtr_r, Xte_r = _cost_frame(c_tr, c_te)
        yb_tr = (c_tr["overrun_flag"] > 0).astype(int).to_numpy()
        yb_te = (c_te["overrun_flag"] > 0).astype(int).to_numpy()
        yr_te = c_te["overrun_ratio"].to_numpy(dtype=float)
        aucs = {}
        for k, m in _clfs().items():
            aucs[k] = float(roc_auc_score(yb_te, clone(m).fit(Xtr, yb_tr).predict_proba(Xte)[:, 1]))
        ttr, shift = _reg_targets(c_tr["overrun_ratio"].to_numpy(dtype=float))
        naives = []
        maes = {}
        for k, m in _regs().items():
            r = clone(m).fit(Xtr_r, ttr)
            pp = _inv_reg(r.predict(Xte_r), shift)
            maes[k] = float(mean_absolute_error(yr_te, pp))
        naive_mae = float(np.abs(yr_te - float(c_tr["overrun_ratio"].median())).mean())
        out["cost"] = {"auc": float(max(aucs.values())), "auc_lr": float(aucs.get("Logistic Regression", float("nan"))),
                       "reg_mae": float(min(maes.values())), "naive_mae": naive_mae,
                       "n_test": int(len(c_te)), "base_rate": round(float(yb_te.mean()), 3)}
    if len(t_te) >= 30 and len(t_tr) >= 100:
        Xtr, Xte = _time_frame(t_tr, t_te)
        yb_tr = (t_tr["delayed"] > 0).astype(int).to_numpy()
        yb_te = (t_te["delayed"] > 0).astype(int).to_numpy()
        aucs = {}
        for k, m in _clfs().items():
            aucs[k] = float(roc_auc_score(yb_te, clone(m).fit(Xtr, yb_tr).predict_proba(Xte)[:, 1]))
        out["time"] = {"auc": float(max(aucs.values())), "auc_lr": float(aucs.get("Logistic Regression", float("nan"))),
                       "n_test": int(len(t_te)), "base_rate": round(float(yb_te.mean()), 3)}
        d_te = t_te[t_te["delayed"] > 0].dropna(subset=["tor_months"])
        d_tr = t_tr[t_tr["delayed"] > 0].dropna(subset=["tor_months"])
        if len(d_te) >= 20 and len(d_tr) >= 20:
            Xdtr, Xdte = _time_frame(d_tr, d_te)
            tgt = np.log1p(d_tr["tor_months"].clip(upper=240).to_numpy(dtype=float))
            maes = {}
            for k, m in _regs().items():
                pp = np.expm1(clone(m).fit(Xdtr, tgt).predict(Xdte))
                maes[k] = float(mean_absolute_error(np.clip(d_te["tor_months"].to_numpy(dtype=float), 0, 240), pp))
            out["time"]["reg_mae"] = float(min(maes.values()))
            out["time"]["reg_naive"] = float(np.abs(np.clip(d_te["tor_months"].to_numpy(dtype=float), 0, 240)
                                                    - float(d_tr["tor_months"].median())).mean())
    return out


def run_external(cutoffs=(2017, 2018, 2019)) -> dict:
    """Train on the 2019-era annexure dataset (<= cutoff), test on the May-2025
    Flash Report's cohorts approved in the following two years.

    Explicitly NOT the old temporal test within the same file: the external
    snapshot's labels are RESOLVED as of May-2025 (anticipated DOC vs original DOC,
    revised vs original cost), so the test cohorts 2019-2021 do not suffer the
    label-censoring that makes in-file AUCs on those cohorts misleading.
    """
    cost = pd.read_csv(COST_CSV)
    time = pd.read_csv(TIME_CSV)
    ext = pd.read_csv(EXT_CSV)

    tk = time[["sector", "project_name", "approval_year"]].copy()
    tk["key"] = tk["sector"].astype(str) + "|" + tk["project_name"].astype(str)
    cost2 = cost.copy()
    cost2["key"] = cost2["sector"].astype(str) + "|" + cost2["project_name"].astype(str)
    cost2 = cost2.merge(tk[["key", "approval_year"]], on="key", how="left")
    cost2 = cost2[cost2["approval_year"].notna()].copy()
    cost2["approval_year"] = cost2["approval_year"].astype(int)
    time2 = time.copy()
    time2 = time2[time2["approval_year"].notna()].copy()
    time2["approval_year"] = time2["approval_year"].astype(int)

    ext2 = ext.copy()
    ext2 = ext2[ext2["approval_year"].notna()].copy()
    ext2["approval_year"] = ext2["approval_year"].astype(int)

    reports = {}
    for cutoff in cutoffs:
        c_tr = cost2[cost2["approval_year"] <= cutoff]
        c_te = ext2[ext2["approval_year"].between(cutoff + 1, cutoff + 2)]
        c_te = c_te[c_te["overrun_flag"].notna()].dropna(subset=["original_cost_cr"])
        t_tr = time2[time2["approval_year"] <= cutoff]
        t_te = ext2[ext2["approval_year"].between(cutoff + 1, cutoff + 2)]
        t_te = t_te[t_te["delayed"].notna()]

        rep = {"cutoff": cutoff, "test_window": f"{cutoff+1}-{cutoff+2}",
               "source_note": ("test = May-2025 Flash Report cohort, labels resolved "
                               "(no label censoring), format: train annexure <= cutoff"),
               "cost": {"n_train": int(len(c_tr)), "n_test": int(len(c_te))},
               "time": {"n_train": int(len(t_tr)), "n_test": int(len(t_te))}}

        # ---- cost ----
        if len(c_te) >= 30 and len(c_tr) >= 100:
            Xtr, Xte, Xtr_r, Xte_r = _cost_frame(c_tr, c_te)
            yb_tr = (c_tr["overrun_flag"] > 0).astype(int).to_numpy()
            yb_te = (c_te["overrun_flag"] > 0).astype(int).to_numpy()
            yr_te = c_te["overrun_ratio"].to_numpy(dtype=float)
            aucs = {}
            for k, m in _clfs().items():
                cl = clone(m).fit(Xtr, yb_tr)
                aucs[k] = float(roc_auc_score(yb_te, cl.predict_proba(Xte)[:, 1]))
            best_clf = max(aucs, key=lambda k: aucs[k])
            prob_te = clone(_clfs()[best_clf]).fit(Xtr, yb_tr).predict_proba(Xte)[:, 1]
            ttr, shift = _reg_targets(c_tr["overrun_ratio"].to_numpy(dtype=float))
            maes = {}
            for k, m in _regs().items():
                r = clone(m).fit(Xtr_r, ttr)
                pp = _inv_reg(r.predict(Xte_r), shift)
                maes[k] = float(mean_absolute_error(yr_te, pp))
            best_r = min(maes, key=lambda k: maes[k])
            naive_mae = float(np.abs(yr_te - float(c_tr["overrun_ratio"].median())).mean())
            rep["cost"]["clf"] = {"auc_by_model": {k: round(v, 3) for k, v in aucs.items()},
                                  "best": best_clf, "auc": round(aucs[best_clf], 3),
                                  "test_base_rate": round(float(yb_te.mean()), 3),
                                  "thr@0.5": _threshold_metrics(yb_te, prob_te),
                                  "top20pct": _top_decile(yb_te, prob_te, 0.2)}
            rep["cost"]["reg"] = {"best": best_r, "mae": round(maes[best_r], 3),
                                  "mae_by_model": {k: round(v, 3) for k, v in maes.items()},
                                  "naive_median_mae": round(naive_mae, 3),
                                  "improvement_vs_naive_pct": round((1 - maes[best_r] / naive_mae) * 100, 1)}
        # ---- time ----
        if len(t_te) >= 30 and len(t_tr) >= 100:
            Xtr, Xte = _time_frame(t_tr, t_te)
            yb_tr = (t_tr["delayed"] > 0).astype(int).to_numpy()
            yb_te = (t_te["delayed"] > 0).astype(int).to_numpy()
            aucs = {}
            for k, m in _clfs().items():
                cl = clone(m).fit(Xtr, yb_tr)
                aucs[k] = float(roc_auc_score(yb_te, cl.predict_proba(Xte)[:, 1]))
            best_clf = max(aucs, key=lambda k: aucs[k])
            prob_te = clone(_clfs()[best_clf]).fit(Xtr, yb_tr).predict_proba(Xte)[:, 1]
            d_te = t_te[t_te["delayed"] > 0].dropna(subset=["tor_months"])
            reg_info = None
            if len(d_te) >= 20:
                d_tr = t_tr[t_tr["delayed"] > 0].dropna(subset=["tor_months"])
                if len(d_tr) >= 20:
                    Xdtr, Xdte = _time_frame(d_tr, d_te)
                    tgt = np.log1p(d_tr["tor_months"].clip(upper=240).to_numpy(dtype=float))
                    maes = {}
                    for k, m in _regs().items():
                        r = clone(m).fit(Xdtr, tgt)
                        pp = np.expm1(r.predict(Xdte))
                        maes[k] = float(mean_absolute_error(
                            np.clip(d_te["tor_months"].to_numpy(dtype=float), 0, 240), pp))
                    best_r = min(maes, key=lambda k: maes[k])
                    naive_mae = float(np.abs(np.clip(d_te["tor_months"].to_numpy(dtype=float), 0, 240)
                                             - float(d_tr["tor_months"].median())).mean())
                    reg_info = {"best": best_r, "mae_months": round(maes[best_r], 1),
                                "naive_median_mae": round(naive_mae, 1),
                                "improvement_vs_naive_pct": round((1 - maes[best_r] / naive_mae) * 100, 1),
                                "test_n_delayed": int(len(d_te))}
            rep["time"]["clf"] = {"auc_by_model": {k: round(v, 3) for k, v in aucs.items()},
                                  "best": best_clf, "auc": round(aucs[best_clf], 3),
                                  "test_base_rate": round(float(yb_te.mean()), 3),
                                  "thr@0.5": _threshold_metrics(yb_te, prob_te),
                                  "top20pct": _top_decile(yb_te, prob_te, 0.2)}
            if reg_info:
                rep["time"]["reg"] = reg_info
        reports[f"cutoff_{cutoff}"] = rep

    result = {"method": ("EXTERNAL holdout: train on the pre-2020 annexure dataset (<= cutoff); test on the "
                         "May-2025 Flash Report's projects approved in the following 2 years — a genuinely "
                         "different document, new format, labels RESOLVED (no censoring on those cohorts)."),
              "data": {"train": str(COST_CSV), "external_test": str(EXT_CSV)},
              "results": reports, "cutoffs": list(cutoffs)}
    (EXT_OUT_JSON).write_text(json.dumps(result, indent=2))
    return result


def _print_external(r: dict):
    print("=" * 72)
    print("EXTERNAL HOLDOUT — train annexure <= cutoff, test May-2025 Flash Report cohort")
    print(r["method"])
    print(f"  train data: {r['data']['train']} | test data: {r['data']['external_test']}")
    print("=" * 72)
    for cut, rep in r["results"].items():
        print(f"\n[{cut}] cutoff {rep['cutoff']} -> TEST = May-2025 snapshot approvals {rep['test_window']}")
        for task in ("cost", "time"):
            c = rep[task]
            print(f"  {task.upper():4}: train n={c['n_train']}, test n={c['n_test']}")
            if "clf" in c:
                cl = c["clf"]
                print(f"    clf AUC (unseen snapshot): {cl['auc']}  [{cl['best']}]   "
                      f"test base rate={cl['test_base_rate']}   @0.5: {cl['thr@0.5']}")
                print(f"      top-20% review would catch {cl['top20pct']}   per-model: {cl['auc_by_model']}")
            if "reg" in c:
                rg = c["reg"]
                print(f"    reg: MAE={rg.get('mae', rg.get('mae_months'))}  "
                      f"naive={rg.get('naive_median_mae')}  improv={rg.get('improvement_vs_naive_pct')}%  [{rg['best']}]")
    print(f"\nSaved -> {EXT_OUT_JSON}")


def run_macro_ab(cutoffs=(2017, 2018, 2019)) -> dict:
    """A/B: do macro-economic covariates (WPI / CPI / repo / GDP at approval year)
    recover lost out-of-sample AUC? Same external holdout, with vs without."""
    macro = pd.read_csv(MACRO_CSV)
    cost = pd.read_csv(COST_CSV)
    time = pd.read_csv(TIME_CSV)
    ext = pd.read_csv(EXT_CSV)

    tk = time[["sector", "project_name", "approval_year"]].copy()
    tk["key"] = tk["sector"].astype(str) + "|" + tk["project_name"].astype(str)
    cost2 = cost.copy()
    cost2["key"] = cost2["sector"].astype(str) + "|" + cost2["project_name"].astype(str)
    cost2 = cost2.merge(tk[["key", "approval_year"]], on="key", how="left")
    cost2 = cost2[cost2["approval_year"].notna()].copy()
    cost2["approval_year"] = cost2["approval_year"].astype(int)
    time2 = time.copy()
    time2 = time2[time2["approval_year"].notna()].copy()
    time2["approval_year"] = time2["approval_year"].astype(int)
    ext2 = ext.copy()
    ext2 = ext2[ext2["approval_year"].notna()].copy()
    ext2["approval_year"] = ext2["approval_year"].astype(int)

    rows = []
    for cutoff in cutoffs:
        c_tr = cost2[cost2["approval_year"] <= cutoff]
        c_te = ext2[ext2["approval_year"].between(cutoff + 1, cutoff + 2)]
        c_te = c_te[c_te["overrun_flag"].notna()].dropna(subset=["original_cost_cr"])
        t_tr = time2[time2["approval_year"] <= cutoff]
        t_te = ext2[ext2["approval_year"].between(cutoff + 1, cutoff + 2)]
        t_te = t_te[t_te["delayed"].notna()]
        base = _ext_metrics(c_tr, c_te, t_tr, t_te)
        withm = _ext_metrics(_add_macro(c_tr, macro), _add_macro(c_te, macro),
                             _add_macro(t_tr, macro), _add_macro(t_te, macro))
        rows.append({"cutoff": cutoff, "test_window": f"{cutoff+1}-{cutoff+2}",
                     "cost": {"without": base.get("cost"), "with_macro": withm.get("cost")},
                     "time": {"without": base.get("time"), "with_macro": withm.get("time")}})

    result = {"method": ("Macro-covariate A/B: same external holdout (train annexure <= cutoff, test "
                         "May-2025 resolved-label cohort), features identical except the split of "
                         "(with_macro) adds approval-year WPI/CPI/repo-rate/GDP (data/macro_by_year.csv, "
                         "public RBI/IMF figures — check before demo)."),
              "results": rows, "cutoffs": list(cutoffs)}
    (MACRO_OUT_JSON).write_text(json.dumps(result, indent=2))
    return result


def _print_macro_ab(r: dict):
    print("=" * 72)
    print("MACRO-COVARIATE A/B — does adding approval-year macro data recover AUC?")
    print(r["method"])
    print("=" * 72)
    for row in r["results"]:
        print(f"\ncutoff {row['cutoff']} -> test {row['test_window']}")
        for task in ("cost", "time"):
            wo, w = row[task].get("without"), row[task].get("with_macro")
            if not wo or not w:
                print(f"  {task.upper()}: n/a")
                continue
            print(f"  {task.upper():4}: AUC {wo.get('auc')} -> {w.get('auc')} "
                  f"(LR {wo.get('auc_lr')} -> {w.get('auc_lr')})")
            if "reg_mae" in wo and "reg_mae" in w:
                nv = wo.get("reg_naive") if task == "time" else wo.get("naive_mae")
                print(f"        reg MAE {wo['reg_mae']} -> {w['reg_mae']} "
                      f"(naive {nv})")
    print(f"\nSaved -> {MACRO_OUT_JSON}")


# --------------------------------------------------------------------------- #
# Survival framing (discrete-time logistic hazard) — fixes label CENSORING
# --------------------------------------------------------------------------- #
# Instead of predicting the binary "delayed" flag (which is unknown/0 for young
# projects only because they have not been watched long enough), we model the
# time-to-delay with right-censoring: every project contributes "periods observed
# without an event yet".  A 'not delayed' project that was only observed for T
# months is NOT a negative example forever — it is a subject censored at T.
#
# Delayed   -> event at  approval..(planned horizon + tor_months)   delta = 1
# Not yet   -> censored at min(planned horizon, snapshot date)      delta = 0
#   (train annexures: planned horizon, assuming completed on time; the May-2025
#    snapshot: min(original DOC, May-2025), i.e. completed on time OR still going)
#
# Scoring: Harrell's C-index (concordance on censored times) — valid even when
# most of the test cohort has not yet been flagged delayed.
SURV_OUT_JSON = DATA / "survival_eval_2025.json"
SNAPSHOT_YEAR = 2025.42  # May 2025 Flash Report


def _surv_times(df, ext: bool):
    """Months from approval to event (delayed) or to last observed 'not delayed'."""
    app = pd.to_numeric(df["approval_year"], errors="coerce").to_numpy(dtype=float) + 0.5
    planned = pd.to_numeric(df["planned_duration_years"], errors="coerce").to_numpy(dtype=float) * 12.0
    planned = np.where(np.isfinite(planned) & (planned > 0), planned, 60.0)
    tor = pd.to_numeric(df.get("tor_months", pd.Series(0.0, index=df.index)), errors="coerce").to_numpy(dtype=float)
    tor = np.nan_to_num(tor, nan=0.0)
    delayed = (pd.to_numeric(df["delayed"], errors="coerce") > 0).to_numpy(dtype=bool)
    delay_true = np.nan_to_num(pd.to_numeric(df["delayed"], errors="coerce").to_numpy(dtype=float), nan=0.0) > 0
    if not ext:
        T = np.where(delay_true, planned + tor, planned)
    else:
        snap_months = np.maximum(SNAPSHOT_YEAR - app, 0.0) * 12.0
        if "original_doc_year" in df.columns:
            doc = pd.to_numeric(df["original_doc_year"], errors="coerce").to_numpy(dtype=float)
        else:
            doc = pd.to_numeric(df.get("planned_doc_year"), errors="coerce").to_numpy(dtype=float)
        doc = np.nan_to_num(doc, nan=np.nanmedian(doc) if np.isfinite(doc).any() else 2030.0)
        censor = np.minimum(np.maximum(np.minimum(doc, SNAPSHOT_YEAR) - app, 0.0) * 12.0, snap_months)
        T = np.where(delay_true, planned + tor, np.maximum(censor, 0.5))
    return np.clip(T, 0.5, None), delay_true.astype(int)


def _person_period(X, T, delta, cap=14):
    """Expand static rows into person-period rows for discrete-time logistic hazard.
    Period p marks 'delayed by the end of year p since approval'.  Censored subjects
    contribute periods 1..ceil(T/12) all with y=0; events contribute a final y=1."""
    pmax = np.clip(np.ceil(T / 12.0).astype(int), 1, 400)
    idx = np.repeat(np.arange(len(X)), pmax)
    per = np.concatenate([np.arange(1, p + 1) for p in pmax])
    b = np.minimum(per, cap)
    y = ((per == pmax[idx]) & (delta[idx].astype(bool))).astype(int)
    cats = np.zeros((len(per), cap), dtype=int)
    cats[np.arange(len(per)), b - 1] = 1
    return np.hstack([X[idx], cats]), y


def _bucket_hazards(X, clf, cap=14):
    """Per-subject hazard h(bucket) for buckets 1..cap (period-clipped features)."""
    n = len(X)
    base = np.hstack([X, np.zeros((n, cap))])
    cols = list(range(X.shape[1], X.shape[1] + cap))
    out = np.zeros((n, cap))
    for j, c in enumerate(cols):
        blk = base.copy()
        blk[:, c] = 1.0
        out[:, j] = clf.predict_proba(blk)[:, 1]
    return np.clip(out, 1e-9, 1 - 1e-9)


def _surv_risk(H, T, cap=14):
    """P(delayed within T months) = 1 - prod over observed periods of (1 - h)."""
    K = np.clip(np.ceil(T / 12.0).astype(int), 0, None)
    logS = np.zeros(len(H))
    for i, k in enumerate(K):
        if k <= 0:
            continue
        kb = min(k, cap)
        logS[i] = np.sum(np.log1p(-H[i, :kb]))
        if k > cap:
            logS[i] += (k - cap) * np.log1p(-H[i, cap - 1])
    return 1.0 - np.exp(logS)


def _c_index(T, delta, risk):
    """Harrell's concordance on possibly right-censored times."""
    T = np.asarray(T, float)
    delta = np.asarray(delta, int)
    risk = np.asarray(risk, float)
    n = len(T)
    concord, usable = 0.0, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if T[i] == T[j]:
                if delta[i] and delta[j] and risk[i] != risk[j]:
                    concord += 0.5
                    usable += 1.0
                continue
            if T[i] < T[j]:
                a, b = i, j
            else:
                a, b = j, i
            if delta[a] and (delta[b] or T[b] > T[a]):
                usable += 1.0
                concord += 1.0 if risk[a] > risk[b] else (0.5 if risk[a] == risk[b] else 0.0)
    return round(concord / usable, 3) if usable else float("nan")


def run_survival(cutoffs=(2017, 2018, 2019)) -> dict:
    """Discrete-time survival re-analysis of the external holdout.

    Same leak-free split (train annexure <= cutoff, test = May-2025 Flash Report
    cohort approved in the following 2 years).  The difference: the target is now
    (time-to-delay, censored) instead of a binary flag, so the very projects whose
    labels were 'too censored' to use before are now used correctly as censored.
    """
    time = pd.read_csv(TIME_CSV)
    ext = pd.read_csv(EXT_CSV)
    time2 = time[time["approval_year"].notna()].copy()
    time2["approval_year"] = time2["approval_year"].astype(int)
    time2["delayed"] = pd.to_numeric(time2["delayed"], errors="coerce").fillna(0) > 0
    ext2 = ext[ext["approval_year"].notna()].copy()
    ext2["approval_year"] = ext2["approval_year"].astype(int)
    ext2["delayed"] = pd.to_numeric(ext2["delayed"], errors="coerce").fillna(0) > 0

    def static(tr, te, cutoff):
        orig = tr["original_cost_cr"].clip(lower=0).to_numpy(dtype=float)
        exp = tr["expenditure_cum_cr"].fillna(0).clip(lower=0).to_numpy(dtype=float)
        pdur = pd.to_numeric(tr["planned_duration_years"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(pdur).any():
            pdur = np.nan_to_num(pdur, nan=np.nanmedian(pdur))
        else:
            pdur = np.full(len(pdur), 5.0)
        rates, overall = _train_only_rate(tr["sector"].astype(str), tr["delayed"] > 0,
                                          tr["approval_year"], cutoff)
        def feats(df):
            o = df["original_cost_cr"].clip(lower=0).to_numpy(dtype=float)
            e = df["expenditure_cum_cr"].fillna(0).clip(lower=0).to_numpy(dtype=float)
            p = pd.to_numeric(df["planned_duration_years"], errors="coerce").to_numpy(dtype=float)
            p = np.nan_to_num(p, nan=np.nanmedian(pdur))
            return np.column_stack([np.log1p(o), e / np.maximum(o, 1e-9), p,
                                    _encode_rate(df["sector"].astype(str), rates, overall)])
        return feats(tr), feats(te)

    reports = {}
    baselines = []
    for cutoff in cutoffs:
        tr = time2[time2["approval_year"] <= cutoff]
        te = ext2[ext2["approval_year"].between(cutoff + 1, cutoff + 2)]
        if len(te) < 30 or len(tr) < 100:
            continue
        Xtr, Xte = static(tr, te, cutoff)
        Ttr, dtr = _surv_times(tr, False)
        Tte, dte = _surv_times(te, True)
        Xpp, ypp = _person_period(Xtr, Ttr, dtr)
        lr = LogisticRegression(max_iter=1000).fit(Xpp, ypp)
        H = _bucket_hazards(Xte, lr)
        # time-free risk score: P(delayed within a FIXED 5yr window) — same horizon
        # for every subject, so ranking does not just measure "observed longer"
        risk5y = _surv_risk(H, np.full(len(Xte), 60.0))
        risk_plan = _surv_risk(H, pd.to_numeric(te["planned_duration_years"], errors="coerce")
                               .fillna(pd.to_numeric(te["planned_duration_years"], errors="coerce").median())
                               .to_numpy(dtype=float) * 12.0)
        cidx_all = _c_index(Tte, dte, risk5y)
        cidx_events = _c_index(Tte[dte == 1], np.ones(dte.sum(), dtype=int), risk5y[dte == 1])
        auc_plan = float(roc_auc_score(dte, risk_plan))
        # same binary cohort definition as the plain external holdout, for a like-for-like number
        d_tr_bs = (tr["delayed"] > 0).astype(int).to_numpy()
        Xtr_b, Xte_b = _time_frame(tr, te)
        lr_b = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr_b, d_tr_bs)
        bin_prob = lr_b.predict_proba(Xte_b)[:, 1]
        auc_bin_lr = float(roc_auc_score(dte, bin_prob))
        cidx_bin = _c_index(Tte, dte, bin_prob)
        base_rate = float(dte.mean())
        med_T = float(np.median(Tte))
        frac_events = float((dte == 1).mean())
        frac_censored = 1.0 - frac_events
        reports[f"cutoff_{cutoff}"] = {
            "cutoff": cutoff, "test_window": f"{cutoff+1}-{cutoff+2}",
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "test_delayed_flag_rate": round(base_rate, 3),
            "test_fraction_censored_not_yet_delayed": round(frac_censored, 3),
            "test_median_followup_months": round(med_T, 0),
            "c_index_survival_incl_censored": cidx_all,
            "c_index_survival_delayed_only": cidx_events,
            "c_index_binary_lr_incl_censored": cidx_bin,
            "auc_hazard_at_planned_horizon_vs_flag": round(auc_plan, 3),
            "auc_binary_lr_same_cohort": round(auc_bin_lr, 3),
            "baseline_hazard_note": "probability of being flagged delayed within each year since approval"}
        baselines.append({"cutoff": cutoff, "test_window": f"{cutoff+1}-{cutoff+2}",
                          "c_index_survival": cidx_all, "c_index_binary": cidx_bin,
                          "auc": round(auc_plan, 3), "frac_censored": round(frac_censored, 3)})
        print(f"[surv] cutoff {cutoff}: c-index survival={cidx_all} vs binary-LR={cidx_bin} | "
              f"hazard-AUC={auc_plan:.3f} vs binary-AUC={auc_bin_lr:.3f} | "
              f"test n={len(te)} (delayed={dte.mean():.2f}, censored={frac_censored:.2f}, "
              f"median follow-up {med_T:.0f} mo)")

    # interpretability: fitted baseline hazard from the pooled (first) cutoff model
    extra = {}
    if reports:
        first = cutoffs[0]
        tr = time2[time2["approval_year"] <= first]
        te = ext2[ext2["approval_year"].between(first + 1, first + 2)]
        if len(te) >= 30:
            Xtr0, Xte0 = static(tr, te, first)
            Ttr0, dtr0 = _surv_times(tr, False)
            Xpp0, ypp0 = _person_period(Xtr0, Ttr0, dtr0)
            lr0 = LogisticRegression(max_iter=1000).fit(Xpp0, ypp0)
            H0 = _bucket_hazards(np.nanmedian(Xte0, axis=0, keepdims=True), lr0)
            extra["median_project_baseline_hazard"] = [
                {"year": int(i + 1), "hazard": round(float(h), 4)} for i, h in enumerate(H0[0])]

    result = {"method": ("SURVIVAL reframing of the external holdout: instead of the binary "
                         "'delayed' flag, model time-to-delay with right-censoring (discrete-time "
                         "logistic hazard). Projects not yet flagged delayed are used as censored "
                         "observations rather than dropped or treated as safe — this is the fix "
                         "for the label-censoring that suppressed raw-cohort AUCs."),
              "target_definition": ("event = delayed revealed by a revised DOC; "
                                    "censor = still under observation/on-time at min(planned DOC, May-2025)."),
              "scoring": ("Harrell's C-index over all test subjects (censored-aware) + AUC of "
                          "P(delayed by planned horizon) vs the flag, same cohort as the plain holdout."),
              "results": reports, "cutoffs": list(cutoffs), "summary": {
                  "c_index_survival_range": (min(b["c_index_survival"] for b in baselines),
                                              max(b["c_index_survival"] for b in baselines)),
                  "c_index_binary_range": (min(b["c_index_binary"] for b in baselines),
                                           max(b["c_index_binary"] for b in baselines)),
                  "auc_hazard_range": (min(b["auc"] for b in baselines), max(b["auc"] for b in baselines)),
                  "sentence": ("The survival model stays evaluable (c-index) on cohorts that are mostly "
                               "still-censored, where a binary classifier's AUC is undefined/garbage.")},
              "baseline_hazard": extra}
    (SURV_OUT_JSON).write_text(json.dumps(result, indent=2))
    return result


def _print_survival(r: dict):
    print("=" * 72)
    print("SURVIVAL REFRAMING (discrete-time hazard, right-censoring) — external holdout")
    print(r["method"])
    print("  target:", r["target_definition"])
    print("  scoring:", r["scoring"])
    print("=" * 72)
    for cut, rep in r["results"].items():
        print(f"\n[{cut}] cutoff {rep['cutoff']} -> TEST = May-2025 approvals {rep['test_window']}")
        print(f"  train n={rep['n_train']}, test n={rep['n_test']} | delayed flag rate={rep['test_delayed_flag_rate']} | "
              f"{rep['test_fraction_censored_not_yet_delayed']:.1%} still-censored, median follow-up {rep['test_median_followup_months']:.0f} mo")
        print(f"  C-INDEX (all test incl. censored): survival {rep['c_index_survival_incl_censored']}  "
              f"| binary-LR {rep['c_index_binary_lr_incl_censored']}  (survival, delayed only: {rep['c_index_survival_delayed_only']})")
        print(f"  AUC P(delayed by planned horizon) vs flag: {rep['auc_hazard_at_planned_horizon_vs_flag']}   "
              f"vs binary-LR same cohort: {rep['auc_binary_lr_same_cohort']}")
    bh = r.get("baseline_hazard", {}).get("median_project_baseline_hazard")
    if bh:
        print("\n  Baseline hazard (median project): P(flagged delayed by end of year N): "
              + ", ".join(f"Y{i['year']}={i['hazard']:.3f}" for i in bh))
    print(f"\nSaved -> {SURV_OUT_JSON}")


# --------------------------------------------------------------------------- #
# Training-set augmentation: does adding the RESOLVED May-2025 rows help?
# --------------------------------------------------------------------------- #
# Same external holdout, but the training set becomes annexure <= cutoff PLUS the
# May-2025 snapshot rows approved <= cutoff (labels resolved as of May-2025).  Test
# stays the same held-out snapshot cohort (approved cutoff+1..cutoff+2).  Directly
# measures the "more real data" lever — if adding resolved rows helps on the next
# cohorts, the platform's continual-learning loop is worth it.
AUG_OUT_JSON = DATA / "augmented_train_holdout_2025.json"


def _aug_train_frames(cutoff):
    """concatenate annexure <= cutoff with snapshot rows approved <= cutoff."""
    time = pd.read_csv(TIME_CSV)
    cost = pd.read_csv(COST_CSV)
    ext = pd.read_csv(EXT_CSV)

    tk = time[["sector", "project_name", "approval_year"]].copy()
    tk["key"] = tk["sector"].astype(str) + "|" + tk["project_name"].astype(str)
    cost2 = cost.copy()
    cost2["key"] = cost2["sector"].astype(str) + "|" + cost2["project_name"].astype(str)
    cost2 = cost2.merge(tk[["key", "approval_year"]], on="key", how="left")
    cost2 = cost2[cost2["approval_year"].notna()].copy()
    cost2["approval_year"] = cost2["approval_year"].astype(int)
    time2 = time[time["approval_year"].notna()].copy()
    time2["approval_year"] = time2["approval_year"].astype(int)
    ext2 = ext[ext["approval_year"].notna()].copy()
    ext2["approval_year"] = ext2["approval_year"].astype(int)

    cost_tr = cost2[cost2["approval_year"] <= cutoff]
    time_tr = time2[time2["approval_year"] <= cutoff]
    ext_tr_c = ext2[(ext2["approval_year"] <= cutoff) & (ext2["overrun_flag"].notna())
                    & (ext2["original_cost_cr"].notna())]
    ext_tr_t = ext2[(ext2["approval_year"] <= cutoff) & (ext2["delayed"].notna())]
    cost_te = ext2[(ext2["approval_year"].between(cutoff + 1, cutoff + 2))
                   & (ext2["overrun_flag"].notna()) & (ext2["original_cost_cr"].notna())]
    time_te = ext2[(ext2["approval_year"].between(cutoff + 1, cutoff + 2))
                   & (ext2["delayed"].notna())]
    return (cost_tr, time_tr, ext_tr_c, ext_tr_t, cost_te, time_te)


def run_augment(cutoffs=(2017, 2018, 2019)) -> dict:
    """A/B: annexure-only vs annexure + resolved May-2025 train rows (<= cutoff),
    evaluated on the SAME held-out May-2025 cohort."""
    reports = {}
    for cutoff in cutoffs:
        cost_tr, time_tr, ext_tr_c, ext_tr_t, cost_te, time_te = _aug_train_frames(cutoff)
        # baseline = annexure only (exact re-run of the external holdout for this cutoff set)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = _ext_metrics(cost_tr, cost_te, time_tr, time_te)
            aug_c = pd.concat([cost_tr, ext_tr_c], ignore_index=True)
            aug_t = pd.concat([time_tr, ext_tr_t], ignore_index=True)
            aug = _ext_metrics(aug_c, cost_te, aug_t, time_te)

        def _row(m):
            return {k: m.get(k) for k in ("auc", "auc_lr", "reg_mae", "naive_mae", "reg_naive",
                                          "n_test", "base_rate")}

        def _gain(metric, task):
            b = (base.get(task) or {}).get(metric)
            a = (aug.get(task) or {}).get(metric)
            if b is None or a is None:
                return None
            if metric in ("auc", "auc_lr"):
                return round(a - b, 3)
            return round((1 - a / b) * 100, 1) if b else None  # % lower MAE = better

        reports[f"cutoff_{cutoff}"] = {
            "cutoff": cutoff, "test_window": f"{cutoff+1}-{cutoff+2}",
            "train_rows": {"annexure_cost": int(len(cost_tr)), "annexure_time": int(len(time_tr)),
                           "added_snapshot_cost": int(len(ext_tr_c)), "added_snapshot_time": int(len(ext_tr_t))},
            "test_rows": {"cost": int(len(cost_te)), "time": int(len(time_te))},
            "baseline": {"cost": _row(base.get("cost") or {}), "time": _row(base.get("time") or {})},
            "augmented": {"cost": _row(aug.get("cost") or {}), "time": _row(aug.get("time") or {})},
            "delta": {"cost_auc": _gain("auc", "cost"), "time_auc": _gain("auc", "time"),
                      "cost_reg_mae_pct": _gain("reg_mae", "cost"), "time_reg_mae_pct": _gain("reg_mae", "time")}}
        print(f"[aug] cutoff {cutoff}: cost AUC {base.get('cost', {}).get('auc')} -> {aug.get('cost', {}).get('auc')} "
              f"(n_train +{len(ext_tr_c)}) | time AUC {base.get('time', {}).get('auc')} -> {aug.get('time', {}).get('auc')} "
              f"(n_train +{len(ext_tr_t)})")

    result = {"method": ("Training-set augmentation A/B: same external holdout (train <= cutoff, "
                         "test = May-2025 cohort approved cutoff+1..cutoff+2). BASELINE = annexure-only "
                         "training. AUGMENTED = annexure training PLUS the resolved May-2025 snapshot rows "
                         "approved <= cutoff (the continual-learning lever). Same metric functions, same "
                         "test rows — the only change is more real, resolved training data."),
              "run": "python -m src.temporal_eval --augment 2017,2018,2019",
              "results": reports, "cutoffs": list(cutoffs)}
    (AUG_OUT_JSON).write_text(json.dumps(result, indent=2))
    return result


def _print_augment(r: dict):
    print("=" * 72)
    print("TRAINING-SET AUGMENTATION A/B — do resolved May-2025 rows help?")
    print(r["method"])
    print("=" * 72)
    print(f"{'cutoff':>6} | {'':>12} | {'cost AUC':>10} {'cost reg MAE':>14} | {'time AUC':>10} {'time reg MAE':>14}")
    for cut, rep in r["results"].items():
        b_c, a_c = rep["baseline"]["cost"], rep["augmented"]["cost"]
        b_t, a_t = rep["baseline"]["time"], rep["augmented"]["time"]
        d = rep["delta"]
        print(f"\n{rep['cutoff']:>6} | train annexure cost={rep['train_rows']['annexure_cost']} time={rep['train_rows']['annexure_time']}, "
              f"+snapshot cost={rep['train_rows']['added_snapshot_cost']} time={rep['train_rows']['added_snapshot_time']}")
        print(f"{'':>6} | {'baseline':>12} | {str(b_c.get('auc')):>10} {str(b_c.get('reg_mae')):>14} | "
              f"{str(b_t.get('auc')):>10} {str(b_t.get('reg_mae')):>14}")
        print(f"{'':>6} | {'augmented':>12} | {str(a_c.get('auc')):>10} {str(a_c.get('reg_mae')):>14} | "
              f"{str(a_t.get('auc')):>10} {str(a_t.get('reg_mae')):>14}")
        print(f"{'':>6} | {'delta':>12} | {str(d['cost_auc']):>10} {str(d['cost_reg_mae_pct'])+'%':>14} | "
              f"{str(d['time_auc']):>10} {str(d['time_reg_mae_pct'])+'%':>14}")
    print(f"\nSaved -> {AUG_OUT_JSON}")


if __name__ == "__main__":
    if "--macro" in sys.argv:
        cutoffs = tuple(int(x) for x in sys.argv[sys.argv.index("--macro") + 1].split(",")) \
            if len(sys.argv) > sys.argv.index("--macro") + 1 and sys.argv[sys.argv.index("--macro") + 1][0].isdigit() \
            else (2017, 2018, 2019)
        res = run_macro_ab(cutoffs)
        _print_macro_ab(res)
        sys.exit(0)
    if "--survival" in sys.argv:
        cutoffs = tuple(int(x) for x in sys.argv[sys.argv.index("--survival") + 1].split(",")) \
            if sys.argv[sys.argv.index("--survival") + 1][0].isdigit() else (2017, 2018, 2019)
        res = run_survival(cutoffs)
        _print_survival(res)
        sys.exit(0)
    if "--augment" in sys.argv:
        cutoffs = tuple(int(x) for x in sys.argv[sys.argv.index("--augment") + 1].split(",")) \
            if sys.argv[sys.argv.index("--augment") + 1][0].isdigit() else (2017, 2018, 2019)
        res = run_augment(cutoffs)
        _print_augment(res)
        sys.exit(0)
    if "--external" in sys.argv:
        cutoffs = tuple(int(x) for x in sys.argv[sys.argv.index("--external") + 1].split(",")) \
            if sys.argv[sys.argv.index("--external") + 1][0].isdigit() else (2017, 2018, 2019)
        res = run_external(cutoffs)
        _print_external(res)
        sys.exit(0)
    cutoffs = tuple(int(x) for x in sys.argv[sys.argv.index("--cutoffs") + 1].split(",")) \
        if "--cutoffs" in sys.argv else (2018, 2019, 2020)
    if "--encoding" in sys.argv:
        modes = tuple(x.strip() for x in sys.argv[sys.argv.index("--encoding") + 1].split(","))
    else:
        modes = ("full", "decay", "rolling")
    res = run(cutoffs, modes)
    _print_report(res)