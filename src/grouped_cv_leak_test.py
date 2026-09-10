"""Grouped-by-state CV leak test for the geo features.

Hypothesis under test (from review):
    State-centroid (lat, lon) is a categorical state ID wearing a continuous
    mask. Under a random/stratified CV split, same-state projects appear in
    both train and test folds, so the tree/linear models memorise "state X
    centroid -> state X historical overrun/delay rate" -- a target-encoded
    state identifier, NOT geography. The external temporal holdout showed
    ~no transfer, which is exactly what this predicts (state base rates drift
    between the old annexure era and the 2019-2021 May-2025 cohort).

Test:
    GroupKFold(5) grouped by STATE. If the AUC/R2 lift from adding geo
    mostly evaporates under grouped splits (vs the ungrouped figures in
    predict.py), the mechanism is confirmed as group leakage. If meaningful
    lift survives grouped CV, part of the geo signal is genuinely spatial.

Run:
    python -m src.grouped_cv_leak_test
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False

from src.geo import extract_state, STATE_COORDINATES

ROOT = "."
COST_CSV = "data/real_mospi_projects.csv"
TIME_CSV = "data/real_mospi_time.csv"

M = 20.0


def _m_rate(groups, target, m=M):
    """m-estimate smoothed group rate over the WHOLE array (used per-train-fold)."""
    g = pd.Series(groups)
    y = np.asarray(target, dtype=float)
    gm = pd.Series(y).groupby(g.values).mean()
    cnt = pd.Series(y).groupby(g.values).size()
    overall = float(y.mean())
    rates = ((cnt * gm + m * overall) / (cnt + m)).to_dict()
    return np.array([rates.get(x, overall) for x in g.values], dtype=float)


def _geo_to_state(lat, lon):
    """Reverse-map a coordinate to the nearest state centroid (crude state proxy)."""
    best, bd = "UNKNOWN", np.inf
    for st, c in STATE_COORDINATES.items():
        if st == "MULTI STATE":
            continue
        d = (np.asarray(lat, dtype=float) - c["lat"]) ** 2 + (np.asarray(lon, dtype=float) - c["lon"]) ** 2
        if d < bd:
            bd, best = d, st
    return best


def _attach_state(df, time_map=None):
    """State per row: extract_state from name, else time-join, else geo-nearest."""
    out = df.copy()
    states = out["project_name"].apply(extract_state)
    if time_map is not None:
        key = out["sector"].astype(str) + "|" + out["project_name"].astype(str)
        joined = key.map(time_map)
        states = states.where(states.notna(), joined)
    lat = pd.to_numeric(out.get("geo_latitude"), errors="coerce")
    lon = pd.to_numeric(out.get("geo_longitude"), errors="coerce")
    needs = states.isna()
    for i in out.index[needs]:
        if not np.isfinite(lat.at[i]) or not np.isfinite(lon.at[i]):
            continue
        states.at[i] = _geo_to_state(lat.at[i], lon.at[i])
    out["state"] = states.fillna("UNKNOWN").astype(str)
    return out


def _cost_frames2(train, test, config, risk_tr=None, risk_te=None):
    """config: 'none', 'geo', 'geo+pace', 'geo+pace+px', 'geo+pace+sch', 'geo+pace+sch+px',
    plus '+rev' variants (revise_log = log1p sanction growth, a base feature).

    'geo+pace+sch' adds the OOF schedule-slippage risk (expected P(delayed))
    fed into the COST model from a time model trained on TRAIN STATES ONLY
    (see _fit_schedule_risk) - never the realized delay label. 'px' adds the
    sector-mapped input-price escalation features (pure function of approval
    year; deterministic per-row, so no fold leak - medians fitted on train).
    'rev' adds revise_log (current revised sanctioned cost vs original - known
    at decision time, leak-free)."""
    def feats(df):
        orig = df["original_cost_cr"].clip(lower=0).to_numpy(dtype=float)
        exp = df["expenditure_cum_cr"].fillna(0).clip(lower=0).to_numpy(dtype=float)
        lat = pd.to_numeric(df.get("geo_latitude"), errors="coerce").fillna(21.1458).to_numpy(dtype=float)
        lon = pd.to_numeric(df.get("geo_longitude"), errors="coerce").fillna(79.0882).to_numpy(dtype=float)
        es = pd.to_numeric(df.get("elapsed_share"), errors="coerce")
        sp = pd.to_numeric(df.get("spend_pace"), errors="coerce")
        pe = pd.to_numeric(df.get("price_esc"), errors="coerce")
        pl = pd.to_numeric(df.get("price_level_approval"), errors="coerce")
        revised = pd.to_numeric(df.get("revised_cost_cr"), errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(revised).all():
            rev = np.where(np.isnan(revised), orig, revised)
        else:
            rev = revised
        gain = np.clip(rev - orig, 0.0, None) / np.maximum(orig, 1e-9)
        return {"log_original_cost": np.log1p(orig), "expenditure_ratio": exp / np.maximum(orig, 1e-9),
                "log_expenditure": np.log1p(exp), "geo_latitude": lat, "geo_longitude": lon,
                "elapsed_share": es, "spend_pace": sp, "price_esc": pe, "price_level_approval": pl,
                "revise_log": np.log1p(gain)}

    def _fit_rate_map(groups, target):
        g = pd.Series(groups)
        y = np.asarray(target, dtype=float)
        gm = pd.Series(y).groupby(g.values).mean()
        cnt = pd.Series(y).groupby(g.values).size()
        overall = float(y.mean())
        return ((cnt * gm + M * overall) / (cnt + M)).to_dict(), overall

    def _apply(groups, rate_map, overall):
        g = pd.Series(groups)
        return np.array([rate_map.get(x, overall) for x in g.values], dtype=float)

    ft = feats(train)
    fy = feats(test)
    es_med = pd.to_numeric(train.get("elapsed_share"), errors="coerce").median()
    sp_med = pd.to_numeric(train.get("spend_pace"), errors="coerce").median()
    pe_med = pd.to_numeric(train.get("price_esc"), errors="coerce").median()
    pl_med = pd.to_numeric(train.get("price_level_approval"), errors="coerce").median()
    for d in (ft, fy):
        d["elapsed_share"] = d["elapsed_share"].fillna(es_med).to_numpy(dtype=float)
        d["spend_pace"] = d["spend_pace"].fillna(sp_med).to_numpy(dtype=float)
        d["price_esc"] = d["price_esc"].fillna(pe_med).to_numpy(dtype=float)
        d["price_level_approval"] = d["price_level_approval"].fillna(pl_med).to_numpy(dtype=float)
    rate_map, overall = _fit_rate_map(train["sector"].astype(str), train["overrun_flag"] > 0)
    rate_map_r, overall_r = _fit_rate_map(train["sector"].astype(str), train["overrun_ratio"])

    def col(base, rate, risk):
        cols = [base["log_original_cost"], base["expenditure_ratio"], base["log_expenditure"], rate]
        if "geo" in config:
            cols += [base["geo_latitude"], base["geo_longitude"]]
        if "pace" in config:
            cols += [base["elapsed_share"], base["spend_pace"]]
        if "px" in config:
            cols += [base["price_esc"], base["price_level_approval"]]
        if "sch" in config:
            cols += [risk]
        if "rev" in config:
            cols += [np.asarray(base["revise_log"], dtype=float)]
        return np.column_stack(cols)

    sr_te = _apply(test["sector"].astype(str), rate_map, overall)
    sr_r_te = _apply(test["sector"].astype(str), rate_map_r, overall_r)
    return col(ft, _apply(train["sector"].astype(str), rate_map, overall), risk_tr), col(fy, sr_te, risk_te), \
        col(ft, _apply(train["sector"].astype(str), rate_map_r, overall_r), risk_tr), col(fy, sr_r_te, risk_te)


def _time_frames2(train, test, config):
    from src.predict import _agency_from_name
    def feats(df):
        orig = df["original_cost_cr"].clip(lower=0).to_numpy(dtype=float)
        exp = df["expenditure_cum_cr"].fillna(0).clip(lower=0).to_numpy(dtype=float)
        pdur = pd.to_numeric(df["planned_duration_years"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(pdur).any():
            pdur = np.nan_to_num(pdur, nan=np.nanmedian(pdur))
        else:
            pdur = np.full(len(pdur), 5.0)
        lat = pd.to_numeric(df.get("geo_latitude"), errors="coerce").fillna(21.1458).to_numpy(dtype=float)
        lon = pd.to_numeric(df.get("geo_longitude"), errors="coerce").fillna(79.0882).to_numpy(dtype=float)
        es = pd.to_numeric(df.get("elapsed_share"), errors="coerce")
        sp = pd.to_numeric(df.get("spend_pace"), errors="coerce")
        return {"log_original_cost": np.log1p(orig), "expenditure_ratio": exp / np.maximum(orig, 1e-9),
                "planned_duration_yrs": pdur, "geo_latitude": lat, "geo_longitude": lon,
                "elapsed_share": es, "spend_pace": sp}

    def _fit_rate_map(groups, target):
        g = pd.Series(groups)
        y = np.asarray(target, dtype=float)
        gm = pd.Series(y).groupby(g.values).mean()
        cnt = pd.Series(y).groupby(g.values).size()
        overall = float(y.mean())
        return ((cnt * gm + M * overall) / (cnt + M)).to_dict(), overall

    def _apply(groups, rate_map, overall):
        g = pd.Series(groups)
        return np.array([rate_map.get(x, overall) for x in g.values], dtype=float)

    ft = feats(train)
    fy = feats(test)
    es_med = pd.to_numeric(train.get("elapsed_share"), errors="coerce").median()
    sp_med = pd.to_numeric(train.get("spend_pace"), errors="coerce").median()
    for d in (ft, fy):
        d["elapsed_share"] = d["elapsed_share"].fillna(es_med).to_numpy(dtype=float)
        d["spend_pace"] = d["spend_pace"].fillna(sp_med).to_numpy(dtype=float)
    rate_map, overall = _fit_rate_map(train["sector"].astype(str), train["delayed"] > 0)
    ag_tr = train["project_name"].apply(_agency_from_name).fillna("UNKNOWN").astype(str)
    ag_te = test["project_name"].apply(_agency_from_name).fillna("UNKNOWN").astype(str)
    agrates, overall_a = _fit_rate_map(ag_tr, train["delayed"] > 0)

    def col(base, rate, ag):
        cols = [base["log_original_cost"], base["expenditure_ratio"], base["planned_duration_yrs"], rate]
        if "geo" in config:
            cols += [base["geo_latitude"], base["geo_longitude"]]
        if "pace" in config:
            cols += [base["elapsed_share"], base["spend_pace"]]
        if "agcy" in config:
            cols += [ag]
        return np.column_stack(cols)

    return (col(ft, _apply(train["sector"].astype(str), rate_map, overall),
                _apply(ag_tr, agrates, overall_a)),
            col(fy, _apply(test["sector"].astype(str), rate_map, overall),
                _apply(ag_te, agrates, overall_a)))


def _clfs():
    m = {"Logistic Regression": LogisticRegression(max_iter=3000, C=1.0),
         "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=8, max_depth=8,
                                                 random_state=42, n_jobs=-1)}
    if HAVE_LGBM:
        m["LightGBM"] = LGBMClassifier(n_estimators=300, num_leaves=8, learning_rate=0.05,
                                       random_state=42, verbose=-1, n_jobs=-1)
    return m


def _regs():
    m = {"Linear Regression": LinearRegression(),
         "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=8, max_depth=8,
                                                random_state=42, n_jobs=-1)}
    if HAVE_LGBM:
        m["LightGBM"] = LGBMRegressor(n_estimators=300, num_leaves=8, learning_rate=0.05,
                                      random_state=42, verbose=-1, n_jobs=-1)
    return m


def _reg_targets(yr):
    ymin = float(np.nanmin(yr))
    shift = max(0.0, -ymin) + 1.0
    return np.log1p(yr + shift), shift


def _inv_reg(pred, shift):
    return np.expm1(pred) - shift


def _time_feat_matrix(df, rate_map=None, overall=None):
    """Time-model feature matrix (pace-inclusive) for arbitrary rows.

    rate_map/overall are the sector-delay-rates fit on the TIME training rows
    (train states); when None, they are fit on df (train side)."""
    orig = df["original_cost_cr"].clip(lower=0).to_numpy(dtype=float)
    exp = df["expenditure_cum_cr"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    pdur = pd.to_numeric(df["planned_duration_years"], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(pdur).any():
        pdur = np.nan_to_num(pdur, nan=np.nanmedian(pdur))
    else:
        pdur = np.full(len(pdur), 5.0)
    lat = pd.to_numeric(df.get("geo_latitude"), errors="coerce").fillna(21.1458).to_numpy(dtype=float)
    lon = pd.to_numeric(df.get("geo_longitude"), errors="coerce").fillna(79.0882).to_numpy(dtype=float)
    es = pd.to_numeric(df.get("elapsed_share"), errors="coerce")
    sp = pd.to_numeric(df.get("spend_pace"), errors="coerce")
    es_med = float(es.median()) if np.isfinite(es.median()) else 0.5
    sp_med = float(sp.median()) if np.isfinite(sp.median()) else 1.0
    es = es.fillna(es_med).to_numpy(dtype=float)
    sp = sp.fillna(sp_med).to_numpy(dtype=float)
    sec = df["sector"].astype(str)
    if rate_map is None:
        y = (df["delayed"] > 0).astype(int).to_numpy()
        g = pd.Series(y).groupby(sec.values).mean()
        cnt = pd.Series(y).groupby(sec.values).size()
        overall = float(y.mean())
        rate_map = ((cnt * g + M * overall) / (cnt + M)).to_dict()
    rate = np.array([rate_map.get(x, overall) for x in sec.values], dtype=float)
    return np.column_stack([np.log1p(orig), exp / np.maximum(orig, 1e-9), pdur, rate, lat, lon, es, sp]), rate_map, overall


def _oof_schedule_risk_fixed(cost_df, time_df):
    """Whole-table OOF-by-state expected P(delayed), fixed per COST row.

    Mirrors predict.py._oof_schedule_risk: GroupKFold(5) by state, each held-out
    row's risk comes from a time model trained on OTHER states' TIME rows only.
    The resulting values are attached to the cost table ONCE and reused across
    ALL grouped-CV folds - exactly how the production cost model trains (the
    feature column is fixed; only the cost model itself is re-split)."""
    states = cost_df["state"].to_numpy()
    base = float((time_df["delayed"] > 0).astype(int).mean())
    risk = np.full(len(cost_df), np.nan)
    if len(time_df) < 100:
        return np.full(len(cost_df), base)
    gkf = GroupKFold(n_splits=5)
    rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=42)
    for tr, va in gkf.split(cost_df, groups=states):
        tt = time_df[time_df["state"].isin(set(states[tr]))]
        if len(tt) < 100:
            risk[va] = base
            continue
        ytt = (tt["delayed"] > 0).astype(int).to_numpy()
        Xtt, rate, overall = _time_feat_matrix(tt)
        Xva, _, _ = _time_feat_matrix(cost_df.iloc[va], rate, overall)
        best_k, best_a = None, -1.0
        for k, m in _clfs().items():
            aucs = []
            for i, j in rskf.split(Xtt, ytt):
                aucs.append(roc_auc_score(ytt[j], clone(m).fit(Xtt[i], ytt[i]).predict_proba(Xtt[j])[:, 1]))
            a = float(np.mean(aucs))
            if a > best_a:
                best_a, best_k = a, k
        clf = clone(_clfs()[best_k]).fit(Xtt, ytt)
        risk[va] = clf.predict_proba(Xva)[:, 1]
    return np.nan_to_num(risk, nan=base)


def main():
    cost = pd.read_csv(COST_CSV)
    time = pd.read_csv(TIME_CSV)

    # state for time rows comes straight from the ']ORG,STATE' suffix
    time = _attach_state(time)
    tm = dict(zip(time["sector"].astype(str) + "|" + time["project_name"].astype(str), time["state"]))

    # pace / slippage features need approval_year + duration
    from src.pace import pace_features, SNAP_YEAR_ANNEXURE
    time = pace_features(time, SNAP_YEAR_ANNEXURE)
    ty = time[["sector", "project_name", "approval_year", "planned_duration_years"]].copy()
    ty["key"] = ty["sector"].astype(str) + "|" + ty["project_name"].astype(str)
    # 1:1 dict join (the time table can repeat a (sector, project) key across
    # annexures; an inner merge would duplicate cost rows and leak across folds)
    ty = ty.drop_duplicates("key")
    ty = ty.dropna(subset=["approval_year"])
    jmap = dict(zip(ty["key"], ty["approval_year"]))
    dmap = dict(zip(ty["key"], ty["planned_duration_years"]))
    cost = cost.copy()
    cost["key"] = cost["sector"].astype(str) + "|" + cost["project_name"].astype(str)
    cost["approval_year"] = cost["key"].map(jmap)
    cost["planned_duration_years"] = cost["key"].map(dmap)
    cost = cost.drop(columns=["key"])
    cost = pace_features(cost, SNAP_YEAR_ANNEXURE)
    cost = _attach_state(cost, time_map=tm)
    from src.price_index import attach_price_index
    cost = attach_price_index(cost)

    # clean/sparse-removal mirroring predict.py
    cost = cost.dropna(subset=["overrun_flag", "original_cost_cr"]).copy()
    time = time.dropna(subset=["delayed"]).copy()

    CONFIGS = ["none", "geo", "geo+pace", "geo+pace+px", "geo+pace+sch", "geo+pace+sch+px",
               "none+rev", "geo+pace+sch+px+rev"]
    TIME_CONFIGS = ["none", "geo", "geo+pace", "geo+pace+agcy"]
    print("=" * 78)
    print("GROUPED-BY-STATE CV LEAK TEST  (GroupKFold by state)")
    print("  configs: none | geo | geo+pace | +px (WPI/fx escalation) | +sch (expected P(delayed), OOF) |")
    print("           +rev (sanction revision so far, base feature) || TIME: +agcy (agency delay rate)")
    print("=" * 78)
    for name, df, ybcol, mode in (("COST", cost, "overrun_flag", "cost"),
                                  ("TIME", time, "delayed", "time")):
        groups = df["state"].to_numpy()
        n_states = int(pd.Series(groups).nunique())
        gkf = GroupKFold(n_splits=5)
        res = {c: {"clf_auc": {k: [] for k in _clfs()}, "reg_mae": [], "reg_r2": []}
               for c in (CONFIGS if mode == "cost" else TIME_CONFIGS)}
        fold_info = []
        sch_risk = None
        if mode == "cost":
            sch_risk = _oof_schedule_risk_fixed(cost, time)
        cfgs = CONFIGS if mode == "cost" else TIME_CONFIGS
        for fold, (tr, va) in enumerate(gkf.split(df, df[ybcol], groups)):
            tdf, vdf = df.iloc[tr], df.iloc[va]
            tr_states = set(tdf["state"])
            overlap = len(tr_states & set(vdf["state"]))
            fold_info.append((len(tr), len(va), len(tr_states), overlap))
            for cfg in cfgs:
                if mode == "cost":
                    risk_tr = None if sch_risk is None else sch_risk[tr]
                    risk_te = None if sch_risk is None else sch_risk[va]
                    Xtr, Xte, Xtr_r, Xte_r = _cost_frames2(tdf, vdf, cfg, risk_tr, risk_te)
                    yb_tr = (tdf["overrun_flag"] > 0).astype(int).to_numpy()
                    yb_te = (vdf["overrun_flag"] > 0).astype(int).to_numpy()
                    for k, m in _clfs().items():
                        auc = roc_auc_score(yb_te, clone(m).fit(Xtr, yb_tr).predict_proba(Xte)[:, 1])
                        res[cfg]["clf_auc"][k].append(float(auc))
                    ok_tr = tdf["overrun_ratio"].notna().to_numpy()
                    ok_te = vdf["overrun_ratio"].notna().to_numpy()
                    if ok_tr.sum() >= 50 and ok_te.sum() >= 30:
                        tgt, shift = _reg_targets(tdf.loc[ok_tr, "overrun_ratio"].to_numpy(dtype=float))
                        Xr_tr = Xtr_r[ok_tr]
                        yr_te = vdf.loc[ok_te, "overrun_ratio"].to_numpy(dtype=float)
                        Xr_te = Xte_r[ok_te]
                        maes, r2s = [], []
                        for k, m in _regs().items():
                            r = clone(m).fit(Xr_tr, tgt)
                            pp = _inv_reg(r.predict(Xr_te), shift)
                            maes.append(mean_absolute_error(yr_te, pp))
                            r2s.append(r2_score(np.log1p(yr_te + shift), r.predict(Xr_te)))
                        res[cfg]["reg_mae"].append(float(np.min(maes)))
                        res[cfg]["reg_r2"].append(float(np.max(r2s)))
                else:
                    Xtr, Xte = _time_frames2(tdf, vdf, cfg)
                    yb_tr = (tdf["delayed"] > 0).astype(int).to_numpy()
                    yb_te = (vdf["delayed"] > 0).astype(int).to_numpy()
                    for k, m in _clfs().items():
                        auc = roc_auc_score(yb_te, clone(m).fit(Xtr, yb_tr).predict_proba(Xte)[:, 1])
                        res[cfg]["clf_auc"][k].append(float(auc))
        print(f"\n### {name} | rows={len(df)} | states={n_states} | GroupKFold(5)")
        for fold, (ntr, nva, nst, ov) in enumerate(fold_info):
            print(f"  fold {fold}: train {ntr}, val {nva}, states_train {nst}, states_shared_with_val {ov}")
        print("  clf AUC (mean over 5 grouped folds):")
        if mode == "cost":
            header = ("    none    | geo         | geo+pace    | +px         | +sch        | +sch+px     | "
                      "px delta | sch delta | px|sch delta\n"
                      "            |             |             | (vs pace)   | (vs pace)   | (vs pace)")
            print(header)
            for k in _clfs():
                a0 = np.mean(res["none"]["clf_auc"][k])
                a1 = np.mean(res["geo"]["clf_auc"][k])
                a2 = np.mean(res["geo+pace"]["clf_auc"][k])
                a4 = np.mean(res["geo+pace+px"]["clf_auc"][k])
                a3 = np.mean(res["geo+pace+sch"]["clf_auc"][k])
                a5 = np.mean(res["geo+pace+sch+px"]["clf_auc"][k])
                print(f"    {a0:6.3f} | {a1:6.3f}      | {a2:6.3f}       | {a4:6.3f}       | {a3:6.3f}       | {a5:6.3f}       | "
                      f"{a4 - a2:+.3f}   | {a3 - a2:+.3f}    | {a5 - a3:+.3f}   {k}")
        else:
            header = ("    none    | geo         | geo+pace    | +agcy       | pace delta | agcy delta\n"
                      "            |             |             |             | (vs geo)   | (vs pace)")
            print(header)
            for k in _clfs():
                a0 = np.mean(res["none"]["clf_auc"][k])
                a1 = np.mean(res["geo"]["clf_auc"][k])
                a2 = np.mean(res["geo+pace"]["clf_auc"][k])
                a6 = np.mean(res["geo+pace+agcy"]["clf_auc"][k])
                print(f"    {a0:6.3f} | {a1:6.3f}      | {a2:6.3f}       | {a6:6.3f}       | {a2 - a1:+.3f}     | {a6 - a2:+.3f}   {k}")
        if res["none"]["reg_mae"]:
            m2 = np.mean(res["geo+pace"]["reg_mae"])
            r2v = np.mean(res["geo+pace"]["reg_r2"])
            if mode == "cost":
                m4 = np.mean(res["geo+pace+px"]["reg_mae"])
                r4 = np.mean(res["geo+pace+px"]["reg_r2"])
                m3 = np.mean(res["geo+pace+sch"]["reg_mae"])
                r3 = np.mean(res["geo+pace+sch"]["reg_r2"])
                m5 = np.mean(res["geo+pace+sch+px"]["reg_mae"])
                r5 = np.mean(res["geo+pace+sch+px"]["reg_r2"])
                print(f"  reg MAE: geo+pace={m2:.3f} | +px={m4:.3f} ({m4 - m2:+.3f}) | +sch={m3:.3f} ({m3 - m2:+.3f}) | "
                      f"+sch+px={m5:.3f} ({m5 - m3:+.3f})")
                print(f"  reg R2 : geo+pace={r2v:.3f} | +px={r4:.3f} | +sch={r3:.3f} | +sch+px={r5:.3f}")

                def _rev_delta(base, plus):
                    out = []
                    for k in _clfs():
                        out.append(np.mean(res[plus]["clf_auc"][k]) - np.mean(res[base]["clf_auc"][k]))
                    mr0 = np.mean(res[base]["reg_mae"]) if res[base]["reg_mae"] else float("nan")
                    mr1 = np.mean(res[plus]["reg_mae"]) if res[plus]["reg_mae"] else float("nan")
                    return out, (mr1 - mr0 if np.isfinite(mr0) and np.isfinite(mr1) else float("nan"))

                for base, plus, tag in (("none", "none+rev", "+rev (vs none)"),
                                        ("geo+pace+sch+px", "geo+pace+sch+px+rev", "+rev (vs sch+px)")):
                    dclf, dreg = _rev_delta(base, plus)
                    print(f"  revise_log clf delta {tag}: LR {dclf[0]:+.3f} / RF {dclf[1]:+.3f} / "
                          f"LGBM {dclf[2]:+.3f} | reg MAE delta {dreg:+.3f}")
            else:
                m0 = np.mean(res["none"]["reg_mae"])
                r0 = np.mean(res["none"]["reg_r2"])
                m2t = np.mean(res["geo+pace"]["reg_mae"])
                r2t = np.mean(res["geo+pace"]["reg_r2"])
                m7 = np.mean(res["geo+pace+agcy"]["reg_mae"])
                r7 = np.mean(res["geo+pace+agcy"]["reg_r2"])
                print(f"  reg MAE: none={m0:.3f} | geo+pace={m2t:.3f} | +agcy={m7:.3f} ({m7 - m2t:+.3f})")
                print(f"  reg R2 : none={r0:.3f} | geo+pace={r2t:.3f} | +agcy={r7:.3f}")

    print("\nRead-out:")
    print("  - geo adds ~0 under grouped CV (confirmed earlier).")
    print("  - a positive 'pace delta' means #1+#2 (spend pace / schedule slippage)")
    print("    carry REAL out-of-sample signal beyond the dominant expenditure_ratio.")
    print("  - a positive '+px delta' means the sector-mapped input-price escalation")
    print("    lifts cost beyond pace. Expect ~0: within this one vintage the price")
    print("    move is a smooth function of approval year (collinear with year); the")
    print("    payoff is future vintages, not this table.")
    print("  - a positive '+sch delta' (COST) means the OOF schedule-slippage risk")
    print("    lifts the cost model honestly (see predict.py._oof_schedule_risk);")
    print("    a flat delta is still expected-utility-positive but should not be")
    print("    oversold on the model card.")
    print("  - '+rev delta' (COST) = log1p sanctioned-cost growth already taken (a")
    print("    decision-time base feature; proxy for revision count in one vintage).")
    print("  - '+agcy delta' (TIME) = OOF implementing-agency delay rate; expect it")
    print("    to overlap sector/pacing signal but add management-share info.")


if __name__ == "__main__":
    main()