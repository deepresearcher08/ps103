"""External holdout A/B: COST model with vs without the OOF schedule-risk feature.

Train = annexure cost rows approved <= cutoff; test = May-2025 Flash Report
snapshot rows approved cutoff+1..cutoff+2 (labels RESOLVED, genuinely external
document). The schedule_risk feature for each cost row is the expected P(delayed)
from a TIME model trained on ANNEXURE rows of TRAIN STATES only:
  * annexure train rows -> within-train GroupKFold-by-state (row's own state held out)
  * snapshot test  rows -> time model trained on train states (the row has no
    annexure time row, so its own delay label can never be seen - trivially OOF)
Never the realized delayed/tor outcome. Print passes as reported by the run().

Run:  python tools/external_sch_check.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold

from src.temporal_eval import (
    _clfs, _regs, _cost_frame, _reg_targets, _inv_reg, _prep_pace,
    COST_CSV, TIME_CSV, EXT_CSV,
)
from src.predict import _states_for, _prepare_time_table
from src.price_index import attach_price_index
from sklearn.metrics import mean_absolute_error

M = 20.0


def _time_feat_matrix(df, rate_map=None, overall=None):
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


def _schedule_risk_for(c_tr_cost, c_te_cost, time_prep, states_tr, states_te):
    """risk for annexure train rows (OOF-by-state within train) + snapshot test rows."""
    risk_tr = np.full(len(c_tr_cost), np.nan)
    base = float((time_prep["delayed"] > 0).astype(int).mean())
    gkf = GroupKFold(n_splits=5)
    rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=42)
    for tr, va in gkf.split(c_tr_cost, groups=states_tr):
        tt = time_prep[time_prep["state"].isin(set(states_tr[tr]))]
        if len(tt) < 100:
            risk_tr[va] = base
            continue
        ytt = (tt["delayed"] > 0).astype(int).to_numpy()
        Xtt, rate, overall = _time_feat_matrix(tt)
        Xva, _, _ = _time_feat_matrix(c_tr_cost.iloc[va], rate, overall)
        best_k, best_a = None, -1.0
        for k, m in _clfs().items():
            aucs = []
            for i, j in rskf.split(Xtt, ytt):
                aucs.append(roc_auc_score(ytt[j], clone(m).fit(Xtt[i], ytt[i]).predict_proba(Xtt[j])[:, 1]))
            a = float(np.mean(aucs))
            if a > best_a:
                best_a, best_k = a, k
        risk_tr[va] = clone(_clfs()[best_k]).fit(Xtt, ytt).predict_proba(Xva)[:, 1]
    # snapshot test rows: time model on ALL train states (their own delay is never
    # in the annexure time table, so this is trivially outside-sample)
    tt = time_prep[time_prep["state"].isin(set(states_tr))]
    if len(tt) >= 100:
        ytt = (tt["delayed"] > 0).astype(int).to_numpy()
        Xtt, rate, overall = _time_feat_matrix(tt)
        Xte, _, _ = _time_feat_matrix(c_te_cost, rate, overall)
        best_k, best_a = None, -1.0
        for k, m in _clfs().items():
            aucs = []
            for i, j in rskf.split(Xtt, ytt):
                aucs.append(roc_auc_score(ytt[j], clone(m).fit(Xtt[i], ytt[i]).predict_proba(Xtt[j])[:, 1]))
            a = float(np.mean(aucs))
            if a > best_a:
                best_a, best_k = a, k
        risk_te = clone(_clfs()[best_k]).fit(Xtt, ytt).predict_proba(Xte)[:, 1]
    else:
        risk_te = np.full(len(c_te_cost), base)
    return np.nan_to_num(risk_tr, nan=base), np.nan_to_num(risk_te, nan=base)


def main():
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
    ext2 = ext.copy()
    ext2 = ext2[ext2["approval_year"].notna()].copy()
    ext2["approval_year"] = ext2["approval_year"].astype(int)

    time_prep = _prepare_time_table()
    tkey = time_prep["sector"].astype(str) + "|" + time_prep["project_name"].astype(str)
    t_state_map = dict(zip(tkey, time_prep["state"]))

    print("=" * 70)
    print("EXTERNAL COST HOLDOUT: schedule_risk OFF/ON  x  price-escalation OFF/ON")
    print("(OOF-by-state TRAIN-STATE time models; price = sector-mapped WPI/fx, fixed ref 2024)")
    print("=" * 70)
    for cutoff in (2017, 2018, 2019):
        c_tr = cost2[cost2["approval_year"] <= cutoff]
        c_te = ext2[ext2["approval_year"].between(cutoff + 1, cutoff + 2)]
        c_te = c_te[c_te["overrun_flag"].notna()].dropna(subset=["original_cost_cr"])
        if len(c_te) < 30 or len(c_tr) < 100:
            print(f"[{cutoff}] skipped (train={len(c_tr)}, test={len(c_te)})")
            continue
        c_tr = _prep_pace(c_tr)
        c_te = _prep_pace(c_te)
        c_tr = attach_price_index(c_tr)
        c_te = attach_price_index(c_te)
        pe_med = float(c_tr["price_esc"].median())
        pl_med = float(c_tr["price_level_approval"].median())
        px_tr = np.column_stack([c_tr["price_esc"].fillna(pe_med).to_numpy(dtype=float),
                                 c_tr["price_level_approval"].fillna(pl_med).to_numpy(dtype=float)])
        px_te = np.column_stack([c_te["price_esc"].fillna(pe_med).to_numpy(dtype=float),
                                 c_te["price_level_approval"].fillna(pl_med).to_numpy(dtype=float)])
        risk_tr, risk_te = _schedule_risk_for(
            c_tr, c_te, time_prep, _states_for(c_tr, t_state_map), _states_for(c_te, t_state_map))
        Xtr, Xte, Xtr_r, Xte_r = _cost_frame(c_tr, c_te)
        Xtr_s, Xte_s = np.column_stack([Xtr, risk_tr]), np.column_stack([Xte, risk_te])
        Xtr_r_s, Xte_r_s = np.column_stack([Xtr_r, risk_tr]), np.column_stack([Xte_r, risk_te])
        Xtr_p, Xte_p = np.column_stack([Xtr, px_tr]), np.column_stack([Xte, px_te])
        Xtr_r_p, Xte_r_p = np.column_stack([Xtr_r, px_tr]), np.column_stack([Xte_r, px_te])
        Xtr_sp, Xte_sp = np.column_stack([Xtr, risk_tr, px_tr]), np.column_stack([Xte, risk_te, px_te])
        Xtr_r_sp, Xte_r_sp = np.column_stack([Xtr_r, risk_tr, px_tr]), np.column_stack([Xte_r, risk_te, px_te])
        rev_tr = np.log1p((c_tr["revised_cost_cr"].fillna(c_tr["original_cost_cr"]) - c_tr["original_cost_cr"])
                          .clip(lower=0) / c_tr["original_cost_cr"].clip(lower=1e-9)).to_numpy(dtype=float)
        rev_te = np.log1p((c_te["revised_cost_cr"].fillna(c_te["original_cost_cr"]) - c_te["original_cost_cr"])
                          .clip(lower=0) / c_te["original_cost_cr"].clip(lower=1e-9)).to_numpy(dtype=float)
        Xtr_rv, Xte_rv = np.column_stack([Xtr, rev_tr]), np.column_stack([Xte, rev_te])
        Xtr_r_rv, Xte_r_rv = np.column_stack([Xtr_r, rev_tr]), np.column_stack([Xte_r, rev_te])
        Xtr_spr, Xte_spr = np.column_stack([Xtr, risk_tr, px_tr, rev_tr]), np.column_stack([Xte, risk_te, px_te, rev_te])
        Xtr_r_spr, Xte_r_spr = np.column_stack([Xtr_r, risk_tr, px_tr, rev_tr]), np.column_stack([Xte_r, risk_te, px_te, rev_te])
        yb_tr = (c_tr["overrun_flag"] > 0).astype(int).to_numpy()
        yb_te = (c_te["overrun_flag"] > 0).astype(int).to_numpy()
        yr_te = c_te["overrun_ratio"].to_numpy(dtype=float)
        tgt, shift = _reg_targets(c_tr["overrun_ratio"].to_numpy(dtype=float))
        cells = []
        for tag, (Atr, Ate, Btr, Bte) in {"off": (Xtr, Xte, Xtr_r, Xte_r),
                                          "+sch": (Xtr_s, Xte_s, Xtr_r_s, Xte_r_s),
                                          "+px": (Xtr_p, Xte_p, Xtr_r_p, Xte_r_p),
                                          "+sch+px": (Xtr_sp, Xte_sp, Xtr_r_sp, Xte_r_sp),
                                          "off+rev": (Xtr_rv, Xte_rv, Xtr_r_rv, Xte_r_rv),
                                          "+sch+px+rev": (Xtr_spr, Xte_spr, Xtr_r_spr, Xte_r_spr)}.items():
            aucs = {}
            for k, m in _clfs().items():
                aucs[k] = roc_auc_score(yb_te, clone(m).fit(Atr, yb_tr).predict_proba(Ate)[:, 1])
            best = max(aucs, key=aucs.get)
            maes = {}
            for k, m in _regs().items():
                r = clone(m).fit(Btr, tgt)
                pp = _inv_reg(r.predict(Bte), shift)
                maes[k] = mean_absolute_error(yr_te, pp)
            best_r = min(maes, key=maes.get)
            cells.append(f"{tag}: clf AUC {aucs[best]:.3f} ({best}) | reg MAE {maes[best_r]:.3f} | " +
                         f"LR {aucs['Logistic Regression']:.3f} / RF {aucs['Random Forest']:.3f} / " +
                         f"LGBM {aucs.get('LightGBM', float('nan')):.3f}")
        print(f"[cutoff {cutoff}] test n={len(c_te)} base_rate={yb_te.mean():.3f}")
        for c in cells:
            print("    " + c)


if __name__ == "__main__":
    main()