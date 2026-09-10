"""Backfill geo coordinates across MoSPI CSVs.

Pipeline:
  1. extract_state directly from project names (src.geo)
  2. cross-dataset propagation (COST <- TIME, SNAPSHOT <- COST+TIME) by normalised key
  3. strict fuzzy fingerprint matching for the truncated SNAPSHOT names
  4. per-sector mean imputation for anything still missing
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.geo import enrich_geo

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

COST_CSV = DATA / "real_mospi_projects.csv"
TIME_CSV = DATA / "real_mospi_time.csv"
SNAP_CSV = DATA / "mospi_may2025_projects.csv"

STOP = {"of", "and", "the", "for", "to", "in", "on", "at", "a", "an", "new",
        "from", "with", "km", "nh", "section", "project", "laning", "lance", "road"}
PREFIX = 6
MULTI_STATE_GEO = (21.1458, 79.0882)


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _fp(name: str) -> set:
    toks = [_norm(t) for t in str(name).split()]
    return {t[:PREFIX] for t in toks if t and t not in STOP and len(t) >= 5}


def _comma_pad(v: float, width: int = 2) -> str:
    return f"{v:0.{width}f}"


def _sector_impute(df: pd.DataFrame) -> dict:
    """Per-sector mean geo of geo'd rows; returns sector -> (lat, lon) + overall mean."""
    lat = pd.to_numeric(df.get("geo_latitude"), errors="coerce")
    lon = pd.to_numeric(df.get("geo_longitude"), errors="coerce")
    sector = df["sector"].astype(str)
    ok = lat.notna() & lon.notna()
    if not ok.any():
        return {}
    lm = lat[ok].groupby(sector[ok]).mean()
    lonm = lon[ok].groupby(sector[ok]).mean()
    return {s: (lm[s], lonm[s]) for s in lm.index}


def propagate_and_impute() -> None:
    # --- load + direct extraction ---
    time_df = pd.read_csv(TIME_CSV)
    enrich_geo(time_df, "project_name")

    cost = pd.read_csv(COST_CSV)
    enrich_geo(cost, "project_name")

    snap = pd.read_csv(SNAP_CSV)
    enrich_geo(snap, "project_name")

    print(f"after direct extract: cost={cost['geo_latitude'].notna().mean():.1%} "
          f"time={time_df['geo_latitude'].notna().mean():.1%} "
          f"snap={snap['geo_latitude'].notna().mean():.1%}")

    # --- TIME lookup for COST ---
    time_lookup = {}
    for _, r in time_df.iterrows():
        if pd.isna(r["geo_latitude"]):
            continue
        geo = (r["geo_latitude"], r["geo_longitude"])
        time_lookup.setdefault("full:" + _norm(r["project_name"]), geo)
        time_lookup.setdefault("pre:" + _norm(r["project_name"].split("[")[0]), geo)

    n_fill_cost = 0
    for i in cost.index[cost["geo_latitude"].isna()]:
        pre = _norm(cost.at[i, "project_name"].split("[")[0])
        hit = time_lookup.get("pre:" + pre) or time_lookup.get("full:" + _norm(cost.at[i, "project_name"]))
        if hit:
            cost.at[i, "geo_latitude"], cost.at[i, "geo_longitude"] = hit
            n_fill_cost += 1
    print(f"cost after TIME-prop: {cost['geo_latitude'].notna().mean():.1%} (+{n_fill_cost})")
    cost.to_csv(COST_CSV, index=False)

    # --- combined lookup (cost+time) for SNAPSHOT ---
    combined = {}
    for src in (cost, time_df):
        for _, r in src.iterrows():
            if pd.isna(r.get("geo_latitude")):
                continue
            geo = (r["geo_latitude"], r["geo_longitude"])
            combined.setdefault("full:" + _norm(r["project_name"]), geo)
            combined.setdefault("pre:" + _norm(str(r["project_name"]).split("[")[0]), geo)

    n_fill_snap = 0
    for i in snap.index[snap["geo_latitude"].isna()]:
        hit = combined.get("full:" + _norm(snap.at[i, "project_name"]))
        if hit is None:
            hit = combined.get("pre:" + _norm(snap.at[i, "project_name"]))
        if hit:
            snap.at[i, "geo_latitude"], snap.at[i, "geo_longitude"] = hit
            n_fill_snap += 1
    print(f"snap after combined-prop: {snap['geo_latitude'].notna().mean():.1%} (+{n_fill_snap})")

    # --- fuzzy fingerprint match for snapshot (strict: >=3 shared tokens) ---
    time_fp_rows = time_df[time_df["geo_latitude"].notna()]
    time_fps = list(time_fp_rows["project_name"].map(_fp))
    time_lats = list(time_fp_rows["geo_latitude"])
    time_lons = list(time_fp_rows["geo_longitude"])

    n_fuzzy = 0
    for i in snap.index[snap["geo_latitude"].isna()]:
        sf = _fp(snap.at[i, "project_name"])
        if not sf:
            continue
        best, bestn = None, 0
        for j, tf in enumerate(time_fps):
            inter = len(sf & tf)
            if inter > bestn:
                bestn, best = inter, j
        if bestn >= 3:
            snap.at[i, "geo_latitude"] = time_lats[best]
            snap.at[i, "geo_longitude"] = time_lons[best]
            n_fuzzy += 1
    print(f"snap after fuzzy-match: {snap['geo_latitude'].notna().mean():.1%} (+{n_fuzzy})")
    snap.to_csv(SNAP_CSV, index=False)

    # --- per-sector mean imputation for residual NaNs ---
    for name, df in (("cost", cost), ("time", time_df), ("snap", snap)):
        means = _sector_impute(df)
        if not means:
            print(f"{name}: no geo rows at all -> MULTI STATE fallback")
            df["geo_latitude"] = df["geo_latitude"].fillna(MULTI_STATE_GEO[0])
            df["geo_longitude"] = df["geo_longitude"].fillna(MULTI_STATE_GEO[1])
        else:
            lat, lon = pd.to_numeric(df["geo_latitude"], errors="coerce"), pd.to_numeric(df["geo_longitude"], errors="coerce")
            sec = df["sector"].astype(str)
            m_lat, m_lon = lat.copy(), lon.copy()
            for i in df.index[lat.isna() | lon.isna()]:
                s = sec.at[i]
                if s in means:
                    m_lat.at[i], m_lon.at[i] = means[s]
                else:
                    m_lat.at[i], m_lon.at[i] = MULTI_STATE_GEO
            df["geo_latitude"], df["geo_longitude"] = m_lat, m_lon
            print(f"{name}: imputed -> geo coverage {df['geo_latitude'].notna().mean():.1%}")
        path = {"cost": COST_CSV, "time": TIME_CSV, "snap": SNAP_CSV}[name]
        df.to_csv(path, index=False)

    # --- final report ---
    print("\nFinal geo coverage")
    for name, path in (("COST", COST_CSV), ("TIME", TIME_CSV), ("SNAP", SNAP_CSV)):
        df = pd.read_csv(path)
        lat = pd.to_numeric(df["geo_latitude"], errors="coerce")
        lon = pd.to_numeric(df["geo_longitude"], errors="coerce")
        uniq = df.assign(k=df["geo_latitude"].round(3).astype(str) + "," + df["geo_longitude"].round(3).astype(str))["k"].nunique()
        print(f"  {name}: n={len(df)}, geo coverage={lat.notna().mean():.1%}, distinct coords={uniq}")


if __name__ == "__main__":
    propagate_and_impute()