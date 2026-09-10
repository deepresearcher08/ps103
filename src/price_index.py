"""Sector-mapped input-price escalation features (user feature #2).

Sources (all public, pre-dating any label in the project tables):
  - WPI, base 2011-12 = 100, FINANCIAL-YEAR averages, labelled by end-FY:
      * metal products (steel proxy)   FY2013-FY2024  (OEA/DPIIT series)
      * cement & lime                  FY2013-FY2024  (OEA/DPIIT series)
  - USD/INR annual average, calendar year, 2013-2025 (FRED "AEXINUS" / RBI HBS)
  - 2025 WPI not included: no fully-finalised value in the source series yet.

REFERENCE_YEAR = 2024 (last fully-final WPI year). The escalation features
measure the input-price level at sanction and the escalation from sanction to a
FIXED reference - so they are identical in training and serving and contain no
future or label information (leak-free by construction).

HONESTY: within the ONE available vintage, approval years cluster 2018-2022 and
every price move is a smooth function of year, so the features are ~collinear
with approval_year and add little in-fold (expect ~0 grouped-CV delta, as with
schedule_risk). Their real payoff is OUT-OF-SAMPLE: a project sanctioned in a
future high/low-price vintage gets a genuinely exogenous feature value.

Mapping: import-equipment / globally-priced sectors use USD/INR; civil-works
sectors use the 50/50 steel-cement construction WPI; default = construction.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "input_prices.csv"
REFERENCE_YEAR = 2024

FX_SECTORS = {"POWER", "PETROLEUM", "COAL", "MINES", "ATOMIC ENERGY", "STEEL"}


@lru_cache(maxsize=1)
def _table() -> pd.DataFrame:
    return pd.read_csv(CSV).set_index("year")


def series_for_sector(sector: str | None) -> str:
    s = str(sector or "").upper().strip()
    return "fx_usd" if s in FX_SECTORS else "construction"


def _construction(year: int) -> float | None:
    t = _table()
    metal, cement = t.loc[year, "wpi_metal"], t.loc[year, "wpi_cement"]
    if pd.isna(metal) or pd.isna(cement):
        return None
    return float(0.5 * (metal + cement))


def index_at(sector: str | None, year: int) -> float | None:
    """Return the sector's input-price index at a year, or None if unknown."""
    t = _table()
    try:
        y = int(year)
        if y not in t.index:
            return None
    except (TypeError, ValueError):
        return None
    if series_for_sector(sector) == "fx_usd":
        v = t.loc[y, "fx_usd"]
        return float(v) if pd.notna(v) else None
    return _construction(y)


def escalations(sector: str | None, approval_year, reference_year: int = REFERENCE_YEAR):
    """(price_esc, price_level_approval) for a row.

    price_esc            = log(index(reference) / index(approval))  <- escalation
    price_level_approval = log(index(approval))                     <- sanction regime
    Both NaN when the approval year is missing or outside the index table
    (callers impute to the training median, exactly like the pace features)."""
    try:
        ay = float(approval_year)
    except (TypeError, ValueError):
        return np.nan, np.nan
    if not np.isfinite(ay) or int(ay) not in _table().index:
        return np.nan, np.nan
    ay = int(ay)
    ia = index_at(sector, ay)
    ir = index_at(sector, reference_year)
    if ia is None or ir is None or ia <= 0:
        return np.nan, np.nan
    return float(np.log(ir / ia)), float(np.log(ia))


def attach_price_index(df: pd.DataFrame, reference_year: int = REFERENCE_YEAR) -> pd.DataFrame:
    """Add price_esc / price_level_approval columns from sector + approval_year."""
    out = df.copy()
    ay = out.get("approval_year", pd.Series(np.nan, index=out.index))
    rows = [escalations(s, a, reference_year)
            for s, a in zip(out["sector"], ay)]
    out["price_esc"] = np.array([r[0] for r in rows], dtype=float)
    out["price_level_approval"] = np.array([r[1] for r in rows], dtype=float)
    return out