"""Parse the real MoSPI Flash Report STATE-WISE time-overrun table (TABLE-8).

TABLE-8 'Extent of time overrun ... (State Wise)' is at PDF index 20 in the May 2024
Flash Report and lists each Indian state/UT with: no. of projects, cost overrun (%),
and Range of T.O.R (time overrun months). This authentic geospatial breakdown powers
the India choropleth in the dashboard.

Output: data/real_mospi_state.csv columns:
  state, n_projects, cost_overrun_pct, tor_min_months, tor_max_months, delay_risk
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
import pdfplumber

logger = logging.getLogger(__name__)

PDF = Path(__file__).resolve().parents[1] / "data/raw/FlashReport_May_2024.pdf"

STATES = {
    "ANDHRA PRADESH", "ARUNACHAL PRAD", "ARUNACHAL PRADESH", "ASSAM", "BIHAR", "CHHATTISGARH",
    "GOA", "GUJARAT", "HARYANA", "HIMACHAL PRADE", "HIMACHAL PRADESH", "JMU AND KSH", "JHARKHAND",
    "JAMMU AND KASHMIR", "JAMMU AND KASH", "KARNATAKA", "KERALA", "MADHYA PRADESH",
    "MAHARASHTRA", "MANIPUR", "MEGHALAYA", "MIZORAM", "NAGALAND", "ODISHA", "PUNJAB",
    "RAJASTHAN", "SIKKIM", "TAMIL NADU", "TRIPUR", "TRIPURA", "UTTAR PRADESH", "WEST BENGAL",
    "ANDAMAN AND NI", "ANDAMAN AND NICOBAR", "DELHI", "LADAKH", "TELANGANA", "UTTARAKHAND",
    "MULTI STATE",
}
# normalize truncated names to canonical choropleth labels
NORM = {
    "ARUNACHAL PRAD": "ARUNACHAL PRADESH", "HIMACHAL PRADE": "HIMACHAL PRADESH",
    "JAMMU AND KASH": "JAMMU AND KASHMIR", "TRIPUR": "TRIPURA", "ANDAMAN AND NI": "ANDAMAN AND NICOBAR",
}


def _cf(val) -> float | None:
    if val is None:
        return None
    s = str(val).replace(",", "").replace("\n", " ").strip()
    parts = s.split()
    if not parts:
        return None
    try:
        return float(parts[0].split("-")[0])
    except ValueError:
        return None


def _nn(val, default=0) -> float:
    v = _cf(val)
    return v if v is not None else default


def parse_state_tor(pdf_path: Path = PDF) -> pd.DataFrame:
    rows, seen = [], set()
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i in range(8, 60):
            text = (pdf.pages[i].extract_text() or "")
            if not ("Extent of time" in text and "State" in text):
                continue
            for table in pdf.pages[i].extract_tables():
                for cells in table:
                    if not cells or len(cells) < 10:
                        continue
                    st = str(cells[1] or "").replace("\n", " ").strip().upper()
                    if st not in STATES:
                        continue
                    canonical = NORM.get(st, st)
                    if canonical in seen:
                        continue
                    seen.add(canonical)
                    tor_range = str(cells[9] or "").replace("\n", " ")
                    mm = re.search(r"(-?\d+)\s*-\s*(-?\d+)", tor_range)
                    tor_min = tor_max = None
                    if mm:
                        tor_min, tor_max = int(mm.group(1)), int(mm.group(2))
                    rows.append({
                        "state": canonical,
                        "n_projects": int(_nn(cells[2])),
                        "cost_overrun_pct": _cf(cells[5]),
                        "tor_min_months": tor_min,
                        "tor_max_months": tor_max,
                    })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["delay_risk"] = (df["tor_max_months"].clip(upper=120)) / 120.0
    return df


def load_state_data() -> pd.DataFrame:
    out = Path(__file__).resolve().parents[1] / "data" / "real_mospi_state.csv"
    if out.exists():
        return pd.read_csv(out)
    df = parse_state_tor()
    if not df.empty:
        df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_state_data()
    print(f"{len(df)} states/UTs")
    print(df[["state", "n_projects", "cost_overrun_pct", "tor_min_months", "tor_max_months"]].to_string() if not df.empty else "empty")