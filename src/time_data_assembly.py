"""Assemble a REAL time-overrun dataset from the MoSPI Flash Report annexures.

Targets (time overrun):
  - delayed (binary): Annexure VII (Delayed Projects) = 1 ; Annexure IV (Ahead of
    Schedule) + V (On Schedule) = 0.
  - tor_months (float regression): Time Overrun in Months from the delayed annexure.

Annexure VII extended 11-column layout:
  0=S.No, 1=Project, 2=Date of Approval, 3=Orig/Rev cost, 4=Anticipated cost,
  5=Cumulative Expenditure, 6=Orig/Rev DOC, 7=Anticipated DOC, 8=Time Overrun (Mo),
  9=Cost Overrun (%), 10=Reasons for Delay.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber

from src.geo import enrich_geo

logger = logging.getLogger(__name__)

VII_MARKER = "details of delayed projects"
IV_MARKER = "details of projects ahead of schedule"
V_MARKER = "details of on schedule projects"
DOC_COLS = {7: 8, 6: 7, 5: 6}  # (orig_idx, ant_idx) per annexure col layout -> n/a


def _clean_float(val) -> float | None:
    if val is None:
        return None
    s = str(val).replace(",", "").strip()
    s = s.split("\n")[0].split("-")[0].strip().strip("-").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _clean_orig_rev(val) -> tuple[float | None, float | None]:
    if val is None:
        return None, None
    parts = str(val).replace(",", "").split("\n")
    orig = _clean_float(parts[0])
    rev = _clean_float(parts[1]) if len(parts) > 1 and parts[1].strip() else None
    return orig, rev


def _approx_year(val) -> int | None:
    """Convert a 'MM/YYYY' string to a year (coarse, for planned-duration use)."""
    if not val:
        return None
    s = str(val).replace("\n", " ").strip()
    # take first token that looks like a date (doc may embed second date)
    for tok in s.split():
        if "/" in tok:
            p = tok.strip()
            if p.count("/") == 1:
                parts = p.split("/")
                if len(parts[1]) == 4:
                    try:
                        return int(parts[1])
                    except ValueError:
                        return None
    return None


def _is_sector_header(cells):  # identical to real_data_assembly
    if not cells or len(cells) < 4:
        return False
    sl = str(cells[0] or "").strip()
    name = str(cells[1] or "").strip()
    rest = cells[3:]
    if sl or not name or not all(not (c or "").strip() for c in rest):
        return False
    letters = [ch for ch in name if ch.isalpha()]
    return bool(letters) and name == name.upper()


def _is_project_row(cells):
    if not cells or len(cells) < 8:
        return False
    sl = str(cells[0] or "").strip()
    name = str(cells[1] or "").strip()
    return sl.isdigit() and bool(name) and name.lower() not in ("total", "grand total")


def _parse_projects(pdf, start: int, stop_markers: tuple[str, ...]) -> list[dict]:
    rows, sector = [], "Unknown"
    for i in range(start, len(pdf.pages)):
        text = (pdf.pages[i].extract_text() or "").lower()
        if i > start and any(m in text for m in stop_markers):
            break
        for table in pdf.pages[i].extract_tables():
            for cells in table:
                if _is_sector_header(cells):
                    sector = str(cells[1]).replace("\n", " ").strip()
                    continue
                if not _is_project_row(cells):
                    continue
                name = (cells[1] or "").replace("\n", " ").strip()
                apx = _clean_float(cells[3])
                ant = _clean_float(cells[4]) if len(cells) > 4 else None
                exp = _clean_float(cells[5]) if len(cells) > 5 else None
                doc_orig = cells[6] if len(cells) > 6 else None
                doc_ant = cells[7] if len(cells) > 7 else None
                ap_year = _approx_year(cells[2])
                doc_year = _approx_year(doc_orig) or _approx_year(doc_ant)
                tor = _clean_float(cells[8]) if len(cells) > 8 else None
                cor = _clean_float(cells[9]) if len(cells) > 9 else None
                reason = (cells[10] if len(cells) > 10 else None)
                rows.append({
                    "sector": sector,
                    "project_name": name,
                    "original_cost_cr": apx,
                    "anticipated_cost_cr": ant,
                    "expenditure_cum_cr": exp,
                    "approval_year": ap_year,
                    "planned_doc_year": doc_year,
                    "tor_months": tor,
                    "cost_overrun_pct": cor,
                    "reasons": None if reason is None else str(reason).replace("\n", " ").strip(),
                })
    return rows


def build_time_dataset(pdf_path: Path) -> pd.DataFrame:
    with pdfplumber.open(str(pdf_path)) as pdf:
        iv = v = vii = None
        for i in range(80, 240):
            t = (pdf.pages[i].extract_text() or "").lower()
            if IV_MARKER in t and iv is None:
                iv = i
            if V_MARKER in t and v is None:
                v = i
            if VII_MARKER in t and vii is None:
                vii = i

        delayed = _parse_projects(pdf, vii, stop_markers=("details of projects without",))
        on_time = _parse_projects(pdf, min(i for i in (iv, v) if i is not None),
                                  stop_markers=("ongoing projects having cost overruns",))

    pos = pd.DataFrame(delayed) if delayed else pd.DataFrame()
    neg = pd.DataFrame(on_time) if on_time else pd.DataFrame()

    if not pos.empty:
        pos["delayed"] = 1
    if not neg.empty:
        neg["delayed"] = 0

    df = pd.concat([pos, neg], ignore_index=True) if (not pos.empty or not neg.empty) \
        else pd.DataFrame()
    df["planned_duration_years"] = None
    m = (df["planned_doc_year"].notna()) & (df["approval_year"].notna())
    df.loc[m, "planned_duration_years"] = df.loc[m, "planned_doc_year"] - df.loc[m, "approval_year"]
    cor = df["cost_overrun_pct"]
    df["cost_overrun_pct"] = df["cost_overrun_pct"].fillna(0)
    df["overrun_ratio"] = df["cost_overrun_pct"] / 100.0
    if "tor_months" not in df.columns:
        df["tor_months"] = np.nan
    if not df.empty:
        enrich_geo(df, "project_name")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pdf = Path(__file__).resolve().parents[1] / "data/raw/FlashReport_May_2024.pdf"
    data = build_time_dataset(pdf)
    print("Total rows:", len(data))
    print("Delayed balance:\n", data["delayed"].value_counts())
    print("\ntor_months describe (delayed only):")
    print(data.loc[data["delayed"] == 1, "tor_months"].describe())
    out = Path(__file__).resolve().parents[1] / "data" / "real_mospi_time.csv"
    data.to_csv(out, index=False)
    print("\nSaved to", out)