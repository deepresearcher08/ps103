"""Assemble a real project-level cost-overrun dataset from the MoSPI Flash Report.

Combines per-project rows from two annexures of the May 2024 Flash Report:
  - ANNEXURE VI  -> projects WITH cost overruns   (overrun_flag = 1, positive)
  - ANNEXURE V   -> projects ON schedule (no cost overrun) (overrun_flag = 0, negative)

Both share the 9-column layout:
  0=S.No, 1=Project, 2=Date of Approval, 3=Original/Revised cost, 4=Anticipated cost,
  5=Cumulative Expenditure, 6=Original/Rev DOC, 7=Anticipated DOC, 8=Cost Overrun (%)

Sector is read from the ALL-CAPS grouping row. This is REAL government data, so the
resulting binary cost-overrun dataset and the SPEC.md model built on it are grounded in
actual MoSPI figures rather than synthetic rows.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pdfplumber

logger = logging.getLogger(__name__)

# Annexure markers (subset match is fine because we locate by section + start page)
VI_MARKER = "ongoing projects having cost overruns"
V_MARKER = "details of on schedule projects"


def _clean_float(val) -> float | None:
    if val is None:
        return None
    s = str(val).replace(",", "").strip()
    # split on newline / dash to take the ORIGINAL cost (first value of "Orig/Rev")
    s = s.split("\n")[0].split("-")[0].strip().strip("-").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _clean_under_flag(val) -> tuple[float | None, float | None]:
    """Return (original_cost, revised_or_orig flag). Col3 may be '762.75' or '762.75\n-'
    or '2,146.86\n2,413.00'. Original = first token; a second token implies a revision."""
    if val is None:
        return None, None
    parts = str(val).replace(",", "").split("\n")
    orig = _clean_float(parts[0])
    rev = None
    if len(parts) > 1 and parts[1].strip():
        rev = _clean_float(parts[1])
    return orig, rev


def _is_sector_header(cells) -> bool:
    if not cells or len(cells) < 4:
        return False
    sl = str(cells[0] or "").strip()
    name = str(cells[1] or "").strip()
    rest = cells[3:]
    empty = all(not (c or "").strip() for c in rest)
    if sl or not name or not empty:
        return False
    letters = [ch for ch in name if ch.isalpha()]
    if not letters:
        return False
    return name == name.upper()


def _is_project_row(cells) -> bool:
    if not cells or len(cells) < 8:
        return False
    sl = str(cells[0] or "").strip()
    name = str(cells[1] or "").strip()
    return sl.isdigit() and bool(name) and name.lower() not in ("total", "grand total")


def _parse_section(pdf, start_page: int, end_before_pages: int) -> list[dict]:
    rows: list[dict] = []
    current_sector = "Unknown"
    for i in range(start_page, min(len(pdf.pages), end_before_pages)):
        text = (pdf.pages[i].extract_text() or "").lower()
        # stop when a later annexure header appears
        for stop in ("details of projects without date", "list of projects ahead of schedule"):
            if i > start_page and stop in text:
                return rows
        for table in pdf.pages[i].extract_tables():
            for cells in table:
                if _is_sector_header(cells):
                    current_sector = str(cells[1]).replace("\n", " ").strip()
                    continue
                if _is_project_row(cells):
                    name = str(cells[1]).replace("\n", " ").strip()
                    orig, rev = _clean_under_flag(cells[3])
                    ant = _clean_float(cells[4])
                    exp = _clean_float(cells[5])
                    cor = _clean_float(cells[8]) if len(cells) > 8 else None
                    if orig is None or ant is None:
                        continue
                    rows.append({
                        "sector": current_sector,
                        "project_name": name,
                        "original_cost_cr": orig,
                        "revised_cost_cr": rev,
                        "anticipated_cost_cr": ant,
                        "expenditure_cum_cr": exp,
                        "cost_overrun_pct": cor,
                    })
    return rows


def locate_section(pdf, marker: str, start_from: int) -> int | None:
    for i in range(start_from, len(pdf.pages)):
        if marker in (pdf.pages[i].extract_text() or "").lower():
            return i
    return None


def build_real_dataset(pdf_path: Path) -> pd.DataFrame:
    """Return a unified real cost-overrun dataset with a binary label."""
    with pdfplumber.open(str(pdf_path)) as pdf:
        # ANNEXURE V start (around idx ~100 by ordering; V precedes VI)
        v_start = locate_section(pdf, V_MARKER, start_from=90)
        vi_start = locate_section(pdf, VI_MARKER, start_from=(v_start or 90) + 1)

        positive = _parse_section(pdf, vi_start or 129, end_before_pages=(vi_start or 129) + 40)
        negative = _parse_section(pdf, v_start or 95, end_before_pages=(vi_start or 129))

    pos_df = pd.DataFrame(positive)
    neg_df = pd.DataFrame(negative)

    if not pos_df.empty:
        pos_df["overrun_flag"] = 1
    if not neg_df.empty:
        neg_df["overrun_flag"] = 0

    df = pd.concat([pos_df, neg_df], ignore_index=True)
    # Derive overrun ratio where anticipated cost available (SPEC.md target).
    df["overrun_ratio"] = (df["anticipated_cost_cr"] - df["original_cost_cr"]) / df["original_cost_cr"]
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pdf = Path(r"C:\Users\soumy\AppData\Local\Temp\opencode\pdfs\FlashReport_May_2024.pdf")
    data = build_real_dataset(pdf)
    print("Total rows:", len(data))
    print("Flag balance:\n", data["overrun_flag"].value_counts())
    print("\nSectors involved:", data["sector"].nunique())
    print(data["sector"].value_counts().head(10).to_string())
    print("\noverrun_ratio describe:")
    print(data["overrun_ratio"].describe())
    out = Path(__file__).resolve().parents[1] / "data" / "real_mospi_projects.csv"
    data.to_csv(out, index=False)
    print("\nSaved to", out)