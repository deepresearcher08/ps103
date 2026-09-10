"""Parse the real MoSPI Flash Report PDF (Annexure VIII) into a project-level dataset.

Extracts per-project rows from "Details of Ongoing Projects having Time and Cost
Overruns w.r.t Original Schedule":
  - sector (from the grouping header row)
  - project name
  - original cost (Rs. crore)
  - anticipated / revised cost (Rs. crore)
  - cumulative expenditure (Rs. crore)
  - cost overrun %
  - time overrun (TOR) in months

This is REAL government data (unlike the synthetic generator). It gives the
positive class (projects with both time & cost overruns). Aggregate tables in the
same report provide sector project counts for building the negative/on-time class.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)

ANNEXURE_MARKER = "having time and cost overruns"
NEXT_MARKER = "without date of commissioning"

# Column meaning for the 10-col Annexure VIII table:
# 0=SI.No, 1=Project, 2=Date of Approval, 3=Original/Revised cost,
# 4=Anticipated cost, 5=Cumulative Expenditure, 6=Orig/Rev DOC, 7=Anticip DOC,
# 8=Cost Overrun (%), 9=TOR (months)


def _clean_float(val: str | None) -> float | None:
    if val is None:
        return None
    s = str(val).replace(",", "").replace(" ", "").strip()
    # Handle "12,320.00 - " style: take first numeric token
    s = s.split("-")[0].strip().strip("-").strip()
    s = s.split("\n")[0].strip()
    try:
        return float(s)
    except ValueError:
        return None


def _is_project_row(cells: list) -> bool:
    """A data row has a numeric serial number in column 0 and a name in column 1."""
    if not cells or len(cells) < 9:
        return False
    sl = str(cells[0] or "").strip()
    name = str(cells[1] or "").strip()
    return sl.isdigit() and bool(name) and name.lower() != "total"


def _is_sector_header(cells: list) -> bool:
    """A sector grouping row: cells[1] is an ALL-CAPS label (e.g. 'ATOMIC ENERGY').

    In Annexure VIII the sector is a bare ALL-CAPS row; the following Title-Case
    rows are implementing agencies / ministry divisions, which we must NOT treat
    as a sector.
    """
    if not cells or len(cells) < 4:
        return False
    sl = str(cells[0] or "").strip()
    name = str(cells[1] or "").strip()
    rest = cells[3:]
    empty = all(not (c or "").strip() for c in rest)
    if sl or not name or not empty:
        return False
    # ALL-CAPS (with spaces/punctuation but no lowercase letters)
    letters = [ch for ch in name if ch.isalpha()]
    if not letters:
        return False
    return name == name.upper()


def parse_annexure_viii(pdf_path: Path) -> list[dict]:
    """Extract project rows from Annexure VIII of a Flash Report PDF."""
    rows: list[dict] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        # locate the real annexure start (skip TOC pages at the front; the data
        # table appears well after the table of contents, ~page 240+)
        start = None
        for i in range(200, len(pdf.pages)):
            text = (pdf.pages[i].extract_text() or "").lower()
            if ANNEXURE_MARKER in text:
                start = i
                break
        if start is None:
            logger.warning("Annexure VIII not found in %s", pdf_path.name)
            return rows

        current_sector = "Unknown"
        grand_total_rows = 0
        for i in range(start, min(len(pdf.pages), start + 60)):
            text = (pdf.pages[i].extract_text() or "").lower()
            if i > start and NEXT_MARKER in text:
                break
            for table in pdf.pages[i].extract_tables():
                for cells in table:
                    if _is_sector_header(cells):
                        current_sector = str(cells[1]).replace("\n", " ").strip()
                        continue
                    if _is_project_row(cells):
                        name = str(cells[1]).replace("\n", " ").strip()
                        orig = _clean_float(cells[3])
                        ant = _clean_float(cells[4])
                        exp = _clean_float(cells[5])
                        cor = _clean_float(cells[8])
                        tor = _clean_float(cells[9])
                        if orig is None or ant is None:
                            continue
                        # skip "Grand Total" aggregation rows carried into cells[1]
                        if name.lower() in ("grand total", "total"):
                            if name.lower() == "grand total":
                                grand_total_rows += 1
                            continue
                        rows.append({
                            "sector": current_sector,
                            "project_name": name,
                            "original_cost_cr": orig,
                            "anticipated_cost_cr": ant,
                            "expenditure_cum_cr": exp,
                            "cost_overrun_pct": cor,
                            "tor_months": tor,
                        })
        logger.info("Parsed %d project rows from Annexure VIII (%s)", len(rows), pdf_path.name)
    return rows


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pdf = Path(__file__).resolve().parents[1] / "data/raw/FlashReport_May_2024.pdf"
    data = parse_annexure_viii(pdf)
    import pandas as pd

    df = pd.DataFrame(data)
    print("Rows:", len(df))
    print("Sectors:", df["sector"].nunique())
    print(df["sector"].value_counts().to_string())
    print("\ncost_overrun_pct describe:")
    print(df["cost_overrun_pct"].describe())
    print("\ntor_months describe:")
    print(df["tor_months"].describe())
