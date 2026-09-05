"""Parse the new-format MoSPI Flash Report (2025+).

The 2025 reports replaced the old annexures (IV/V/VII) with a single
"Table 7: Project List: Ongoing Projects" spanning state -> sector -> project
rows. That table has NO explicit delayed/TOR/cost-overrun columns, so we DERIVE
the labels from the shown dates and costs, which is self-consistent inside the
snapshot:

  delayed          = anticipated DOC later than original (sanctioned) DOC
  tor_months       = (anticipated DOC - original DOC) in months
  cost_overrun_pct = (max(revised, anticipated) - original) / original * 100   (>= 0)
  overrun_flag     = cost_overrun_pct > 0
  planned duration = original_doc_year - approval_year

Parsing is position-based (pdfplumber words grouped into lines by 'top', bucketed
by x-band) because the state/sector bands wrap and extract_table mis-splits them.
State is captured best-effort from the left band; sector wraps are resolved by
matching against the known sector list.

CLI:
    python -m src.flash2025_parser              # parse -> data/mospi_may2025_projects.csv
"""
from __future__ import annotations

import re
from pathlib import Path

import pdfplumber
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "FR_May2025.pdf"
OUT = ROOT / "data" / "mospi_may2025_projects.csv"

HEADER_MARK = "Project List: Ongoing Projects"

KNOWN_SECTORS = {
    "ATOMIC ENERGY", "CIVIL AVIATION", "COAL", "DEPARTMENT OF HIGHER EDUCATION",
    "DPIIT", "FINANCE", "HEALTH AND FAMILY WELFARE", "HOME AFFAIRS", "MINES",
    "PETROLEUM", "POWER", "RAILWAYS", "RENEWABLE ENERGY",
    "ROAD TRANSPORT AND HIGHWAYS", "SHIPPING AND PORTS", "STEEL", "SOCIAL JUSTICE",
    "TELECOMMUNICATIONS", "URBAN DEVELOPMENT", "WATER RESOURCES", "NEW & RENEWABLE ENERGY",
    "RAILWAY", "MINISTRY OF RAILWAYS", "AGRICULTURE", "COAL AND LIGNITE",
}


def _num(v) -> float | None:
    if v is None:
        return None
    s = str(v).replace(",", "").strip().split("\n")[0].strip()
    if not s or s.upper() in ("N.A.", "NA", "-", "--", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _snap_sector(v: str) -> str:
    """Snap a possibly-fragmented sector label to its canonical known name."""
    if not v:
        return v
    vw = _norm(v).split()
    flat = _norm(v).replace(" ", "")
    for s in KNOWN_SECTORS:
        sw = s.split()
        flat_s = s.replace(" ", "")
        if flat in flat_s or flat_s in flat:
            return s
        if vw == sw[: len(vw)]:
            return s
        if vw == sw[-len(vw):] and len(vw) >= 2:
            return s
    return v


def _month_year(v) -> tuple[int, int] | None:
    if not v:
        return None
    s = str(v).strip().strip("(){}").split("\n")[0].strip()
    m = re.search(r"(\d{1,2})[/-](\d{4})", s)
    if m:
        return int(m.group(2)), int(m.group(1))
    m = re.search(r"\((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-\s/](\d{4})\)", s, re.I)
    if m:
        months = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                  "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
        return int(m.group(2)), months[m.group(1).lower()[:3]]
    return None


def _iso(d):
    return (d[0] * 12 + d[1]) if d else None


def _words_in_band(words, lo, hi):
    return " ".join(w["text"] for w in sorted(words, key=lambda w: w["x0"])
                    if lo <= w["x0"] < hi).strip()


def _lines(page):
    ws = page.extract_words(x_tolerance=1)
    ws.sort(key=lambda w: (round(w["top"]), w["x0"]))
    lines = []
    cur_top, cur = None, []
    for w in ws:
        if cur_top is None or abs(w["top"] - cur_top) <= 3.0:
            if cur_top is None:
                cur_top = w["top"]
            cur.append(w)
        else:
            lines.append((cur_top, cur))
            cur, cur_top = [w], w["top"]
    if cur:
        lines.append((cur_top, cur))
    return lines


def _skip_line(txt):
    if not txt.strip():
        return True
    low = txt.lower()
    if any(k in low for k in ("mospi_", "table:-7", "project list: ongoing",
                              "abbreviations", "total", "date ", "commissioning",
                              "cumulative", "physical", "approval (mm/yyyy)",
                              "state ", "sector sl", "sl no")):
        return True
    # standalone page number / stray year token
    if re.fullmatch(r"\d{1,3}|Page \d+", txt.strip()):
        return True
    return False


def _norm(s):
    return re.sub(r"\s+", " ", s or "").strip().upper().rstrip(":")


def _merge_sector(buf: str, frag: str) -> str:
    f = _norm(frag)
    if not f or f in ("N.A.", "NA"):
        return buf
    b = _norm(buf)
    cand = (b + " " + f).strip() if b else f
    if cand == b:
        return b
    if cand in KNOWN_SECTORS:
        return cand
    if f in KNOWN_SECTORS:
        return f
    if b in KNOWN_SECTORS:
        fw = set(f.split())
        return b if fw.issubset(b.split()) else f
    return f


def parse_flash2025(path=RAW) -> pd.DataFrame:
    records = []
    pending_state, pending_sector = "", ""
    current = None
    name_buf = ""
    with pdfplumber.open(path) as pdf:
        for idx, page in enumerate(pdf.pages):
            if idx == 1 or HEADER_MARK not in (page.extract_text() or ""):
                continue
            for _top, words in _lines(page):
                txt = " ".join(w["text"] for w in words).strip()
                if _skip_line(txt):
                    continue
                st = _words_in_band(words, 30, 62).upper()
                se = _words_in_band(words, 62, 120).upper()
                sl = _words_in_band(words, 120, 146)
                name = _words_in_band(words, 187, 296)
                appr = _words_in_band(words, 296, 344)
                doc = _words_in_band(words, 344, 445)
                cost = _words_in_band(words, 439, 515)
                expend = _words_in_band(words, 505, 560)
                prog = _words_in_band(words, 555, 1000)
                sl_no = int(sl) if sl.isdigit() else None
                if sl_no is not None:
                    if current is not None:
                        records.append(current)
                        current = None
                    if st:
                        pending_state = st
                    if se and se.upper() != "N.A.":
                        pending_sector = _merge_sector(pending_sector, se)
                    name_line = (name_buf + " " + name).strip() if (name_buf and name) else (name_buf or name)
                    current = {"state": pending_state or "",
                               "sector": pending_sector or "",
                               "sl_no": sl_no,
                               "name_line": name_line,
                               "name_buf_seen": bool(name_buf),
                               "approval": appr, "doc": doc,
                               "cost": cost, "expend": expend, "prog": prog}
                    name_buf = ""
                    continue
                if st:
                    pending_state = pending_state + " " + st if (pending_state and st not in pending_state) else (pending_state or st)
                    pending_state = re.sub(r"\s+", " ", pending_state).strip()
                if se and se.upper() != "N.A.":
                    pending_sector = _merge_sector(pending_sector, se)
                if current is None:
                    if name and "(" not in name:
                        name_buf = (name_buf + " " + name).strip()
                else:
                    if name and "(" not in name and "(" not in current["name_line"]:
                        current["name_line"] = (current["name_line"] + " " + name).strip()
                    if appr and not current["approval"]:
                        current["approval"] = appr
                    if doc:
                        current["doc"] = (current["doc"] + " " + doc).strip()
                    if cost:
                        current["cost"] = (current["cost"] + " " + cost).strip()
                    if expend:
                        current["expend"] = (current["expend"] + " " + expend).strip()
                    if prog:
                        current["prog"] = (current["prog"] + " " + prog).strip()
        if current is not None:
            records.append(current)
    # ---- assemble ----
    rows = []
    for r in records:
        name_part = re.split(r"\s*\(", r["name_line"], maxsplit=1)[0].strip().upper()
        if len(name_part) < 4:
            continue
        appr = _month_year(r["approval"])
        m_o = re.search(r"(\d{1,2}[/-]\d{4}|\([A-Z][a-z]+[-\s/]\d{4}\))", r["doc"])
        orig_doc = _month_year(m_o.group(1)) if m_o else None
        m_r = re.search(r"\(([^)]+)\)", r["doc"])
        rev_doc = _month_year(m_r.group(1)) if m_r else None
        m_a = re.search(r"\{([^}]+)\}", r["doc"])
        ant_doc = _month_year(m_a.group(1)) if m_a else None

        m_co = re.search(r"^\s*([\d,]+\.?\d*)", r["cost"])
        orig_cost = _num(m_co.group(1)) if m_co else None
        m_cr = re.search(r"\(([^)]+)\)", r["cost"])
        rev_cost = _num(m_cr.group(1)) if m_cr else None
        m_ca = re.search(r"\{([^}]+)\}", r["cost"])
        ant_cost = _num(m_ca.group(1)) if m_ca else None
        expend = _num(r["expend"])
        prog = _num(r["prog"])

        eff = rev_cost if rev_cost is not None else ant_cost
        ovr = ((eff - orig_cost) / orig_cost * 100) if (orig_cost and eff and orig_cost > 0) else 0.0
        ovr = max(0.0, round(ovr, 4))

        oi, ai = _iso(orig_doc), _iso(ant_doc)
        delayed, tor, p_dur = None, None, None
        if ant_doc and orig_doc:
            delayed = int(ai > oi)
            tor = float(max(ai - oi, 0))
            if appr:
                p_dur = orig_doc[0] - appr[0]

        rows.append({
            "sector": (r["sector"] or "").upper(),
            "project_name": name_part,
            "original_cost_cr": orig_cost,
            "revised_cost_cr": rev_cost,
            "anticipated_cost_cr": ant_cost,
            "expenditure_cum_cr": expend,
            "approval_year": appr[0] if appr else None,
            "original_doc_year": orig_doc[0] if orig_doc else None,
            "tor_months": tor,
            "delayed": delayed,
            "planned_duration_years": p_dur,
            "cost_overrun_pct": ovr,
            "overrun_flag": int(ovr > 0),
            "overrun_ratio": round(ovr / 100.0, 6),
            "physical_progress": prog,
        })
    df = pd.DataFrame(rows)
    df["sector"] = df["sector"].map(_snap_sector)
    df = df.drop_duplicates(subset=["sector", "project_name"], keep="last").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = parse_flash2025()
    df.to_csv(OUT, index=False)
    print(f"parsed {len(df)} projects -> {OUT}")
    print(f"  sectors: {df['sector'].nunique()}")
    print(f"  sector set: {sorted(df['sector'].dropna().unique())[:30]}")
    print(f"  delayed rate: {(df['delayed'] > 0).mean():.3f} (n={int((df['delayed'] > 0).sum())})")
    print(f"  overrun flag rate: {(df['overrun_flag'] > 0).mean():.3f}")
    print(f"  approval years {int(df['approval_year'].min())}..{int(df['approval_year'].max())}")
    for probe in ("POLAVARAM IRRIGATION PROJECT", "SARDAR SAROVAR PROJECT"):
        hit = df[df["project_name"].str.contains(probe, case=False)]
        if len(hit):
            r = hit.iloc[0]
            print(f"  sample: {probe} | delayed={r['delayed']} tor={r['tor_months']} "
                  f"overrun%={r['cost_overrun_pct']} exp/orig={(r['expenditure_cum_cr'] or 0)/r['original_cost_cr']:.2f}")