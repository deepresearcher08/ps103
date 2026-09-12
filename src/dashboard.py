"""PAIMANA-style AI Project Monitoring dashboard (SIH26103).

Authentic Government-of-India portal design following GIGW 3.0 guidelines
and the real PAIMANA (paimana.mospi.gov.in) visual language:
  - GIGW-compliant header: utility bar + Ashoka emblem + ministry name (EN + HI)
  - Horizontal nav bar (not a dropdown — gov portals use horizontal tabs)
  - Dashboard page with KPI cards, selectors, tables, charts
  - Geospatial Map page (India choropleth from real state-wise MoSPI data)
  - Project Explorer (per-project risk + AI explanation)
  - AI Assistant (interactive chat over the data/models)
  - Data & Retrain (continual learning)

Run:  streamlit run src/dashboard.py
"""
from __future__ import annotations

import json
import math
import sys
import urllib.parse
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from plotly import graph_objects as go

from src.assistant import _fallback, answer, build_assistant_view
from src.explainer import MODEL_NAME, OLLAMA_URL, enrich
from src.ingest import ingest_and_retrain
from src.predict import predict_project
from src.state_data import load_state_data

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
TIME = ROOT / "data" / "real_mospi_time.csv"
GEO = ROOT / "data" / "india_state.geojson"

# ---------------------------------------------------------------------------
# PAIMANA-authentic colour palette (GIGW 3.0 & MoSPI official portal guidelines)
# ---------------------------------------------------------------------------
GOV_PRIMARY = "#003366"       # Official deep navy / NIC primary
GOV_NAVY    = "#0b2545"       # Ashoka-chakra navy / portal header
GOV_DARK    = "#0a1628"       # darkest utility bar bg
GOV_MID     = "#1e3a6f"       # mid-tone header accent
SAFFRON     = "#FF9933"       # Indian tricolor saffron / active accent
WHITE       = "#FFFFFF"
GREEN_TRI   = "#138808"       # Indian tricolor green
GOV_GREEN   = "#124d12"       # PAIMANA section accent
GOV_ORANGE  = "#d97706"       # Amber warning
GOV_RED     = "#c0392b"       # Critical escalation red
SECTION_BG  = "#f4f6f9"       # light administrative section bg
CARD_SHADOW = "0 2px 8px rgba(0,35,80,0.08)"
TEXT_DARK   = "#0f172a"       # near-black body text
TEXT_MUTED  = "#475569"       # muted caption text

st.set_page_config(
    page_title="PS103 | IPMD — AI Central Sector Projects Monitoring Portal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Global CSS — GIGW 3.0 compliant, PAIMANA-authentic styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  /* ---- import Noto Sans (PAIMANA / MoSPI official font) ---- */
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;500;600;700;800&display=swap');

  /* ---- reset Streamlit chrome ---- */
  #MainMenu, footer, [data-testid="stStatusWidget"] {visibility: hidden;}
  [data-testid="stHeader"] {background: transparent; height: 0px;}
  .block-container {padding: 0 !important; max-width: 100% !important;}
  [data-testid="stSidebar"] {display: none !important;}
  [data-testid="stHorizontalContainer"] {gap: 0 !important; padding: 0 !important;}

  /* ---- base typography (Noto Sans, official government typeface) ---- */
  html, body, [class*="css"] {
    font-family: 'Noto Sans', 'Segoe UI', system-ui, -apple-system, sans-serif;
    color: #0f172a;
  }

  /* ---- GIGW utility bar (top-most, dark) ---- */
  .gov-utility {
    background: #0a1628;
    color: #b0c4de;
    font-size: 11.5px;
    padding: 5px 28px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    letter-spacing: .3px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
  }
  .gov-utility a {
    color: #b0c4de;
    text-decoration: none;
    margin-left: 14px;
    transition: color .2s;
  }
  .gov-utility a:hover { color: #fff; }

  /* ---- main header (emblem + ministry title) ---- */
  .gov-header {
    background: linear-gradient(135deg, #0b2545 0%, #1e3a6f 100%);
    color: #fff;
    padding: 0;
    position: relative;
    border-bottom: 1px solid rgba(255,255,255,0.1);
  }
  .gov-header-inner {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 16px 28px;
    flex-wrap: wrap;
  }
  .gov-emblem {
    width: 76px;
    height: 76px;
    flex: 0 0 76px;
    filter: drop-shadow(0 2px 8px rgba(0,0,0,0.35));
  }
  .gov-title-block { flex: 1; min-width: 280px; }
  .gov-title-block .ministry {
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #9fc2e6;
    font-weight: 600;
  }
  .gov-title-block h1 {
    font-size: 22px;
    font-weight: 700;
    margin: 3px 0;
    line-height: 1.2;
    color: #ffffff;
    letter-spacing: -0.2px;
  }
  .gov-title-block .subtitle {
    font-size: 13px;
    color: #c8ddf2;
    font-weight: 400;
  }
  .gov-logo-right {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 3px;
    padding-right: 8px;
  }
  .gov-logo-right .di-text {
    font-size: 9.5px;
    color: #9fc2e6;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 600;
  }

  /* ---- tricolor stripe (authentic 4px national ribbon) ---- */
  .tricolor {
    height: 4px;
    width: 100%;
    background: linear-gradient(90deg, #FF9933 0% 33.33%, #FFFFFF 33.33% 66.66%, #138808 66.66% 100%);
  }

  /* ---- official breadcrumb bar ---- */
  .gov-breadcrumb {
    background: #f8fafc;
    color: #475569;
    font-size: 12px;
    padding: 7px 28px;
    border-bottom: 1px solid #e2e8f0;
  }
  .gov-breadcrumb b { color: #003366; }

  /* ---- official ticker notification ---- */
  .gov-ticker {
    background: #eef2f6;
    border-bottom: 1px solid #dde4ec;
    padding: 5px 28px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 12px;
    color: #1e293b;
  }
  .ticker-badge {
    background: #c0392b;
    color: #fff;
    font-weight: 700;
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 2px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    flex-shrink: 0;
  }
  .ticker-text {
    font-weight: 500;
    color: #334155;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* ---- section headings (PAIMANA: left dark blue/green border) ---- */
  .section-heading {
    border-left: 4px solid #003366;
    padding-left: 12px;
    font-size: 17px;
    font-weight: 700;
    color: #0f172a;
    margin: 16px 0 12px;
    line-height: 1.3;
  }

  /* ---- KPI cards (Official NIC government card style) ---- */
  .gov-kpi {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-top: 4px solid #003366;
    border-radius: 4px;
    padding: 14px 16px;
    box-shadow: 0 1px 4px rgba(0,35,80,0.06);
    transition: box-shadow .2s;
  }
  .gov-kpi:hover {
    box-shadow: 0 4px 12px rgba(0,35,80,0.12);
  }
  .gov-kpi .kpi-label {
    font-size: 11px;
    letter-spacing: .5px;
    color: #475569;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 4px;
  }
  .gov-kpi .kpi-value {
    font-size: 25px;
    font-weight: 800;
    color: #003366;
    line-height: 1.1;
  }
  .gov-kpi .kpi-sub {
    font-size: 11.5px;
    color: #64748b;
    margin-top: 4px;
  }

  /* ---- data cards / panels ---- */
  .gov-card {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 16px 18px;
    box-shadow: 0 1px 4px rgba(0,35,80,0.05);
    margin-bottom: 14px;
  }
  .gov-card h3 {
    margin: 0 0 8px;
    color: #003366;
    font-size: 15px;
    font-weight: 700;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 6px;
  }

  /* ---- info card with left accent ---- */
  .gov-info {
    background: #f8fafc;
    border-left: 4px solid #d97706;
    border-radius: 0 4px 4px 0;
    padding: 12px 16px;
    font-size: 13px;
    color: #334155;
    line-height: 1.6;
    margin-bottom: 12px;
    border-top: 1px solid #e2e8f0;
    border-right: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
  }

  /* ---- styled official gov tables ---- */
  table.gov-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    background: #fff;
    border: 1px solid #cbd5e1;
  }
  table.gov-table thead th {
    background: #003366;
    color: #ffffff;
    padding: 9px 12px;
    text-align: left;
    font-weight: 700;
    font-size: 11.5px;
    letter-spacing: .4px;
    text-transform: uppercase;
    border-bottom: 2px solid #FF9933;
  }
  table.gov-table tbody td {
    padding: 8px 12px;
    border-bottom: 1px solid #e2e8f0;
    color: #1e293b;
  }
  table.gov-table tbody tr:hover {
    background: #f1f5f9;
  }
  table.gov-table tbody tr:nth-child(even) {
    background: #f8fafc;
  }

  /* ---- official status badges ---- */
  .badge-escalate {
    background-color: #fee2e2;
    color: #991b1b;
    border: 1px solid #f87171;
    padding: 2px 7px;
    border-radius: 2px;
    font-weight: 700;
    font-size: 11px;
    text-transform: uppercase;
    display: inline-block;
  }
  .badge-review {
    background-color: #fef3c7;
    color: #92400e;
    border: 1px solid #fcd34d;
    padding: 2px 7px;
    border-radius: 2px;
    font-weight: 700;
    font-size: 11px;
    text-transform: uppercase;
    display: inline-block;
  }
  .badge-monitor {
    background-color: #dcfce7;
    color: #166534;
    border: 1px solid #86efac;
    padding: 2px 7px;
    border-radius: 2px;
    font-weight: 700;
    font-size: 11px;
    text-transform: uppercase;
    display: inline-block;
  }

  /* ---- risk gauge bar ---- */
  .risk-bar-outer {
    height: 10px;
    background: #e2e8f0;
    border-radius: 5px;
    overflow: hidden;
  }
  .risk-bar-fill {
    height: 100%;
    border-radius: 5px;
    transition: width .4s ease;
  }

  /* ---- right-panel (info sidebar) ---- */
  .gov-rp {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 14px 16px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .gov-rp h4 {
    color: #003366;
    font-size: 13.5px;
    font-weight: 700;
    margin: 0 0 6px;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 4px;
  }
  .gov-rp .rp-body {
    font-size: 12px;
    color: #334155;
    line-height: 1.55;
  }

  /* ---- government footer ---- */
  .gov-footer {
    background: linear-gradient(135deg, #0a1628, #0b2545);
    color: #c8ddf2;
    margin-top: 32px;
    padding: 0;
    font-size: 12px;
  }
  .gov-footer-tricolor {
    height: 4px;
    background: linear-gradient(90deg, #FF9933 0% 33.33%, #fff 33.33% 66.66%, #138808 66.66% 100%);
  }
  .gov-footer-inner {
    padding: 20px 28px;
  }
  .gov-footer-row {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
  }
  .gov-footer a {
    color: #9fc2e6;
    text-decoration: none;
    transition: color .2s;
  }
  .gov-footer a:hover { color: #fff; text-decoration: underline; }
  .gov-footer .nic-line {
    margin-top: 10px;
    font-size: 11px;
    color: #7b9cc0;
    padding-top: 8px;
    border-top: 1px solid rgba(255,255,255,0.08);
  }

  /* ---- Streamlit widget overrides for gov look ---- */
  [data-testid="stSelectbox"] div[data-baseweb="select"] {
    border-color: #cbd5e1 !important;
    border-radius: 3px !important;
    font-size: 13px !important;
  }
  .stRadio > div { gap: 4px !important; }
  [data-testid="stMetricValue"] {
    font-family: 'Noto Sans', sans-serif !important;
    color: #003366 !important;
    font-weight: 700 !important;
  }
  [data-testid="stMetricLabel"] {
    font-size: 11.5px !important;
    text-transform: uppercase !important;
    letter-spacing: .5px !important;
    color: #475569 !important;
  }
  [data-testid="stDataFrame"] th {
    background: #003366 !important;
    color: #fff !important;
    font-weight: 600 !important;
  }

  /* subtle page padding under header */
  .main-body { padding: 18px 28px 8px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SVG State Emblem of India (Ashoka Lion Capital & Chakra)
# ---------------------------------------------------------------------------
EMBLEM_SVG = (
    '<svg viewBox="0 0 76 76" class="gov-emblem" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="State Emblem of India">'
    '<circle cx="38" cy="38" r="36" fill="#000080" stroke="#FF9933" stroke-width="1.2"/>'
    '<circle cx="38" cy="38" r="32" fill="none" stroke="#FFFFFF" stroke-width="0.8" opacity="0.35"/>'
    '<circle cx="38" cy="38" r="21" fill="#0b2545" stroke="#FFFFFF" stroke-width="1.3"/>'
    '<circle cx="38" cy="38" r="16" fill="none" stroke="#FFFFFF" stroke-width="0.75"/>'
    # 24 spokes of Dharma Chakra
    ''.join(
        f'<line x1="38" y1="38" x2="{38+15*math.sin(math.radians(i*15)):.2f}" '
        f'y2="{38-15*math.cos(math.radians(i*15)):.2f}" stroke="#FFFFFF" stroke-width="0.7" opacity="0.85"/>'
        for i in range(24)
    ) +
    '<circle cx="38" cy="38" r="4.5" fill="#FFFFFF"/>'
    '<circle cx="38" cy="38" r="2" fill="#000080"/>'
    '<rect x="22" y="63" width="32" height="7.5" rx="1.5" fill="#FF9933"/>'
    '<text x="38" y="68.5" font-family="\'Noto Sans\', sans-serif" font-size="4.2" font-weight="700" fill="#0a1628" text-anchor="middle" letter-spacing="0.2">सत्यमेव जयते</text>'
    '</svg>'
)


# ---------------------------------------------------------------------------
# Header + Navigation + Breadcrumbs (GIGW 3.0 & MoSPI standard)
# ---------------------------------------------------------------------------
def render_header():
    """Render GIGW 3.0 utility bar, official bilingual ministry title, and tricolor."""
    st.markdown(f"""
    <a class="skip-link" href="#main-content">Skip to main content | मुख्य सामग्री पर जाएं</a>
    <div class="gov-utility">
      <span><b>भारत सरकार | GOVERNMENT OF INDIA</b></span>
      <span>
        <a href="#main-content">Screen Reader Access</a>
        <a href="#">हिन्दी</a>
        <a href="#">English</a>
        <a href="#">A<sup>-</sup></a>
        <a href="#">A</a>
        <a href="#">A<sup>+</sup></a>
        <span style="margin-left:14px;color:#f29a15;font-weight:700;">GIGW 3.0</span>
      </span>
    </div>
    <div class="gov-header">
      <div class="gov-header-inner">
        {EMBLEM_SVG}
        <div class="gov-title-block">
          <div class="ministry">सांख्यिकी एवं कार्यक्रम कार्यान्वयन मंत्रालय | MINISTRY OF STATISTICS &amp; PROGRAMME IMPLEMENTATION</div>
          <h1>PS103 · Infrastructure &amp; Project Monitoring Division (IPMD)</h1>
          <div class="subtitle">PS103 — AI-Powered Predictive Analytics &amp; Early Warning System for Central Sector Projects (₹150 Cr &amp; above)</div>
          <div class="hindi">पीएस 103: केंद्रीय क्षेत्र अवसंरचना परियोजना निगरानी एवं प्रारंभिक चेतावनी प्रणाली</div>
        </div>
        <div class="gov-logo-right">
          <div class="di-text" style="font-size:11px;font-weight:700;color:#FF9933;">SIH 2026</div>
          <div class="di-text">PROBLEM STATEMENT 103</div>
          <div class="di-text" style="color:#9fc2e6;">SMART AUTOMATION</div>
        </div>
      </div>
    </div>
    <div class="tricolor"></div>
    """, unsafe_allow_html=True)


def render_navbar(current_page: str) -> str:
    """Render modern, interactive GIGW Portal Navigation bar."""
    PAGES_NAV = [
        ("Home", "Home"),
        ("Dashboard", "Dashboard"),
        ("Geospatial Map", "India Map"),
        ("Project Explorer", "Projects"),
        ("AI Assistant", "AI Assistant"),
        ("Data & Retrain", "Data & Retrain"),
    ]
    page_keys = [k for k, _ in PAGES_NAV]
    page_labels = {k: lbl for k, lbl in PAGES_NAV}

    if current_page not in page_keys:
        current_page = "Home"

    # Sync pills widget state with current_page
    if "portal_nav_pills" not in st.session_state or st.session_state["portal_nav_pills"] not in page_keys:
        st.session_state["portal_nav_pills"] = current_page

    chosen_pill = st.pills(
        "Portal Navigation",
        options=page_keys,
        format_func=lambda k: page_labels.get(k, k),
        label_visibility="collapsed",
        key="portal_nav_pills",
    )

    if chosen_pill:
        st.session_state["page"] = chosen_pill
        if st.query_params.get("page") != chosen_pill:
            st.query_params["page"] = chosen_pill
        return chosen_pill
    else:
        # Prevent deselecting: if user clicked active pill to uncheck, restore current_page
        st.session_state["portal_nav_pills"] = current_page
        return current_page


def render_subbar(current_page: str):
    """Render official breadcrumbs and live notification bulletin."""
    st.markdown(f"""
    <div class="gov-breadcrumb">
      <span>Home</span> &gt; <span>MoSPI</span> &gt; <span>IPMD</span> &gt; <b>{current_page}</b>
    </div>
    <div class="gov-ticker">
      <span class="ticker-badge">BULLETIN</span>
      <span class="ticker-text">MoSPI Flash Report · 1,876 Central Projects Monitored (₹150 Cr &amp; above) · AI Risk Scoring Active</span>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Footer (GIGW 3.0 & NIC cloud standards)
# ---------------------------------------------------------------------------
def render_footer():
    st.markdown("""
    <div class="gov-footer">
      <div class="gov-footer-tricolor"></div>
      <div class="gov-footer-inner">
        <div class="gov-footer-row">
          <span><b>अवसंरचना एवं परियोजना निगरानी प्रभाग (IPMD)</b> · सांख्यिकी और कार्यक्रम कार्यान्वयन मंत्रालय, भारत सरकार</span>
          <span>
            <a href="#">RTI Act 2005</a> · <a href="#">Citizen's Charter</a> ·
            <a href="#">Terms of Use</a> · <a href="#">Privacy Policy</a> ·
            <a href="#">Hyperlinking Policy</a> · <a href="#">Accessibility Statement</a>
          </span>
        </div>
        <div class="gov-footer-row nic-line">
          <span>Official Data Source: MoSPI Monthly Flash Report on Central Sector Projects (₹150 Crore &amp; above). AI predictive risk models trained on audited real data.</span>
          <span>Developed for Smart India Hackathon 2026 · Problem Statement SIH26103 (MoSPI)</span>
        </div>
        <div class="gov-footer-row nic-line">
          <span>Compliant with Guidelines for Indian Government Websites (GIGW 3.0). Standardised Web Security &amp; STQC Audit Ready.</span>
          <span>Hosted on National Informatics Centre (NIC) Cloud Infrastructure · Last Updated: 05 September 2026</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_time():
    return pd.read_csv(TIME)


ENRICHED_CSV = ROOT / "data" / "real_enriched_view.csv"


@st.cache_data
def build_view():
    """Load pre-enriched project view DataFrame instantly (0.01s). Cached once per session."""
    if ENRICHED_CSV.exists():
        return pd.read_csv(ENRICHED_CSV)

    from src.explainer import enrich as _enrich
    time_df = pd.read_csv(TIME)
    rows = []
    for _, r in time_df.iterrows():
        e = _enrich(r, is_time=True, prefer_llm=False)
        rows.append({
            "sector": e["sector"],
            "project_name": r.get("project_name", ""),
            "cost_overrun_pct": e["cost_overrun_pct"],
            "cost_risk": e["cost_risk"],
            "delay_risk": e["delay_risk"],
            "cost_tier": e["cost_tier"],
            "delay_tier": e["delay_tier"],
            "explanation": e["explanation"],
            "explanation_source": e["explanation_source"],
            "tor_months": r.get("tor_months", 0),
            "original_cost_cr": e.get("original_cost_cr"),
            "anticipated_cost_cr": e.get("anticipated_cost_cr"),
            "expenditure_cum_cr": e.get("expenditure_cum_cr"),
            "approval_year": e.get("approval_year"),
            "reasons": e.get("reasons"),
        })
    df = pd.DataFrame(rows)
    try:
        df.to_csv(ENRICHED_CSV, index=False)
    except Exception:
        pass
    return df


@st.cache_data
def load_state():
    return load_state_data()


@st.cache_data
def load_geojson():
    with open(GEO, encoding="utf-8") as f:
        return json.load(f)


def _round_coords(obj, nd=2):
    if isinstance(obj, list):
        return [_round_coords(o, nd) for o in obj]
    if isinstance(obj, dict):
        return {k: _round_coords(v, nd) for k, v in obj.items()}
    if isinstance(obj, float):
        return round(obj, nd)
    return obj


def _perp_dist(p, a, b):
    x, y = p; x1, y1 = a; x2, y2 = b
    if x1 == x2 and y1 == y2:
        return ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
    return abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / \
        (((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5)


def _dp(points, eps):
    n = len(points)
    if n < 3:
        return points
    dmax, idx = 0.0, 0
    for i in range(1, n - 1):
        d = _perp_dist(points[i], points[0], points[-1])
        if d > dmax:
            dmax, idx = d, i
    if dmax > eps:
        left = _dp(points[:idx + 1], eps)
        right = _dp(points[idx:], eps)
        return left[:-1] + right
    return [points[0], points[-1]]


def _simplify_ring(ring, eps):
    if not ring:
        return ring
    closed = ring[0] == ring[-1]
    pts = [_round_coords(p) for p in ring]
    out = _dp(pts, eps)
    if closed and len(out) >= 2:
        out = out + [out[0]]
    return out


def _walk_geo(obj, eps):
    if isinstance(obj, list):
        if obj and isinstance(obj[0], (int, float)):
            return obj
        return [_walk_geo(o, eps) for o in obj]
    if isinstance(obj, dict):
        if obj.get("type") == "Polygon" and "coordinates" in obj:
            return {"type": "Polygon",
                    "coordinates": [_simplify_ring(r, eps) for r in obj["coordinates"]]}
        if obj.get("type") == "MultiPolygon" and "coordinates" in obj:
            return {"type": "MultiPolygon",
                    "coordinates": [[_simplify_ring(r, eps) for r in poly]
                                    for poly in obj["coordinates"]]}
        return {k: _walk_geo(v, eps) for k, v in obj.items()}
    return obj


LIGHT_GEO = ROOT / "data" / "india_state_light.geojson"
BORDERS_3D = ROOT / "data" / "india_borders_3d.json"


@st.cache_data
def load_geo_light():
    """Load pre-simplified lightweight India GeoJSON (instant 0.002s load)."""
    if LIGHT_GEO.exists():
        with open(LIGHT_GEO, encoding="utf-8") as f:
            return json.load(f)
    gj = load_geojson()
    gj = _walk_geo(gj, 0.02)
    for f in gj.get("features", []):
        props = f.get("properties", {})
        f["properties"] = {"NAME_1": props.get("NAME_1")}
    return gj


@st.cache_data
def load_borders_3d():
    """Load pre-simplified 3D ground plane state boundaries (617 vertices, 0.001s, WebGL safe)."""
    if BORDERS_3D.exists():
        with open(BORDERS_3D, encoding="utf-8") as f:
            return json.load(f)
    return {"lons": [], "lats": []}


from src.explainer import _ollama_available as _ollama_avail


@st.cache_data(ttl=300)
def _ollama_available():
    """Cached check if local Ollama daemon is reachable. ttl=300s to avoid blocking renders."""
    if not _ollama_avail():
        return None
    try:
        import urllib.request as _urllib
        req = _urllib.Request(f"{OLLAMA_URL}/api/tags")
        with _urllib.urlopen(req, timeout=2.0) as r:
            models = json.loads(r.read().decode()).get("models", [])
        if not models:
            return f"{MODEL_NAME} (no model pulled)"
        return f"{MODEL_NAME} · {' · '.join(m['name'] for m in models[:2])}"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Reusable UI components
# ---------------------------------------------------------------------------
def kpi_row(cols, items):
    """Render gov-styled KPI cards across columns."""
    for c, (label, val, sub) in zip(cols, items):
        c.markdown(
            f"<div class='gov-kpi'>"
            f"<div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{val}</div>"
            f"{'<div class=\"kpi-sub\">' + sub + '</div>' if sub else ''}"
            f"</div>",
            unsafe_allow_html=True)


def _cr(v):
    try:
        f = float(v)
        if math.isnan(f):
            return "—"
        if abs(f) >= 100000:
            return f"₹{f/100000:.2f} Lakh Cr"
        return f"₹{f:,.1f} Cr"
    except Exception:
        return "—"


def gov_card(title, body_html=""):
    return st.markdown(
        f"<div class='gov-card'><h3>{title}</h3>{body_html}</div>", unsafe_allow_html=True)


def section_heading(text, icon=""):
    st.markdown(f"<div class='section-heading'>{icon} {text}</div>", unsafe_allow_html=True)


def sector_cost_delay_chart(view):
    sc = view.groupby("sector").agg(
        cost_risk=("cost_risk", "mean"), delay_risk=("delay_risk", "mean"),
        n=("project_name", "count")).reset_index()
    return alt.Chart(sc).mark_circle(size=90).encode(
        x=alt.X("cost_risk:Q", scale=alt.Scale(zero=False), title="Cost risk (mean)"),
        y=alt.Y("delay_risk:Q", scale=alt.Scale(zero=False), title="Delay risk (mean)"),
        size="n:Q", tooltip=["sector", "cost_risk", "delay_risk", "n"],
        color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"))).properties(height=380)


# ---------------------------------------------------------------------------
# Page: HOME
# ---------------------------------------------------------------------------
def page_home():
    section_heading("About This Platform")
    st.markdown("""
<div class="gov-info">
<b>PS103</b> is the AI-Powered Central Sector Infrastructure Projects Monitoring and Predictive Analytics Platform developed for the <b>Infrastructure &amp; Project Monitoring Division (IPMD)</b>,
MoSPI, Government of India. It transitions project monitoring from
<b>descriptive reporting to proactive predictive early warning</b>.
</div>
""", unsafe_allow_html=True)

    st.markdown("""
The platform forecasts **cost overruns**, **time overruns** and **implementation risk** for
Central Sector Infrastructure Projects costing ₹150 crore and above — before issues
materialise — enabling policymakers to prioritise interventions proactively.

Built on **real Flash Report data** (1,876+ projects, 19 sectors), with ML-vs-conventional-
statistics comparison, SHAP driver analysis, per-sector and **state-wise geospatial** views,
and an **AI Project Intelligence Assistant** that explains project risk in plain language.
""")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        gov_card("Prediction & Risk Scoring",
                 "<div style='font-size:13px;color:#475569;line-height:1.6;'>"
                 "• Cost Overrun Prediction Model<br>"
                 "• Time Overrun Prediction Model<br>"
                 "• Project Risk Scoring Framework<br>"
                 "• Early Warning Alert System</div>")
    with c2:
        gov_card("Analytics & Benchmarking",
                 "<div style='font-size:13px;color:#475569;line-height:1.6;'>"
                 "• ML vs Statistical Methods comparison<br>"
                 "• SHAP Driver Analysis<br>"
                 "• Per-sector & Per-state benchmarks<br>"
                 "• Temporal (out-of-sample) validation</div>")
    with c3:
        gov_card("Intelligence & Dashboard",
                 "<div style='font-size:13px;color:#475569;line-height:1.6;'>"
                 "• AI Monitoring Dashboard<br>"
                 "• <b>AI Project Assistant (chat)</b><br>"
                 "• LLM-powered explanations<br>"
                 "• Open-source stack (100%)</div>")


# ---------------------------------------------------------------------------
# Page: DASHBOARD
# ---------------------------------------------------------------------------
def page_dashboard():
    view = build_view()
    section_heading("Dashboard — Central Sector Infrastructure Projects")

    # ---- Interactive Filter Bar with Dropdowns (Replaces boring static buttons) ----
    with st.container():
        st.markdown(
            "<div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px 16px 4px; margin-bottom:12px;'>"
            "<div style='font-size:11px; font-weight:700; color:#003366; text-transform:uppercase; letter-spacing:0.6px; margin-bottom:6px;'>"
            "Filter Projects"
            "</div>",
            unsafe_allow_html=True
        )
        f1, f2, f3, f4 = st.columns([1.5, 1.3, 1.3, 1.3])
        with f1:
            all_sectors = ["All Sectors"] + sorted(view["sector"].dropna().unique().tolist())
            sel_sector = st.selectbox("Sector", all_sectors, index=0)
        with f2:
            sel_tier = st.selectbox(
                "Risk Level",
                ["All Levels", "Critical (≥ 0.70)", "High (0.45 – 0.70)", "Moderate (0.25 – 0.45)", "Low (< 0.25)"],
                index=0
            )
        with f3:
            sel_delay = st.selectbox(
                "Schedule",
                ["All", "Delayed Only", "On Schedule"],
                index=0
            )
        with f4:
            sel_scale = st.selectbox(
                "Budget Size",
                ["All (≥ ₹150 Cr)", "Mega (≥ ₹1,000 Cr)", "Major (₹150 – ₹1,000 Cr)"],
                index=0
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Apply interactive filters to view
    filtered = view.copy()
    if sel_sector != "All Sectors":
        filtered = filtered[filtered["sector"] == sel_sector]
    if "Critical" in sel_tier:
        filtered = filtered[(filtered["cost_risk"] >= 0.70) | (filtered["delay_risk"] >= 0.70)]
    elif "High" in sel_tier:
        filtered = filtered[((filtered["cost_risk"] >= 0.45) & (filtered["cost_risk"] < 0.70)) |
                             ((filtered["delay_risk"] >= 0.45) & (filtered["delay_risk"] < 0.70))]
    elif "Moderate" in sel_tier:
        filtered = filtered[((filtered["cost_risk"] >= 0.25) & (filtered["cost_risk"] < 0.45)) |
                             ((filtered["delay_risk"] >= 0.25) & (filtered["delay_risk"] < 0.45))]
    elif "Low" in sel_tier:
        filtered = filtered[(filtered["cost_risk"] < 0.25) & (filtered["delay_risk"] < 0.25)]

    if "Delayed" in sel_delay:
        filtered = filtered[filtered["tor_months"] > 0]
    elif "On Schedule" in sel_delay:
        filtered = filtered[filtered["tor_months"] <= 0]

    if "Mega" in sel_scale:
        filtered = filtered[filtered["original_cost_cr"] >= 1000]
    elif "Major" in sel_scale:
        filtered = filtered[(filtered["original_cost_cr"] >= 150) & (filtered["original_cost_cr"] < 1000)]

    # Compute live portfolio aggregates
    n_filt = len(filtered)
    tot_orig = filtered["original_cost_cr"].sum()
    tot_ant = filtered["anticipated_cost_cr"].sum()
    tot_overrun = max(0.0, tot_ant - tot_orig)
    avg_overrun_pct = filtered["cost_overrun_pct"].mean() if n_filt else 0.0
    avg_delay_m = filtered["tor_months"].mean() if n_filt else 0.0
    delayed_count = int((filtered["tor_months"] > 0).sum())
    crit_count = int(((filtered["cost_risk"] >= 0.45) | (filtered["delay_risk"] >= 0.45)).sum())

    # Live KPI Row
    kpi_row(
        st.columns(5),
        [
            ("Projects Shown", f"{n_filt:,}", f"{n_filt/len(view)*100:.0f}% of {len(view):,} total"),
            ("Sanctioned Cost", _cr(tot_orig), f"Anticipated: {_cr(tot_ant)}"),
            ("Budget Exceeded By", _cr(tot_overrun), f"Avg overrun: {avg_overrun_pct:.1f}%"),
            ("Delayed Projects", f"{delayed_count:,}", f"Avg delay: {avg_delay_m:.0f} months"),
            ("High/Critical Risk", f"{crit_count:,}", "Risk score ≥ 0.45"),
        ]
    )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # ---- Interactive Analytics Tabs ----
    t_viz, t_bench, t_list = st.tabs([
        "Charts & Visualisations",
        "Sector Benchmarks",
        "Project List"
    ])

    with t_viz:
        c_ctrl1, c_ctrl2 = st.columns([1.8, 2.2])
        with c_ctrl1:
            viz_type = st.selectbox(
                "Chart Type",
                [
                    "Top Sectors by Budget Overrun (₹ Cr)",
                    "Average Delay by Sector (Months)",
                    "Cost Overrun % Distribution",
                    "Cost Risk vs Delay Risk (Bubble Chart)",
                ],
                index=0
            )

        if "Budget Overrun" in viz_type:
            sec_grp = filtered.groupby("sector").agg(
                n_proj=("project_name", "count"),
                tot_orig=("original_cost_cr", "sum"),
                tot_ant=("anticipated_cost_cr", "sum")
            ).reset_index()
            sec_grp["overrun_cr"] = (sec_grp["tot_ant"] - sec_grp["tot_orig"]).clip(lower=0)
            sec_grp = sec_grp.sort_values("overrun_cr", ascending=False).head(10)

            fig = px.bar(
                sec_grp,
                x="sector", y="overrun_cr",
                color="overrun_cr",
                color_continuous_scale="Reds",
                labels={"overrun_cr": "Fiscal Overrun Exposure (₹ Cr)", "sector": "Sector"},
                title="Top 10 Sectors by Cumulative Fiscal Overrun Exposure (₹ Crore)",
                text_auto=".2s"
            )
            fig.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True, key="viz_exposure")

        elif "Bubble" in viz_type:
            plot_df = (filtered.head(150) if n_filt > 150 else filtered).copy()
            if len(plot_df):
                plot_df["bubble_size"] = np.clip(np.sqrt(plot_df["original_cost_cr"].fillna(150)), 6, 22)
            fig = px.scatter(
                plot_df,
                x="cost_risk", y="delay_risk",
                size="bubble_size",
                size_max=22,
                color="sector",
                hover_name="project_name",
                hover_data={
                    "cost_overrun_pct": ":.1f",
                    "tor_months": True,
                    "original_cost_cr": ":,.1f",
                },
                labels={
                    "cost_risk": "Cost Risk Index (0 - 1)",
                    "delay_risk": "Delay Risk Index (0 - 1)",
                    "original_cost_cr": "Sanctioned ₹ Cr",
                    "sector": "Sector"
                },
                title=f"Cost Risk vs Delay Risk ({len(plot_df)} projects plotted)",
            )
            fig.update_layout(
                height=480,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", y=-0.2),
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True, key="viz_scatter")

        elif "Average Delay" in viz_type:
            sec_delay = filtered.groupby("sector").agg(
                avg_delay=("tor_months", "mean"),
                n_proj=("project_name", "count")
            ).reset_index().sort_values("avg_delay", ascending=False).head(12)

            fig = px.bar(
                sec_delay,
                y="sector", x="avg_delay",
                orientation="h",
                color="avg_delay",
                color_continuous_scale="Oranges",
                labels={"avg_delay": "Average Delay (Months)", "sector": "Sector"},
                title="Sector-wise Average Implementation Delay (Months)",
                text_auto=".1f"
            )
            fig.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True, key="viz_delay")

        else:
            fig = px.histogram(
                filtered,
                x="cost_overrun_pct",
                nbins=35,
                color="cost_tier",
                title="Cost Overrun % Frequency Distribution Across Portfolio",
                labels={"cost_overrun_pct": "Cost Overrun (%)", "count": "Project Count"},
                color_discrete_map={"Critical": "#c0392b", "High": "#d97706", "Medium": "#f59e0b", "Low": "#166534"}
            )
            fig.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True, key="viz_hist")

    with t_bench:
        section_heading("Sector Benchmarks — Average Risk, Delay & Budget Overrun")
        sec_table = filtered.groupby("sector").agg(
            Projects=("project_name", "count"),
            Avg_Cost_Risk=("cost_risk", "mean"),
            Avg_Delay_Risk=("delay_risk", "mean"),
            Avg_Overrun_Pct=("cost_overrun_pct", "mean"),
            Avg_Delay_Mo=("tor_months", "mean"),
            Total_Sanctioned_Cr=("original_cost_cr", "sum"),
            Total_Anticipated_Cr=("anticipated_cost_cr", "sum"),
        ).round(2)
        sec_table["Fiscal_Overrun_Cr"] = (sec_table["Total_Anticipated_Cr"] - sec_table["Total_Sanctioned_Cr"]).clip(lower=0).round(1)
        sec_table = sec_table.sort_values("Fiscal_Overrun_Cr", ascending=False)
        st.dataframe(
            sec_table.rename(columns={
                "Avg_Cost_Risk": "Cost Risk (0-1)",
                "Avg_Delay_Risk": "Delay Risk (0-1)",
                "Avg_Overrun_Pct": "Avg Overrun (%)",
                "Avg_Delay_Mo": "Avg Delay (Mo)",
                "Total_Sanctioned_Cr": "Sanctioned (₹ Cr)",
                "Total_Anticipated_Cr": "Anticipated (₹ Cr)",
                "Fiscal_Overrun_Cr": "Overrun Gap (₹ Cr)",
            }),
            use_container_width=True
        )

    with t_list:
        section_heading("Project List")
        q_col1, q_col2, q_col3 = st.columns([2.2, 1.8, 1.0])
        with q_col1:
            kw = st.text_input("Search Project", placeholder="e.g. NHAI, Metro, Power, Dam...")
        with q_col2:
            sort_by = st.selectbox(
                "Sort By",
                [
                    "Highest Cost Overrun %",
                    "Longest Delay (Months)",
                    "Highest Cost Risk Score",
                    "Highest Delay Risk Score",
                    "Largest Budget (₹ Cr)",
                ]
            )
        with q_col3:
            page_size = st.selectbox("Page Size", [10, 25, 50, 100], index=1)

        disp = filtered.copy()
        if kw.strip():
            disp = disp[disp["project_name"].astype(str).str.lower().str.contains(kw.strip().lower(), na=False)]

        if "Overrun %" in sort_by:
            disp = disp.sort_values("cost_overrun_pct", ascending=False)
        elif "Delay" in sort_by:
            disp = disp.sort_values("tor_months", ascending=False)
        elif "Cost Risk" in sort_by:
            disp = disp.sort_values("cost_risk", ascending=False)
        elif "Delay Risk" in sort_by:
            disp = disp.sort_values("delay_risk", ascending=False)
        elif "Largest" in sort_by:
            disp = disp.sort_values("original_cost_cr", ascending=False)

        st.caption(f"Displaying **{min(len(disp), page_size)}** of **{len(disp)}** matching projects:")
        disp_table = disp.head(page_size)[[
            "project_name", "sector", "original_cost_cr", "anticipated_cost_cr",
            "cost_overrun_pct", "tor_months", "cost_tier", "delay_tier"
        ]].rename(columns={
            "project_name": "Project Name & Agency",
            "sector": "Sector",
            "original_cost_cr": "Sanctioned (₹ Cr)",
            "anticipated_cost_cr": "Anticipated (₹ Cr)",
            "cost_overrun_pct": "Overrun (%)",
            "tor_months": "Delay (Mo)",
            "cost_tier": "Cost Tier",
            "delay_tier": "Delay Tier",
        })
        st.dataframe(disp_table, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: GEOSPATIAL MAP (3D WebGL & 2D GIGW Administrative Views)
# ---------------------------------------------------------------------------
_GEO_NAME_MAP = {
    "ANDHRA PRADESH": "Andhra Pradesh",
    "ARUNACHAL PRADESH": "Arunachal Pradesh", "ASSAM": "Assam", "BIHAR": "Bihar",
    "CHHATTISGARH": "Chhattisgarh", "GOA": "Goa", "GUJARAT": "Gujarat",
    "HARYANA": "Haryana", "HIMACHAL PRADESH": "Himachal Pradesh",
    "JAMMU AND KASHMIR": "Jammu and Kashmir", "JHARKHAND": "Jharkhand",
    "KARNATAKA": "Karnataka", "KERALA": "Kerala", "MADHYA PRADESH": "Madhya Pradesh",
    "MAHARASHTRA": "Maharashtra", "MANIPUR": "Manipur", "MEGHALAYA": "Meghalaya",
    "MIZORAM": "Mizoram", "NAGALAND": "Nagaland", "ODISHA": "Orissa", "PUNJAB": "Punjab",
    "RAJASTHAN": "Rajasthan", "SIKKIM": "Sikkim", "TAMIL NADU": "Tamil Nadu",
    "TRIPURA": "Tripura", "UTTAR PRADESH": "Uttar Pradesh", "WEST BENGAL": "West Bengal",
    "ANDAMAN AND NICOBAR": "Andaman and Nicobar", "DELHI": "Delhi", "LADAKH": "Ladakh",
    "TELANGANA": "Telangana", "UTTARAKHAND": "Uttaranchal",
}

_ACTION_COLOR = {"Escalate": "#c0392b", "Review": "#d97706", "Monitor": "#166534"}

STATE_COORDINATES = {
    "ANDHRA PRADESH": {"lat": 15.9129, "lon": 79.7400, "abbr": "AP", "capital": "Amaravati"},
    "ARUNACHAL PRADESH": {"lat": 28.2180, "lon": 94.7278, "abbr": "AR", "capital": "Itanagar"},
    "ASSAM": {"lat": 26.2006, "lon": 92.9376, "abbr": "AS", "capital": "Dispur"},
    "BIHAR": {"lat": 25.0961, "lon": 85.3131, "abbr": "BR", "capital": "Patna"},
    "CHHATTISGARH": {"lat": 21.2787, "lon": 81.8661, "abbr": "CG", "capital": "Raipur"},
    "GOA": {"lat": 15.2993, "lon": 74.1240, "abbr": "GA", "capital": "Panaji"},
    "GUJARAT": {"lat": 22.2587, "lon": 71.1924, "abbr": "GJ", "capital": "Gandhinagar"},
    "HARYANA": {"lat": 29.0588, "lon": 76.0856, "abbr": "HR", "capital": "Chandigarh"},
    "HIMACHAL PRADESH": {"lat": 31.1048, "lon": 77.1734, "abbr": "HP", "capital": "Shimla"},
    "JAMMU AND KASHMIR": {"lat": 33.7782, "lon": 76.5762, "abbr": "JK", "capital": "Srinagar/Jammu"},
    "JHARKHAND": {"lat": 23.6102, "lon": 85.2799, "abbr": "JH", "capital": "Ranchi"},
    "KARNATAKA": {"lat": 15.3173, "lon": 75.7139, "abbr": "KA", "capital": "Bengaluru"},
    "KERALA": {"lat": 10.8505, "lon": 76.2711, "abbr": "KL", "capital": "Thiruvananthapuram"},
    "MADHYA PRADESH": {"lat": 22.9734, "lon": 78.6569, "abbr": "MP", "capital": "Bhopal"},
    "MAHARASHTRA": {"lat": 19.7515, "lon": 75.7139, "abbr": "MH", "capital": "Mumbai"},
    "MANIPUR": {"lat": 24.6637, "lon": 93.9063, "abbr": "MN", "capital": "Imphal"},
    "MEGHALAYA": {"lat": 25.4670, "lon": 91.3662, "abbr": "ML", "capital": "Shillong"},
    "MIZORAM": {"lat": 23.1645, "lon": 92.9376, "abbr": "MZ", "capital": "Aizawl"},
    "NAGALAND": {"lat": 26.1584, "lon": 94.5624, "abbr": "NL", "capital": "Kohima"},
    "ODISHA": {"lat": 20.9517, "lon": 85.0985, "abbr": "OD", "capital": "Bhubaneswar"},
    "PUNJAB": {"lat": 31.1471, "lon": 75.3412, "abbr": "PB", "capital": "Chandigarh"},
    "RAJASTHAN": {"lat": 27.0238, "lon": 74.2179, "abbr": "RJ", "capital": "Jaipur"},
    "SIKKIM": {"lat": 27.5330, "lon": 88.5122, "abbr": "SK", "capital": "Gangtok"},
    "TAMIL NADU": {"lat": 11.1271, "lon": 78.6569, "abbr": "TN", "capital": "Chennai"},
    "TELANGANA": {"lat": 18.1124, "lon": 79.0193, "abbr": "TS", "capital": "Hyderabad"},
    "TRIPURA": {"lat": 23.9408, "lon": 91.9882, "abbr": "TR", "capital": "Agartala"},
    "UTTAR PRADESH": {"lat": 26.8467, "lon": 80.9462, "abbr": "UP", "capital": "Lucknow"},
    "UTTARAKHAND": {"lat": 30.0668, "lon": 79.0193, "abbr": "UK", "capital": "Dehradun"},
    "WEST BENGAL": {"lat": 22.9868, "lon": 87.8550, "abbr": "WB", "capital": "Kolkata"},
    "DELHI": {"lat": 28.7041, "lon": 77.1025, "abbr": "DL", "capital": "New Delhi (HQ)"},
    "LADAKH": {"lat": 34.1526, "lon": 77.5771, "abbr": "LA", "capital": "Leh"},
    "MULTI STATE": {"lat": 21.1458, "lon": 79.0882, "abbr": "MS", "capital": "Pan-India Corridors"},
}


def _state_action(r):
    """Practical triage for a state row: Escalate > Review > Monitor."""
    if r["cost_overrun_pct"] >= 50 or r.get("delay_risk", 0) >= 0.9 or (r.get("tor_max_months") and r["tor_max_months"] >= 150):
        return "Escalate"
    if r["cost_overrun_pct"] >= 25 or r.get("delay_risk", 0) >= 0.6 or (r.get("tor_max_months") and r["tor_max_months"] >= 60):
        return "Review"
    return "Monitor"


@st.cache_data
def _load_geo_ultralight():
    """Extra-simplified GeoJSON (eps=0.05) for 3D view — lowest GPU memory footprint."""
    gj = load_geojson()
    gj = _walk_geo(gj, 0.05)   # coarser than the 2D view (0.02) — fine for 3D backdrop
    for feat in gj.get("features", []):
        feat["properties"] = {"n": feat["properties"].get("NAME_1")}
    return gj


def render_3d_india_map(
    df: pd.DataFrame,
    metric: str,
    pitch: float,
    elevation_scale: float,
    map_style_name: str,
    show_arcs: bool,
    show_borders: bool,
):
    """Render 3D India Map via Plotly 3D — lightweight, no WebGL OOM.

    Features:
      - State boundary outlines on ground plane
      - Vertical pillars proportional to overrun/delay
      - Status markers: Red (Escalate) / Amber (Review) / Green (Monitor)
      - Connection arcs from New Delhi HQ to state capitals
      - Full 3D rotation and hover inspection
    """
    fig = go.Figure()

    # Themes
    themes = {
        "NIC Official Light (Carto Positron)": {
            "bg": "#f8fafc", "grid": "#e2e8f0", "border": "#94a3b8", "text": "#0f172a", "pillar": "#475569"
        },
        "Command Center Dark (Dark Matter)": {
            "bg": "#0a1628", "grid": "#1e3a6f", "border": "#38bdf8", "text": "#f8fafc", "pillar": "#64748b"
        },
        "Road Voyager (High Contrast)": {
            "bg": "#fdfbf7", "grid": "#e7e5e4", "border": "#78716c", "text": "#1c1917", "pillar": "#57534e"
        },
    }
    cur_theme = themes.get(map_style_name, themes["NIC Official Light (Carto Positron)"])

    # 1. State borders on z=0 plane if enabled (617 vertices, 0.001s instant load, zero WebGL freeze)
    if show_borders:
        try:
            b_data = load_borders_3d()
            b_lons = b_data.get("lons", [])
            b_lats = b_data.get("lats", [])
            if b_lons:
                fig.add_trace(go.Scatter3d(
                    x=b_lons, y=b_lats, z=[0.0] * len(b_lons),
                    mode="lines",
                    line=dict(color=cur_theme["border"], width=1.5),
                    hoverinfo="none",
                    name="State Boundaries",
                    showlegend=False,
                ))
        except Exception:
            pass

    # 2. Extract state metrics and build 3D vertical pillars & cap markers
    pillar_x, pillar_y, pillar_z = [], [], []
    cap_x, cap_y, cap_z, cap_colors, cap_sizes, cap_text, cap_abbrs = [], [], [], [], [], [], []
    ground_x, ground_y = [], []
    arc_x, arc_y, arc_z = [], [], []  # batched arc segments

    hub_lon, hub_lat = 77.2090, 28.6139  # New Delhi HQ

    for _, r in df.iterrows():
        st_name = str(r["state"]).strip().upper()
        if st_name not in STATE_COORDINATES:
            continue
        c = STATE_COORDINATES[st_name]
        lon, lat = c["lon"], c["lat"]

        overrun = float(r.get("cost_overrun_pct", 0) or 0)
        delay_m = float(r.get("tor_max_months", 0) or 0)
        n_p = int(r.get("n_projects", 0) or 0)
        act = _state_action(r)
        color = _ACTION_COLOR[act]

        if metric == "cost_overrun_pct":
            val = max(1.0, overrun)
        elif metric == "delay_risk":
            val = max(1.0, delay_m)
        else:
            val = max(1.0, float(n_p))

        h = val * elevation_scale

        # Pillar segment (from z=0 to z=h)
        pillar_x.extend([lon, lon, None])
        pillar_y.extend([lat, lat, None])
        pillar_z.extend([0.0, h, None])

        ground_x.append(lon)
        ground_y.append(lat)

        cap_x.append(lon)
        cap_y.append(lat)
        cap_z.append(h)
        cap_colors.append(color)
        cap_sizes.append(max(6, min(18, 6 + int(math.sqrt(n_p) * 2))))
        cap_abbrs.append(c["abbr"])
        cap_text.append(
            f"<b>{st_name.title()}</b> ({c['abbr']})<br>"
            f"Capital: {c['capital']}<br>"
            f"Projects: <b>{n_p}</b><br>"
            f"Overrun: <b>{overrun:.1f}%</b><br>"
            f"Delay: <b>{int(r.get('tor_min_months', 0) or 0)}–{int(delay_m)} months</b><br>"
            f"Action: <b>{act}</b>"
        )

        # 3. Connection arcs (New Delhi HQ -> State) — collected into batch lists
        if show_arcs:
            n_arc = 10
            t = np.linspace(0, 1, n_arc)
            a_lons = hub_lon + (lon - hub_lon) * t
            a_lats = hub_lat + (lat - hub_lat) * t
            dist = math.sqrt((lon - hub_lon) ** 2 + (lat - hub_lat) ** 2)
            peak_z = max(15.0, h + dist * 1.8)
            a_zs = (1 - t) * 0.0 + t * h + 4 * peak_z * t * (1 - t)

            arc_x.extend(a_lons.tolist() + [None])
            arc_y.extend(a_lats.tolist() + [None])
            arc_z.extend(a_zs.tolist() + [None])

    # Add batched arc trace (single trace for ALL arcs — OOM-safe)
    if arc_x:
        fig.add_trace(go.Scatter3d(
            x=arc_x, y=arc_y, z=arc_z,
            mode="lines",
            line=dict(color="rgba(255, 153, 51, 0.45)", width=2),
            hoverinfo="none",
            name="HQ Connections",
            showlegend=False,
        ))

    # Add 3D Vertical Pillar Lines
    if pillar_x:
        fig.add_trace(go.Scatter3d(
            x=pillar_x, y=pillar_y, z=pillar_z,
            mode="lines",
            line=dict(color=cur_theme["pillar"], width=4),
            hoverinfo="none",
            name="Pillars",
            showlegend=False,
        ))

    # Add Ground Base Pins
    if ground_x:
        fig.add_trace(go.Scatter3d(
            x=ground_x, y=ground_y, z=[0.0] * len(ground_x),
            mode="markers",
            marker=dict(size=4, color="#94a3b8", opacity=0.7),
            hoverinfo="none",
            name="Base Points",
            showlegend=False,
        ))

    # Add Pillar Caps with State Labels
    if cap_x:
        fig.add_trace(go.Scatter3d(
            x=cap_x, y=cap_y, z=cap_z,
            mode="markers+text",
            marker=dict(
                size=cap_sizes,
                color=cap_colors,
                symbol="diamond",
                opacity=0.92,
                line=dict(color="#ffffff", width=1)
            ),
            text=cap_abbrs,
            textposition="top center",
            textfont=dict(color=cur_theme["text"], size=10, family="Segoe UI, sans-serif"),
            hovertext=cap_text,
            hoverinfo="text",
            name="States/UTs",
            showlegend=False,
        ))

    # Add New Delhi HQ marker
    fig.add_trace(go.Scatter3d(
        x=[hub_lon], y=[hub_lat], z=[0.0],
        mode="markers+text",
        marker=dict(size=12, color="#FF9933", symbol="cross", line=dict(color="#ffffff", width=2)),
        text=["DELHI HQ"],
        textposition="bottom center",
        textfont=dict(color="#FF9933", size=11, family="Segoe UI, sans-serif"),
        hovertext="<b>MoSPI / IPMD Headquarters</b><br>New Delhi",
        hoverinfo="text",
        name="Delhi HQ",
        showlegend=False,
    ))

    # Calculate 3D Camera Angles from Pitch
    theta = math.radians(pitch)
    eye_x = 1.3
    eye_y = -1.6 * math.cos(theta)
    eye_z = max(0.5, 1.8 * math.sin(theta))

    z_title = {
        "cost_overrun_pct": "Overrun %",
        "delay_risk": "Delay (Months)",
        "n_projects": "Projects Count"
    }.get(metric, metric)

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Longitude (°E)", range=[67, 98],
                       backgroundcolor=cur_theme["bg"], gridcolor=cur_theme["grid"]),
            yaxis=dict(title="Latitude (°N)", range=[7, 37],
                       backgroundcolor=cur_theme["bg"], gridcolor=cur_theme["grid"]),
            zaxis=dict(title=z_title,
                       backgroundcolor=cur_theme["bg"], gridcolor=cur_theme["grid"]),
            camera=dict(
                eye=dict(x=eye_x, y=eye_y, z=eye_z),
                center=dict(x=0, y=0, z=-0.1)
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1.2, z=0.65)
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=560,
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True, key="map_3d")
    st.caption(
        "**Controls:** Drag to rotate · Right-click to pan · Scroll to zoom. "
        "Monitor · Review · Escalate · Delhi HQ"
    )


def page_map():
    section_heading("India Map — State-wise Cost & Time Overrun")
    st.caption("State-wise data from MoSPI Flash Report (Central Sector Projects, ₹150 Cr & above). "
               "View risk in 3D or 2D map, and drill down into individual state details.")
    df = load_state().copy()
    df["geo_name"] = df["state"].map(_GEO_NAME_MAP)

    nat_avg = df["cost_overrun_pct"].mean()
    tot_projects = int(df["n_projects"].sum())
    worst = df.loc[df["cost_overrun_pct"].idxmax()]
    flagged = df.apply(lambda r: _state_action(r) != "Monitor", axis=1).sum()

    kpi_row(st.columns(4), [
        ("States/UTs Covered", f"{len(df)}", "Including multi-state corridors"),
        ("Total Projects", f"{tot_projects:,}", "₹150 Cr & above"),
        ("States Needing Action", f"{int(flagged)}", "Escalate or Review"),
        ("Highest Overrun", f"{worst['state'].title()} ({worst['cost_overrun_pct']:.1f}%)", "Budget revision recommended"),
    ])

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    ctrl1, ctrl2 = st.columns([1.4, 1.6])
    with ctrl1:
        map_view_mode = st.radio(
            "Map View",
            ["3D Map", "2D Choropleth"],
            horizontal=True,
            key="map_view_mode_radio",
        )
    with ctrl2:
        metric = st.radio(
            "Colour By",
            ["cost_overrun_pct", "delay_risk", "n_projects"],
            horizontal=True,
            key="map_metric_radio",
            format_func=lambda m: {
                "cost_overrun_pct": "Cost Overrun (%)",
                "delay_risk": "Delay (Months)",
                "n_projects": "Project Count"
            }[m]
        )

    selected = None

    if map_view_mode == "3D Map":
        # Read all 3D settings BEFORE rendering so sliders don't cause double renders
        with st.expander("3D Map Settings", expanded=False):
            c_p1, c_p2, c_p3 = st.columns(3)
            with c_p1:
                pitch_val = st.slider("Camera Angle", min_value=20, max_value=70, value=45, step=5,
                                      key="map_pitch_slider")
            with c_p2:
                elev_scale = st.slider("Pillar Height", min_value=0.5, max_value=3.0, value=1.0, step=0.25,
                                       key="map_elev_slider")
            with c_p3:
                map_style_opt = st.selectbox(
                    "Theme",
                    ["NIC Official Light (Carto Positron)", "Command Center Dark (Dark Matter)", "Road Voyager (High Contrast)"],
                    key="map_theme_select",
                )
            perf_col1, perf_col2 = st.columns(2)
            with perf_col1:
                show_arcs_opt = st.checkbox(
                    "Show connection arcs (Delhi HQ → States)",
                    value=False,
                    key="map_arcs_check",
                    help="Arcs from Central HQ to state capitals"
                )
            with perf_col2:
                show_borders_opt = st.checkbox(
                    "Show state borders",
                    value=True,
                    key="map_borders_check",
                    help="State boundary outlines on ground plane"
                )

        render_3d_india_map(df, metric, pitch_val, elev_scale, map_style_opt, show_arcs_opt, show_borders_opt)
    else:
        # 2D Plotly GeoJSON view (loads lightweight 185KB vector boundary)
        gj = load_geo_light()
        feature_names = {f["properties"]["NAME_1"] for f in gj["features"]}
        plot_df = df[df["geo_name"].isin(feature_names)].copy()

        fig = go.Figure(go.Choropleth(
            geojson=gj, locations=plot_df["geo_name"], z=plot_df[metric], featureidkey="properties.NAME_1",
            colorscale="Blues", marker_line_color="#ffffff", marker_line_width=0.6,
            colorbar_title=metric, hovertemplate="<b>%{location}</b><br>Value: %{z:.2f}<extra></extra>",
            selected=dict(marker=dict(opacity=1)),
            unselected=dict(marker=dict(opacity=0.6))))
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=520, paper_bgcolor="rgba(0,0,0,0)")
        # on_select="rerun" removed — it caused a rerun on every map click/hover
        st.plotly_chart(fig, use_container_width=True, key="map_2d")
        selected = None  # click-to-drill-down via selectbox below

    # State Selector Drill-down for both 3D and 2D views
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    all_states = sorted(df["state"].unique().tolist())
    state_picklist = ["-- Select a State --"] + [s.title() for s in all_states]
    default_idx = 0
    if selected:
        for idx_s, s_name in enumerate(all_states):
            if _GEO_NAME_MAP.get(s_name) == selected or s_name.title() == selected:
                default_idx = idx_s + 1
                break

    sel_state_choice = st.selectbox(
        "🔎 Select State for Details",
        state_picklist,
        index=default_idx
    )

    if sel_state_choice != "-- Select a State --":
        s_target = sel_state_choice.upper()
        srow = df[df["state"] == s_target]
        if not srow.empty:
            r = srow.iloc[0]
            rank_over = int((df["cost_overrun_pct"] > r["cost_overrun_pct"]).sum()) + 1
            share = r["n_projects"] / tot_projects * 100
            action = _state_action(r)
            acol = _ACTION_COLOR[action]
            coord_info = STATE_COORDINATES.get(s_target, {})
            capital_name = coord_info.get("capital", "State Hub")

            section_heading(f"State Details — {sel_state_choice} (Capital: {capital_name})")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Central Projects", f"{r['n_projects']}")
            k2.metric("Cost Overrun", f"{r['cost_overrun_pct']:.1f}%")
            k3.metric("Delay (TOR) Range", f"{r['tor_min_months']}–{r['tor_max_months']} mo")
            k4.metric("Recommended Action", action)

            st.markdown(
                f"<div class='gov-info' style='border-left: 4px solid {acol};'>"
                f"<b>Summary for {sel_state_choice}:</b> Accounts for <b>{share:.1f}%</b> of central sector projects. "
                f"Ranks <b>#{rank_over}</b> of {len(df)} states/UTs by cost overrun. "
                f"Delay risk score: <b>{r['delay_risk']:.2f}</b>.<br/>"
                f"<b>Action:</b> <span style='color:{acol};font-weight:700;'>{action}</span> — "
                f"{'Escalation to Secretary-level review recommended. Budget revision and land acquisition clearance needed.' if action=='Escalate' else 'Quarterly milestone review recommended. Nodal agency to submit revised expenditure schedule.' if action=='Review' else 'Routine monitoring. Progress within sanctioned limits.'}"
                f"</div>", unsafe_allow_html=True)

    # --- top-5 overrun chart + priority watchlist ---
    c1, c2 = st.columns([1.1, 1.9], gap="medium")
    with c1:
        section_heading("Top Cost Overrun States")
        top = df.nlargest(5, "cost_overrun_pct")
        chart = alt.Chart(top).mark_bar(color=GOV_PRIMARY).encode(
            y=alt.Y("state:N", sort="-x", title=""),
            x=alt.X("cost_overrun_pct:Q", title="Cost Overrun (%)"),
            tooltip=["state", "cost_overrun_pct", "n_projects"]).properties(height=240)
        st.altair_chart(chart, use_container_width=True)
        st.caption(f"National average overrun: **{nat_avg:.1f}%** across all monitored states and Union Territories.")
    with c2:
        section_heading("State Priority Watchlist")
        wt = df.copy()
        wt["action"] = wt.apply(_state_action, axis=1)
        wt["score"] = wt["cost_overrun_pct"] + wt["delay_risk"] * 100
        wt = wt.sort_values("score", ascending=False).head(12)

        def _badge_html(act):
            if act == "Escalate":
                return "<span class='badge-escalate'>ESCALATE</span>"
            elif act == "Review":
                return "<span class='badge-review'>REVIEW</span>"
            return "<span class='badge-monitor'>MONITOR</span>"

        rows_html = "".join(
            f"<tr><td><b>{r['state'].title()}</b></td><td>{r['n_projects']}</td>"
            f"<td><b>{r['cost_overrun_pct']:.1f}%</b></td>"
            f"<td>{r['tor_min_months']}–{r['tor_max_months']} mo</td>"
            f"<td>{_badge_html(r['action'])}</td></tr>"
            for _, r in wt.iterrows())
        st.markdown(
            "<table class='gov-table'><thead><tr><th>State / UT</th><th>Projects</th><th>Cost Overrun</th>"
            "<th>Delay Range</th><th>Action</th></tr></thead><tbody>" + rows_html + "</tbody></table>",
            unsafe_allow_html=True)

    # ---- MoSPI Decision Support Card ----
    work = df.copy()
    work["exposure"] = work["n_projects"] * work["cost_overrun_pct"]
    tot_exp = work["exposure"].sum()
    top_con = work.nlargest(5, "exposure")
    pct_top = top_con["exposure"].sum() / tot_exp * 100
    triage = work.apply(_state_action, axis=1)
    n_esc = int((triage == "Escalate").sum())

    section_heading("MoSPI Decision Support & Recommendations")
    esc_states = work[triage == "Escalate"]["state"].tolist()
    top3 = " · ".join(s.title() for s in top_con["state"].head(3))
    st.markdown(
        f"<div class='gov-info'>"
        f"<b>Overrun Concentration:</b> Just <b>{len(top_con)}</b> states account for "
        f"<b>{pct_top:.0f}%</b> of total cost overrun impact. "
        f"Top contributors: <b>{top3}</b>.<br/>"
        f"<b>Escalation Required:</b> <b>{n_esc}</b> states need "
        f"<span style='color:#c0392b;font-weight:700;'>Escalation</span> "
        f"({' · '.join(s.title() for s in esc_states[:6])}{' …' if n_esc > 6 else ''}).<br/>"
        f"<b>Recommendation:</b> Prioritise budget revision and quarterly inter-ministerial reviews for flagged states. "
        f"Resolving issues in {top_con['state'].iloc[0].title()} alone could address "
        f"<b>{top_con['exposure'].iloc[0]/tot_exp*100:.0f}%</b> of total overrun."
        f"</div>", unsafe_allow_html=True)

    with st.expander("View Complete State-wise MoSPI Audited Data Table"):
        wt_full = df.copy()
        wt_full["action"] = wt_full.apply(_state_action, axis=1)
        st.dataframe(
            wt_full[["state", "n_projects", "cost_overrun_pct",
                     "tor_min_months", "tor_max_months", "delay_risk", "action"]]
            .rename(columns={
                "state": "State / UT",
                "n_projects": "Projects Count",
                "cost_overrun_pct": "Cost Overrun (%)",
                "tor_min_months": "Min Delay (mo)",
                "tor_max_months": "Max Delay (mo)",
                "delay_risk": "Delay Risk Index",
                "action": "Triage Directive"
            })
            .sort_values("Cost Overrun (%)", ascending=False),
            use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: PROJECT EXPLORER
# ---------------------------------------------------------------------------
def page_explorer():
    view = build_view()
    section_heading("Project Explorer — Risk Profile & AI Analysis")
    st.caption("Select a project to view its risk assessment, predicted cost & delay, "
               "and an AI explanation of risk factors.")
    sector = st.selectbox("Filter by Sector", ["All"] + sorted(view["sector"].dropna().unique().tolist()))
    rows_in = view if sector == "All" else view[view["sector"] == sector]
    proj = st.selectbox("Select Project", rows_in["project_name"].tolist())
    idx = view.index[view["project_name"] == proj][0]
    pr = view.loc[idx]

    # header card
    ap = f" · Approved {int(pr['approval_year'])}" if pd.notna(pr["approval_year"]) else ""
    st.markdown(f"<div class='gov-card'><h3>{proj}</h3>"
                f"<div style='color:#64748b; font-size:13px;'>{pr['sector']}{ap}</div></div>",
                unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Original Cost", _cr(pr["original_cost_cr"]))
    k2.metric("Revised/Anticipated Cost", _cr(pr["anticipated_cost_cr"]))
    k3.metric("Spent So Far", _cr(pr["expenditure_cum_cr"]))
    k4.metric("Cost Overrun", f"{pr['cost_overrun_pct']:.1f}%")

    g1, g2 = st.columns(2)
    g1.markdown(_risk_gauge("Cost Risk", pr["cost_risk"], pr["cost_tier"]), unsafe_allow_html=True)
    g2.markdown(_risk_gauge("Delay Risk", pr["delay_risk"], pr["delay_tier"]), unsafe_allow_html=True)

    section_heading("AI Predictions")
    try:
        est = predict_project({
            "sector": pr.get("sector"),
            "original_cost_cr": pr["original_cost_cr"],
            "expenditure_cum_cr": pr["expenditure_cum_cr"],
            "approval_year": pr.get("approval_year"),
            "planned_duration_years": pr.get("tor_months") and (pr["tor_months"] / 12)
                                     or 5.0,
        })
        ec1, ec2, ec3, ec4, ec5 = st.columns(5)
        c = est.get("cost") or {}
        t = est.get("time") or {}
        ec1.metric("Overrun Probability", f"{c.get('overrun_prob', 0)*100:.0f}%")
        ec2.metric("Predicted Overrun", _cr(c.get("overrun_cr")))
        ec3.metric("Potential Extra Cost", _cr(c.get("fiscal_exposure_cr")), help="Probability × Estimated overrun amount")
        ec4.metric("Predicted Total Cost", _cr(c.get("anticipated_cr")))
        ec5.metric("Delay Probability", f"{t.get('delay_prob', 0)*100:.0f}%")
        if t.get("delay_months") is not None:
            st.caption(f"Model accuracy: overrun ±{c.get('mae_ratio', 0):.2f} pts, "
                       f"delay ±{t.get('mae_months', 0):.0f} months. "
                       f"The 'Potential Extra Cost' metric flags projects where high "
                       f"probability meets large budget impact.")
            used_c = c.get("schedule_used", True)
            used_t = t.get("schedule_used", True)
            if not (used_c and used_t):
                st.caption("Schedule not provided (approval year / planned duration) — "
                           "risk scored by the schedule-blind model with base features only, "
                           "not assumed mid-flight.")
    except Exception as _e:  # model not present -> degrade gracefully
        st.caption("Predictive models not trained yet — run `python src/predict.py` "
                   "or add data on the 'Data & Retrain' page.")

    # persisted AI explanation (survives rerun) else template
    ai_key = f"expl_{proj}"
    explanation = st.session_state.get(ai_key, pr["explanation"])
    source = st.session_state.get(ai_key + "_src", pr["explanation_source"])

    section_heading("Risk Explanation")
    st.markdown(explanation)
    if pd.notna(pr["reasons"]) and str(pr["reasons"]) not in ("nan", ""):
        st.caption(f"Detected causes: {pr['reasons']}")

    if st.button("Regenerate explanation with AI"):
        src_row = load_time()[load_time()["project_name"] == proj]
        if len(src_row):
            with st.spinner("Running risk analysis..."):
                e = enrich(src_row.iloc[0], is_time=True, prefer_llm=True)
            st.session_state[ai_key] = e["explanation"]
            st.session_state[ai_key + "_src"] = e["explanation_source"]
            st.rerun()
    st.caption(f"Explanation source: {source}")


def _risk_gauge(label, score, tier):
    color = "#c0392b" if score >= 0.45 else "#e67e22" if score >= 0.25 else "#1e7e34"
    return (f"<div class='gov-card'><h3>{label} — {tier}</h3>"
            f"<div class='risk-bar-outer'>"
            f"<div class='risk-bar-fill' style='width:{min(100, score*100):.0f}%;background:{color};'></div></div>"
            f"<div style='font-size:13px;color:#64748b;margin-top:6px;'>Score {score:.2f} / 1.00</div></div>")


# ---------------------------------------------------------------------------
# Page: AI ASSISTANT
# ---------------------------------------------------------------------------
def _reset_history():
    st.session_state.history = [{"role": "assistant",
                                 "content": "IPMD Project Intelligence Assistant. "
                                            "Query the portfolio by sector, project, risk tier, "
                                            "or budget."}]


def page_assistant():
    section_heading("AI Project Intelligence Assistant")
    st.caption("Ask about the portfolio, sectors, top risks, budgets, or any single project. "
               "Powered by tool-using LLM agent over real MoSPI data.")
    view = build_view()
    time_df = load_time()

    # Cache assistant context in session state — rebuilding on every chat message wastes time
    if "assistant_ctx" not in st.session_state:
        st.session_state["assistant_ctx"] = build_assistant_view(view, time_df)
    ctx = st.session_state["assistant_ctx"]

    if "history" not in st.session_state:
        _reset_history()

    header = st.columns([5, 1])
    with header[1]:
        if st.button("New conversation", use_container_width=True):
            st.session_state.history = []
            st.session_state["assistant_ctx"] = build_assistant_view(view, time_df)
    with header[0]:
        st.caption(f"{st.session_state.history and 'Active conversation' or 'New'} — "
                   f"{len(st.session_state.history)//2} exchange(s)")

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if q := st.chat_input("Ask about the project portfolio..."):
        st.session_state.history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Querying project records..."):
                try:
                    text, table = answer(q, ctx)
                except Exception:
                    fb_text, fb_table = _fallback(q, ctx)
                    text = f"Sorry — something went wrong mid-query. Let me re-answer from the records:\n\n{fb_text}"
                    table = fb_table
            st.markdown(text)
            if table is not None:
                try:
                    st.dataframe(table, use_container_width=True)
                except Exception:
                    st.markdown(table.to_string())
            st.session_state.history.append({"role": "assistant", "content": text})


# ---------------------------------------------------------------------------
# Page: DATA & RETRAIN
# ---------------------------------------------------------------------------
def page_data():
    section_heading("Data Ingestion & Continual Learning")
    st.markdown("""
<div class="gov-info">
<b>Purpose:</b> Add the latest monthly Flash Report data and retrain the predictive models
in place. Models are leak-free (OOF target encoding, no outcome-derived features) and
versioned on every retrain.<br>
<b>Hard estimates for officials:</b> each project gets a <i>predicted ₹ cost overrun</i> and
<i>predicted delay (months)</i> on the <b>Project Explorer</b> page.
</div>
""", unsafe_allow_html=True)

    # ---- out-of-sample (temporal) validation ----
    tjson = ROOT / "data" / "temporal_eval_results.json"
    if tjson.exists():
        try:
            tres = json.loads(tjson.read_text())
            section_heading("How Good on NEW Data? (Out-of-Sample Validation)")
            st.caption("Temporal holdout: models trained only on projects approved up to a cutoff, "
                       "then scored on projects approved in the following 2 years — data the "
                       "model never saw. Sector encodings rebuilt from train only (no leakage). "
                       "**Logistic regression leads on unseen data in every window.**")
            cols = st.columns(3)
            for c, (cut, rep) in zip(cols, list(tres["results"].items())):
                test_win = rep["test_window"]
                with c:
                    st.markdown(f"**Train ≤ {rep['cutoff']} → test {test_win}**")
                    for task in ("time", "cost"):
                        m = rep[task]
                        if "clf" in m:
                            cl = m["clf"]
                            st.markdown(f"*{task.title()}:* AUC **{cl['auc']}** "
                                        f"({cl['best']})")
                            st.caption(f"test n={m['n_test']} · label rate {cl['test_base_rate']*100:.0f}%")
                            lift = cl.get("top20pct")
                            if lift:
                                st.caption(f"top-20% review catches {lift['recall@top']*100:.0f}% "
                                           f"of {'delays' if task=='time' else 'overruns'} "
                                           f"(lift {lift['lift']}×)")
                        if "reg" in m:
                            rg = m["reg"]
                            v = rg.get("mae_months", rg.get("mae"))
                            nv = rg.get("naive_median_mae", rg.get("naive_median_mae"))
                            st.caption(f"reg MAE {v} vs naive {nv} "
                                       f"({rg['improvement_vs_naive_pct']:.0f}% better)")
            lc = tres.get("label_censoring", {})
            cr = lc.get("cost_overrun_base_rate_by_approval", {})
            if cr:
                yrs = sorted(cr)[-5:]
                seq = " → ".join(f"{y}: {cr[y]['rate']*100:.0f}%" for y in yrs)
                st.markdown(
                    f"<div class='gov-info'>"
                    f"<b>Label censoring applies to BOTH tasks:</b> the cost-overrun <i>flag</i> only "
                    f"appears once a formal revised estimate is sanctioned, so its base rate on "
                    f"recent approvals collapses just like the delay label (<code>{seq}</code> for "
                    f"approval years {yrs[0]}–{yrs[-1]}). The fairest columns are the "
                    f"2019-2020 test window (cost ≈ 0.71, time ≈ 0.80 AUC). The "
                    f"<b>deployment metric is overrun/delay MAGNITUDE + ranking</b>."
                    f"</div>", unsafe_allow_html=True)
            st.markdown("---")
        except Exception:
            pass

    ej = ROOT / "data" / "external_holdout_2025.json"
    if ej.exists():
        try:
            eres = json.loads(ej.read_text())
            section_heading("New-Document Holdout: May-2025 Flash Report")
            st.caption("Train on pre-cutoff annexure data; TEST on May-2025 snapshot "
                       "(different report format, labels **resolved** as of May-2025). "
                       "**Strongest generalisation check available.**")
            cols = st.columns(len(eres["results"]))
            for c, (cut, rep) in zip(cols, eres["results"].items()):
                with c:
                    st.markdown(f"**Train ≤ {rep['cutoff']} → test {rep['test_window']}**")
                    for task in ("time", "cost"):
                        m = rep[task]
                        if "clf" in m:
                            cl = m["clf"]
                            st.markdown(f"*{task.title()}* AUC **{cl['auc']}** · reg MAE "
                                        f"{m.get('reg', {}).get('mae', m.get('reg', {}).get('mae_months', '-'))} "
                                        f"(naive {m.get('reg', {}).get('naive_median_mae', '-')})")
                            lift = cl.get("top20pct")
                            if lift:
                                st.caption(f"top-20% catches {lift['recall@top']*100:.0f}% "
                                           f"(lift {lift['lift']}×) · test n={m['n_test']}")
            st.markdown(
                "<div class='gov-info'>"
                "<b>Reading:</b> Time-delay detection transfers to the new document "
                "(AUC ~0.74-0.81) with magnitudes 29-34% more accurate than the naive median. "
                "Cost <b>classification</b> is honestly weaker out-of-document (AUC ~0.64-0.72) — "
                "the deployable metric stays ranking + magnitude (top-20% catches 37-45% of "
                "actual overruns).</div>", unsafe_allow_html=True)
            mj = ROOT / "data" / "macro_experiment.json"
            if mj.exists():
                try:
                    mres = json.loads(mj.read_text())
                    st.caption("**Macro-covariate test (WPI/CPI/repo/GDP):** "
                               "adding macro data did **not** recover lost AUC — cost moved "
                               f"≤0.02 and *hurt* time (e.g. LR AUC {mres['results'][0]['time']['without']['auc_lr']:.2f} "
                               f"→ {mres['results'][0]['time']['with_macro']['auc_lr']:.2f}). "
                               "The degradation is label censoring, not a missing macro signal.")
                except Exception:
                    pass
            sj = ROOT / "data" / "survival_eval_2025.json"
            if sj.exists():
                try:
                    sres = json.loads(sj.read_text())
                    st.caption("**Survival reframing (discrete-time hazard, right-censored):** "
                               "on the most-censored cohort (2021-22, "
                               f"{sres['results']['cutoff_2020']['test_fraction_censored_not_yet_delayed']*100:.0f}% "
                               "still not flagged), survival C-index "
                               f"{sres['results']['cutoff_2020']['c_index_survival_incl_censored']} "
                               f"beats binary {sres['results']['cutoff_2020']['c_index_binary_lr_incl_censored']}.")
                except Exception:
                    pass
            aj = ROOT / "data" / "augmented_train_holdout_2025.json"
            if aj.exists():
                try:
                    ares = json.loads(aj.read_text())
                    r7 = ares["results"]["cutoff_2017"]
                    st.caption(f"**Augmentation test:** "
                               f"cost AUC {r7['baseline']['cost']['auc']:.3f} → "
                               f"**{r7['augmented']['cost']['auc']:.3f}** (+{r7['train_rows']['added_snapshot_cost']} rows) "
                               f"— more cost data helps. Time: "
                               f"{r7['baseline']['time']['auc']:.3f} → {r7['augmented']['time']['auc']:.3f} "
                               f"(label mismatch — harmonize definition first).")
                except Exception:
                    pass
            st.markdown("---")
        except Exception:
            pass

    section_heading("Add Project Data")
    with st.form("add_project_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Project name (unique)")
            sector = st.text_input("Sector", value="ROAD TRANSPORT AND HIGHWAYS")
            orig = st.number_input("Original cost (₹ Cr)", min_value=0.0, value=1000.0)
        with c2:
            exp = st.number_input("Expenditure to date (₹ Cr)", min_value=0.0, value=400.0)
            ant = st.number_input("Anticipated/revised cost (₹ Cr)", min_value=0.0, value=1150.0)
            ap_year = st.number_input("Approval year", min_value=1980, max_value=2026, value=2019)
        with c3:
            plan_year = st.number_input("Planned completion year", min_value=1980, max_value=2040, value=2025)
            tor = st.number_input("Time overrun reported (months; 0 = on schedule)",
                                  min_value=0.0, value=0.0)
            reasons = st.text_input("Reason (optional)")
        submitted = st.form_submit_button("Add project & retrain")

    if submitted:
        delayed = 1 if tor > 0 else 0
        cost_overrun_pct = (ant - orig) / orig * 100 if orig > 0 else 0.0
        overrun_ratio = cost_overrun_pct / 100.0
        pdur = (plan_year - ap_year) if plan_year >= ap_year else 0.0
        new_row = pd.DataFrame([{
            "sector": sector, "project_name": name, "original_cost_cr": orig,
            "anticipated_cost_cr": ant, "expenditure_cum_cr": exp,
            "approval_year": float(ap_year), "planned_doc_year": float(plan_year),
            "tor_months": tor, "cost_overrun_pct": cost_overrun_pct, "reasons": reasons,
            "delayed": delayed, "planned_duration_years": pdur, "overrun_ratio": overrun_ratio,
        }])
        cost_new_row = None
        if c := new_row[["sector", "project_name", "original_cost_cr", "anticipated_cost_cr",
                         "expenditure_cum_cr", "cost_overrun_pct", "overrun_ratio"]].copy():
            cost_new_row = c.assign(revised_cost_cr=ant, overrun_flag=delayed)
        try:
            report = ingest_and_retrain(time_new=new_row, cost_new=cost_new_row, verbose=False)
            st.success(f"Project added and models retrained (version {report['model']['version']}).")
            m = report["model"]
            st.info(f"Cost: clf AUC {m['cost']['clf_auc']:.3f} (5x2 repeated-stratified CV), "
                    f"reg MAE {m['cost']['reg_mae']:.3f} "
                    f"· Time: clf AUC {m['time']['clf_auc']:.3f}, reg MAE {m['time']['reg_mae']:.0f} mo")
            st.caption("CV figures hold out random projects (states shared across folds). "
                       "External temporal holdout (unseen resolved-label cohorts 2019-2021): "
                       f"cost {m.get('external_holdout_reference', {}).get('cost')} — "
                       "the number to quote for generalization.")
        except Exception as e:
            st.error(f"Could not retrain: {e}")

    st.markdown("---")
    section_heading("Upload Flash Report / CSV")
    import tempfile
    up = st.file_uploader("Upload Flash Report PDF or new-project CSV",
                          type=["pdf", "csv"], key="ingest_upload")
    if up is not None and st.button("Ingest uploaded file & retrain"):
        ext = up.name.rsplit(".", 1)[-1].lower()
        tmp = Path(tempfile.gettempdir()) / f"ingest_{up.name}"
        tmp.write_bytes(up.getvalue())
        if ext == "pdf":
            report = ingest_and_retrain(pdf_path=tmp, verbose=False)
        else:
            csv = pd.read_csv(tmp)
            report = ingest_and_retrain(time_new=csv, cost_new=csv, verbose=False)
        st.success("Uploaded; models retrained. " + f"cost +{report['added_cost']} / "
                   f"time +{report['added_time']} net new rows.")


# ---------------------------------------------------------------------------
# Right-panel contextual content per page
# ---------------------------------------------------------------------------
def _right_panel(page):
    if page == "Dashboard":
        st.markdown('<div class="gov-rp"><h4>Portfolio Quick View</h4>'
                     '<div class="rp-body">Central Sector Projects monitored: <b>1,876</b><br>'
                     'Avg cost overrun: ~62%<br>Avg delay risk: ~0.55</div></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="gov-rp"><h4>Top-Risk Sectors</h4>'
                     '<div class="rp-body">Railways · Petroleum · Road · Coal<br>'
                     '<span style="color:#64748b;">See sector benchmark table →</span></div></div>',
                    unsafe_allow_html=True)
    elif page == "Project Explorer":
        st.markdown('<div class="gov-rp"><h4>How to Read</h4>'
                     '<div class="rp-body">Select a project above. '
                     'The card shows its risk profile, predicted ₹ cost '
                     'overrun, delay estimate (with bounds), and a '
                     'plain-language AI explanation.</div></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="gov-rp"><h4>Models (Leak-Free)</h4>'
                     '<div class="rp-body">Cost clf AUC ~0.85 · reg MAE ~0.16<br>'
                     'Time clf AUC ~0.83 · reg MAE ~21 mo</div></div>',
                    unsafe_allow_html=True)
    elif page == "Geospatial Map":
        st.markdown('<div class="gov-rp"><h4>Using This Map</h4>'
                     '<div class="rp-body">• Colour by delay risk, overrun %, or project count<br>'
                     '• Click any state for drill-down<br>'
                     '• Use the watchlist to prioritise Escalate / Review / Monitor</div></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="gov-rp"><h4>Practical Triage</h4>'
                     '<div class="rp-body">States with ≥50% overrun or '
                     'high delay risk → <b style="color:#c0392b;">Escalate</b><br>'
                     '<b style="color:#e67e22;">Review</b> = quarterly milestone<br>'
                     '<b style="color:#1e7e34;">Monitor</b> = routine tracking</div></div>',
                    unsafe_allow_html=True)
    elif page == "AI Assistant":
        st.markdown('<div class="gov-rp"><h4>Ask About</h4>'
                     '<div class="rp-body">• Portfolio overview<br>'
                     '• Sector comparisons<br>• Top at-risk projects<br>'
                     '• Specific project detail</div></div>',
                    unsafe_allow_html=True)
        _llm_status = _ollama_available()
        if not _llm_status:
            st.markdown('<div class="gov-rp"><h4>LLM Status</h4>'
                         '<div class="rp-body" style="color:#c0392b;">● Offline — '
                         'Deterministic engine active</div></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="gov-rp"><h4>LLM Status</h4>'
                         f'<div class="rp-body" style="color:#1e7e34;">● Live · {MODEL_NAME}</div></div>',
                        unsafe_allow_html=True)
    elif page == "Data & Retrain":
        st.markdown('<div class="gov-rp"><h4>Continual Learning</h4>'
                     '<div class="rp-body">Add a project or upload '
                     'a Flash Report PDF/CSV, then retrain on all accumulated data. '
                     'Models are versioned, feature/label leak-free (out-of-fold encodings), '
                     'and validated against unseen cohorts (external temporal holdout).</div></div>',
                    unsafe_allow_html=True)
    elif page == "Home":
        st.markdown('<div class="gov-rp"><h4>SIH 2026 · PS103</h4>'
                     '<div class="rp-body">Smart India Hackathon 2026<br>'
                     'Problem Statement SIH26103<br>'
                     'Theme: Smart Automation<br>'
                     '<b>AI for Infrastructure Monitoring</b></div></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="gov-rp"><h4>Data Source</h4>'
                     '<div class="rp-body">MoSPI Monthly Flash Report<br>'
                     'Central Sector Projects<br>'
                     '₹150 Crore & above<br>'
                     'Real data (not synthetic)</div></div>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main app: navigation + layout
# ---------------------------------------------------------------------------
def main():
    pages = {
        "Home": page_home,
        "Dashboard": page_dashboard,
        "Geospatial Map": page_map,
        "Project Explorer": page_explorer,
        "AI Assistant": page_assistant,
        "Data & Retrain": page_data,
    }

    # Navigation state: only read query_params on the very first load (page not yet set)
    if "page" not in st.session_state:
        qp = st.query_params.get("page", "Home")
        st.session_state["page"] = qp if qp in pages else "Home"

    # Render GIGW 3.0 Government Portal Header
    render_header()

    # Render interactive Horizontal Government Navigation Bar
    active_page = render_navbar(st.session_state["page"])

    # Render breadcrumb bar and notification ticker
    render_subbar(active_page)

    page = active_page

    # Layout: main + right panel
    main_col, right_col = st.columns([3.0, 1.0], gap="small")
    with main_col:
        st.markdown('<div class="main-body" id="main-content">', unsafe_allow_html=True)
        pages[page]()
        st.markdown('</div>', unsafe_allow_html=True)
    with right_col:
        _right_panel(page)

    render_footer()


if __name__ == "__main__":
    main()

