"""PAIMANA-style AI Project Monitoring dashboard (SIH26103).

Government-portal look mirroring the PAIMANA website:
  - top ministry header bar
  - horizontal/sidebar navigation
  - Dashboard page with KPI cards, selectors, tables, charts
  - Geospatial Map page (India choropleth from real state-wise MoSPI data)
  - Project Explorer (per-project risk + AI explanation)
  - AI Assistant (interactive chat over the data/models)

Run:  streamlit run src/dashboard.py
"""
from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st
from plotly import graph_objects as go

from src.assistant import answer, build_assistant_view
from src.explainer import enrich
from src.state_data import load_state_data

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
TIME = ROOT / "data" / "real_mospi_time.csv"
GEO = ROOT / "data" / "india_state.geojson"

GOV_NAVY = "#0b2545"
GOV_BLUE = "#0b3d66"
ACCENT = "#e8a33d"
SAFFRON = "#FF9933"
GREEN = "#138808"
CHAKRA = "#000080"

st.set_page_config(page_title="MoSPI | IPMD — AI Project Monitoring", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------- global theme ----------
st.markdown("""
<style>
  #MainMenu, footer {visibility: hidden;}
  .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1180px;}
  [data-testid="stHeader"] {background: #0b2545; height: 0px;}
  .mos-head {background: linear-gradient(180deg,#0b2545,#0b3d66); color:#fff;
             padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; width:100%;}
  .mos-utils {background:#0b2545; color:#bcd4ee; font-size:12px; padding:6px 22px;
              display:flex; justify-content:space-between; letter-spacing:.3px;}
  .mos-utils a {color:#bcd4ee; text-decoration:none; margin-left:14px;}
  .mos-utils a:hover {color:#fff;}
  .mos-main {display:flex; align-items:center; gap:18px; padding:14px 22px; flex-wrap:wrap;}
  .mos-emblem {width:64px; height:64px; flex:0 0 auto;}
  .mos-id {flex:1; min-width:260px;}
  .mos-id .gov{font-size:11px; letter-spacing:1.5px; color:#9fc2e6;}
  .mos-id h1{font-size:22px; font-weight:700; margin:2px 0; line-height:1.1;}
  .mos-id .dept{font-size:13px; color:#d7e7f7;}
  .mos-tricolor{height:7px; width:100%;
    background:linear-gradient(90deg, #FF9933 0 33%, #ffffff 33% 66%, #138808 66% 100%);}
  .mos-tabs{background:#f4f6f8; border-bottom:3px solid #0b3d66; display:flex;
            flex-wrap:wrap; gap:4px; padding:8px 22px 0;}
  .mos-tab{border:none; background:#e3e9f1; color:#123; padding:9px 18px; cursor:pointer;
           font-weight:600; font-size:14px; border-radius:6px 6px 0 0;}
  .mos-tab.active{background:#0b3d66; color:#fff;}
  .mos-tab:hover{background:#c9d6e4;}
  .mos-bread{background:#eef2f6; color:#244; font-size:12.5px; padding:7px 22px;
             border-bottom:1px solid #dde4ec;}
  .mos-bread b{color:#0b3d66;}
  .mos-foot{background:#0b2545; color:#cfe0f0; font-size:12.5px; margin-top:34px;
            padding:16px 22px;}
  .mos-foot .row{display:flex; justify-content:space-between; flex-wrap:wrap; gap:10px;}
  .mos-foot a {color:#9fc2e6; text-decoration:none;}
  .mos-card{background:#fff; border:1px solid #e2e8f0; border-top:4px solid #0b3d66;
            border-radius:4px; padding:14px 16px; margin-bottom:12px;
            box-shadow:0 1px 2px rgba(0,0,0,.04);}
  .mos-card h3{margin:0 0 6px; color:#0b3d66; font-size:16px;}
  .mos-kpi{background:#fff; border:1px solid #e2e8f0; border-left:5px solid #e8a33d;
           border-radius:4px; padding:12px 14px; box-shadow:0 1px 2px rgba(0,0,0,.04);}
  .mos-kpi .l{font-size:11.5px; color:#556; letter-spacing:.4px;}
  .mos-kpi .v{font-size:24px; font-weight:700; color:#0b3d66;}
  .css-1r6slb0, [data-testid="stSidebar"]{display:none;}
</style>
""", unsafe_allow_html=True)

EMBLEM_SVG = (
    '<svg viewBox="0 0 64 64" class="mos-emblem" xmlns="http://www.w3.org/2000/svg">'
    '<circle cx="32" cy="32" r="30" fill="#000080"/>'
    '<circle cx="32" cy="32" r="15" fill="none" stroke="#fff" stroke-width="1.4"/>'
    '<line x1="32" y1="18" x2="32" y2="46" stroke="#fff" stroke-width="1.1" opacity=".6"/>'
    '<line x1="18" y1="32" x2="46" y2="32" stroke="#fff" stroke-width="1.1" opacity=".6"/>'
    '<line x1="22" y1="22" x2="42" y2="42" stroke="#fff" stroke-width="1.1" opacity=".6"/>'
    '<line x1="42" y1="22" x2="22" y2="42" stroke="#fff" stroke-width="1.1" opacity=".6"/>'
    '<circle cx="32" cy="32" r="6" fill="#fff"/>'
    '</svg>')

st.markdown(f"""
<div class="mos-head">
  <div class="mos-utils">
    <span><b>GOVERNMENT OF INDIA</b></span>
    <span><a href="#">गृह पृष्ठ</a><a href="#">मराठी</a><a href="#">हिंदी</a>
          <a href="#">English</a><a href="#">Accessibility</a><a href="#">A- | A | A+</a></span>
  </div>
  <div class="mos-main">
    {EMBLEM_SVG}
    <div class="mos-id">
      <div class="gov">MINISTRY OF STATISTICS &amp; PROGRAMME IMPLEMENTATION · भारत सरकार</div>
      <h1>Infrastructure &amp; Project Monitoring Division</h1>
      <div class="dept">AI-Powered Predictive Analytics &amp; Early Warning for Central Sector Projects
        (₹150 Cr &amp; above) · PS SIH26103</div>
    </div>
  </div>
  <div class="mos-tricolor"></div>
</div>
<div class="mos-bread">Home / <b>Project Monitoring &amp; Decision Support</b></div>
""", unsafe_allow_html=True)



# ---------------- data loading ----------------
@st.cache_data
def load_time():
    return pd.read_csv(TIME)


@st.cache_data
def build_view(time):
    from src.explainer import enrich as _enrich
    rows = []
    for _, r in time.iterrows():
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
    return pd.DataFrame(rows)


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


@st.cache_data
def load_geo_light(eps=0.02):
    """Load + aggressively simplify the India GeoJSON (Douglas-Peucker decimation +
    coordinate rounding + property pruning) so the choropleth stays responsive in-browser."""
    gj = load_geojson()
    gj = _walk_geo(gj, eps)
    for f in gj.get("features", []):
        props = f.get("properties", {})
        f["properties"] = {"NAME_1": props.get("NAME_1")}
    return gj


# ---------------- nav (horizontal government tab bar) ----------------
PAGES = {
    "Home": "🏠 Home",
    "Dashboard": "📊 Monitoring Dashboard",
    "Geospatial Map": "🗺️ Geospatial Map",
    "Project Explorer": "🔍 Project Explorer",
    "AI Assistant": "🤖 AI Assistant",
}

def _nav_bar():
    col = st.columns(len(PAGES))[0]
    cols = st.columns(len(PAGES))
    for c, key in zip(cols, PAGES):
        label = f"{PAGES[key]}"
        active = st.session_state.get("page", "Home") == key
        cls = "mos-tab active" if active else "mos-tab"
        with c:
            if st.button(label, key=f"nav_{key}", type="secondary", use_container_width=True,
                         help=key):
                st.session_state["page"] = key
                st.rerun()

if "page" not in st.session_state:
    st.session_state["page"] = "Home"
_nav_bar()
page = st.session_state["page"]


def kpi_row(cols, items):
    for c, (label, val) in zip(cols, items):
        c.markdown(
            f"<div class='mos-kpi'><div class='l'>{label}</div><div class='v'>{val}</div></div>",
            unsafe_allow_html=True)


def mos_card(title, body_html=""):
    return st.markdown(
        f"<div class='mos-card'><h3>{title}</h3>{body_html}</div>", unsafe_allow_html=True)


def sector_cost_delay_chart(view):
    sc = view.groupby("sector").agg(
        cost_risk=("cost_risk", "mean"), delay_risk=("delay_risk", "mean"),
        n=("project_name", "count")).reset_index()
    return alt.Chart(sc).mark_circle(size=90).encode(
        x=alt.X("cost_risk:Q", scale=alt.Scale(zero=False), title="Cost risk (mean)"),
        y=alt.Y("delay_risk:Q", scale=alt.Scale(zero=False), title="Delay risk (mean)"),
        size="n:Q", tooltip=["sector", "cost_risk", "delay_risk", "n"],
        color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"))).properties(height=380)


# ---------------- pages ----------------
def page_home():
    st.subheader("About")
    st.markdown("""
The **Infrastructure & Project Monitoring Division (IPMD)**, MoSPI, monitors Central Sector
Infrastructure Projects costing ₹150 crore and above. This AI platform moves monitoring from
**descriptive to predictive**: it forecasts **cost overruns**, **time overruns** and
**implementation risk** before issues materialise, so policymakers can prioritise
interventions.

Built on **real Flash Report data**, with an ML-vs-conventional-statistics comparison, SHAP
driver analysis, per-sector and **state-wise geospatial** views, and an **AI assistant** that
explains project risk in plain language.
""")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.markdown("#### Deliverables covered")
    c1.markdown("- Cost Overrun Model\n- Time Overrun Model\n- Risk Scoring\n- Early Warning")
    c2.markdown("#### Analytics")
    c2.markdown("- ML vs statistical comparison\n- SHAP driver analysis\n- Per-sector benchmarks\n- State-wise geospatial map")
    c3.markdown("#### Intelligence")
    c3.markdown("- AI Monitoring Dashboard\n- **AI Assistant (chat)**\n- LLM explanations\n- Open-source stack")


def page_dashboard():
    view = build_view(load_time())
    st.subheader("Monitored Portfolio — Central Sector Projects")
    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        st.markdown("**Report**"); st.caption("Monthly Flash Report · May 2024")
    with colB:
        st.markdown("**Breakdown**"); st.caption("Central Sector · ₹150 Cr & above")

    kpi_row(
        st.columns(5),
        [
            ("Projects monitored", f"{len(view):,}"),
            ("Avg cost overrun", f"{view['cost_overrun_pct'].mean():.1f}%"),
            ("Avg cost risk", f"{view['cost_risk'].mean():.2f}"),
            ("Avg delay risk", f"{view['delay_risk'].mean():.2f}"),
            ("Hi-risk (cost)", f"{(view['cost_risk']>=0.45).sum():,}"),
        ])
    st.markdown("---")

    left, right = st.columns(2)
    with left:
        st.markdown("**Sector benchmark (mean risk)**")
        bench = view.groupby("sector")[["cost_risk", "delay_risk"]].mean().round(3)\
            .sort_values("delay_risk", ascending=False)
        st.dataframe(bench, use_container_width=True)
    with right:
        st.markdown("**Cost vs delay risk by sector**")
        st.altair_chart(sector_cost_delay_chart(view), use_container_width=True)


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


def page_map():
    st.subheader("Geospatial View — State-wise Cost & Time Overrun")
    st.caption("Real state-wise position from the MoSPI Flash Report (May 2024). Click a state to inspect it.")
    df = load_state().copy()
    gj = load_geo_light()
    df["geo_name"] = df["state"].map(_GEO_NAME_MAP)

    metric = st.radio("Colour by", ["delay_risk", "cost_overrun_pct", "n_projects"],
                      horizontal=True, format_func=lambda m: {
                          "delay_risk": "Delay risk (0–1)",
                          "cost_overrun_pct": "Cost overrun (%)",
                          "n_projects": "No. of projects"}[m])

    # only colour features present in the boundary file (avoids blank/odd regions)
    feature_names = {f["properties"]["NAME_1"] for f in gj["features"]}
    plot_df = df[df["geo_name"].isin(feature_names)].copy()

    fig = go.Figure(go.Choropleth(
        geojson=gj, locations=plot_df["geo_name"], z=plot_df[metric], featureidkey="properties.NAME_1",
        colorscale="Blues", marker_line_color="white", marker_line_width=0.4,
        colorbar_title=metric, hovertemplate="%{location}<br>%{z:.2f}<extra></extra>",
        selected=dict(marker=dict(opacity=1)),
        unselected=dict(marker=dict(opacity=0.55))))
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=620)
    ev = st.plotly_chart(fig, use_container_width=True, on_select="rerun")

    # clicked state -> drill-down
    selected = None
    try:
        pts = ev["selection"]["points"]
        if pts:
            selected = pts[0].get("location")
    except Exception:
        selected = None

    if selected:
        srow = plot_df[plot_df["geo_name"] == selected]
        if not srow.empty:
            r = srow.iloc[0]
            st.markdown(f"### {selected}")
            k1, k2, k3 = st.columns(3)
            k1.metric("Projects", f"{r['n_projects']}")
            k2.metric("Cost overrun", f"{r['cost_overrun_pct']:.1f}%")
            k3.metric("Delay (TOR) range", f"{r['tor_min_months']}–{r['tor_max_months']} mo")
            st.markdown(f"State-wise delay risk **{r['delay_risk']:.2f}** — "
                        f"{'elevated' if r['delay_risk'] >= 0.5 else 'moderate' if r['delay_risk'] >= 0.3 else 'low'} exposure.")
        else:
            st.caption(f"'{selected}' is drawn from the boundary file but has no project table "
                       f"entry (post-census state); click another state.")
    st.caption("Note: some post-census states (e.g. Telangana, Ladakh) are not present in the "
               "boundary set used; they remain visible in the table below.")

    with st.expander("View state-wise table"):
        st.dataframe(df[["state", "n_projects", "cost_overrun_pct",
                         "tor_min_months", "tor_max_months"]], use_container_width=True)


def page_explorer():
    view = build_view(load_time())
    st.subheader("Project Explorer")
    st.caption("Select a project to see its risk profile and a plain-language explanation of why it is at risk.")
    sector = st.selectbox("Sector", ["All"] + sorted(view["sector"].dropna().unique().tolist()))
    rows_in = view if sector == "All" else view[view["sector"] == sector]
    proj = st.selectbox("Project", rows_in["project_name"].tolist())
    idx = view.index[view["project_name"] == proj][0]
    pr = view.loc[idx]

    # header card
    ap = f" · approved {int(pr['approval_year'])}" if pd.notna(pr["approval_year"]) else ""
    st.markdown(f"<div class='mos-card'><h3>{proj}</h3>"
                f"<div style='color:#556; font-size:13px;'>{pr['sector']}{ap}</div></div>",
                unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Original cost", _cr(pr["original_cost_cr"]))
    k2.metric("Estimated cost", _cr(pr["anticipated_cost_cr"]))
    k3.metric("Spent to date", _cr(pr["expenditure_cum_cr"]))
    k4.metric("Cost overrun", f"{pr['cost_overrun_pct']:.1f}%")

    g1, g2 = st.columns(2)
    g1.markdown(_risk_gauge("Cost risk", pr["cost_risk"], pr["cost_tier"]), unsafe_allow_html=True)
    g2.markdown(_risk_gauge("Delay risk", pr["delay_risk"], pr["delay_tier"]), unsafe_allow_html=True)

    # persisted AI explanation (survives rerun) else template
    ai_key = f"expl_{proj}"
    explanation = st.session_state.get(ai_key, pr["explanation"])
    source = st.session_state.get(ai_key + "_src", pr["explanation_source"])

    st.markdown("### Why this project is classified as at risk")
    st.markdown(explanation)
    if pd.notna(pr["reasons"]) and str(pr["reasons"]) not in ("nan", ""):
        st.caption(f"Recorded causes in report: {pr['reasons']}")

    if st.button("♻️ Regenerate explanation with AI (Ollama)"):
        src_row = load_time()[load_time()["project_name"] == proj]
        if len(src_row):
            e = enrich(src_row.iloc[0], is_time=True, prefer_llm=True)
            st.session_state[ai_key] = e["explanation"]
            st.session_state[ai_key + "_src"] = e["explanation_source"]
            st.rerun()
    st.caption(f"Explanation source: {source}")


def _cr(v):
    try:
        return f"₹{float(v):,.1f} Cr" if pd.notna(v) and v else "—"
    except Exception:
        return "—"


def _risk_gauge(label, score, tier):
    color = "#c0392b" if score >= 0.45 else "#e8a33d" if score >= 0.25 else "#138808"
    return (f"<div class='mos-card'><h3>{label} — {tier}</h3>"
            f"<div style='height:10px;background:#eef2f6;border-radius:5px;'>"
            f"<div style='height:10px;width:{min(100, score*100):.0f}%;background:{color};border-radius:5px;'></div></div>"
            f"<div style='font-size:13px;color:#556;margin-top:6px;'>score {score:.2f} / 1.00</div></div>")


def _reset_history():
    st.session_state.history = [{"role": "assistant",
                                 "content": "Hello! Ask me about at-risk projects, sector "
                                            "comparisons, budgets, or a specific project."}]


def page_assistant():
    st.subheader("AI Assistant — Project Intelligence")
    st.caption("Ask about the portfolio, sectors, top risks, budgets, or any single project.")
    view = build_view(load_time())
    time = load_time()
    ctx = build_assistant_view(view, time)

    if "history" not in st.session_state:
        _reset_history()

    header = st.columns([5, 1])
    with header[1]:
        if st.button("🆕 New conversation", use_container_width=True):
            st.session_state.history = []
            st.rerun()
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
            text, table = answer(q, ctx)
            st.markdown(text)
            if table is not None:
                st.dataframe(table, use_container_width=True)
            st.session_state.history.append({"role": "assistant", "content": text})


pages = {
    "Home": page_home,
    "Dashboard": page_dashboard,
    "Geospatial Map": page_map,
    "Project Explorer": page_explorer,
    "AI Assistant": page_assistant,
}
pages[page]()


# ---------------- footer ----------------
st.markdown("""
<div class="mos-foot">
  <div class="row">
    <span><b>Infrastructure &amp; Project Monitoring Division (IPMD)</b> · MoSPI, Government of India</span>
    <span><a href="#">Help</a>·<a href="#">Feedback</a>·<a href="#">Terms</a>·<a href="#">Privacy</a>·<a href="#">Copyright</a></span>
  </div>
  <div class="row" style="margin-top:8px; color:#8fb3d8; font-size:11.5px;">
    <span>Data: MoSPI Monthly Flash Report (Central Sector Projects, ₹150 Cr &amp; above). Predictive models built in-house on real data.</span>
    <span>Content owned by Ministry · Powered by open-source AI · Designed for SIH 2026 · PS SIH26103</span>
  </div>
</div>
""", unsafe_allow_html=True)
