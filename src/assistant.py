"""AI Assistant — an LLM-driven project-intelligence agent over the real MoSPI data.

Approach: the user question + dataset are given to an 8B-class local LLM (Ollama).
The model calls one of the query tools below (structured), and we execute it against
the real data, returning a live answer + optional table. This correctly handles free
questions like "how many airport-related projects are there?".

When Ollama is unavailable (demo fallback), a deterministic query engine answers the
same range of questions so the chat still works.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from src.explainer import _ollama, SYSTEM

# tool schemas for Ollama function-calling (supported by llama3.1+/qwen2.5 tool models)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_projects",
            "description": "Count or list projects matching a keyword in their name or sector (e.g. 'airport', 'railway', 'coal').",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "substring to match"},
                    "field": {"type": "string", "enum": ["project_name", "sector"], "default": "project_name"},
                    "list": {"type": "boolean", "description": "return a table of matches", "default": False},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["keyword"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rank_projects",
            "description": "Rank projects (optionally filtered by a keyword) by a metric to answer assessing/comparative questions, e.g. 'which airport project has the highest cost overrun'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": ["cost_overrun_pct", "cost_risk", "delay_risk", "tor_months", "original_cost_cr", "expenditure_cum_cr"], "description": "metric to rank by"},
                    "keyword": {"type": "string", "description": "optional filter substring in name/sector"},
                    "n": {"type": "integer", "default": 5},
                    "ascending": {"type": "boolean", "default": False},
                },
                "required": ["metric"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sector_stats",
            "description": "Get aggregate risk/overrun stats for a sector, or all sectors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sector": {"type": "string", "description": "sector name, or 'ALL'"},
                },
                "required": ["sector"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "top_risk_projects",
            "description": "Top-N projects by combined cost+delay risk.",
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "integer", "default": 10}},
                "required": [],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "project_detail",
            "description": "Risk details for a single named project.",
            "parameters": {
                "type": "object",
                "properties": {"project_name": {"type": "string"}},
                "required": ["project_name"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "portfolio_overview",
            "description": "High-level portfolio summary (count, avg overrun, risk).",
            "parameters": {"type": "object", "properties": {}},
        }
    },
]


# ---------------- query tools (executed against real data) ----------------
def tool_search(ctx, args):
    kw = str(args.get("keyword", "")).lower()
    field = args.get("field", "project_name")
    view = ctx["view"]
    col = "project_name" if field == "project_name" else "sector"
    mask = view[col].astype(str).str.lower().str.contains(kw, na=False)
    sub = view[mask]
    if args.get("list"):
        df = sub[["project_name", "sector", "cost_risk", "delay_risk", "cost_overrun_pct"]]\
            .head(int(args.get("limit", 10)))
        return (f"{len(sub)} project(s) match '{kw}'.", df)
    return (f"There are **{len(sub)}** projects matching '{kw}' in {field}.", None)


def tool_sector(ctx, args):
    view = ctx["view"]
    s = args.get("sector", "ALL")
    if s.upper() == "ALL":
        df = view.groupby("sector").agg(
            n=("project_name", "count"), mean_cost_risk=("cost_risk", "mean"),
            mean_delay_risk=("delay_risk", "mean"),
            mean_overrun=("cost_overrun_pct", "mean")).sort_values("n", ascending=False)
        return ("Sector-wise summary:", df)
    sub = view[view["sector"].str.lower() == s.lower()]
    if sub.empty:
        return (f"No sector named '{s}'.", None)
    return (f"Sector **{s}** has {len(sub)} projects; mean cost risk "
            f"{sub['cost_risk'].mean():.2f}, mean delay risk {sub['delay_risk'].mean():.2f}, "
            f"mean overrun {sub['cost_overrun_pct'].mean():.1f}%.", None)


def tool_top(ctx, args):
    view = ctx["view"].copy()
    view["combined"] = (view["cost_risk"] + view["delay_risk"]) / 2
    n = int(args.get("n", 10))
    top = view.nlargest(n, "combined")[["project_name", "sector", "cost_risk",
                                        "delay_risk", "combined"]]
    return (f"Top {n} projects by combined risk:", top)


def tool_rank(ctx, args):
    view = ctx["view"].copy()
    metric = args.get("metric", "cost_overrun_pct")
    if metric not in view.columns:
        return f"Unknown metric '{metric}'.", None
    kw = str(args.get("keyword") or "").lower()
    if kw:
        mask = ((view["project_name"].astype(str).str.lower().str.contains(kw, na=False)) |
                (view["sector"].astype(str).str.lower().str.contains(kw, na=False)))
        view = view[mask]
        if view.empty:
            return (f"No projects match '{kw}'.", None)
    n = int(args.get("n", 5))
    ascending = bool(args.get("ascending", False))
    ranked = view.sort_values(metric, ascending=ascending).head(n)
    cols = [c for c in ["project_name", "sector", metric, "cost_risk", "delay_risk",
                        "cost_overrun_pct", "original_cost_cr"] if c in ranked.columns]
    label = metric
    return (f"Projects ranked by **{label}**" + (f" among those matching '{kw}'" if kw else "") +
            ": " + ranked["project_name"].astype(str).head(3).str.cat(sep=", ") + "...",
            ranked[cols])


def tool_detail(ctx, args):
    view = ctx["view"]
    name = args.get("project_name", "")
    row = view[view["project_name"].str.lower() == name.lower()]
    if row.empty:
        # fuzzy: keyword match
        row = view[view["project_name"].astype(str).str.contains(name, case=False, na=False)]
    if row.empty:
        return (f"Couldn't find project '{name}'.", None)
    r = row.iloc[0]
    fmt_cr = lambda v: f"₹{float(v):,.1f} Cr" if pd.notna(v) and v else "n/a"
    line = (f"**{r['sector']} — {r['project_name']}**  \n"
            f"- Original cost: {fmt_cr(r.get('original_cost_cr'))}  \n"
            f"- Estimated (anticipated) cost: {fmt_cr(r.get('anticipated_cost_cr'))}  \n"
            f"- Expenditure to date: {fmt_cr(r.get('expenditure_cum_cr'))}  \n"
            f"- Cost overrun: {r['cost_overrun_pct']:.1f}%  \n"
            f"- Cost risk: {r['cost_risk']:.2f} ({r['cost_tier']})  \n"
            f"- Delay risk: {r['delay_risk']:.2f} ({r['delay_tier']})")
    if pd.notna(r.get("tor_months")) and r.get("tor_months"):
        line += f"  \n- TOR: {r['tor_months']:.0f} months"
    if r.get("reasons") and str(r["reasons"]) not in ("nan", ""):
        line += f"  \n- Noted causes: {r['reasons']}"
    return line, None


def tool_overview(ctx, args):
    p = ctx["portfolio"]
    return (f"**{p['n']:,}** projects on monitor; avg cost overrun **{p['mean_cost_overrun']:.1f}%**; "
            f"avg cost risk {p['mean_cost_risk']:.2f}, avg delay risk {p['mean_delay_risk']:.2f}; "
            f"**{p['critical_high_cost']}** projects at critical/high cost risk.", None)


TOOL_DISPATCH = {
    "search_projects": tool_search,
    "sector_stats": tool_sector,
    "top_risk_projects": tool_top,
    "rank_projects": tool_rank,
    "project_detail": tool_detail,
    "portfolio_overview": tool_overview,
}


def build_assistant_view(view: pd.DataFrame, raw_time: pd.DataFrame) -> dict:
    view = view.copy()
    view["combined"] = (view["cost_risk"] + view["delay_risk"]) / 2
    return {
        "view": view,
        "portfolio": {
            "n": int(len(view)),
            "mean_cost_overrun": float(view["cost_overrun_pct"].mean()),
            "mean_cost_risk": float(view["cost_risk"].mean()),
            "mean_delay_risk": float(view["delay_risk"].mean()),
            "critical_high_cost": int((view["cost_risk"] >= 0.45).sum()),
            "critical_high_delay": int((view["delay_risk"] >= 0.45).sum()),
        },
    }


def _dispatch(name, args, ctx):
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return "Unknown tool.", None
    return fn(ctx, args)


# ---------------- LLM tool-calling path ----------------
def _llm_choose_tool(q: str):
    """Ask Ollama to pick a tool + args (JSON). Returns (tool_name, args) or None."""
    user = ("You are a data assistant. Given the question, respond ONLY with a JSON object: "
            '{"name": "<tool>", "arguments": {...}} choosing from: '
            "search_projects, sector_stats, top_risk_projects, rank_projects, project_detail, portfolio_overview. "
            f'Question: "{q}"')
    out = _ollama([{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": user}], tools=TOOLS)
    if not out:
        return None
    # try structured tool_calls first, else parse JSON from text
    if isinstance(out, dict) and out.get("tool_calls"):
        tc = out["tool_calls"][0]["function"]
        return tc["name"], json.loads(tc.get("arguments") or "{}")
    try:
        body = re.search(r"\{.*\}", out, re.S)
        if body:
            d = json.loads(body.group(0))
            return d["name"], d.get("arguments", {})
    except Exception:
        pass
    return None


# ---------------- deterministic fallback query engine ----------------
_FIELD_KEYWORDS = {
    "airport": "airport", "air": "airport", "rail": "railway", "highway": "highway",
    "road": "road", "coal": "coal", "power": "power", "port": "port", "metro": "metro",
    "health": "HEALTH", "education": "EDUCATION", "petroleum": "petroleum",
}


def _fallback(q, ctx):
    ql = q.lower()
    # explain / tell-me-about a specific project
    if re.search(r"(explain|tell me about|about the project|why is .* at risk|what.*(happened|is wrong|status).*)", ql):
        proj = _find_project_in(q, ctx["view"])
        if proj:
            text, _ = tool_detail(ctx, {"project_name": proj})
            expl = ctx["view"].loc[ctx["view"]["project_name"] == proj, "explanation"]
            tail = expl.iloc[0] if not expl.empty else ""
            return (text + "\n\n**Plain-language breakdown:**\n" + tail), None
    assess = re.search(r"(highest|lowest|most|least|largest|smallest|max|min|biggest|rank|which .* (overrun|risk|cost|spend))", ql)
    if assess:
        metric = "cost_overrun_pct"
        if re.search(r"risk", ql):
            metric = "cost_risk" if "cost risk" in ql or "cost" in ql else "delay_risk"
        if re.search(r"(tor|delay)", ql) and "cost" not in ql:
            metric = "tor_months"
        if re.search(r"(budget|cost|spend|expensive|original)", ql) and "overrun" not in ql and "risk" not in ql:
            metric = "original_cost_cr"
        ascending = bool(re.search(r"(lowest|least|smallest|min)", ql))
        kw = _find_keyword(ql)
        ranked, tab = tool_rank(ctx, {"metric": metric, "keyword": kw or "",
                                      "n": 5, "ascending": ascending})
        return ranked, tab
    want_list = bool(re.search(r"(list|show|which|name)", ql))
    if re.search(r"how many|count|number of", ql) or (want_list and _find_keyword(ql)):
        kw = _find_keyword(ql)
        view = ctx["view"]
        if kw:
            mask = ((view["project_name"].astype(str).str.lower().str.contains(kw, na=False)) |
                    (view["sector"].astype(str).str.lower().str.contains(kw, na=False)))
            n = int(mask.sum())
            if want_list:
                df = view[mask][["project_name", "sector", "cost_risk",
                                 "delay_risk", "cost_overrun_pct"]].head(15)
                return (f"**{n}** project(s) relate to '{kw}'. Showing up to 15:", df)
            return (f"**{n}** project(s) relate to '{kw}' (matched in project name or sector).", None)
        p = ctx["portfolio"]
        return f"The portfolio monitors **{p['n']:,}** projects.", None
    if re.search(r"(top|riskiest|worst|at[- ]risk)", ql):
        n = re.search(r"(\d+)", q)
        n = int(n.group(1)) if n else 10
        return tool_top(ctx, {"n": n})
    if re.search(r"(sector|per sector|by sector)", ql):
        kw = _find_keyword(ql)
        if kw:
            return tool_sector(ctx, {"sector": kw})
        return tool_sector(ctx, {"sector": "ALL"})
    if re.search(r"(overall|portfolio|summary|total|status)", ql):
        return tool_overview(ctx, {})
    return ("I can answer questions like: 'How many airport-related projects?', "
            "'List airport projects', 'Top 10 at-risk projects', 'Railways sector summary', "
            "'Overall portfolio status', or 'Tell me about <project name>'.", None)


def _find_keyword(ql: str):
    for k in sorted(_FIELD_KEYWORDS, key=len, reverse=True):
        if k in ql:
            return _FIELD_KEYWORDS[k]
    return None


def _find_project_in(q: str, view: pd.DataFrame):
    ql = q.lower()
    # (a) full-name substring in query
    for _, row in view.iterrows():
        name = str(row["project_name"]).lower()
        if name and name in ql and len(name) > 6:
            return row["project_name"]
    # (b) query keyword appears inside a project name (e.g. 'bhavini' in 'BHAVINI reactor')
    tokens = {w for w in re.findall(r"[a-z0-9]{4,}", ql) if w not in
              {"explain", "about", "project", "tell", "what", "which", "status", "overrun",
              "risk", "cost", "delay", "the", "this", "that", "portfolio", "sector", "work"}}
    if tokens:
        best, best_name = 0, None
        for _, row in view.iterrows():
            name = str(row["project_name"]).lower()
            hit = sum(1 for t in tokens if t in name)
            if hit > best:
                best, best_name = hit, row["project_name"]
        if best_name is not None:
            return best_name
    return None


def answer(q: str, ctx: dict) -> tuple[str, pd.DataFrame | None]:
    """Return (markdown_text, optional_table). Tries the LLM agent, falls back to engine."""
    choice = _llm_choose_tool(q)
    if choice:
        name, args = choice
        text, table = _dispatch(name, args, ctx)
        # let the LLM phrase a final answer in prose when possible
        if table is not None:
            prose = _ollama([{"role": "system", "content": SYSTEM},
                             {"role": "user", "content":
                              f"The data result is:\n{text}\n{table.head(10).to_string()}\n"
                              f"Answer the user's original question concisely: {q}"}])
            if prose:
                text = prose
        return text, table
    return _fallback(q, ctx)