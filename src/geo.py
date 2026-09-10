"""Shared geo utilities: state extraction from project names + centroid lookup.

MoSPI Flash Report project names always include the state/UT at the end
(e.g. "...AAI,TAMIL NADU"). This module extracts that and maps to
lat/lon centroids for use in ML features and dashboard geoplots.
"""
from __future__ import annotations

import re
from typing import Optional

# --- State centroid database (centroid of each state/UT) ---
# Keys are UPPER-CASE canonical names matching MoSPI conventions.
STATE_COORDINATES: dict[str, dict] = {
    "ANDHRA PRADESH":       {"lat": 15.9129, "lon": 79.7400},
    "ARUNACHAL PRADESH":    {"lat": 28.2180, "lon": 94.7278},
    "ASSAM":                {"lat": 26.2006, "lon": 92.9376},
    "BIHAR":                {"lat": 25.0961, "lon": 85.3131},
    "CHHATTISGARH":         {"lat": 21.2787, "lon": 81.8661},
    "GOA":                  {"lat": 15.2993, "lon": 74.1240},
    "GUJARAT":              {"lat": 22.2587, "lon": 71.1924},
    "HARYANA":              {"lat": 29.0588, "lon": 76.0856},
    "HIMACHAL PRADESH":     {"lat": 31.1048, "lon": 77.1734},
    "JAMMU AND KASHMIR":    {"lat": 33.7782, "lon": 76.5762},
    "JHARKHAND":            {"lat": 23.6102, "lon": 85.2799},
    "KARNATAKA":            {"lat": 15.3173, "lon": 75.7139},
    "KERALA":               {"lat": 10.8505, "lon": 76.2711},
    "MADHYA PRADESH":       {"lat": 22.9734, "lon": 78.6569},
    "MAHARASHTRA":          {"lat": 19.7515, "lon": 75.7139},
    "MANIPUR":              {"lat": 24.6637, "lon": 93.9063},
    "MEGHALAYA":            {"lat": 25.4670, "lon": 91.3662},
    "MIZORAM":              {"lat": 23.1645, "lon": 92.9376},
    "NAGALAND":             {"lat": 26.1584, "lon": 94.5624},
    "ODISHA":               {"lat": 20.9517, "lon": 85.0985},
    "PUNJAB":               {"lat": 31.1471, "lon": 75.3412},
    "RAJASTHAN":            {"lat": 27.0238, "lon": 74.2179},
    "SIKKIM":               {"lat": 27.5330, "lon": 88.5122},
    "TAMIL NADU":           {"lat": 11.1271, "lon": 78.6569},
    "TELANGANA":            {"lat": 18.1124, "lon": 79.0193},
    "TRIPURA":              {"lat": 23.9408, "lon": 91.9882},
    "UTTAR PRADESH":        {"lat": 26.8467, "lon": 80.9462},
    "UTTARAKHAND":          {"lat": 30.0668, "lon": 79.0193},
    "WEST BENGAL":          {"lat": 22.9868, "lon": 87.8550},
    "DELHI":                {"lat": 28.7041, "lon": 77.1025},
    "LADAKH":               {"lat": 34.1526, "lon": 77.5771},
    "MULTI STATE":          {"lat": 21.1458, "lon": 79.0882},
    "PUDUCHERRY":           {"lat": 11.9416, "lon": 79.8083},
    "CHANDIGARH":           {"lat": 30.7333, "lon": 76.7794},
    "DADRA AND NAGAR HAVELI": {"lat": 20.1809, "lon": 73.0158},
    "DAMAN AND DIU":        {"lat": 20.3974, "lon": 72.8328},
    "LAKSHADWEEP":          {"lat": 10.5667, "lon": 72.6417},
    "ANDAMAN AND NICOBAR":  {"lat": 11.7401, "lon": 92.7297},
    "MEGHALAYA":            {"lat": 25.4670, "lon": 91.3662},
}

# Build regex pattern: longest names first so "ARUNACHAL PRADESH" matches before "PRADESH"
_STATE_NAMES_SORTED = sorted(STATE_COORDINATES.keys(), key=len, reverse=True)
_STATE_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(s) for s in _STATE_NAMES_SORTED) + r')\b',
    re.IGNORECASE,
)

# Normalized (whitespace-stripped, upper) state name -> canonical name.
# PDF extraction injects stray spaces into truncated state names
# (e.g. "MAHARASHTR A" or "CHHATTISGA RH"), so we match on whitespace-free form.
_NORMALIZED_STATES = {}
for _canon in STATE_COORDINATES:
    _NORMALIZED_STATES[re.sub(r"\s+", "", _canon)] = _canon
_STATE_PARTS = sorted(_NORMALIZED_STATES.keys(), key=len, reverse=True)

# Also handle abbreviations in project names (e.g. "...TAMIL NADU" or "...TN)")
_ABBREV_MAP = {
    "AP": "ANDHRA PRADESH", "AR": "ARUNACHAL PRADESH", "AS": "ASSAM",
    "BR": "BIHAR", "CG": "CHHATTISGARH", "GA": "GOA", "GJ": "GUJARAT",
    "HR": "HARYANA", "HP": "HIMACHAL PRADESH", "JK": "JAMMU AND KASHMIR",
    "JH": "JHARKHAND", "KA": "KARNATAKA", "KL": "KERALA",
    "MP": "MADHYA PRADESH", "MH": "MAHARASHTRA", "MN": "MANIPUR",
    "ML": "MEGHALAYA", "MZ": "MIZORAM", "NL": "NAGALAND", "OD": "ODISHA",
    "PB": "PUNJAB", "RJ": "RAJASTHAN", "SK": "SIKKIM", "TN": "TAMIL NADU",
    "TS": "TELANGANA", "TR": "TRIPURA", "UP": "UTTAR PRADESH",
    "UK": "UTTARAKHAND", "WB": "WEST BENGAL", "DL": "DELHI",
    "LA": "LADAKH", "PY": "PUDUCHERRY", "CH": "CHANDIGARH",
    "AN": "ANDAMAN AND NICOBAR", "LD": "LAKSHADWEEP",
}


def _match_normalized(token: str) -> Optional[str]:
    """Match a possibly space-mangled state token against the known states."""
    norm = re.sub(r"\s+", "", token).upper()
    if not norm:
        return None
    for part in _STATE_PARTS:
        # exact whitespace-free equality OR the token is a clean prefix
        # (PDF truncates at cell edges, so partial matches are common).
        if part == norm or part.startswith(norm):
            return _NORMALIZED_STATES[part]
    return None


def _extract_from_suffix(name: str) -> Optional[str]:
    """Parse the '...]ORG,STATE' tail that Flash Report project names carry."""
    m = re.search(r"\]\s*([^\]\[]+)$", name)
    if not m:
        return None
    tail = m.group(1)
    # state is everything after the LAST comma (org prefix can contain spaces)
    if "," in tail:
        state_part = tail.rsplit(",", 1)[1].strip()
    else:
        state_part = tail.strip()
    if not state_part:
        return None
    hit = _match_normalized(state_part)
    if hit:
        return hit
    # fall back to full-name regex inside the tail (e.g. suffix missing comma)
    m2 = _STATE_PATTERN.search(state_part)
    if m2:
        return m2.group(1).upper()
    return None


def extract_state(project_name) -> Optional[str]:
    """Extract state/UT name from a MoSPI project name string.

    Tries the ']ORG,STATE' suffix first (most reliable, handles PDF space
    mangling), then full-name regex, then abbreviation lookup.
    """
    if project_name is None or (isinstance(project_name, float) and __import__("math").isnan(project_name)):
        return None
    name = str(project_name)
    # 1. ']ORG,STATE' suffix (most projects carry it)
    s = _extract_from_suffix(name)
    if s:
        return s
    # 2. Full state name anywhere in the string
    m = _STATE_PATTERN.search(name)
    if m:
        return m.group(1).upper()
    # 3. Abbreviation after comma or bracket: e.g. '...]AAI,TN' or '...]NPCIL,KA'
    m3 = re.search(r"\]\s*[A-Z0-9]*,\s*([A-Z]{2})\b", name)
    if m3:
        abbr = m3.group(1)
        if abbr in _ABBREV_MAP:
            return _ABBREV_MAP[abbr]
    return None


def state_to_geo(state) -> tuple[Optional[float], Optional[float]]:
    """Map a state name to (latitude, longitude) centroid. Returns (None, None) if unknown."""
    if state is None or (isinstance(state, float) and __import__('math').isnan(state)):
        return None, None
    key = str(state).upper().strip()
    coords = STATE_COORDINATES.get(key)
    if coords:
        return coords["lat"], coords["lon"]
    return None, None


def enrich_geo(df, project_name_col: str = "project_name"):
    """Add geo_latitude and geo_longitude columns to a DataFrame by extracting
    state from project_name and looking up centroid coordinates.

    Modifies df in-place and returns it.
    """
    states = df[project_name_col].apply(extract_state)
    lats, lons = [], []
    for s in states:
        lat, lon = state_to_geo(s)
        lats.append(lat)
        lons.append(lon)
    df["geo_latitude"] = lats
    df["geo_longitude"] = lons
    return df
