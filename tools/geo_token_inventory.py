"""Inventory candidate location tokens in MoSPI project names.

Prints the most frequent uppercase multi-word sequences (with enough length to
be real place names) so we can build a district/city centroid table targeted at
what actually appears, instead of hand-writing 700+ Indian districts.
"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

FILES = {"COST": "data/real_mospi_projects.csv",
         "TIME": "data/real_mospi_time.csv",
         "SNAP": "data/mospi_may2025_projects.csv"}

STOP = set("""
A AN AND OR OF THE IN AT FOR ON FROM WITH TO BY AS LT IPR
C O N E P R I S D T M X L K U V G J H F B S
NEW INDIA INDIAN NATIONAL INTERNATIONAL PROJECT PROJECTS SYSTEM SYSTEMS
WORKS WORK DEVELOPMENT CONSTRUCTION EXTENSION EXPANSION UPGRADATION UPGRADE
MASTER PLAN OPERATING OPERATION OPERATIONS CAPACITY POWER PLANT PLANTS
ENHANCEMENT IMPROVEMENT MODERNISATION REHABILITATION RESTORATION REVITALISATION
COMPLEX COMPLEXES CAMPUS CAMPUSES PHASE PHASES PHASE-I PHASE-1 STAGE STAGE-I
FACILITY FACILITIES BUILDING BUILDINGS TERMINAL TERMINALS AIRPORT AERODROME
PORT PATAN PORT AIRPORT TOWNSHIP HOUSING HEADQUARTERS OFFICE OFFICES BLOCK BLOCK
SECTION SECTIONS UNIT UNITS TYPE NOS NUMBER NO UNDER COMPLETION BEDDED BED
PROGRAMME PROGRAM CIVIL WORK WORKS MAINTENANCE OPERATION MAINTENANCE
GENERATION TRANSMISSION DISTRIBUTION SUPPLY TOWN CITY AREA REGION STATE
DISTRICT ROAD ROADS HIGHWAY ODISHA BIHAR etc etc
""".split())


def norm(tok):
    return re.sub(r"[^A-Z0-9]", "", str(tok).upper())


def candidates(name, max_words=5):
    """All contiguous uppercase runs of 2..max_words words (joined), deduped."""
    if not isinstance(name, str):
        return []
    toks = [t for t in re.split(r"[^A-Za-z0-9]+", name) if t and t.isupper() and len(t) >= 3]
    out = []
    for n in range(2, min(max_words, len(toks)) + 1):
        for i in range(0, len(toks) - n + 1):
            seq = toks[i:i + n]
            if any(t in STOP or norm(t) in set(norm(s) for s in STOP) for t in seq):
                continue
            out.append(" ".join(seq))
    return out


def main():
    ctr = Counter()
    for label, f in FILES.items():
        df = pd.read_csv(f)
        for name in df["project_name"].dropna():
            for cand in candidates(name):
                ctr[(label, cand)] += 0  # placeholder to keep tuple distinct
                ctr[cand] += 1
    # top candidates by overall frequency
    for cand, n in ctr.most_common(700):
        if len(cand) < 4:
            continue
        print(f"{n:5d}  {cand}")


if __name__ == "__main__":
    main()