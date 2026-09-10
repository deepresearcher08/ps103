"""Backfill geo_latitude/longitude from district/town-level project-name hits.

District match (when the project name carries a city/district/HQ token) replaces
the STATE-CENTROID value; otherwise the state centroid is retained (no data
fabricated). Also prints coverage + token distribution.

Run:
    python tools/geocode_backfill.py            # dry run (no writes)
    python tools/geocode_backfill.py --write   # backup + overwrite CSVs
"""
from __future__ import annotations

import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.district_geo import geocode_project

FILES = ["data/real_mospi_projects.csv",
         "data/real_mospi_time.csv",
         "data/mospi_may2025_projects.csv"]


def main():
    write = "--write" in sys.argv
    for f in FILES:
        df = pd.read_csv(f)
        before = len(df)
        before_distinct = df[["geo_latitude", "geo_longitude"]].drop_duplicates().shape[0]
        hits, tokens = [], Counter()
        lat_v, lon_v = [], []
        for i, name in enumerate(df["project_name"]):
            lat, lon, tok = geocode_project(name)
            hits.append(lat is not None)
            if tok:
                tokens[tok] += 1
            if lat is not None:
                lat_v.append(lat)
                lon_v.append(lon)
            else:
                lat_v.append(None)
                lon_v.append(None)
        df2 = df.copy()
        df2["geo_latitude"] = df2["geo_latitude"].astype("float64")
        df2["geo_longitude"] = df2["geo_longitude"].astype("float64")
        got = [i for i, h in enumerate(hits) if h]
        df2.loc[got, "geo_latitude"] = [lat_v[i] for i in got]
        df2.loc[got, "geo_longitude"] = [lon_v[i] for i in got]
        after_distinct = df2[["geo_latitude", "geo_longitude"]].drop_duplicates().shape[0]
        print("=" * 70)
        print(f"{f}  rows={before}")
        print(f"  district/town hits : {len(got)} / {before}  ({100 * len(got) / before:.1f}%)")
        print(f"  distinct coords    : {before_distinct} -> {after_distinct}")
        print("  top matched tokens :", dict(tokens.most_common(12)))
        if write:
            bak = Path(f).with_suffix(Path(f).suffix + ".statecentroid.bak")
            shutil.copy2(f, bak)
            df2.to_csv(f, index=False)
            print(f"  WROTE {f} (backup -> {bak})")


if __name__ == "__main__":
    main()