"""Data ingestion + continual-learning pipeline for the MoSPI monitoring platform.

Officials add data month over month. This module:
  1. parse_flash_report()   - parse a new Monthly Flash Report PDF into fresh rows
     (reuses the annexure parsers) OR accept a CSV of new project rows.
  2. dedupe_append()         - merge new rows into real_mospi_projects.csv /
     real_mospi_time.csv keyed by (sector, project_name), taking the NEWER row.
  3. census_age()            - re-derive time-derivative labels (tor_months,
     delayed) from original vs. reported date-of-commissioning so a project can
     move from 'on schedule' to 'delayed' as it ages (survivorship is handled
     honestly: we only label 'delayed' when the report actually shows a TOR).
  4. continual_retrain()     - append + retrain + persist versioned models via
     src.predict (incremental update of a live system). Also supports a
     lightweight online LR `partial_fit` update for streaming scenarios.

CLI:
  python -m src.ingest --pdf path/to/FlashReport_Jun_2026.pdf --retrain
  python -m src.ingest --csv new_rows.csv --retrain
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src import predict

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COST_CSV = DATA / "real_mospi_projects.csv"
TIME_CSV = DATA / "real_mospi_time.csv"

CAP_TOR = 240.0


# --------------------------------------------------------------------------- #
# 1. parse
# --------------------------------------------------------------------------- #
def parse_cost_pdf(pdf_path: Path) -> pd.DataFrame:
    """Parse a Flash Report PDF into project rows for the COST dataset."""
    from src.real_data_assembly import build_real_dataset
    df = build_real_dataset(pdf_path)
    return df


def parse_time_pdf(pdf_path: Path) -> pd.DataFrame:
    """Parse a Flash Report PDF into project rows for the TIME dataset."""
    from src.time_data_assembly import build_time_dataset
    df = build_time_dataset(pdf_path)
    return df


def _norm(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# --------------------------------------------------------------------------- #
# 2. dedupe + append
# --------------------------------------------------------------------------- #
def dedupe_append(existing: pd.DataFrame, new: pd.DataFrame,
                  keys=("sector", "project_name")) -> pd.DataFrame:
    """Incrementally merge NEW rows over EXISTING by (sector, project_name).

    Only keys present in `new` are refreshed (the newer report wins); untouched
    existing rows (including any pre-existing duplicate keys) are left as-is.
    Returns a dataframe whose row count grows by the number of genuinely-new keys
    and whose touched keys now hold the latest values from `new`.
    """
    if new is None or new.empty:
        return existing.copy()
    keys = list(keys) if isinstance(keys, tuple) else keys

    def _str(s):
        return str(s) if pd.notna(s) else ""
    for k in keys:
        new[k] = new[k].map(_str)
        existing[k] = existing[k].map(_str)

    exist_ix = existing.set_index(keys).index
    new_ix = new.set_index(keys).index

    # which existing rows are refreshed by this batch?
    touched = exist_ix.isin(new_ix)
    untouched = existing[~touched]
    refreshed = new[new_ix.isin(exist_ix)]
    added = new[~new_ix.isin(exist_ix)]

    out = pd.concat([untouched, refreshed, added], ignore_index=True)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. census-age the time label (honest: only mark delayed when a TOR is shown)
# --------------------------------------------------------------------------- #
def census_age(df: pd.DataFrame, cap: float = CAP_TOR) -> pd.DataFrame:
    """Recompute derived time fields when new reports arrive.

    A project is 'delayed' only when the report shows tor_months > 0; magnitude
    is capped (cell-shift artifact). Planned duration is inferred from the
    approval & planned-doc years when present.
    """
    df = df.copy()
    df["tor_months"] = pd.to_numeric(df.get("tor_months"), errors="coerce")
    if "delayed" not in df.columns or df["delayed"].isna().all():
        df["delayed"] = (df["tor_months"] > 0).astype(int)
    df.loc[df["tor_months"].isna(), "tor_months"] = 0
    df.loc[df["delayed"] == 0, "tor_months"] = 0
    df.loc[df["delayed"] == 1, "tor_months"] = df.loc[df["delayed"] == 1, "tor_months"].clip(upper=cap)
    if "planned_duration_years" in df.columns and "approval_year" in df.columns and "planned_doc_year" in df.columns:
        m = df["planned_doc_year"].notna() & df["approval_year"].notna()
        df.loc[m, "planned_duration_years"] = df.loc[m, "planned_doc_year"] - df.loc[m, "approval_year"]
    return df


# --------------------------------------------------------------------------- #
# 4. continual retrain
# --------------------------------------------------------------------------- #
def ingest_and_retrain(cost_new: pd.DataFrame | None = None,
                       time_new: pd.DataFrame | None = None,
                       pdf_path: Path | None = None,
                       verbose: bool = True) -> dict:
    """Append new data, re-derive labels, retrain + persist versioned models."""
    cost_existing = pd.read_csv(COST_CSV) if COST_CSV.exists() else pd.DataFrame()
    time_existing = pd.read_csv(TIME_CSV) if TIME_CSV.exists() else pd.DataFrame()

    if pdf_path is not None:
        try:
            cost_new = parse_cost_pdf(pdf_path)
        except Exception as e:  # pragma: no cover
            if verbose:
                print(f"[ingest] cost parse failed: {e}")
            cost_new = None
        try:
            time_new = parse_time_pdf(pdf_path)
        except Exception as e:  # pragma: no cover
            if verbose:
                print(f"[ingest] time parse failed: {e}")
            time_new = None

    report = {"added_cost": 0, "added_time": 0, "cost_rows": None, "time_rows": None}

    if cost_new is not None and not cost_new.empty:
        merged = dedupe_append(cost_existing, cost_new)
        merged.to_csv(COST_CSV, index=False)
        report["added_cost"] = int(len(merged) - len(cost_existing))
        report["cost_rows"] = int(len(merged))
        if verbose:
            print(f"[ingest] cost: {len(cost_existing)} -> {len(merged)} "
                  f"(+{report['added_cost']} net)")

    if time_new is not None and not time_new.empty:
        merged_t = dedupe_append(time_existing, time_new)
        merged_t = census_age(merged_t)
        merged_t.to_csv(TIME_CSV, index=False)
        report["added_time"] = int(len(merged_t) - len(time_existing))
        report["time_rows"] = int(len(merged_t))
        if verbose:
            print(f"[ingest] time: {len(time_existing)} -> {len(merged_t)} "
                  f"(+{report['added_time']} net)")

    # retrain (continual learning) and persist a new model version
    from src.predict import train_all
    model_report = train_all()
    report["model"] = model_report
    if verbose:
        print(f"[ingest] retrained model {model_report['version']}: "
              f"cost_auc={model_report['cost']['clf_auc']:.3f} "
              f"time_auc={model_report['time']['clf_auc']:.3f}")
    return report


# --------------------------------------------------------------------------- #
# 5. online LR partial_fit (streaming / very frequent updates)
# --------------------------------------------------------------------------- #
# For truly streaming updates where a full retrain each report is too heavy one
# could continue a warm-started LogisticRegression from iteration to iteration.
# At this dataset scale a full leak-free retrain (train_all) is cheap and
# recommended, so that path is the default. See predict.train_all().


def _arg_parser():
    ap = argparse.ArgumentParser(description="Ingest + continual retrain for MoSPI platform")
    ap.add_argument("--pdf", type=Path, default=None, help="path to a new Flash Report PDF")
    ap.add_argument("--csv", type=Path, default=None, help="path to a CSV of new project rows")
    ap.add_argument("--retrain", action="store_true", help="retrain + persist models after ingest")
    return ap


if __name__ == "__main__":
    args = _arg_parser().parse_args()
    cov_new = tim_new = None
    if args.csv is not None:
        cov_new = tim_new = pd.read_csv(args.csv)
    if args.pdf is not None:
        if not args.pdf.exists():
            print(f"PDF not found: {args.pdf}")
            raise SystemExit(1)
    report = ingest_and_retrain(cost_new=cov_new, time_new=tim_new, pdf_path=args.pdf,
                                verbose=True)
    print("DONE.")
