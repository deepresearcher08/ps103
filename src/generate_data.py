"""Synthetic dataset generator for the MoSPI Project Monitoring Platform (SIH26103).

Implements the exact schema, probabilistic distributions, target calculation logic,
and validation rules defined in SPEC2.md. The generation parameters are read from the
machine-readable ``dataset_spec.json`` configuration file.

Usage:
    python -m src.generate_data [--config dataset_spec.json] [--output data/MoSPI_projects_synthetic.csv]
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "dataset_spec.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "MoSPI_projects_synthetic.csv"


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _date_to_ordinal(d: date) -> int:
    return d.toordinal()


def _ordinal_to_date(ordinal: int) -> date:
    return date.fromordinal(int(ordinal))


class MoSPISyntheticGenerator:
    """Generates the synthetic infrastructure project dataset from a spec dict."""

    def __init__(self, spec: dict):
        self.spec = spec
        meta = spec["dataset_metadata"]
        self.seed = int(meta["random_seed"])
        self.n = int(meta["total_records"])
        self.rng = np.random.default_rng(self.seed)
        self.fields = spec["fields"]

        # Sector -> possible executing agencies (from SPEC2 section 1 linkage).
        self._default_agency_link = {
            "Highways": ["NHAI", "PWD"],
            "Railways": ["Railways"],
            "Power & Energy": ["NTPC", "PWD"],
            "Urban Metro": ["DMRC", "PWD"],
            "Water Resources": ["CWD", "PWD"],
            "Ports & Shipping": ["PWD"],
        }

    # ------------------------------------------------------------------ helpers
    def _categorical(self, field_cfg: dict, size: int | None = None) -> np.ndarray:
        size = size or self.n
        options = field_cfg["options"]
        weights_key = "weights"
        if field_cfg.get("weighted_by"):
            weights_key = "weights"
        weights = field_cfg.get(weights_key)
        if weights is None:
            weights = [1.0 / len(options)] * len(options)
        return self.rng.choice(options, size=size, p=weights)

    @staticmethod
    def _clip(val: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return np.clip(val, lo, hi)

    # ------------------------------------------------------------------ fields
    def _project_ids(self) -> list[str]:
        pattern = self.fields["project_id"]["pattern"]
        ids = []
        for i in range(1, self.n + 1):
            ids.append(pattern.format(id=i))
        # Emulate "unique sequential/random" — keep sequential for traceability.
        return ids

    def _sectors(self) -> np.ndarray:
        return self._categorical(self.fields["sector"])

    def _state_regions(self) -> np.ndarray:
        return self._categorical(self.fields["state_region"])

    def _executing_agencies(self, sectors: np.ndarray) -> np.ndarray:
        slot_domains = {s: self._default_agency_link.get(s, ["PWD"]) for s in self.fields["sector"]["options"]}
        agencies = np.empty(self.n, dtype=object)
        for s in self.fields["sector"]["options"]:
            mask = sectors == s
            if mask.any():
                agencies[mask] = self.rng.choice(slot_domains[s], size=int(mask.sum()))
        return agencies

    def _start_dates(self) -> list[date]:
        cfg = self.fields["start_date"]
        lo = _date_to_ordinal(date.fromisoformat(cfg["min"]))
        hi = _date_to_ordinal(date.fromisoformat(cfg["max"]))
        ordinals = self.rng.integers(lo, hi + 1, size=self.n)
        return [_ordinal_to_date(o) for o in ordinals]

    def _completion_dates_from(self, start_dates: list[date]) -> list[date]:
        cfg = self.fields["original_completion_date"]
        mu = cfg["duration_mean_months"]
        sigma = cfg["duration_std_months"]
        dmin = cfg["duration_min_months"]
        dmax = cfg["duration_max_months"]
        durations_months = self.rng.normal(mu, sigma, size=self.n)
        durations_months = self._clip(durations_months, dmin, dmax).astype(int)
        return [sd + timedelta(days=int(m * 30.44)) for sd, m in zip(start_dates, durations_months)]

    def _original_budget(self) -> np.ndarray:
        cfg = self.fields["original_budget_cr"]
        vals = self.rng.lognormal(cfg["mean"], cfg["sigma"], size=self.n)
        return self._clip(vals, cfg["min"], cfg["max"])

    def _revised_budget(self, original: np.ndarray,
                        delay_months: np.ndarray | None = None) -> np.ndarray:
        """Cost side calibrated to real MoSPI Flash Reports.

        Real-world: ~25% of projects carry a cost overrun vs original cost; overall
        COR ~20%; overrun magnitudes are right-skewed with a long tail (some >100%);
        a small share are descoped *below* original. This replaces the earlier
        unrealistic "80% get a 1.05-1.85x multiplier" assumption.
        """
        cfg = self.fields["revised_budget_cr"]
        n = self.n
        revised = original.copy()

        # ~25% get a genuine cost overrun, drawn from an exponential/bounded tail.
        over = self.rng.random(n) < cfg["prob_overrun"]
        p_over = cfg["overrun_cor_pct"]
        cor_pct = self.rng.exponential(scale=p_over["mean_cor_pct"], size=n)
        cor_pct = self._clip(cor_pct, p_over["min_cor_pct"], p_over["max_cor_pct"])
        revised = np.where(over, original * (1.0 + cor_pct / 100.0), revised)

        # ~2% descoped below original (legitimate re-costing, as seen in PAIMANA).
        desc = cfg["prob_descope"]
        desc_pct = self.rng.uniform(cfg["descope_pct"]["min"], cfg["descope_pct"]["max"], size=n)
        revised = np.where(self.rng.random(n) < desc, original * (1.0 + desc_pct / 100.0), revised)

        if delay_months is not None:
            # Validation: delay == 0  ->  revised == original.
            revised = np.where(delay_months <= 0, original, revised)
            # Real-world linkage (SPEC2): a project that is BOTH highly delayed
            # (>12mo) AND carrying a cost overrun has an overrun of at least +15%.
            # Real MoSPI shows most delayed projects have NO cost overrun, so this
            # only raises the floor where an overrun already exists, keeping the
            # ~25% overrun rate anchored to reality.
            high = delay_months > 12
            had_overrun = (revised > original) & high
            if had_overrun.any():
                cor = self.rng.exponential(scale=p_over["mean_cor_pct"], size=int(had_overrun.sum()))
                cor = self._clip(cor, 15.0, p_over["max_cor_pct"])
                revised[had_overrun] = original[had_overrun] * (1.0 + cor / 100.0)
        return revised

    def _expenditure(self, revised: np.ndarray, physical: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (expenditure_todate_cr, financial_progress_pct) from capped beta fraction."""
        cfg = self.fields["expenditure_todate_cr"]
        frac = self.rng.beta(cfg["alpha"], cfg["beta"], size=self.n)
        max_frac = np.clip(physical / 100.0 + 0.25, 0.0, 1.0)
        frac = np.minimum(frac, max_frac)
        expenditure = frac * revised
        financial_pct = np.minimum(frac * 100.0, physical + 25.0)
        return expenditure, financial_pct

    def _physical_progress(self) -> np.ndarray:
        cfg = self.fields["physical_progress_pct"]
        return self.rng.uniform(cfg["min"], cfg["max"], size=self.n)

    def _land_acquisition(self) -> np.ndarray:
        cfg = self.fields["land_acquisition_pct"]
        vals = self.rng.beta(cfg["alpha"], cfg["beta"], size=self.n) * cfg["scale"]
        return self._clip(vals, 0.0, 100.0)

    def _env_status(self) -> np.ndarray:
        return self._categorical(self.fields["env_clearance_status"])

    def _litigation(self) -> np.ndarray:
        cfg = self.fields["litigation_active"]
        return self.rng.random(self.n) < cfg["probability_true"]

    def _contractor_tier(self) -> np.ndarray:
        return self._categorical(self.fields["contractor_tier"])

    def _geo(self, cfg) -> np.ndarray:
        return self.rng.uniform(cfg["min"], cfg["max"], size=self.n)

    # ------------------------------------------------------------- delay target
    def _delay_months(self, land: np.ndarray, env: np.ndarray, litigation: np.ndarray,
                      contractor: np.ndarray, region: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Realistic delay target.

        Keeps the documented MoSPI penalty drivers (land, clearance, litigation,
        contractor, North-East) but folds in:
          - strong additive base noise (unobserved project risk),
          - interaction effects (e.g. litigation + pending clearance amplify each other),
          - a per-project multiplier on the deterministic part.
        So the target is genuinely hard to predict — no model can read it off a
        deterministic sum of the features (matches real MoSPI: R^2 ~ 0.11 for delay).
        """
        w = self.spec["delay_calculation_weights"]
        penalties = w["penalties_in_months"]
        base_noise_std = w["base_noise_std"]
        unexplained_mo = w["unexplained_risk_month_abs"]
        interact_scale = w["interaction_penalty_scale"]

        p = penalties
        risk = np.zeros(self.n)
        risk += np.where(land < 50, p["land_acquisition_below_50pct"], 0)
        risk += np.where((land >= 50) & (land < 80), p["land_acquisition_below_80pct"], 0)
        risk += np.where(env == "Pending", p["env_clearance_pending"], 0)
        risk += np.where(env == "In-Review", p["env_clearance_in_review"], 0)
        risk += np.where(litigation, p["active_litigation_true"], 0)
        risk += np.where(contractor == "Tier 3", p["contractor_tier_3"], 0)
        risk += np.where(contractor == "Tier 2", p["contractor_tier_2"], 0)
        risk += np.where(region == "North-East", p["region_northeast"], 0)

        # Interaction: two or more major blockers amplify each other.
        major_blockers = (
            (land < 60).astype(float) + (env == "Pending").astype(float)
            + litigation.astype(float) + (contractor == "Tier 3").astype(float)
        )
        interaction = (major_blockers >= 2) * interact_scale * risk

        # Per-project scale on the deterministic part (heterogeneity).
        project_scale = self.rng.lognormal(0.0, 0.5, size=self.n)

        noise = self.rng.normal(0.0, base_noise_std, size=self.n)
        unexplained = self.rng.exponential(scale=unexplained_mo / 2.0, size=self.n)

        delay = risk * project_scale + interaction + noise + unexplained
        return self._clip(delay, 0, 60).astype(int)

    def _risk_level(self, delay: np.ndarray, cost_overrun_pct: np.ndarray) -> np.ndarray:
        thresholds = self.spec["risk_thresholds"]
        # Ordered evaluation from highest to lowest; returns first matching tier.
        order = ["Critical", "High Risk", "Medium Risk", "Low Risk"]
        out = np.full(self.n, "Low Risk", dtype=object)
        assigned = np.zeros(self.n, dtype=bool)
        for tier in order:
            t = thresholds[tier]
            mask = (~assigned) & (delay >= t["min_delay_months"]) & (cost_overrun_pct >= t["min_cost_overrun_pct"])
            out[mask] = tier
            assigned |= mask
        return out

    # -------------------------------------------------------------- generation
    def build(self) -> pd.DataFrame:
        df = pd.DataFrame()
        df["project_id"] = self._project_ids()
        sectors = self._sectors()
        df["sector"] = sectors
        region = self._state_regions()
        df["state_region"] = region
        df["executing_agency"] = self._executing_agencies(sectors)

        start = self._start_dates()
        df["start_date"] = start
        df["original_completion_date"] = self._completion_dates_from(start)

        df["physical_progress_pct"] = self._physical_progress()
        land = self._land_acquisition()
        # Land Constraint: physical>60 => land>75. Enforce before delay uses it.
        over_60 = df["physical_progress_pct"] > 60
        low_land = over_60 & (land <= 75)
        if low_land.any():
            land[low_land] = self.rng.uniform(75, 100, size=int(low_land.sum()))
        df["land_acquisition_pct"] = land

        env = self._env_status()
        df["env_clearance_status"] = env
        litigation = self._litigation()
        df["litigation_active"] = litigation
        contractor = self._contractor_tier()
        df["contractor_tier"] = contractor

        df["original_budget_cr"] = self._original_budget()
        original = df["original_budget_cr"].to_numpy()

        # ---- targets -------------------------------------------------------
        delay = self._delay_months(land, env, litigation, contractor, region, original)
        df["delay_months"] = delay

        df["revised_budget_cr"] = self._revised_budget(original, delay)
        revised = df["revised_budget_cr"].to_numpy()

        physical = df["physical_progress_pct"].to_numpy()
        expenditure, financial_pct = self._expenditure(revised, physical)
        df["expenditure_todate_cr"] = expenditure
        df["financial_progress_pct"] = financial_pct
        # Hard-clip to enforce physical-financial gap constraint (fixes float drift).
        df["financial_progress_pct"] = np.minimum(
            df["financial_progress_pct"].to_numpy(),
            df["physical_progress_pct"].to_numpy() + 25.0,
        )

        cost_overrun_pct = (revised - original) / original * 100
        df["cost_overrun_pct"] = cost_overrun_pct
        df["risk_level"] = self._risk_level(delay, cost_overrun_pct)

        df["geo_latitude"] = self._geo(self.fields["geo_latitude"])
        df["geo_longitude"] = self._geo(self.fields["geo_longitude"])

        # revised_completion_date = original_completion_date + delay_months.
        # When delay == 0 the revised date equals the original (cost-linkage rule).
        df["revised_completion_date"] = [
            oc + timedelta(days=int(m * 30.44))
            for oc, m in zip(df["original_completion_date"], delay)
        ]
        return df



def validate(df: pd.DataFrame, spec: dict, eps: float = 1e-6) -> dict:
    """Run the SPEC2 section-3 validation constraints; return pass/fail summary."""
    rules = spec["validation_rules"]
    checks = {}

    checks["chronology_start_lt_orig"] = bool((df["start_date"] < df["original_completion_date"]).all())
    # When delay == 0 the cost-linkage rule sets revised == original;
    # chronology "original < revised" only applies where delay > 0.
    has_delay = df["delay_months"] > 0
    checks["chronology_orig_lt_revised"] = bool(
        (df.loc[has_delay, "original_completion_date"] < df.loc[has_delay, "revised_completion_date"]).all()
    ) if has_delay.any() else True

    zero_delay = df["delay_months"] == 0
    checks["zero_delay_cost_equal"] = bool(
        np.isclose(df.loc[zero_delay, "revised_budget_cr"].to_numpy(),
                   df.loc[zero_delay, "original_budget_cr"].to_numpy()).all()
    )

    # Real-world linkage: among projects that are BOTH highly delayed (>12mo) AND
    # carrying a cost overrun, the overrun is at least 15%. (Real MoSPI data shows
    # most delayed projects have no cost overrun, so this is scoped to overrun ones.)
    high_delay_overrun = (df["delay_months"] > 12) & (df["revised_budget_cr"] > df["original_budget_cr"])
    if high_delay_overrun.any():
        checks["high_delay_cost_gte_15pct"] = bool(
            (df.loc[high_delay_overrun, "revised_budget_cr"]
             >= 1.15 * df.loc[high_delay_overrun, "original_budget_cr"]).all()
        )
    else:
        checks["high_delay_cost_gte_15pct"] = True

    checks["physical_financial_gap"] = bool(
        (df["financial_progress_pct"] - df["physical_progress_pct"] <= 25.0 + eps).all()
    )

    mask = df["physical_progress_pct"] > 60
    checks["land_constraint"] = bool((df.loc[mask, "land_acquisition_pct"] > 75).all())

    return checks


def write_summary(df: pd.DataFrame, checks: dict, output: Path) -> None:
    print(f"Generated: {len(df)} rows")
    print(f"Shape: {df.shape[0]} x {df.shape[1]}")
    print(f"Columns: {list(df.columns)}")
    print("\nValidation checks:")
    for k, v in checks.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"\nOutput written to: {output}")
    print("\nrisk_level distribution:")
    print(df["risk_level"].value_counts().to_string())
    print("\ndelay_months describe:")
    print(df["delay_months"].describe().to_string())
    print("\nNulls per column:")
    print(df.isnull().sum().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MoSPI synthetic project dataset")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows", type=int, default=None, help="Override total_records")
    args = parser.parse_args()

    spec = load_config(args.config)
    if args.rows is not None:
        spec["dataset_metadata"]["total_records"] = args.rows

    gen = MoSPISyntheticGenerator(spec)
    df = gen.build()
    checks = validate(df, spec)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    write_summary(df, checks, args.output)


if __name__ == "__main__":
    main()
