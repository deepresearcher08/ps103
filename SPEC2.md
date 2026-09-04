This technical specification defines the exact schema, probabilistic distribution, constraints, and target calculation logic required to build a production-grade synthetic dataset for the MoSPI Project Monitoring Platform (SIH26103).

### 1. Complete Feature Schema & Distributions

| Field Name | Data Type | Permissible Values / Range | Probability / Distribution | Business Constraint |
| --- | --- | --- | --- | --- |
| `project_id` | String | `PRJ-IN-2026-[0-9]{4}` | Unique Sequential / Random | Primary Key |
| `sector` | Categorical | Highway, Railway, Power, Metro, Water, Ports | Uniform (16.6% each) | Determines baseline risk weight |
| `state_region` | Categorical | North, South, East, West, North-East | Uniform (20% each) | North-East adds inherent delay factor |
| `executing_agency` | Categorical | NHAI, Railways, NTPC, DMRC, PWD, CWD | Linked to `sector` | Tier-based performance tracking |
| `start_date` | Date | `2018-01-01` to `2025-12-31` | Uniform Random | Must precede completion dates |
| `original_completion_date` | Date | `start_date` + (12 to 72 months) | Normal Distribution ($\mu=36, \sigma=12$) | Planned baseline timeline |
| `original_budget_cr` | Float | ₹50.00 Cr – ₹15,000.00 Cr | Log-Normal ($\mu=5.5, \sigma=1.2$) | Original allocation |
| `revised_budget_cr` | Float | $\ge$ `original_budget_cr` | 80% multiplier ($1.05\times - 1.85\times$) | Cost overruns |
| `expenditure_todate_cr` | Float | $0.00$ to `revised_budget_cr` | Beta Distribution ($\alpha=2, \beta=2$) | $\le$ `revised_budget_cr` |
| `physical_progress_pct` | Float | $0.00\% - 100.00\%$ | Uniform Random | Tracks real ground progress |
| `financial_progress_pct` | Float | $0.00\% - 100.00\%$ | `expenditure_todate` / `revised_budget` | Correlates with physical progress |
| `land_acquisition_pct` | Float | $0.00\% - 100.00\%$ | Beta ($\alpha=5, \beta=2$) skewing high | Key blocker if $< 80\%$ |
| `env_clearance_status` | Categorical | Cleared (75%), In-Review (15%), Pending (10%) | Categorical Weights | Blocks construction if Pending |
| `litigation_active` | Boolean | True (15%), False (85%) | Bernoulli ($p=0.15$) | Adds heavy delay penalty |
| `contractor_tier` | Categorical | Tier 1 (50%), Tier 2 (35%), Tier 3 (15%) | Categorical Weights | Performance capability index |
| `geo_latitude` | Float | $8.4^\circ\text{N} - 35.5^\circ\text{N}$ | Bounded Random | India geographical bounds |
| `geo_longitude` | Float | $68.7^\circ\text{E} - 97.2^\circ\text{E}$ | Bounded Random | India geographical bounds |
| `delay_months` | Integer | $0 - 60\text{ months}$ | Derived Target Variable | Deterministic + Gaussian Noise |
| `risk_level` | Categorical | Low Risk, Medium Risk, High Risk, Critical | Derived Classification Label | Threshold based on delay & cost |

---

### 2. Machine-Readable Configuration File (`dataset_spec.json`)

You can parse this JSON specification directly into your Python dataset generator pipeline.

```json
{
  "dataset_metadata": {
    "name": "MoSPI_Infrastructure_Projects_Synthetic_v1",
    "total_records": 10000,
    "random_seed": 42,
    "target_variables": ["delay_months", "risk_level"]
  },
  "fields": {
    "project_id": {"type": "string", "pattern": "PRJ-IN-2026-{id:04d}"},
    "sector": {
      "type": "categorical",
      "options": ["Highways", "Railways", "Power & Energy", "Urban Metro", "Water Resources", "Ports & Shipping"],
      "weights": [0.30, 0.25, 0.20, 0.10, 0.10, 0.05]
    },
    "original_budget_cr": {"type": "float", "distribution": "lognormal", "mean": 5.8, "sigma": 1.0, "min": 50.0, "max": 20000.0},
    "land_acquisition_pct": {"type": "float", "distribution": "beta", "alpha": 4.0, "beta": 1.5, "scale": 100.0},
    "env_clearance_status": {
      "type": "categorical",
      "options": ["Cleared", "In-Review", "Pending"],
      "weights": [0.75, 0.15, 0.10]
    },
    "active_litigation": {"type": "boolean", "probability_true": 0.15},
    "contractor_rating": {
      "type": "categorical",
      "options": ["Tier-1 High Reliability", "Tier-2 Average", "Tier-3 High Risk"],
      "weights": [0.50, 0.35, 0.15]
    }
  },
  "delay_calculation_weights": {
    "base_noise_std": 2.0,
    "penalties_in_months": {
      "land_acquisition_below_50pct": 18,
      "land_acquisition_below_80pct": 8,
      "env_clearance_pending": 12,
      "env_clearance_in_review": 4,
      "active_litigation_true": 24,
      "contractor_tier_3": 10,
      "contractor_tier_2": 3,
      "region_northeast": 6
    }
  },
  "risk_thresholds": {
    "Critical": {"min_delay_months": 24, "min_cost_overrun_pct": 40.0},
    "High Risk": {"min_delay_months": 12, "min_cost_overrun_pct": 20.0},
    "Medium Risk": {"min_delay_months": 4, "min_cost_overrun_pct": 5.0},
    "Low Risk": {"min_delay_months": 0, "min_cost_overrun_pct": 0.0}
  }
}

```

---

### 3. Business Rules & Validation Constraints

To keep the dataset realistic and prevent logical anomalies, enforce these validation rules during generation:

* **Chronology Rule:** `start_date` < `original_completion_date` < `revised_completion_date`.
* **Cost Overrun Linkage:** `revised_budget_cr` must be equal to `original_budget_cr` if `delay_months` == 0. If `delay_months` > 12, `revised_budget_cr` must be at least 15% higher than `original_budget_cr`.
* **Physical vs. Financial Dependency:** `financial_progress_pct` cannot exceed `physical_progress_pct` by more than 25 percentage points (prevents impossible front-loading of funds).
* **Land Constraint:** If `physical_progress_pct` > 60%, `land_acquisition_pct` must be > 75%.