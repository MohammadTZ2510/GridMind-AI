"""
config.py
=========
Central configuration for the whole project.

Every module imports its paths, constants and settings from here, so important
values (random seed, column names, algorithm limits, ...) are defined ONCE.

All data used by this project is SIMULATED DATA. It is not real utility data.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

# =============================================================================
# 1. PATHS (always relative to the project root - nothing is hard-coded to one PC)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
SCHEDULES_DIR = OUTPUT_DIR / "schedules"
METRICS_DIR = OUTPUT_DIR / "metrics"
CHARTS_DIR = OUTPUT_DIR / "charts"

# Input files (created earlier by dataset_generator.py - never modified here)
FEEDERS_CSV = DATA_DIR / "feeders.csv"
DEMAND_CSV = DATA_DIR / "hourly_demand.csv"
SUPPLY_CSV = DATA_DIR / "hourly_supply.csv"
WEATHER_CSV = DATA_DIR / "weather_time.csv"

# Output files
PREDICTIONS_CSV = PREDICTIONS_DIR / "demand_predictions.csv"
CSP_SCHEDULES_CSV = SCHEDULES_DIR / "csp_schedules.csv"
HC_SCHEDULES_CSV = SCHEDULES_DIR / "hill_climbing_schedules.csv"
INFEASIBLE_CSV = SCHEDULES_DIR / "infeasible_scenarios.csv"
REGRESSION_METRICS_CSV = METRICS_DIR / "regression_metrics.csv"
SCHEDULING_METRICS_CSV = METRICS_DIR / "scheduling_metrics.csv"
SCHEDULING_SUMMARY_CSV = METRICS_DIR / "scheduling_summary.csv"
EXPERIMENT_SUMMARY_TXT = METRICS_DIR / "experiment_summary.txt"

CHART_ACTUAL_VS_PREDICTED = CHARTS_DIR / "actual_vs_predicted_demand.png"
CHART_RUNTIME = CHARTS_DIR / "csp_vs_hill_climbing_runtime.png"
CHART_SHORTAGE = CHARTS_DIR / "csp_vs_hill_climbing_shortage.png"
CHART_OUTAGE_BY_AREA = CHARTS_DIR / "outage_hours_by_area.png"

# =============================================================================
# 2. FIXED PROJECT SIZE (must match dataset_generator.py)
# =============================================================================
NUM_SCENARIOS = 50
NUM_AREAS = 5
FEEDERS_PER_AREA = 2
NUM_FEEDERS = NUM_AREAS * FEEDERS_PER_AREA        # 10
HOURS_PER_SCENARIO = 24

SCENARIO_IDS = [f"S{i:03d}" for i in range(1, NUM_SCENARIOS + 1)]

EXPECTED_ROWS = {
    "feeders": NUM_FEEDERS,                                        # 10
    "demand": NUM_SCENARIOS * HOURS_PER_SCENARIO * NUM_FEEDERS,    # 12,000
    "supply": NUM_SCENARIOS * HOURS_PER_SCENARIO,                  # 1,200
    "weather": NUM_SCENARIOS * HOURS_PER_SCENARIO,                 # 1,200
}

HOSPITAL_FEEDER_ID = "F001"

# Binary decision encoding used everywhere: 1 = ON (served), 0 = OFF (outage)
ON = 1
OFF = 0

# =============================================================================
# 3. REPRODUCIBILITY
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# 4. COLUMN NAMES (the expected CSV headers)
# =============================================================================
REQUIRED_COLUMNS = {
    "feeders": [
        "feeder_id", "area", "feeder_name", "feeder_capacity_mw", "priority",
        "priority_weight", "is_critical", "critical_type", "max_total_outage_hours",
        "max_consecutive_outage_hours", "protected_hours", "base_demand_mw",
        "demand_variation_pct",
    ],
    "demand": [
        "scenario_id", "hour", "time_label", "feeder_id", "area", "day_type", "season",
        "temperature_c", "humidity_pct", "previous_hour_demand_mw", "actual_demand_mw",
        "demand_category", "feeder_capacity_mw",
    ],
    "supply": [
        "scenario_id", "hour", "time_label", "available_supply_mw", "reserve_mw",
        "usable_supply_mw", "total_demand_mw", "shortage_mw", "supply_status",
    ],
    "weather": [
        "scenario_id", "hour", "time_label", "day_type", "day_of_week", "season",
        "temperature_c", "humidity_pct", "is_weekend", "is_peak_hour",
        "is_morning_peak", "is_evening_peak",
    ],
}

# If a CSV ever uses a different header, list the mapping here (old name -> expected name).
# The data loader applies it and reports every mapping it used. Empty = headers match exactly.
COLUMN_ALIASES: dict[str, dict[str, str]] = {
    "feeders": {}, "demand": {}, "supply": {}, "weather": {},
}

TARGET_COLUMN = "actual_demand_mw"

# =============================================================================
# 5. REGRESSION (DemandPredictor) SETTINGS
# =============================================================================
# Only information that is known when the forecast is made is used as input.
CATEGORICAL_FEATURES = ["day_type", "season", "feeder_id"]
NUMERIC_FEATURES = [
    "hour", "hour_sin", "hour_cos", "previous_hour_demand_mw",
    "temperature_c", "humidity_pct", "is_peak_hour",
]
MODEL_N_ESTIMATORS = 200
MODEL_MIN_SAMPLES_LEAF = 3
MODEL_MAX_FEATURES = 0.6
N_CV_FOLDS = 5            # scenario-grouped cross-validation (GroupKFold)

# =============================================================================
# 6. SCHEDULING TOLERANCES
# =============================================================================
DEMAND_TOLERANCE = 1e-9          # float tolerance used inside the CSP search
SUPPLY_TOLERANCE = 1e-6          # float tolerance for "served <= usable supply"

# =============================================================================
# 7. CSP BACKTRACKING SETTINGS
# =============================================================================
CSP_MAX_NODES = 300_000          # search limit (reported if reached)
CSP_MAX_SECONDS = 30.0           # wall-clock limit per scenario (reported if reached)

# =============================================================================
# 8. HILL CLIMBING SETTINGS
# =============================================================================
HC_NUM_RESTARTS = 5              # restart 0 = the CSP schedule itself; 1..4 = perturbed starts
HC_MAX_ITERATIONS = 500          # maximum accepted improvements per restart
HC_PERTURBATION_MOVES = 8        # random valid moves used to create a perturbed start
HC_IMPROVEMENT_EPSILON = 1e-12   # a neighbour must beat the current score by more than this

# Objective-function weights (the score is MINIMISED). They sum to 1.
HC_WEIGHT_UNMET = 0.5            # share of the score from unserved demand
HC_WEIGHT_FAIRNESS = 0.3         # share from outage-hour imbalance between areas
HC_WEIGHT_PRIORITY = 0.2         # share from priority-weighted unserved demand


# =============================================================================
# 9. HELPER FUNCTIONS
# =============================================================================
def ensure_output_dirs() -> None:
    """Create all output folders if they do not exist yet."""
    for folder in (PREDICTIONS_DIR, SCHEDULES_DIR, METRICS_DIR, CHARTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python's and NumPy's global generators (individual modules also use seeded generators)."""
    random.seed(seed)
    np.random.seed(seed)
