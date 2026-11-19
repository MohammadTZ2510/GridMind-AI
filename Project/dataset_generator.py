#!/usr/bin/env python3
"""
dataset_generator.py
====================
AI-Based Electricity Demand Prediction and Fair Load-Shedding Scheduler
University Artificial Intelligence Laboratory Project

*** ALL DATA PRODUCED BY THIS SCRIPT IS SIMULATED DATA. ***
It is NOT real measurement data from K-Electric or any other utility, and it is
NOT connected to any real electricity grid. It is for educational use only.

What this script does
---------------------
Generates four CSV files plus a metadata summary in one run:

    data/feeders.csv                   (10 rows)    static feeder metadata
    data/hourly_demand.csv             (12,000 rows) Regression dataset
    data/hourly_supply.csv             (1,200 rows)  network-wide supply
    data/weather_time.csv              (1,200 rows)  weather / time features
    data/metadata/dataset_summary.txt                summary + validation report

It does NOT implement Regression, CSP or Hill Climbing, and it does NOT create
schedules or model predictions.

Random-seed strategy (fully reproducible)
-----------------------------------------
One master seed (42) is used. `set_random_seed()` seeds Python's `random` and
NumPy's legacy global generator, and then creates independent, named generators:

    * Python  random.Random(seed + 1)      -> scenario layout (season, day type, ...)
    * Python  random.Random(seed + 2)      -> supply-stress level assignment
    * NumPy   default_rng([seed, 1])       -> weather noise
    * NumPy   default_rng([seed, 2])       -> fixed per-feeder profile jitter
    * NumPy   default_rng([seed, 3])       -> demand noise
    * NumPy   default_rng([seed, 4])       -> supply generation

Every generator is consumed in a fixed order, so the CSV files are identical on
every run (only the timestamp inside dataset_summary.txt changes).

Demand model (documented formula)
---------------------------------
    demand = base_demand_mw
             x time_of_day_profile[category][hour]   (category-specific shape)
             x feeder_profile_jitter[hour]           (fixed small per-feeder difference)
             x seasonal_factor[season]
             x temperature_factor                    (cooling / heating response)
             x day_type_factor[category][day_type]
             x scenario_feeder_factor                (small daily level shift)
             x (1 + AR(1) noise)                     (autocorrelated, bounded noise)
    then clipped to [5% , 98%] of the feeder capacity and rounded to 3 decimals.

    temperature_factor = 1 + sensitivity[category] *
                         ( 0.010 * max(0, T_eff - 27) + 0.004 * max(0, 18 - T_eff) )
    T_eff = temperature_c + 0.05 * (humidity_pct - 50)       (humid air feels hotter)

Supply model (documented)
-------------------------
For every scenario-hour, a supply "regime" is drawn (Markov-style persistence so
neighbouring hours are related, and shortages are more likely in peak hours and
in high-stress scenarios):

    Regime 0 (comfortable): available = 106%-116% of total demand
    Regime 1 (tight)      : available =  90%-100% of total demand
    Regime 2 (severe)     : available =  75%- 90% of total demand

    reserve_mw      = available x U(2%, 5%)
    usable_supply   = available - reserve
    shortage_mw     = max(0, total_demand - usable_supply)      (3 decimals)
    supply_status   = Sufficient if shortage == 0
                      Tight      if 0 < shortage <= 10% of total demand
                      Shortage   if shortage > 10% of total demand

Binary model note for later phases: an ON feeder serves its full demand, an OFF
feeder serves zero. No partial feeder switching is modelled.

Requirements:  pip install pandas numpy
Run:           python dataset_generator.py
"""

from __future__ import annotations

import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# 1. FIXED PROJECT CONFIGURATION
# =============================================================================
PROJECT_NAME = "AI-Based Electricity Demand Prediction and Fair Load-Shedding Scheduler"
RANDOM_SEED = 42
NUM_SCENARIOS = 50
NUM_AREAS = 5
FEEDERS_PER_AREA = 2
NUM_FEEDERS = NUM_AREAS * FEEDERS_PER_AREA          # 10
HOURS_PER_SCENARIO = 24
POWER_UNIT = "MW"

SCENARIO_IDS = [f"S{i:03d}" for i in range(1, NUM_SCENARIOS + 1)]
HOURS = list(range(HOURS_PER_SCENARIO))

EXPECTED_FEEDER_ROWS = NUM_FEEDERS                                   # 10
EXPECTED_DEMAND_ROWS = NUM_SCENARIOS * HOURS_PER_SCENARIO * NUM_FEEDERS  # 12,000
EXPECTED_SUPPLY_ROWS = NUM_SCENARIOS * HOURS_PER_SCENARIO            # 1,200
EXPECTED_WEATHER_ROWS = NUM_SCENARIOS * HOURS_PER_SCENARIO           # 1,200

# Peak-hour definitions
MORNING_PEAK_HOURS = set(range(6, 10))     # 06:00 - 09:59 (hours 6,7,8,9)
EVENING_PEAK_HOURS = set(range(17, 22))    # 17:00 - 21:59 (hours 17..21)

# Scenario layout (counts are fixed; their order is shuffled with the seed)
SEASON_COUNTS = {"Summer": 20, "Winter": 15, "Monsoon": 15}         # sums to 50
DAY_TYPE_COUNTS = {"Weekday": 36, "Weekend": 14}                    # sums to 50
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
WEEKEND_NAMES = ["Saturday", "Sunday"]

SEASON_PARAMS = {
    # temp_range = hard clipping range, tmin/tmax = range of the daily min/max
    "Summer":  {"temp_range": (30.0, 42.0), "tmin": (30.0, 33.0), "tmax": (36.0, 42.0),
                "humidity_base": (48.0, 62.0)},
    "Winter":  {"temp_range": (15.0, 28.0), "tmin": (15.0, 19.0), "tmax": (23.0, 28.0),
                "humidity_base": (42.0, 58.0)},
    "Monsoon": {"temp_range": (25.0, 35.0), "tmin": (25.0, 28.0), "tmax": (30.0, 35.0),
                "humidity_base": (70.0, 82.0)},
}
HUMIDITY_MIN, HUMIDITY_MAX = 30.0, 90.0

# Demand model constants
SEASONAL_FACTOR = {"Summer": 1.05, "Monsoon": 1.00, "Winter": 0.95}
TEMP_SENSITIVITY = {"Residential": 1.0, "Commercial": 0.8, "Industrial": 0.4, "Hospital": 0.5}
DAY_TYPE_FACTOR = {
    "Residential": {"Weekday": 1.00, "Weekend": 1.06},
    "Commercial":  {"Weekday": 1.00, "Weekend": 0.75},
    "Industrial":  {"Weekday": 1.00, "Weekend": 0.82},
    "Hospital":    {"Weekday": 1.00, "Weekend": 1.00},
}
AR_RHO = 0.7                 # correlation of demand noise between consecutive hours
MIN_DEMAND_FRACTION = 0.05   # demand never below 5% of capacity (keeps it positive)
MAX_DEMAND_FRACTION = 0.98   # demand never above 98% of capacity

# Time-of-day profiles (24 values each, index = hour). Multipliers of base demand.
TIME_PROFILES = {
    "Residential": [0.55, 0.50, 0.47, 0.45, 0.46, 0.52,          # 00-05 low
                    0.75, 0.92, 1.00, 0.88,                      # 06-09 morning peak
                    0.80, 0.78, 0.76, 0.75, 0.74, 0.76, 0.82,    # 10-16 moderate
                    0.98, 1.15, 1.30, 1.28, 1.15,                # 17-21 evening peak
                    0.90, 0.70],                                 # 22-23 decreasing
    "Commercial":  [0.35, 0.33, 0.32, 0.32, 0.34, 0.40,
                    0.55, 0.75, 0.95, 1.10,
                    1.20, 1.25, 1.25, 1.22, 1.25, 1.22, 1.15,
                    1.00, 0.90, 0.85, 0.78, 0.65,
                    0.50, 0.40],
    "Industrial":  [0.80, 0.80, 0.80, 0.80, 0.82, 0.88,
                    0.95, 1.05, 1.10, 1.10,
                    1.12, 1.12, 1.10, 1.08, 1.10, 1.12, 1.10,
                    1.05, 1.00, 0.95, 0.92, 0.88,
                    0.85, 0.82],
    "Hospital":    [0.90, 0.89, 0.88, 0.88, 0.89, 0.92,
                    0.96, 1.00, 1.03, 1.05,
                    1.06, 1.07, 1.07, 1.06, 1.06, 1.05, 1.04,
                    1.03, 1.03, 1.02, 1.00, 0.97,
                    0.94, 0.92],
}

# Demand category thresholds (fraction of feeder capacity)
CAT_LOW_LIMIT = 0.40
CAT_MEDIUM_LIMIT = 0.65
CAT_HIGH_LIMIT = 0.85

# Supply model constants
SUPPLY_REGIME_RATIO = {0: (1.06, 1.16), 1: (0.90, 1.00), 2: (0.75, 0.90)}
# Regime probabilities (comfortable, tight, severe) by (stress level, is_peak_hour)
SUPPLY_REGIME_PROBS = {
    ("Low", True):     (0.55, 0.35, 0.10),
    ("Low", False):    (0.85, 0.13, 0.02),
    ("Medium", True):  (0.30, 0.40, 0.30),
    ("Medium", False): (0.65, 0.27, 0.08),
    ("High", True):    (0.10, 0.40, 0.50),
    ("High", False):   (0.40, 0.40, 0.20),
}
STRESS_COUNTS = {"Low": 17, "Medium": 17, "High": 16}   # sums to 50
REGIME_PERSISTENCE = 0.5          # chance the previous hour's regime is kept
RESERVE_FRACTION_RANGE = (0.02, 0.05)
SHORTAGE_THRESHOLD_FRACTION = 0.10   # Tight: shortage <= 10% of total demand

# Output paths (script always works relative to its own folder)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
FEEDERS_CSV = DATA_DIR / "feeders.csv"
DEMAND_CSV = DATA_DIR / "hourly_demand.csv"
SUPPLY_CSV = DATA_DIR / "hourly_supply.csv"
WEATHER_CSV = DATA_DIR / "weather_time.csv"
SUMMARY_TXT = METADATA_DIR / "dataset_summary.txt"

# Column definitions (exact order of the CSV files)
FEEDER_COLUMNS = [
    "feeder_id", "area", "feeder_name", "feeder_capacity_mw", "priority",
    "priority_weight", "is_critical", "critical_type", "max_total_outage_hours",
    "max_consecutive_outage_hours", "protected_hours", "base_demand_mw",
    "demand_variation_pct",
]
DEMAND_COLUMNS = [
    "scenario_id", "hour", "time_label", "feeder_id", "area", "day_type", "season",
    "temperature_c", "humidity_pct", "previous_hour_demand_mw", "actual_demand_mw",
    "demand_category", "feeder_capacity_mw",
]
SUPPLY_COLUMNS = [
    "scenario_id", "hour", "time_label", "available_supply_mw", "reserve_mw",
    "usable_supply_mw", "total_demand_mw", "shortage_mw", "supply_status",
]
WEATHER_COLUMNS = [
    "scenario_id", "hour", "time_label", "day_type", "day_of_week", "season",
    "temperature_c", "humidity_pct", "is_weekend", "is_peak_hour",
    "is_morning_peak", "is_evening_peak",
]


class DatasetValidationError(Exception):
    """Raised when a generated dataset fails one or more validation checks."""


# =============================================================================
# 2. HELPERS
# =============================================================================
def time_label(hour: int) -> str:
    """Return a readable time label such as '07:00'."""
    return f"{hour:02d}:00"


def format_protected_hours(hours: list[int]) -> str:
    """Protected hours as a consistent comma-separated string, e.g. '08,09,10'."""
    return ",".join(f"{h:02d}" for h in sorted(hours))


def classify_demand(demand: pd.Series, capacity: pd.Series) -> pd.Series:
    """Vectorised demand category based on demand / feeder capacity."""
    ratio = demand / capacity
    conditions = [ratio < CAT_LOW_LIMIT, ratio < CAT_MEDIUM_LIMIT, ratio < CAT_HIGH_LIMIT]
    return pd.Series(np.select(conditions, ["Low", "Medium", "High"], default="Peak"),
                     index=demand.index)


def classify_supply_status(shortage: pd.Series, total_demand: pd.Series) -> pd.Series:
    """Sufficient / Tight / Shortage according to the documented threshold."""
    conditions = [shortage <= 0.0, shortage <= SHORTAGE_THRESHOLD_FRACTION * total_demand]
    return pd.Series(np.select(conditions, ["Sufficient", "Tight"], default="Shortage"),
                     index=shortage.index)


# =============================================================================
# 3. RANDOM SEED
# =============================================================================
def set_random_seed(seed: int = RANDOM_SEED) -> dict:
    """Seed all generators and return independent, named generators."""
    random.seed(seed)
    np.random.seed(seed)  # legacy global NumPy generator, seeded for completeness
    return {
        "layout_py": random.Random(seed + 1),
        "stress_py": random.Random(seed + 2),
        "weather": np.random.default_rng([seed, 1]),
        "profile": np.random.default_rng([seed, 2]),
        "demand": np.random.default_rng([seed, 3]),
        "supply": np.random.default_rng([seed, 4]),
    }


# =============================================================================
# 4. TABLE 1 - FEEDERS
# =============================================================================
def generate_feeders() -> pd.DataFrame:
    """Static feeder metadata (identical for all 50 scenarios)."""
    # (id, area, category, capacity, base, priority, weight, total_out, consec_out, protected, var%)
    spec = [
        ("F001", "North Karachi", "Hospital",    16.0,  9.0, 5, 5.0, 0, 0, list(range(24)), 3.0),
        ("F002", "North Karachi", "Industrial",  36.0, 20.0, 3, 2.0, 4, 2, [9, 10, 11, 12, 13, 14], 5.0),
        ("F003", "Gulshan",       "Residential", 28.0, 15.0, 3, 2.0, 5, 2, [], 10.0),
        ("F004", "Gulshan",       "Commercial",  32.0, 17.0, 4, 3.0, 3, 1, [11, 12, 13, 14], 8.0),
        ("F005", "Saddar",        "Commercial",  34.0, 18.0, 4, 3.0, 3, 1, [12, 13, 14, 15], 8.0),
        ("F006", "Saddar",        "Residential", 22.0, 12.0, 2, 1.5, 6, 2, [], 10.0),
        ("F007", "Clifton",       "Residential", 30.0, 16.0, 3, 2.0, 5, 2, [20, 21], 10.0),
        ("F008", "Clifton",       "Commercial",  28.0, 15.0, 3, 2.0, 4, 2, [11, 12, 13], 8.0),
        ("F009", "Malir",         "Industrial",  38.0, 21.0, 4, 3.0, 3, 1, [8, 9, 10, 11, 12, 13, 14, 15], 5.0),
        ("F010", "Malir",         "Residential", 20.0, 11.0, 2, 1.5, 6, 2, [], 10.0),
    ]
    rows = []
    for fid, area, category, cap, base, prio, weight, tot_out, con_out, prot, var in spec:
        rows.append({
            "feeder_id": fid,
            "area": area,
            "feeder_name": f"{area} {category} Feeder {fid[-3:]}",
            "feeder_capacity_mw": cap,
            "priority": prio,
            "priority_weight": weight,
            "is_critical": category == "Hospital",
            "critical_type": category,
            "max_total_outage_hours": tot_out,
            "max_consecutive_outage_hours": con_out,
            "protected_hours": format_protected_hours(prot),
            "base_demand_mw": base,
            "demand_variation_pct": var,
        })
    return pd.DataFrame(rows, columns=FEEDER_COLUMNS)


# =============================================================================
# 5. TABLE 4 - WEATHER AND TIME
# =============================================================================
def generate_weather_time(rngs: dict) -> pd.DataFrame:
    """Scenario-level hourly weather-like and time features (1,200 rows)."""
    rng = rngs["weather"]
    layout = rngs["layout_py"]

    # Controlled random layout: fixed counts, seeded shuffle.
    seasons = [s for s, n in SEASON_COUNTS.items() for _ in range(n)]
    day_types = [d for d, n in DAY_TYPE_COUNTS.items() for _ in range(n)]
    layout.shuffle(seasons)
    layout.shuffle(day_types)

    rows = []
    for idx, scenario_id in enumerate(SCENARIO_IDS):
        season = seasons[idx]
        day_type = day_types[idx]
        day_of_week = layout.choice(WEEKDAY_NAMES if day_type == "Weekday" else WEEKEND_NAMES)
        params = SEASON_PARAMS[season]

        t_min = rng.uniform(*params["tmin"])
        t_max = rng.uniform(*params["tmax"])
        t_mean = (t_min + t_max) / 2.0
        humidity_base = rng.uniform(*params["humidity_base"])

        temp_noise = 0.0
        hum_noise = 0.0
        for hour in HOURS:
            # Daily temperature curve: minimum at 03:00, maximum at 15:00.
            curve = 0.5 * (1.0 + np.cos(2.0 * np.pi * (hour - 15) / 24.0))
            temp_clean = t_min + (t_max - t_min) * curve
            temp_noise = 0.8 * temp_noise + rng.normal(0.0, 0.25)   # smooth AR(1) noise
            temperature = float(np.clip(temp_clean + temp_noise, *params["temp_range"]))

            # Humidity falls when the day is hotter than average, plus smooth noise.
            hum_noise = 0.8 * hum_noise + rng.normal(0.0, 1.0)
            humidity = float(np.clip(humidity_base - 0.9 * (temperature - t_mean) + hum_noise,
                                     HUMIDITY_MIN, HUMIDITY_MAX))

            is_morning = hour in MORNING_PEAK_HOURS
            is_evening = hour in EVENING_PEAK_HOURS
            rows.append({
                "scenario_id": scenario_id,
                "hour": hour,
                "time_label": time_label(hour),
                "day_type": day_type,
                "day_of_week": day_of_week,
                "season": season,
                "temperature_c": round(temperature, 1),
                "humidity_pct": round(humidity, 1),
                "is_weekend": day_type == "Weekend",
                "is_peak_hour": is_morning or is_evening,
                "is_morning_peak": is_morning,
                "is_evening_peak": is_evening,
            })
    return pd.DataFrame(rows, columns=WEATHER_COLUMNS)


# =============================================================================
# 6. TABLE 2 - HOURLY DEMAND
# =============================================================================
def generate_hourly_demand(feeders: pd.DataFrame, weather: pd.DataFrame, rngs: dict) -> pd.DataFrame:
    """One row per scenario x hour x feeder (12,000 rows)."""
    rng = rngs["demand"]
    profile_rng = rngs["profile"]
    feeder_records = feeders.to_dict("records")

    # Fixed (scenario-independent) small hourly shape difference per feeder, so two
    # feeders of the same category never share exactly the same daily profile.
    jitter = {}
    for f in feeder_records:
        amplitude = 0.01 if f["critical_type"] == "Hospital" else 0.04
        jitter[f["feeder_id"]] = 1.0 + profile_rng.uniform(-amplitude, amplitude, HOURS_PER_SCENARIO)

    weather_by_scenario = {
        sid: grp.sort_values("hour").reset_index(drop=True)
        for sid, grp in weather.groupby("scenario_id")
    }

    records = []
    for scenario_id in SCENARIO_IDS:
        w = weather_by_scenario[scenario_id]
        temps = w["temperature_c"].to_numpy(dtype=float)
        hums = w["humidity_pct"].to_numpy(dtype=float)
        season = w["season"].iloc[0]
        day_type = w["day_type"].iloc[0]

        # Effective temperature combines temperature and humidity (heat discomfort).
        t_eff = temps + 0.05 * (hums - 50.0)

        demand_by_feeder = {}
        previous_by_feeder = {}
        for f in feeder_records:
            fid = f["feeder_id"]
            category = f["critical_type"]
            cap = float(f["feeder_capacity_mw"])
            base = float(f["base_demand_mw"])
            limit = f["demand_variation_pct"] / 100.0      # maximum noise fraction
            sigma = limit / 2.0                            # noise std-dev
            profile = np.array(TIME_PROFILES[category], dtype=float)

            temp_factor = 1.0 + TEMP_SENSITIVITY[category] * (
                0.010 * np.maximum(0.0, t_eff - 27.0) + 0.004 * np.maximum(0.0, 18.0 - t_eff)
            )
            scenario_feeder_factor = float(np.clip(rng.normal(1.0, 0.02), 0.95, 1.05))

            # AR(1) noise: consecutive hours are correlated (no wild hourly jumps).
            noise = float(np.clip(rng.normal(0.0, sigma), -limit, limit))
            demand = np.empty(HOURS_PER_SCENARIO)
            for hour in HOURS:
                noise = AR_RHO * noise + np.sqrt(1.0 - AR_RHO ** 2) * rng.normal(0.0, sigma)
                noise = float(np.clip(noise, -limit, limit))
                raw = (base * profile[hour] * jitter[fid][hour]
                       * SEASONAL_FACTOR[season]
                       * temp_factor[hour]
                       * DAY_TYPE_FACTOR[category][day_type]
                       * scenario_feeder_factor
                       * (1.0 + noise))
                demand[hour] = np.clip(raw, MIN_DEMAND_FRACTION * cap, MAX_DEMAND_FRACTION * cap)
            demand = np.round(demand, 3)

            # Previous-hour feature: hour h uses hour h-1 of the SAME feeder and scenario.
            # Hour 0 uses a separate initial value built only from the feeder's base demand,
            # the category's typical late-evening level and a small random variation.
            # Hour 23 of THIS scenario is never used (no future information).
            initial_prev = base * profile[HOURS_PER_SCENARIO - 1] * rng.uniform(0.95, 1.05)
            initial_prev = round(float(np.clip(initial_prev, MIN_DEMAND_FRACTION * cap,
                                               MAX_DEMAND_FRACTION * cap)), 3)
            previous = np.empty(HOURS_PER_SCENARIO)
            previous[0] = initial_prev
            previous[1:] = demand[:-1]

            demand_by_feeder[fid] = demand
            previous_by_feeder[fid] = previous

        # Assemble rows in the order scenario -> hour -> feeder.
        for hour in HOURS:
            for f in feeder_records:
                fid = f["feeder_id"]
                records.append({
                    "scenario_id": scenario_id,
                    "hour": hour,
                    "time_label": time_label(hour),
                    "feeder_id": fid,
                    "area": f["area"],
                    "day_type": day_type,
                    "season": season,
                    "temperature_c": float(temps[hour]),
                    "humidity_pct": float(hums[hour]),
                    "previous_hour_demand_mw": float(previous_by_feeder[fid][hour]),
                    "actual_demand_mw": float(demand_by_feeder[fid][hour]),
                    "demand_category": "",   # assigned below from the final demand values
                    "feeder_capacity_mw": float(f["feeder_capacity_mw"]),
                })

    demand_df = pd.DataFrame(records, columns=DEMAND_COLUMNS)
    demand_df["demand_category"] = classify_demand(demand_df["actual_demand_mw"],
                                                   demand_df["feeder_capacity_mw"])
    return demand_df


# =============================================================================
# 7. TABLE 3 - HOURLY SUPPLY
# =============================================================================
def generate_hourly_supply(demand: pd.DataFrame, weather: pd.DataFrame, rngs: dict) -> pd.DataFrame:
    """One row per scenario x hour: network-wide supply (1,200 rows)."""
    rng = rngs["supply"]
    stress_py = rngs["stress_py"]

    # Scenario stress level controls how often shortages occur (mix of easy/hard days).
    stress_levels = [s for s, n in STRESS_COUNTS.items() for _ in range(n)]
    stress_py.shuffle(stress_levels)
    stress_by_scenario = dict(zip(SCENARIO_IDS, stress_levels))

    total_demand = (demand.groupby(["scenario_id", "hour"])["actual_demand_mw"]
                    .sum().round(3))
    peak_flag = weather.set_index(["scenario_id", "hour"])["is_peak_hour"]

    rows = []
    for scenario_id in SCENARIO_IDS:
        stress = stress_by_scenario[scenario_id]
        regime = None
        for hour in HOURS:
            is_peak = bool(peak_flag.loc[(scenario_id, hour)])
            if regime is None or rng.random() >= REGIME_PERSISTENCE:
                regime = int(rng.choice(3, p=SUPPLY_REGIME_PROBS[(stress, is_peak)]))
            ratio = rng.uniform(*SUPPLY_REGIME_RATIO[regime])
            reserve_fraction = rng.uniform(*RESERVE_FRACTION_RANGE)

            demand_total = float(total_demand.loc[(scenario_id, hour)])
            available = round(demand_total * ratio, 3)
            reserve = round(available * reserve_fraction, 3)
            usable = round(available - reserve, 3)
            shortage = round(max(0.0, demand_total - usable), 3)

            rows.append({
                "scenario_id": scenario_id,
                "hour": hour,
                "time_label": time_label(hour),
                "available_supply_mw": available,
                "reserve_mw": reserve,
                "usable_supply_mw": usable,
                "total_demand_mw": demand_total,
                "shortage_mw": shortage,
                "supply_status": "",
            })

    supply_df = pd.DataFrame(rows, columns=SUPPLY_COLUMNS)
    supply_df["supply_status"] = classify_supply_status(supply_df["shortage_mw"],
                                                        supply_df["total_demand_mw"])
    return supply_df


# =============================================================================
# 8. VALIDATION
# =============================================================================
def validate_datasets(feeders: pd.DataFrame, demand: pd.DataFrame,
                      supply: pd.DataFrame, weather: pd.DataFrame) -> list[tuple[str, bool]]:
    """Run every quality check. Raises DatasetValidationError if any check fails."""
    # --- Column structure must be exact before any other check can run ----------
    for name, df, cols in [("feeders", feeders, FEEDER_COLUMNS), ("hourly_demand", demand, DEMAND_COLUMNS),
                           ("hourly_supply", supply, SUPPLY_COLUMNS), ("weather_time", weather, WEATHER_COLUMNS)]:
        if list(df.columns) != cols:
            raise DatasetValidationError(f"{name}: columns {list(df.columns)} do not match required {cols}")

    results: list[tuple[str, bool]] = []

    def check(description: str, condition) -> None:
        results.append((description, bool(condition)))

    close = lambda a, b: np.isclose(a, b, rtol=0.0, atol=1e-6)  # noqa: E731

    # --- Feeders ------------------------------------------------------------------
    check("feeders.csv has exactly 10 rows", len(feeders) == EXPECTED_FEEDER_ROWS)
    check("feeder_id values are unique", feeders["feeder_id"].is_unique)
    check("exactly 5 unique areas", feeders["area"].nunique() == NUM_AREAS)
    check("each area has exactly 2 feeders", (feeders.groupby("area").size() == FEEDERS_PER_AREA).all())
    check("exactly one hospital feeder", (feeders["critical_type"] == "Hospital").sum() == 1)
    hospital = feeders[feeders["feeder_id"] == "F001"].iloc[0]
    check("F001 is the critical hospital feeder",
          bool(hospital["is_critical"]) and hospital["critical_type"] == "Hospital")
    check("only F001 is critical", feeders.loc[feeders["is_critical"], "feeder_id"].tolist() == ["F001"])
    check("F001 has the highest priority (5) and no other feeder has 5",
          hospital["priority"] == 5 and (feeders["priority"] == 5).sum() == 1)
    check("F001 has zero permitted outage hours",
          hospital["max_total_outage_hours"] == 0 and hospital["max_consecutive_outage_hours"] == 0)
    check("priority values are within 1-5", feeders["priority"].between(1, 5).all())
    check("priority_weight values are positive", (feeders["priority_weight"] > 0).all())
    check("base demand is below feeder capacity",
          (feeders["base_demand_mw"] < feeders["feeder_capacity_mw"]).all())
    check("max consecutive outage <= max total outage",
          (feeders["max_consecutive_outage_hours"] <= feeders["max_total_outage_hours"]).all())
    check("critical_type values are valid",
          feeders["critical_type"].isin(["Hospital", "Residential", "Commercial", "Industrial", "None"]).all())

    protected_ok = True
    for _, row in feeders.iterrows():
        text = row["protected_hours"]
        hours = [int(x) for x in text.split(",")] if text != "" else []
        formatted_ok = text == format_protected_hours(hours)
        in_range = all(0 <= h < HOURS_PER_SCENARIO for h in hours) and len(set(hours)) == len(hours)
        feasible = row["max_total_outage_hours"] <= HOURS_PER_SCENARIO - len(hours)
        if not (formatted_ok and in_range and feasible):
            protected_ok = False
    check("protected_hours are well-formatted, valid and leave room for allowed outages", protected_ok)

    # --- Row counts -----------------------------------------------------------------
    check("hourly_demand.csv has exactly 12,000 rows", len(demand) == EXPECTED_DEMAND_ROWS)
    check("hourly_supply.csv has exactly 1,200 rows", len(supply) == EXPECTED_SUPPLY_ROWS)
    check("weather_time.csv has exactly 1,200 rows", len(weather) == EXPECTED_WEATHER_ROWS)

    # --- Duplicates ---------------------------------------------------------------------
    check("no duplicate scenario-hour-feeder rows in demand",
          not demand.duplicated(["scenario_id", "hour", "feeder_id"]).any())
    check("no duplicate scenario-hour rows in supply", not supply.duplicated(["scenario_id", "hour"]).any())
    check("no duplicate scenario-hour rows in weather", not weather.duplicated(["scenario_id", "hour"]).any())

    # --- Missing values -----------------------------------------------------------------
    for name, df in [("feeders", feeders), ("hourly_demand", demand),
                     ("hourly_supply", supply), ("weather_time", weather)]:
        check(f"{name}: no missing values", not df.isna().any().any())

    # --- Scenario / hour structure ---------------------------------------------------------
    for name, df in [("hourly_demand", demand), ("hourly_supply", supply), ("weather_time", weather)]:
        check(f"{name}: scenario ids are exactly S001-S050", sorted(df["scenario_id"].unique()) == SCENARIO_IDS)
        check(f"{name}: every scenario has exactly 24 hours (0-23)",
              (df.groupby("scenario_id")["hour"].nunique() == HOURS_PER_SCENARIO).all()
              and df["hour"].between(0, 23).all())
    check("each scenario-hour has exactly 10 feeder demand records",
          (demand.groupby(["scenario_id", "hour"]).size() == NUM_FEEDERS).all()
          and demand.groupby(["scenario_id", "hour"]).ngroups == EXPECTED_SUPPLY_ROWS)
    check("each scenario-hour has exactly one supply record",
          (supply.groupby(["scenario_id", "hour"]).size() == 1).all())
    check("each scenario-hour has exactly one weather/time record",
          (weather.groupby(["scenario_id", "hour"]).size() == 1).all())

    # --- Demand vs feeders --------------------------------------------------------------------
    check("every demand feeder_id exists in feeders.csv", demand["feeder_id"].isin(feeders["feeder_id"]).all())
    merged_f = demand.merge(feeders[["feeder_id", "area", "feeder_capacity_mw"]], on="feeder_id",
                            how="left", suffixes=("", "_feeder"))
    check("demand area names match feeder metadata", (merged_f["area"] == merged_f["area_feeder"]).all())
    check("demand feeder capacity matches feeders.csv",
          close(merged_f["feeder_capacity_mw"], merged_f["feeder_capacity_mw_feeder"]).all())

    # --- Demand vs weather ----------------------------------------------------------------------
    merged_w = demand.merge(
        weather[["scenario_id", "hour", "temperature_c", "humidity_pct", "season", "day_type"]],
        on=["scenario_id", "hour"], how="left", suffixes=("", "_w"))
    check("demand temperature matches weather_time.csv", close(merged_w["temperature_c"], merged_w["temperature_c_w"]).all())
    check("demand humidity matches weather_time.csv", close(merged_w["humidity_pct"], merged_w["humidity_pct_w"]).all())
    check("demand season matches weather_time.csv", (merged_w["season"] == merged_w["season_w"]).all())
    check("demand day_type matches weather_time.csv", (merged_w["day_type"] == merged_w["day_type_w"]).all())

    # --- Demand values -----------------------------------------------------------------------------
    check("all actual demand values are positive", (demand["actual_demand_mw"] > 0).all())
    check("no actual demand exceeds feeder capacity",
          (demand["actual_demand_mw"] <= demand["feeder_capacity_mw"]).all())
    check("previous-hour demand is positive and within capacity",
          (demand["previous_hour_demand_mw"] > 0).all()
          and (demand["previous_hour_demand_mw"] <= demand["feeder_capacity_mw"]).all())
    ordered = demand.sort_values(["scenario_id", "feeder_id", "hour"]).reset_index(drop=True)
    shifted = ordered.groupby(["scenario_id", "feeder_id"])["actual_demand_mw"].shift(1)
    later_hours = ordered["hour"] > 0
    check("previous_hour_demand_mw equals the same feeder's previous-hour demand (hours 1-23)",
          close(ordered.loc[later_hours, "previous_hour_demand_mw"], shifted[later_hours]).all())
    max_step = (ordered.groupby(["scenario_id", "feeder_id"])["actual_demand_mw"].diff().abs()
                / ordered["feeder_capacity_mw"]).max()
    check("no extreme hour-to-hour demand jumps (<= 30% of capacity)", max_step <= 0.30)
    check("demand_category matches demand / capacity thresholds",
          (demand["demand_category"] == classify_demand(demand["actual_demand_mw"],
                                                        demand["feeder_capacity_mw"])).all())
    check("demand is not identical across all feeders/scenarios", demand["actual_demand_mw"].nunique() > 1000)

    # --- Supply -----------------------------------------------------------------------------------------
    check("available supply is always positive", (supply["available_supply_mw"] > 0).all())
    check("reserve is non-negative", (supply["reserve_mw"] >= 0).all())
    check("usable_supply = available_supply - reserve",
          close(supply["usable_supply_mw"], supply["available_supply_mw"] - supply["reserve_mw"]).all())
    check("shortage is never negative", (supply["shortage_mw"] >= 0).all())
    expected_shortage = np.maximum(0.0, supply["total_demand_mw"] - supply["usable_supply_mw"]).round(3)
    check("shortage = max(0, total_demand - usable_supply)", close(supply["shortage_mw"], expected_shortage).all())
    sums = (demand.groupby(["scenario_id", "hour"])["actual_demand_mw"].sum().rename("sum_demand").reset_index())
    merged_s = supply.merge(sums, on=["scenario_id", "hour"], how="left")
    check("total_demand_mw equals the sum of the 10 feeder demands",
          close(merged_s["total_demand_mw"], merged_s["sum_demand"]).all())
    check("supply_status is consistent with shortage and threshold",
          (supply["supply_status"] == classify_supply_status(supply["shortage_mw"], supply["total_demand_mw"])).all())
    check("supply data contains Sufficient, Tight and Shortage hours",
          set(supply["supply_status"].unique()) == {"Sufficient", "Tight", "Shortage"})

    # --- Weather / time ------------------------------------------------------------------------------------
    for season, params in SEASON_PARAMS.items():
        temps = weather.loc[weather["season"] == season, "temperature_c"]
        low, high = params["temp_range"]
        check(f"{season} temperatures within {low:.0f}-{high:.0f} C", temps.between(low, high).all())
    check("humidity within 30%-90%", weather["humidity_pct"].between(HUMIDITY_MIN, HUMIDITY_MAX).all())
    temp_steps = weather.sort_values(["scenario_id", "hour"]).groupby("scenario_id")["temperature_c"].diff().abs()
    check("temperature changes smoothly between hours (<= 3 C)", temp_steps.max() <= 3.0)
    check("season, day_type and day_of_week are constant within each scenario",
          (weather.groupby("scenario_id")[["season", "day_type", "day_of_week"]].nunique() == 1).all().all())
    check("is_weekend matches day_type", (weather["is_weekend"] == (weather["day_type"] == "Weekend")).all())
    check("day_of_week matches day_type",
          weather["day_of_week"].isin(WEEKEND_NAMES).eq(weather["day_type"] == "Weekend").all())
    check("morning peak flag is hours 6-9",
          (weather["is_morning_peak"] == weather["hour"].isin(MORNING_PEAK_HOURS)).all())
    check("evening peak flag is hours 17-21",
          (weather["is_evening_peak"] == weather["hour"].isin(EVENING_PEAK_HOURS)).all())
    check("is_peak_hour = morning peak OR evening peak",
          (weather["is_peak_hour"] == (weather["is_morning_peak"] | weather["is_evening_peak"])).all())
    check("peak/weekend flags are Boolean",
          all(weather[c].dtype == bool for c in ["is_weekend", "is_peak_hour", "is_morning_peak", "is_evening_peak"]))

    failed = [desc for desc, ok in results if not ok]
    if failed:
        message = "Dataset validation FAILED. Failed checks:\n  - " + "\n  - ".join(failed)
        raise DatasetValidationError(message)
    return results


# =============================================================================
# 9. SAVING AND SUMMARY
# =============================================================================
def save_datasets(feeders: pd.DataFrame, demand: pd.DataFrame,
                  supply: pd.DataFrame, weather: pd.DataFrame) -> None:
    """Create folders if necessary and write the four CSV files (only these files)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    # lineterminator="\n" keeps files byte-identical on every operating system.
    feeders.to_csv(FEEDERS_CSV, index=False, lineterminator="\n", encoding="utf-8")
    demand.to_csv(DEMAND_CSV, index=False, lineterminator="\n", encoding="utf-8")
    supply.to_csv(SUPPLY_CSV, index=False, lineterminator="\n", encoding="utf-8")
    weather.to_csv(WEATHER_CSV, index=False, lineterminator="\n", encoding="utf-8")


def generate_summary(feeders: pd.DataFrame, demand: pd.DataFrame, supply: pd.DataFrame,
                     weather: pd.DataFrame, validation_results: list[tuple[str, bool]]) -> str:
    """Build the metadata summary text and write data/metadata/dataset_summary.txt."""
    line = "=" * 70
    status_counts = supply["supply_status"].value_counts()
    shortage_hours = int((supply["shortage_mw"] > 0).sum())
    scenarios_with_shortage = int(supply.loc[supply["shortage_mw"] > 0, "scenario_id"].nunique())
    scenario_info = weather.drop_duplicates("scenario_id")

    lines = [
        line,
        f" DATASET SUMMARY - {PROJECT_NAME}",
        line,
        "",
        "*** ALL DATA IN THIS PROJECT IS SIMULATED DATA. ***",
        "It is NOT actual measurement data from K-Electric or any other electricity",
        "company and is NOT connected to a real electricity grid. Educational use only.",
        "",
        "-" * 70,
        " CONFIGURATION",
        "-" * 70,
        f"Generated on          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Random seed           : {RANDOM_SEED}",
        f"Number of scenarios   : {NUM_SCENARIOS}",
        f"Number of areas       : {NUM_AREAS}",
        f"Number of feeders     : {NUM_FEEDERS} ({FEEDERS_PER_AREA} per area)",
        f"Hours per scenario    : {HOURS_PER_SCENARIO} (time slots 0-23)",
        f"Power unit            : {POWER_UNIT}",
        f"Season mix            : {', '.join(f'{k}={v}' for k, v in scenario_info['season'].value_counts().items())}",
        f"Day-type mix          : {', '.join(f'{k}={v}' for k, v in scenario_info['day_type'].value_counts().items())}",
        "",
        "-" * 70,
        " ROW COUNTS",
        "-" * 70,
        f"feeders.csv           : {len(feeders):>6,} rows",
        f"hourly_demand.csv     : {len(demand):>6,} rows",
        f"hourly_supply.csv     : {len(supply):>6,} rows",
        f"weather_time.csv      : {len(weather):>6,} rows",
        "",
        "-" * 70,
        " COLUMNS",
        "-" * 70,
        f"feeders.csv       : {', '.join(FEEDER_COLUMNS)}",
        f"hourly_demand.csv : {', '.join(DEMAND_COLUMNS)}",
        f"hourly_supply.csv : {', '.join(SUPPLY_COLUMNS)}",
        f"weather_time.csv  : {', '.join(WEATHER_COLUMNS)}",
        "",
        "-" * 70,
        " DEMAND STATISTICS (feeder-level actual_demand_mw, MW)",
        "-" * 70,
        f"Minimum : {demand['actual_demand_mw'].min():.3f}",
        f"Maximum : {demand['actual_demand_mw'].max():.3f}",
        f"Mean    : {demand['actual_demand_mw'].mean():.3f}",
        "Demand category counts: " + ", ".join(
            f"{c}={int((demand['demand_category'] == c).sum())}" for c in ["Low", "Medium", "High", "Peak"]),
        "",
        "-" * 70,
        " SUPPLY STATISTICS (network available_supply_mw, MW)",
        "-" * 70,
        f"Minimum : {supply['available_supply_mw'].min():.3f}",
        f"Maximum : {supply['available_supply_mw'].max():.3f}",
        f"Mean    : {supply['available_supply_mw'].mean():.3f}",
        f"Total shortage hours (scenario-hours with shortage_mw > 0) : {shortage_hours}",
        f"Scenarios containing at least one shortage hour            : {scenarios_with_shortage} of {NUM_SCENARIOS}",
        f"Largest hourly shortage (MW)                               : {supply['shortage_mw'].max():.3f}",
        f"Sufficient scenario-hours : {int(status_counts.get('Sufficient', 0))}",
        f"Tight scenario-hours      : {int(status_counts.get('Tight', 0))}",
        f"Shortage scenario-hours   : {int(status_counts.get('Shortage', 0))}",
        f"(Tight = 0 < shortage <= {SHORTAGE_THRESHOLD_FRACTION:.0%} of total demand; "
        f"Shortage = more than {SHORTAGE_THRESHOLD_FRACTION:.0%})",
        "",
        "-" * 70,
        " FEEDER INFORMATION",
        "-" * 70,
        f"Critical feeders : {int(feeders['is_critical'].sum())} "
        f"({', '.join(feeders.loc[feeders['is_critical'], 'feeder_id'])} - Hospital, never scheduled for outage)",
        "Note: protected_hours is an empty string for feeders without protected hours. When",
        "      loading feeders.csv with pandas, use keep_default_na=False (or dtype=str for",
        "      that column) so empty strings are not converted to NaN.",
        "",
        "-" * 70,
        " VALIDATION RESULTS",
        "-" * 70,
    ]
    for description, passed in validation_results:
        lines.append(f"[{'PASS' if passed else 'FAIL'}] {description}")
    passed_count = sum(1 for _, ok in validation_results if ok)
    lines += ["", f"Checks passed: {passed_count} / {len(validation_results)}",
              "OVERALL RESULT: ALL VALIDATION CHECKS PASSED" if passed_count == len(validation_results)
              else "OVERALL RESULT: VALIDATION FAILED", line, ""]

    text = "\n".join(lines)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_TXT.write_text(text, encoding="utf-8")
    return text


# =============================================================================
# 10. MAIN
# =============================================================================
def main() -> None:
    banner = "=" * 50
    print(banner)
    print(" AI ELECTRICITY LOAD-SHEDDING DATA GENERATOR")
    print(" (all generated data is SIMULATED DATA)")
    print(banner)
    print()

    try:
        print("[1/6] Initializing random seed...")
        rngs = set_random_seed(RANDOM_SEED)

        print("[2/6] Generating feeder metadata...")
        feeders = generate_feeders()

        print("[3/6] Generating weather and time data...")
        weather = generate_weather_time(rngs)

        print("[4/6] Generating hourly demand data...")
        demand = generate_hourly_demand(feeders, weather, rngs)

        print("[5/6] Generating hourly supply data...")
        supply = generate_hourly_supply(demand, weather, rngs)

        print("[6/6] Validating and saving datasets...")
        validation_results = validate_datasets(feeders, demand, supply, weather)
        save_datasets(feeders, demand, supply, weather)
        generate_summary(feeders, demand, supply, weather, validation_results)
    except DatasetValidationError as error:
        print()
        print(banner)
        print(" DATASET GENERATION FAILED - VALIDATION ERROR")
        print(banner)
        print(error)
        print("\nNo dataset files were saved.")
        sys.exit(1)
    except Exception as error:  # unexpected problem: report clearly, never fail silently
        print()
        print(banner)
        print(" DATASET GENERATION FAILED - UNEXPECTED ERROR")
        print(banner)
        print(f"{type(error).__name__}: {error}")
        sys.exit(1)

    status_counts = supply["supply_status"].value_counts()
    print()
    print(banner)
    print(" DATASET GENERATION COMPLETED SUCCESSFULLY")
    print(banner)
    print()
    print(f"Feeders:          {len(feeders)} rows")
    print(f"Hourly Demand:    {len(demand)} rows")
    print(f"Hourly Supply:    {len(supply)} rows")
    print(f"Weather & Time:   {len(weather)} rows")
    print()
    print(f"Scenarios:        {NUM_SCENARIOS}")
    print(f"Areas:            {NUM_AREAS}")
    print(f"Feeders:          {NUM_FEEDERS}")
    print(f"Hours:            {HOURS_PER_SCENARIO}")
    print(f"Random Seed:      {RANDOM_SEED}")
    print()
    print(f"Supply status:    Sufficient={int(status_counts.get('Sufficient', 0))}, "
          f"Tight={int(status_counts.get('Tight', 0))}, Shortage={int(status_counts.get('Shortage', 0))}")
    print()
    print(f"All {len(validation_results)} validation checks passed.")
    print()
    print(f"Files saved successfully in: {DATA_DIR}")
    print(f"Summary report:              {SUMMARY_TXT}")
    print(banner)


if __name__ == "__main__":
    main()