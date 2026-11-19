"""
data_loader.py
==============
Loads the four existing CSV files, validates them, and returns one clean
`DataBundle` object that the other modules use.

Nothing is ever regenerated or modified here. Important rows are never dropped
silently: if something is wrong, a `DataValidationError` lists every problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


class DataValidationError(Exception):
    """Raised when the input CSV files do not match what the project expects."""


def parse_protected_hours(text: str) -> frozenset[int]:
    """Convert a protected-hours string such as '08,09,10' into a set of ints {8, 9, 10}.

    An empty string means the feeder has no protected hours.
    """
    text = "" if text is None else str(text).strip()
    if text == "":
        return frozenset()
    return frozenset(int(part) for part in text.split(","))


@dataclass
class DataBundle:
    """All validated input tables plus a log of what was checked."""

    feeders: pd.DataFrame
    demand: pd.DataFrame
    supply: pd.DataFrame
    weather: pd.DataFrame
    column_mappings: list[str] = field(default_factory=list)
    validation_log: list[str] = field(default_factory=list)

    @property
    def scenario_ids(self) -> list[str]:
        return sorted(self.supply["scenario_id"].unique())

    @property
    def feeder_ids(self) -> list[str]:
        return sorted(self.feeders["feeder_id"].unique())

    @property
    def areas(self) -> list[str]:
        return list(dict.fromkeys(self.feeders.sort_values("feeder_id")["area"]))


class DataLoader:
    """Reads and validates the four project CSV files."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else config.DATA_DIR
        self._paths = {
            "feeders": self.data_dir / "feeders.csv",
            "demand": self.data_dir / "hourly_demand.csv",
            "supply": self.data_dir / "hourly_supply.csv",
            "weather": self.data_dir / "weather_time.csv",
        }

    # ------------------------------------------------------------------ loading
    def load(self) -> DataBundle:
        """Read the CSV files, map headers if needed, validate, and return a DataBundle."""
        tables: dict[str, pd.DataFrame] = {}
        mappings: list[str] = []
        for name, path in self._paths.items():
            if not path.exists():
                raise DataValidationError(
                    f"Required file not found: {path}. Run dataset_generator.py first "
                    f"(it is not run automatically so existing data is never overwritten)."
                )
            # keep_default_na=False: an empty protected_hours cell must stay '' (not NaN).
            frame = pd.read_csv(path, keep_default_na=False, na_values=[])
            frame, used = self._apply_column_aliases(name, frame)
            mappings.extend(used)
            tables[name] = frame

        # 'protected_hours' must be text so '08,09' keeps its leading zeros.
        tables["feeders"]["protected_hours"] = tables["feeders"]["protected_hours"].astype(str)

        bundle = DataBundle(
            feeders=tables["feeders"], demand=tables["demand"],
            supply=tables["supply"], weather=tables["weather"],
            column_mappings=mappings,
        )
        self.validate(bundle)
        return bundle

    @staticmethod
    def _apply_column_aliases(name: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Rename columns using config.COLUMN_ALIASES (transparently logged)."""
        used = []
        aliases = config.COLUMN_ALIASES.get(name, {})
        rename = {old: new for old, new in aliases.items() if old in frame.columns}
        for old, new in rename.items():
            used.append(f"{name}.csv: column '{old}' treated as '{new}'")
        return frame.rename(columns=rename), used

    # --------------------------------------------------------------- validation
    def validate(self, bundle: DataBundle) -> None:
        """Run every data check. Raises DataValidationError listing ALL failed checks."""
        failures: list[str] = []
        log: list[str] = []

        def check(description: str, condition: bool, detail: str = "") -> None:
            if condition:
                log.append(f"[PASS] {description}")
            else:
                log.append(f"[FAIL] {description} {detail}".strip())
                failures.append(f"{description} {detail}".strip())

        tables = {"feeders": bundle.feeders, "demand": bundle.demand,
                  "supply": bundle.supply, "weather": bundle.weather}

        # 1. Required columns (must be present, otherwise nothing else can be trusted)
        for name, frame in tables.items():
            missing = [c for c in config.REQUIRED_COLUMNS[name] if c not in frame.columns]
            if missing:
                raise DataValidationError(f"{name}.csv is missing required columns: {missing}")
            log.append(f"[PASS] {name}.csv has all {len(config.REQUIRED_COLUMNS[name])} required columns")

        feeders, demand, supply, weather = (tables[k] for k in ("feeders", "demand", "supply", "weather"))

        # 2. Row counts
        for name, frame in tables.items():
            check(f"{name}.csv has {config.EXPECTED_ROWS[name]} rows",
                  len(frame) == config.EXPECTED_ROWS[name], f"(found {len(frame)})")

        # 3. Missing values (empty strings in protected_hours are allowed by design)
        for name, frame in tables.items():
            columns = [c for c in frame.columns if not (name == "feeders" and c == "protected_hours")]
            blank = frame[columns].isna() | (frame[columns].astype(str) == "")
            check(f"{name}.csv has no missing values", not blank.any().any())

        # 4. Data types
        numeric_columns = {
            "feeders": ["feeder_capacity_mw", "priority", "priority_weight", "max_total_outage_hours",
                        "max_consecutive_outage_hours", "base_demand_mw", "demand_variation_pct"],
            "demand": ["hour", "temperature_c", "humidity_pct", "previous_hour_demand_mw",
                       "actual_demand_mw", "feeder_capacity_mw"],
            "supply": ["hour", "available_supply_mw", "reserve_mw", "usable_supply_mw",
                       "total_demand_mw", "shortage_mw"],
            "weather": ["hour", "temperature_c", "humidity_pct"],
        }
        for name, columns in numeric_columns.items():
            bad = [c for c in columns if not pd.api.types.is_numeric_dtype(tables[name][c])]
            check(f"{name}.csv numeric columns are numeric", not bad, f"(not numeric: {bad})")
        check("feeders.is_critical is boolean", pd.api.types.is_bool_dtype(feeders["is_critical"]))
        for column in ["is_weekend", "is_peak_hour", "is_morning_peak", "is_evening_peak"]:
            check(f"weather.{column} is boolean", pd.api.types.is_bool_dtype(weather[column]))

        # Stop here if basic structure is broken; relationship checks would only raise confusing errors.
        if failures:
            raise DataValidationError("Data validation failed:\n  - " + "\n  - ".join(failures))

        # 5. Duplicates
        check("no duplicate feeder_id in feeders.csv", feeders["feeder_id"].is_unique)
        check("no duplicate scenario-hour-feeder rows in demand",
              not demand.duplicated(["scenario_id", "hour", "feeder_id"]).any())
        check("no duplicate scenario-hour rows in supply", not supply.duplicated(["scenario_id", "hour"]).any())
        check("no duplicate scenario-hour rows in weather", not weather.duplicated(["scenario_id", "hour"]).any())

        # 6. Feeder metadata
        check(f"{config.NUM_FEEDERS} feeders", feeders["feeder_id"].nunique() == config.NUM_FEEDERS)
        check(f"{config.NUM_AREAS} areas with {config.FEEDERS_PER_AREA} feeders each",
              feeders["area"].nunique() == config.NUM_AREAS
              and (feeders.groupby("area").size() == config.FEEDERS_PER_AREA).all())
        hospital_rows = feeders[feeders["feeder_id"] == config.HOSPITAL_FEEDER_ID]
        check(f"{config.HOSPITAL_FEEDER_ID} exists, is critical and is the only Hospital feeder",
              len(hospital_rows) == 1 and bool(hospital_rows.iloc[0]["is_critical"])
              and hospital_rows.iloc[0]["critical_type"] == "Hospital"
              and (feeders["critical_type"] == "Hospital").sum() == 1
              and feeders["is_critical"].sum() == 1)
        check("hospital feeder has zero permitted outage hours",
              len(hospital_rows) == 1 and int(hospital_rows.iloc[0]["max_total_outage_hours"]) == 0)
        check("max consecutive outage <= max total outage",
              (feeders["max_consecutive_outage_hours"] <= feeders["max_total_outage_hours"]).all())
        check("base demand below feeder capacity",
              (feeders["base_demand_mw"] < feeders["feeder_capacity_mw"]).all())
        try:
            parsed = [parse_protected_hours(t) for t in feeders["protected_hours"]]
            hours_ok = all(0 <= h < config.HOURS_PER_SCENARIO for s in parsed for h in s)
        except ValueError:
            hours_ok = False
        check("protected_hours strings are valid (hours 0-23)", hours_ok)

        # 7. Scenario / hour / feeder structure
        expected_scenarios = config.SCENARIO_IDS
        for name, frame in (("demand", demand), ("supply", supply), ("weather", weather)):
            check(f"{name}: scenario ids are exactly S001-S050",
                  sorted(frame["scenario_id"].unique()) == expected_scenarios)
            check(f"{name}: every scenario has exactly {config.HOURS_PER_SCENARIO} hours (0-23)",
                  (frame.groupby("scenario_id")["hour"].nunique() == config.HOURS_PER_SCENARIO).all()
                  and frame["hour"].between(0, config.HOURS_PER_SCENARIO - 1).all())
        check("every demand feeder_id exists in feeders.csv", demand["feeder_id"].isin(feeders["feeder_id"]).all())
        check(f"every scenario-hour has exactly {config.NUM_FEEDERS} feeder demand records",
              (demand.groupby(["scenario_id", "hour"]).size() == config.NUM_FEEDERS).all())

        # 8. Cross-table consistency
        merged = demand.merge(feeders[["feeder_id", "area", "feeder_capacity_mw"]],
                              on="feeder_id", how="left", suffixes=("", "_feeder"))
        check("demand area and feeder capacity match feeders.csv",
              (merged["area"] == merged["area_feeder"]).all()
              and np.allclose(merged["feeder_capacity_mw"], merged["feeder_capacity_mw_feeder"]))
        check("demand is positive and within feeder capacity",
              (demand["actual_demand_mw"] > 0).all()
              and (demand["actual_demand_mw"] <= demand["feeder_capacity_mw"]).all())
        joined_weather = demand.merge(
            weather[["scenario_id", "hour", "temperature_c", "humidity_pct", "season", "day_type"]],
            on=["scenario_id", "hour"], how="left", suffixes=("", "_w"))
        check("demand temperature/humidity/season/day_type match weather_time.csv",
              np.allclose(joined_weather["temperature_c"], joined_weather["temperature_c_w"])
              and np.allclose(joined_weather["humidity_pct"], joined_weather["humidity_pct_w"])
              and (joined_weather["season"] == joined_weather["season_w"]).all()
              and (joined_weather["day_type"] == joined_weather["day_type_w"]).all())

        # 9. Supply arithmetic
        sums = demand.groupby(["scenario_id", "hour"])["actual_demand_mw"].sum().rename("sum_demand").reset_index()
        merged_supply = supply.merge(sums, on=["scenario_id", "hour"], how="left")
        check("supply.total_demand_mw equals the sum of feeder demands",
              np.allclose(merged_supply["total_demand_mw"], merged_supply["sum_demand"], atol=1e-6, rtol=0))
        check("usable_supply_mw = available_supply_mw - reserve_mw",
              np.allclose(supply["usable_supply_mw"], supply["available_supply_mw"] - supply["reserve_mw"],
                          atol=1e-6, rtol=0))
        check("shortage_mw = max(0, total_demand_mw - usable_supply_mw)",
              np.allclose(supply["shortage_mw"],
                          np.maximum(0.0, supply["total_demand_mw"] - supply["usable_supply_mw"]),
                          atol=1e-3, rtol=0))
        check("supply values are valid (available > 0, reserve >= 0, shortage >= 0)",
              (supply["available_supply_mw"] > 0).all() and (supply["reserve_mw"] >= 0).all()
              and (supply["shortage_mw"] >= 0).all())

        bundle.validation_log = log
        if failures:
            raise DataValidationError("Data validation failed:\n  - " + "\n  - ".join(failures))
