"""
environment.py
==============
`LoadSheddingEnvironment` - the simulated electricity network from the proposal.

It stores areas, feeders, hourly demand/supply, outage constraints, the hospital
feeder, protected time slots and the outage history, and it can load ONE scenario
(24 hourly slots) at a time.

Schedule representation
-----------------------
A schedule is a NumPy array of shape (num_feeders, 24) with binary values:

        schedule[i, t] = 1  ->  feeder i is ON  (its full demand is served)
        schedule[i, t] = 0  ->  feeder i is OFF (its full demand is NOT served)

Feeder rows follow the sorted feeder_id order (F001 ... F010). No partial
switching is modelled.

Two demand "bases"
------------------
* "planning" : the demand FORECAST (regression output). The schedulers plan with
               this, because in practice the real demand is not known in advance.
* "actual"   : the simulated true demand, used afterwards to measure how good a
               schedule really was.

Feeder-level decisions vs. area-level fairness
----------------------------------------------
The proposal talks about area-hour decisions. Our simulated network has 2 feeders
per area, so we decide ON/OFF per FEEDER-hour (needed to protect the hospital
feeder F001 individually) and then aggregate feeder outage hours per AREA:

    area_outage_hours = mean outage hours of the area's SHEDDABLE feeders

The hospital feeder can never be switched off, so it is excluded from this mean;
otherwise its area (North Karachi) would look artificially "better treated" just
because of the hospital. This also removes any dependence on how many feeders an
area has, so raw totals never favour an area simply for having more feeders.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import config
from .data_loader import DataBundle, parse_protected_hours


def longest_off_run(row) -> int:
    """Length of the longest run of consecutive OFF (0) values in one feeder's 24-hour row."""
    longest = current = 0
    for value in row:
        if value == config.OFF:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return longest


class LoadSheddingEnvironment:
    """Holds the static network description and one loaded scenario."""

    def __init__(self, data: DataBundle) -> None:
        feeders = data.feeders.sort_values("feeder_id").reset_index(drop=True)

        # ---- static feeder information (arrays are indexed by feeder position 0..9) ----
        self.n_hours = config.HOURS_PER_SCENARIO
        self.feeder_ids: list[str] = feeders["feeder_id"].tolist()
        self.n_feeders = len(self.feeder_ids)
        self.feeder_index = {fid: i for i, fid in enumerate(self.feeder_ids)}
        self.feeder_names: list[str] = feeders["feeder_name"].tolist()
        self.feeder_area: list[str] = feeders["area"].tolist()
        self.areas: list[str] = list(dict.fromkeys(self.feeder_area))   # order of first appearance
        self.area_index = {a: k for k, a in enumerate(self.areas)}
        self.capacity = feeders["feeder_capacity_mw"].to_numpy(dtype=float)
        self.priority = feeders["priority"].to_numpy(dtype=int)
        self.priority_weight = feeders["priority_weight"].to_numpy(dtype=float)
        self.is_critical = feeders["is_critical"].to_numpy(dtype=bool)
        self.max_total_outage = feeders["max_total_outage_hours"].to_numpy(dtype=int)
        self.max_consecutive_outage = feeders["max_consecutive_outage_hours"].to_numpy(dtype=int)
        self.protected_hours: list[frozenset[int]] = [parse_protected_hours(t) for t in feeders["protected_hours"]]
        self.hospital_index = self.feeder_index[config.HOSPITAL_FEEDER_ID]

        # must_stay_on[i, t] is True when the CSP domain of x[i,t] is reduced to {1}:
        # hospital feeder, protected hours, or a feeder that is allowed no outage at all.
        self.must_stay_on = np.zeros((self.n_feeders, self.n_hours), dtype=bool)
        for i in range(self.n_feeders):
            if self.is_critical[i] or i == self.hospital_index or self.max_total_outage[i] == 0:
                self.must_stay_on[i, :] = True
            for hour in self.protected_hours[i]:
                self.must_stay_on[i, hour] = True

        # Area aggregation matrix (areas x feeders): each SHEDDABLE feeder has weight 1/(number
        # of sheddable feeders in its area). area_outage = area_weights @ feeder_outage_hours.
        self.sheddable = (~self.must_stay_on).any(axis=1)
        self.area_weights = np.zeros((len(self.areas), self.n_feeders))
        for k, area in enumerate(self.areas):
            members = [i for i in range(self.n_feeders) if self.feeder_area[i] == area and self.sheddable[i]]
            for i in members:
                self.area_weights[k, i] = 1.0 / len(members)

        # ---- pre-arranged per-scenario arrays (built once, so loading a scenario is instant) ----
        self._actual_by_scenario: dict[str, np.ndarray] = {}
        self._supply_by_scenario: dict[str, dict[str, np.ndarray]] = {}
        self._build_scenario_arrays(data)
        self._predictions_by_scenario: dict[str, np.ndarray] = {}
        self.planning_source = "actual demand (no forecast loaded)"

        # ---- currently loaded scenario ----
        self.scenario_id: Optional[str] = None
        self.actual_demand = np.zeros((self.n_feeders, self.n_hours))
        self.planning_demand = np.zeros((self.n_feeders, self.n_hours))
        self.available_supply = np.zeros(self.n_hours)
        self.reserve = np.zeros(self.n_hours)
        self.usable_supply = np.zeros(self.n_hours)
        self.outage_history: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ set-up
    def _build_scenario_arrays(self, data: DataBundle) -> None:
        """Reshape the long CSV tables into (feeders x hours) arrays, one per scenario."""
        ordered = data.demand.sort_values(["scenario_id", "feeder_id", "hour"])
        for scenario_id, group in ordered.groupby("scenario_id"):
            if list(group["feeder_id"].unique()) != self.feeder_ids:
                raise ValueError(f"Scenario {scenario_id}: feeder list does not match feeders.csv")
            self._actual_by_scenario[scenario_id] = (
                group["actual_demand_mw"].to_numpy(dtype=float).reshape(self.n_feeders, self.n_hours))
        for scenario_id, group in data.supply.sort_values(["scenario_id", "hour"]).groupby("scenario_id"):
            self._supply_by_scenario[scenario_id] = {
                "available": group["available_supply_mw"].to_numpy(dtype=float),
                "reserve": group["reserve_mw"].to_numpy(dtype=float),
                "usable": group["usable_supply_mw"].to_numpy(dtype=float),
            }

    def set_predictions(self, predictions: pd.DataFrame, column: str = "regression_prediction_mw") -> None:
        """Store demand forecasts (from DemandPredictor) so scenarios are planned with FORECAST demand."""
        ordered = predictions.sort_values(["scenario_id", "feeder_id", "hour"])
        self._predictions_by_scenario = {}
        for scenario_id, group in ordered.groupby("scenario_id"):
            if len(group) != self.n_feeders * self.n_hours:
                raise ValueError(f"Predictions for {scenario_id} are incomplete ({len(group)} rows)")
            values = np.maximum(group[column].to_numpy(dtype=float), 0.0)
            self._predictions_by_scenario[scenario_id] = values.reshape(self.n_feeders, self.n_hours)
        self.planning_source = f"forecast column '{column}'"

    # ---------------------------------------------------------------- scenarios
    def load_scenario(self, scenario_id: str, use_predictions: bool = True) -> None:
        """Load one scenario (all 24 hourly slots). Clears the outage history."""
        if scenario_id not in self._actual_by_scenario:
            raise KeyError(f"Unknown scenario id: {scenario_id}")
        self.scenario_id = scenario_id
        self.actual_demand = self._actual_by_scenario[scenario_id].copy()
        supply = self._supply_by_scenario[scenario_id]
        self.available_supply = supply["available"].copy()
        self.reserve = supply["reserve"].copy()
        self.usable_supply = supply["usable"].copy()
        if use_predictions and scenario_id in self._predictions_by_scenario:
            self.planning_demand = self._predictions_by_scenario[scenario_id].copy()
        else:
            self.planning_demand = self.actual_demand.copy()
        self.outage_history = {}

    def demand_for(self, basis: str) -> np.ndarray:
        """Return the demand matrix for basis 'planning' (forecast) or 'actual'."""
        if basis == "planning":
            return self.planning_demand
        if basis == "actual":
            return self.actual_demand
        raise ValueError("basis must be 'planning' or 'actual'")

    # ------------------------------------------------------------- constraints
    def get_feeder_constraints(self) -> dict[str, dict]:
        """Return every feeder's constraints as a dictionary (useful for reports and tests)."""
        constraints = {}
        for i, fid in enumerate(self.feeder_ids):
            constraints[fid] = {
                "area": self.feeder_area[i],
                "priority": int(self.priority[i]),
                "priority_weight": float(self.priority_weight[i]),
                "is_critical": bool(self.is_critical[i]),
                "max_total_outage_hours": int(self.max_total_outage[i]),
                "max_consecutive_outage_hours": int(self.max_consecutive_outage[i]),
                "protected_hours": sorted(self.protected_hours[i]),
                "capacity_mw": float(self.capacity[i]),
            }
        return constraints

    def get_violations(self, schedule: np.ndarray, basis: str = "planning",
                       stop_at_first: bool = False) -> list[str]:
        """List every HARD-constraint violation of a schedule (empty list = valid).

        Hard constraints (checked separately from any objective score):
          1. schedule has shape (feeders, 24) and contains only 0/1 values
          2. hospital feeder is always ON
          3. every protected hour is ON
          4. total OFF hours <= max_total_outage_hours (per feeder)
          5. consecutive OFF hours <= max_consecutive_outage_hours (per feeder)
          6. served demand <= usable supply in every hour
        """
        problems: list[str] = []

        def add(message: str) -> bool:
            problems.append(message)
            return stop_at_first

        schedule = np.asarray(schedule)
        if schedule.shape != (self.n_feeders, self.n_hours):
            problems.append(f"schedule has shape {schedule.shape}, expected {(self.n_feeders, self.n_hours)}")
            return problems
        if not np.isin(schedule, (config.ON, config.OFF)).all():
            problems.append("schedule contains values other than 0 (OFF) and 1 (ON)")
            return problems

        if (schedule[self.hospital_index] != config.ON).any():
            if add(f"hospital feeder {config.HOSPITAL_FEEDER_ID} is OFF in at least one hour"):
                return problems
        if (schedule[self.must_stay_on] != config.ON).any():
            if add("a protected / must-stay-on feeder-hour is OFF"):
                return problems

        outage_hours = (schedule == config.OFF).sum(axis=1)
        for i in np.where(outage_hours > self.max_total_outage)[0]:
            if add(f"{self.feeder_ids[i]}: {outage_hours[i]} outage hours > limit {self.max_total_outage[i]}"):
                return problems
        for i in range(self.n_feeders):
            run = longest_off_run(schedule[i])
            if run > self.max_consecutive_outage[i]:
                if add(f"{self.feeder_ids[i]}: {run} consecutive outage hours > limit "
                       f"{self.max_consecutive_outage[i]}"):
                    return problems

        served = (schedule * self.demand_for(basis)).sum(axis=0)
        for t in np.where(served > self.usable_supply + config.SUPPLY_TOLERANCE)[0]:
            if add(f"hour {t}: served demand {served[t]:.3f} MW > usable supply {self.usable_supply[t]:.3f} MW"):
                return problems
        return problems

    def is_valid(self, schedule: np.ndarray, basis: str = "planning") -> bool:
        """True when the schedule satisfies ALL hard constraints (see get_violations)."""
        return len(self.get_violations(schedule, basis, stop_at_first=True)) == 0

    # ---------------------------------------------------------------- shortage
    def calculate_shortage(self, schedule: np.ndarray, basis: str = "actual") -> dict:
        """Hour-by-hour supply/demand figures for a schedule.

        For every hour t:
            served_demand[t]     = sum of demand of feeders that are ON
            unmet_demand[t]      = sum of demand of feeders that are OFF
            network_shortage[t]  = max(0, total_demand[t] - usable_supply[t])
                                   (how much the network is short BEFORE any switching)
            supply_overload[t]   = max(0, served_demand[t] - usable_supply[t])
                                   (> 0 means the schedule would draw more than the usable supply)
            excess_outage[t]     = unmet_demand[t] - network_shortage[t]
                                   (outage beyond the unavoidable minimum; >= 0 for valid schedules)

        Note: network shortage and unmet demand are different things. Switching feeders OFF
        is the way the shortage is removed, and the binary model usually has to switch off a
        little more demand than the exact shortage because feeders are all-or-nothing.
        """
        demand = self.demand_for(basis)
        schedule = np.asarray(schedule)
        served = (schedule * demand).sum(axis=0)
        unmet = ((1 - schedule) * demand).sum(axis=0)
        total = demand.sum(axis=0)
        network_shortage = np.maximum(0.0, total - self.usable_supply)
        overload = np.maximum(0.0, served - self.usable_supply)
        return {
            "total_demand_mw": total,
            "served_demand_mw": served,
            "unmet_demand_mw": unmet,
            "network_shortage_mw": network_shortage,
            "supply_overload_mw": overload,
            "excess_outage_mw": unmet - network_shortage,
            "total_unmet_mwh": float(unmet.sum()),                 # 1 hour slots: MW summed = MWh
            "mean_unmet_per_hour_mw": float(unmet.mean()),
            "total_network_shortage_mwh": float(network_shortage.sum()),
            "overload_hours": int((overload > config.SUPPLY_TOLERANCE).sum()),
        }

    # ----------------------------------------------------------- outage hours
    def get_feeder_outage_hours(self, schedule: np.ndarray) -> np.ndarray:
        """Number of OFF hours for each feeder."""
        return (np.asarray(schedule) == config.OFF).sum(axis=1)

    def get_area_outage_hours(self, schedule: np.ndarray, per_feeder: bool = True) -> pd.Series:
        """Outage hours per area.

        per_feeder=True  (default, used for fairness): mean outage hours of the area's sheddable feeders.
        per_feeder=False : plain sum of the outage hours of all feeders in the area.
        """
        feeder_hours = self.get_feeder_outage_hours(schedule).astype(float)
        if per_feeder:
            values = self.area_weights @ feeder_hours
        else:
            values = np.array([feeder_hours[[i for i in range(self.n_feeders)
                                             if self.feeder_area[i] == area]].sum() for area in self.areas])
        return pd.Series(values, index=self.areas, name="outage_hours")

    def outage_imbalance(self, schedule: np.ndarray) -> float:
        """Fairness metric: standard deviation of the area outage hours (0 = perfectly equal).

            imbalance = sqrt( mean_k (a_k - mean(a))^2 )     where a_k = area_outage_hours of area k
        """
        return float(np.std(self.area_weights @ self.get_feeder_outage_hours(schedule).astype(float)))

    # ---------------------------------------------------------- outage history
    def record_outage_history(self, method: str, schedule: np.ndarray) -> None:
        """Remember a method's schedule for the loaded scenario."""
        self.outage_history[method] = np.asarray(schedule).copy()

    def schedule_to_frame(self, schedule: np.ndarray) -> pd.DataFrame:
        """Long table (one row per feeder-hour) describing a schedule, using ACTUAL demand for served/unmet."""
        rows = []
        for i, fid in enumerate(self.feeder_ids):
            for t in range(self.n_hours):
                on = schedule[i, t] == config.ON
                rows.append({
                    "scenario_id": self.scenario_id,
                    "hour": t,
                    "feeder_id": fid,
                    "area": self.feeder_area[i],
                    "predicted_demand_mw": round(float(self.planning_demand[i, t]), 3),
                    "actual_demand_mw": round(float(self.actual_demand[i, t]), 3),
                    "decision": "ON" if on else "OFF",
                    "served_demand_mw": round(float(self.actual_demand[i, t]), 3) if on else 0.0,
                    "unmet_demand_mw": 0.0 if on else round(float(self.actual_demand[i, t]), 3),
                })
        return pd.DataFrame(rows).sort_values(["hour", "feeder_id"]).reset_index(drop=True)
