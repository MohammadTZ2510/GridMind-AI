"""
scheduler_base.py
=================
The common `Scheduler` base class. `CSPScheduler` and `HillClimbingScheduler`
both inherit from it and override `generate_schedule()`.

Shared logic kept here (so it is not duplicated):
  * schedule validation       -> validate_schedule()
  * shortage calculation      -> calculate_shortage()
  * outage-hour calculation   -> outage_hours_by_feeder(), outage_hours_by_area()
  * area-level fairness       -> area_fairness()
  * runtime measurement       -> timer()
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from .environment import LoadSheddingEnvironment

# Possible values of ScheduleResult.status
STATUS_FEASIBLE = "FEASIBLE"
STATUS_INFEASIBLE = "INFEASIBLE"
STATUS_LIMIT_REACHED = "LIMIT_REACHED"     # search stopped by a node/time limit, outcome unknown


@dataclass
class ScheduleResult:
    """What a scheduler returns for one scenario."""

    scenario_id: str
    method: str
    status: str
    schedule: Optional[np.ndarray] = None       # None unless a VALID schedule was found
    reason: str = ""                            # explanation for infeasible / limit cases
    cause: str = ""                             # 'supply', 'constraint interaction', 'search limit', ...
    runtime_s: float = 0.0
    details: dict = field(default_factory=dict)  # method-specific statistics

    @property
    def is_feasible(self) -> bool:
        return self.status == STATUS_FEASIBLE and self.schedule is not None


class Scheduler(ABC):
    """Base class: common interface and shared utilities for all scheduling methods."""

    def __init__(self, environment: LoadSheddingEnvironment, name: str) -> None:
        self.env = environment
        self.name = name

    # ----------------------------------------------------------- main interface
    @abstractmethod
    def generate_schedule(self) -> ScheduleResult:
        """Create a schedule for the scenario currently loaded in the environment."""

    # -------------------------------------------------------- shared utilities
    def validate_schedule(self, schedule: np.ndarray, basis: str = "planning") -> list[str]:
        """Return the list of hard-constraint violations (empty list = valid)."""
        return self.env.get_violations(schedule, basis)

    def calculate_shortage(self, schedule: np.ndarray, basis: str = "actual") -> dict:
        """Served / unmet / shortage figures (see LoadSheddingEnvironment.calculate_shortage)."""
        return self.env.calculate_shortage(schedule, basis)

    def outage_hours_by_feeder(self, schedule: np.ndarray) -> np.ndarray:
        return self.env.get_feeder_outage_hours(schedule)

    def outage_hours_by_area(self, schedule: np.ndarray) -> pd.Series:
        return self.env.get_area_outage_hours(schedule)

    def area_fairness(self, schedule: np.ndarray) -> float:
        """Outage imbalance between areas (standard deviation of area outage hours; lower = fairer)."""
        return self.env.outage_imbalance(schedule)

    @staticmethod
    @contextmanager
    def timer() -> Iterator[dict]:
        """Measure wall-clock runtime:   with Scheduler.timer() as t: ...;   t['seconds']"""
        record = {"seconds": 0.0}
        start = time.perf_counter()
        try:
            yield record
        finally:
            record["seconds"] = time.perf_counter() - start

    def _result(self, status: str, schedule: Optional[np.ndarray], runtime_s: float,
                reason: str = "", cause: str = "", **details) -> ScheduleResult:
        """Small helper so subclasses build results in one consistent way."""
        return ScheduleResult(
            scenario_id=str(self.env.scenario_id), method=self.name, status=status,
            schedule=None if schedule is None else np.asarray(schedule, dtype=int),
            reason=reason, cause=cause, runtime_s=runtime_s, details=dict(details),
        )
