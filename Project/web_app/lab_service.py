"""Explicit, bounded live algorithm demonstrations; never write project outputs."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.csp_scheduler import CSPScheduler
from src.data_loader import DataLoader
from src.environment import LoadSheddingEnvironment
from src.hill_climbing_scheduler import HillClimbingScheduler

_lock = threading.Lock()


def run_lab(scenario_id: str, method: str = "CSP") -> dict:
    """Run real code with an optional observer; results are separate from saved runs."""
    if method not in ("CSP", "HC"):
        raise ValueError("Choose CSP or HC")
    if not _lock.acquire(blocking=False):
        raise RuntimeError("An algorithm demonstration is already running. Try again when it finishes.")
    try:
        data = DataLoader().load()
        if scenario_id not in data.scenario_ids:
            raise ValueError("Unknown scenario")
        env = LoadSheddingEnvironment(data)
        env.set_predictions(pd.read_csv(ROOT / "outputs/predictions/demand_predictions.csv"))
        env.load_scenario(scenario_id)
        events = []
        count = 0
        started = time.perf_counter()

        def observe(event):
            nonlocal count
            count += 1
            if len(events) < 600:
                events.append({"step": count, "elapsed_ms": round((time.perf_counter() - started) * 1000, 3), **event})

        # HC begins from an actual fresh CSP schedule, not an invented starting state.
        csp = CSPScheduler(env, event_callback=observe if method == "CSP" else None).generate_schedule()
        result = csp
        if method == "HC" and csp.is_feasible:
            observe({"event": "csp_start", "status": csp.status,
                     "runtime_s": csp.runtime_s, "nodes": csp.details.get("nodes")})
            result = HillClimbingScheduler(env, initial_schedule=csp.schedule,
                                          event_callback=observe).generate_schedule()
        elif method == "HC":
            observe({"event": "skipped", "reason": "No feasible CSP starting schedule", "status": csp.status})
        summary = dict(result.details)
        if result.is_feasible:
            summary["constraint_violations"] = len(env.get_violations(result.schedule, "planning"))
            summary["actual_unmet_mwh"] = env.calculate_shortage(result.schedule, "actual")["total_unmet_mwh"]
        return {
            "scenario_id": scenario_id, "method": method, "status": result.status,
            "feeder_ids": env.feeder_ids,
            "final_schedule": result.schedule.tolist() if result.is_feasible else None,
            "runtime_s": result.runtime_s, "total_runtime_s": time.perf_counter() - started,
            "reason": result.reason, "summary": summary,
            "trace": events, "trace_truncated": count > len(events),
            "event_count": count, "trace_limit": 600,
            "note": "Fresh execution using saved forecasts. Trace records real events, capped at 600. "
                    "Historical CSV metrics are from the original saved experiment; timings may differ. "
                    "No detailed search history was stored with those CSVs. No datasets or saved outputs were modified.",
        }
    finally:
        _lock.release()
