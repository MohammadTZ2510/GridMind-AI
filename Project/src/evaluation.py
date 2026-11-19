"""
evaluation.py
=============
`EvaluationReport` - runs CSP and Hill Climbing on the SAME seeded scenarios, collects
metrics, compares the methods, creates the charts and writes all result files.

Metrics (per scenario and method)
---------------------------------
Planning is done with the FORECAST demand; results are measured with the ACTUAL demand.
  * runtime_s                 wall-clock seconds (Hill Climbing: improvement step only;
                              total_runtime_incl_csp_s adds the CSP start)
  * actual_unmet_mwh          sum over hours of the demand of OFF feeders (MWh per day)
  * mean_unmet_per_hour_mw    actual_unmet_mwh / 24
  * network_shortage_mwh      sum over hours of max(0, total demand - usable supply): the minimum that
                              had to be shed. excess_outage_mwh = unmet - shortage is over-shedding
                              caused by the all-or-nothing (binary) feeders and by forecast error.
  * supply_overload_hours     hours in which the served ACTUAL demand exceeded usable supply (can only
                              happen because the plan used forecast demand)
  * outage hours by feeder / by area (area = mean over its sheddable feeders)
  * area_outage_imbalance     FAIRNESS = standard deviation of the area outage hours (lower = fairer)
  * hard_constraint_violations number of violated hard constraints on the planning demand (must be 0)
Feasibility counts (feasible / infeasible / unresolved) are reported, never hidden.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import config, visualization
from .csp_scheduler import CSPScheduler
from .environment import LoadSheddingEnvironment
from .hill_climbing_scheduler import HillClimbingScheduler
from .scheduler_base import STATUS_FEASIBLE, ScheduleResult

# (metric column, label, True if a LOWER value is better)
SUMMARY_METRICS = [
    ("runtime_s", "Runtime (s)", True),
    ("total_runtime_incl_csp_s", "Runtime incl. CSP start (s)", True),
    ("actual_unmet_mwh", "Total unmet demand (MWh/day, actual)", True),
    ("mean_unmet_per_hour_mw", "Mean unmet demand per hour (MW)", True),
    ("excess_outage_mwh", "Over-shedding beyond network shortage (MWh/day)", True),
    ("area_outage_imbalance", "Outage-hour imbalance between areas (std, hours)", True),
    ("total_outage_hours", "Total feeder outage hours per day", True),
    ("supply_overload_hours", "Hours with actual served demand > usable supply", True),
    ("hard_constraint_violations", "Hard-constraint violations (planning demand)", True),
    ("objective_score", "Objective score (planning demand)", True),
]


class EvaluationReport:
    """Runs the experiments and reports the results."""

    def __init__(self, environment: LoadSheddingEnvironment, predictions: pd.DataFrame,
                 regression_metrics: pd.DataFrame, scenario_ids: Optional[list[str]] = None,
                 log: Callable[[str], None] = print) -> None:
        self.env = environment
        self.predictions = predictions
        self.regression_metrics = regression_metrics
        self.scenario_ids = list(scenario_ids) if scenario_ids is not None else list(config.SCENARIO_IDS)
        self.log = log
        self.rows: list[dict] = []
        self.csp_frames: list[pd.DataFrame] = []
        self.hc_frames: list[pd.DataFrame] = []
        self.infeasible_rows: list[dict] = []
        self.results = pd.DataFrame()
        self.summary = pd.DataFrame()

    # ---------------------------------------------------------------- metrics
    def _metric_row(self, result: ScheduleResult, extra_runtime: float = 0.0) -> dict:
        """Metrics of one method on the loaded scenario. Infeasible results only record their status."""
        env = self.env
        row = {"scenario_id": result.scenario_id, "method": result.method, "status": result.status,
               "feasible": result.is_feasible, "runtime_s": result.runtime_s,
               "total_runtime_incl_csp_s": result.runtime_s + extra_runtime}
        row.update({k: result.details.get(k) for k in ("nodes", "backtracks", "restarts", "iterations")})
        if not result.is_feasible:
            return row

        schedule = result.schedule
        planned = env.calculate_shortage(schedule, "planning")
        actual = env.calculate_shortage(schedule, "actual")
        feeder_hours = env.get_feeder_outage_hours(schedule)
        area_hours = env.get_area_outage_hours(schedule)
        violations = env.get_violations(schedule, "planning")
        row.update({
            "planned_unmet_mwh": planned["total_unmet_mwh"],
            "actual_unmet_mwh": actual["total_unmet_mwh"],
            "mean_unmet_per_hour_mw": actual["mean_unmet_per_hour_mw"],
            "network_shortage_mwh": actual["total_network_shortage_mwh"],
            "excess_outage_mwh": actual["total_unmet_mwh"] - actual["total_network_shortage_mwh"],
            "supply_overload_hours": actual["overload_hours"],
            "total_outage_hours": int(feeder_hours.sum()),
            "area_outage_imbalance": env.outage_imbalance(schedule),
            "hard_constraint_violations": len(violations),
            "objective_score": result.details.get("final_score"),
        })
        for fid, hours in zip(env.feeder_ids, feeder_hours):
            row[f"outage_hours_{fid}"] = int(hours)
        for area, hours in area_hours.items():
            row[f"area_outage_{area}"] = float(hours)
        return row

    # ------------------------------------------------------------------ trials
    def run_trials(self) -> pd.DataFrame:
        """Run CSP and then Hill Climbing (from the CSP schedule) on every scenario."""
        self.env.set_predictions(self.predictions)
        for number, scenario_id in enumerate(self.scenario_ids, start=1):
            self.env.load_scenario(scenario_id)
            csp_result = CSPScheduler(self.env).generate_schedule()
            csp_row = self._metric_row(csp_result)
            self.rows.append(csp_row)

            if not csp_result.is_feasible:
                self.infeasible_rows.append({
                    "scenario_id": scenario_id, "status": csp_result.status, "cause": csp_result.cause,
                    "reason": csp_result.reason, "csp_nodes": csp_result.details.get("nodes"),
                    "csp_runtime_s": round(csp_result.runtime_s, 6),
                    "total_required_shed_mwh": csp_result.details.get("total_required_shed_mwh"),
                    "hill_climbing": "not run (no valid starting schedule)"})
                self.log(f"  [{number:02d}/{len(self.scenario_ids)}] {scenario_id}: CSP {csp_result.status} "
                         f"({csp_result.cause}) - Hill Climbing skipped")
                continue

            hc_result = HillClimbingScheduler(self.env, initial_schedule=csp_result.schedule).generate_schedule()
            # Hill Climbing's initial score IS the CSP schedule's objective score (same score function).
            csp_row["objective_score"] = hc_result.details["initial_score"]
            self.rows.append(self._metric_row(hc_result, extra_runtime=csp_result.runtime_s))
            self.env.record_outage_history("CSP", csp_result.schedule)
            self.env.record_outage_history("HillClimbing", hc_result.schedule)
            self.csp_frames.append(self.env.schedule_to_frame(csp_result.schedule))
            self.hc_frames.append(self.env.schedule_to_frame(hc_result.schedule))
            self.log(f"  [{number:02d}/{len(self.scenario_ids)}] {scenario_id}: CSP ok ({csp_result.runtime_s*1000:.1f} ms, "
                     f"{csp_result.details.get('nodes')} nodes) | Hill Climbing score "
                     f"{hc_result.details['initial_score']:.4f} -> {hc_result.details['final_score']:.4f} "
                     f"({hc_result.runtime_s*1000:.0f} ms)")
        self.results = pd.DataFrame(self.rows)
        return self.results

    # ------------------------------------------------------------- comparison
    def compare_methods(self) -> pd.DataFrame:
        """Mean/std of every metric for both methods on the scenarios where BOTH ran, plus paired wins."""
        if self.results.empty:
            raise RuntimeError("Call run_trials() first")
        feasible = self.results[self.results["feasible"]]
        csp = feasible[feasible["method"] == "CSP"].set_index("scenario_id")
        hc = feasible[feasible["method"] == "HillClimbing"].set_index("scenario_id")
        common = csp.index.intersection(hc.index)
        csp, hc = csp.loc[common], hc.loc[common]

        rows = []
        for column, label, lower_is_better in SUMMARY_METRICS:
            csp_values = csp[column].astype(float)
            hc_values = hc[column].astype(float)
            diff = hc_values - csp_values
            rows.append({
                "metric": column, "description": label, "n_scenarios": len(common),
                "csp_mean": csp_values.mean(), "csp_std": csp_values.std(ddof=0),
                "hc_mean": hc_values.mean(), "hc_std": hc_values.std(ddof=0),
                "hc_minus_csp_mean": diff.mean(),
                "hc_better_in": int((diff < -1e-9).sum()) if lower_is_better else int((diff > 1e-9).sum()),
                "hc_equal_in": int((diff.abs() <= 1e-9).sum()),
                "hc_worse_in": int((diff > 1e-9).sum()) if lower_is_better else int((diff < -1e-9).sum()),
            })
        self.summary = pd.DataFrame(rows)
        return self.summary

    # ------------------------------------------------------------------ charts
    def create_charts(self) -> list[Path]:
        """Create the four chart files in outputs/charts/."""
        if self.results.empty:
            raise RuntimeError("Call run_trials() first")
        feeder = "F003" if "F003" in self.env.feeder_ids else self.env.feeder_ids[1]
        scenario = self.scenario_ids[0]
        return [
            visualization.plot_actual_vs_predicted(self.predictions, config.CHART_ACTUAL_VS_PREDICTED,
                                                   scenario, feeder),
            visualization.plot_runtime(self.results, config.CHART_RUNTIME),
            visualization.plot_shortage(self.results, config.CHART_SHORTAGE),
            visualization.plot_outage_by_area(self.results, self.env.areas, config.CHART_OUTAGE_BY_AREA),
        ]

    # ------------------------------------------------------------------ saving
    def save_outputs(self) -> None:
        """Write schedules, infeasible-scenario list and metric tables."""
        config.ensure_output_dirs()
        csp_all = pd.concat(self.csp_frames, ignore_index=True) if self.csp_frames else pd.DataFrame()
        hc_all = pd.concat(self.hc_frames, ignore_index=True) if self.hc_frames else pd.DataFrame()
        csp_all.to_csv(config.CSP_SCHEDULES_CSV, index=False)
        hc_all.to_csv(config.HC_SCHEDULES_CSV, index=False)
        infeasible_columns = ["scenario_id", "status", "cause", "reason", "csp_nodes", "csp_runtime_s",
                              "total_required_shed_mwh", "hill_climbing"]
        pd.DataFrame(self.infeasible_rows, columns=infeasible_columns).to_csv(config.INFEASIBLE_CSV, index=False)
        self.regression_metrics.to_csv(config.REGRESSION_METRICS_CSV, index=False)
        self.results.round(6).to_csv(config.SCHEDULING_METRICS_CSV, index=False)
        self.summary.round(6).to_csv(config.SCHEDULING_SUMMARY_CSV, index=False)

    def write_summary_text(self, extra_lines: Optional[list[str]] = None) -> str:
        """Human-readable experiment summary (every number comes from the results tables)."""
        results, summary = self.results, self.summary
        csp_rows = results[results["method"] == "CSP"]
        n_total = len(csp_rows)
        n_feasible = int(csp_rows["feasible"].sum())
        n_infeasible = int((csp_rows["status"] == "INFEASIBLE").sum())
        n_unresolved = int((csp_rows["status"] == "LIMIT_REACHED").sum())
        overall = self.regression_metrics[self.regression_metrics["fold"] == "overall"]

        lines = ["=" * 78, " EXPERIMENT SUMMARY - AI-Based Electricity Demand Prediction and Fair Load-Shedding",
                 "=" * 78, "", "*** ALL DATA IS SIMULATED DATA - NOT real utility measurements. ***", "",
                 f"Generated on : {time.strftime('%Y-%m-%d %H:%M:%S')}",
                 f"Random seed  : {config.RANDOM_SEED}",
                 f"Scenarios    : {n_total}   (areas: {config.NUM_AREAS}, feeders: {config.NUM_FEEDERS}, "
                 f"hours: {config.HOURS_PER_SCENARIO})",
                 f"Planning demand: {self.env.planning_source} (scenario-grouped out-of-fold predictions)",
                 "", "-" * 78, " 1. DEMAND FORECASTING (all 50 scenarios, out-of-fold, MW)", "-" * 78]
        for row in overall.itertuples():
            lines.append(f"{row.model:<28} MAE = {row.MAE_MW:7.3f}   RMSE = {row.RMSE_MW:7.3f}")
        reg = overall[overall["model"].str.startswith("Regression")].iloc[0]
        prev = overall[overall["model"].str.startswith("Previous")].iloc[0]
        mean_b = overall[overall["model"].str.startswith("Mean")].iloc[0]
        lines.append(f"Regression MAE is {100*(1-reg.MAE_MW/prev.MAE_MW):.1f}% lower than the previous-hour baseline "
                     f"and {100*(1-reg.MAE_MW/mean_b.MAE_MW):.1f}% lower than the mean-demand baseline.")

        lines += ["", "-" * 78, " 2. FEASIBILITY OF THE CSP SCHEDULER", "-" * 78,
                  f"Feasible scenarios (valid schedule found) : {n_feasible} of {n_total}",
                  f"Infeasible scenarios (proven)             : {n_infeasible}",
                  f"Unresolved scenarios (search limit)       : {n_unresolved}"]
        for row in self.infeasible_rows:
            lines.append(f"  - {row['scenario_id']} [{row['status']}, {row['cause']}]: {row['reason']}")

        lines += ["", "-" * 78, f" 3. CSP vs HILL CLIMBING (same {int(summary['n_scenarios'].iloc[0]) if len(summary) else 0} "
                  "feasible scenarios; mean +/- std)", "-" * 78]
        for row in summary.itertuples():
            lines.append(f"{row.description:<52} CSP {row.csp_mean:9.4f} +/- {row.csp_std:7.4f} | "
                         f"HC {row.hc_mean:9.4f} +/- {row.hc_std:7.4f} | HC better/equal/worse: "
                         f"{row.hc_better_in}/{row.hc_equal_in}/{row.hc_worse_in}")

        by_metric = summary.set_index("metric")
        lines += ["", "Interpretation (computed from the results above):"]
        unmet = by_metric.loc["actual_unmet_mwh"]
        fair = by_metric.loc["area_outage_imbalance"]
        lines.append(f"  * Hill Climbing changed mean unmet demand from {unmet.csp_mean:.1f} to {unmet.hc_mean:.1f} MWh/day "
                     f"(lower in {unmet.hc_better_in} of {int(unmet.n_scenarios)} scenarios).")
        lines.append(f"  * It changed the mean outage-hour imbalance from {fair.csp_mean:.3f} to {fair.hc_mean:.3f} hours "
                     f"(fairer in {fair.hc_better_in}, equal in {fair.hc_equal_in}, less fair in {fair.hc_worse_in}).")
        lines.append("  * Hill Climbing is a local search: it stops at a local optimum and is not claimed to be globally optimal.")
        feasible_rows = results[results["feasible"]]
        median_csp = feasible_rows[feasible_rows["method"] == "CSP"]["runtime_s"].median() * 1000
        median_hc = feasible_rows[feasible_rows["method"] == "HillClimbing"]["runtime_s"].median() * 1000
        slowest = feasible_rows[feasible_rows["method"] == "CSP"].nlargest(1, "runtime_s").iloc[0]
        lines.append(f"  * Median runtime: CSP {median_csp:.1f} ms, Hill Climbing improvement step {median_hc:.1f} ms. "
                     f"The CSP mean is raised by a few tight scenarios (slowest: {slowest['scenario_id']}, "
                     f"{slowest['runtime_s']:.2f} s, {int(slowest['nodes'])} search nodes).")
        violations = int(results["hard_constraint_violations"].fillna(0).sum())
        lines.append(f"  * Hard-constraint violations in all returned schedules (planning demand): {violations}.")
        overload = by_metric.loc["supply_overload_hours"]
        lines.append(f"  * Because plans use FORECAST demand, the actual demand exceeded usable supply in "
                     f"{overload.csp_mean:.2f} (CSP) / {overload.hc_mean:.2f} (HC) hours per scenario on average.")
        if extra_lines:
            lines += [""] + extra_lines
        lines += ["", "Fairness definition: area outage hours = mean outage hours of the area's sheddable feeders; "
                  "imbalance = standard deviation of the 5 area values (0 = perfectly equal).",
                  "Unmet demand = demand of feeders that are OFF; network shortage = max(0, demand - usable supply).",
                  "=" * 78, ""]
        text = "\n".join(lines)
        config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
        config.EXPERIMENT_SUMMARY_TXT.write_text(text, encoding="utf-8")
        return text
