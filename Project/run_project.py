"""
run_project.py
==============
Main entry point:   python run_project.py

Runs the complete AL-2002 pipeline on the EXISTING simulated datasets
(they are never regenerated or modified):

 1. configuration + seeds        6. CSP Backtracking on every scenario
 2. load and validate data       7. Hill Climbing from each valid CSP schedule
 3. Regression + baselines       8. compare methods
 4. demand predictions           9. charts
 5. load scenarios               10. save schedules, metrics, reports

Optional:   python run_project.py --max-scenarios 10     (quick demo on the first 10 scenarios;
            the regression step always uses all 50 scenarios)

ALL DATA IS SIMULATED. This is a decision-support simulation, not a grid controller.
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from src import config
from src.data_loader import DataLoader, DataValidationError
from src.demand_predictor import DemandPredictor
from src.environment import LoadSheddingEnvironment
from src.evaluation import EvaluationReport

BANNER = "=" * 70


def step(number: int, total: int, text: str) -> None:
    print(f"\n[{number}/{total}] {text}")


def main(max_scenarios: int | None = None) -> int:
    started = time.perf_counter()
    print(BANNER)
    print(" AL-2002  AI-Based Electricity Demand Prediction and Fair Load-Shedding")
    print(" (ALL DATA IS SIMULATED - not real utility data)")
    print(BANNER)

    total_steps = 8

    # ---- 1. configuration and seeds ------------------------------------------------------
    step(1, total_steps, "Initialising configuration and random seed...")
    config.set_global_seed(config.RANDOM_SEED)
    config.ensure_output_dirs()
    print(f"      project root : {config.PROJECT_ROOT}")
    print(f"      random seed  : {config.RANDOM_SEED}")

    # ---- 2. data -------------------------------------------------------------------------
    step(2, total_steps, "Loading and validating the existing datasets (no data is regenerated)...")
    try:
        data = DataLoader().load()
    except DataValidationError as error:
        print(f"\nDATA ERROR: {error}")
        return 1
    passed = sum(line.startswith("[PASS]") for line in data.validation_log)
    print(f"      feeders={len(data.feeders)}  demand rows={len(data.demand)}  "
          f"supply rows={len(data.supply)}  weather rows={len(data.weather)}")
    print(f"      all {passed} data validation checks passed")
    for mapping in data.column_mappings:
        print(f"      column mapping used: {mapping}")

    # ---- 3/4. regression -------------------------------------------------------------------
    step(3, total_steps, "Training Regression (scenario-grouped cross-validation) and baselines...")
    predictor = DemandPredictor()
    predictions, regression_metrics = predictor.cross_validate_by_scenario(data.demand, data.weather)
    overall = regression_metrics[regression_metrics["fold"] == "overall"]
    for row in overall.itertuples():
        print(f"      {row.model:<28} MAE = {row.MAE_MW:6.3f} MW   RMSE = {row.RMSE_MW:6.3f} MW")
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(config.PREDICTIONS_CSV, index=False)
    print(f"      saved {len(predictions)} predictions -> {config.PREDICTIONS_CSV.relative_to(config.PROJECT_ROOT)}")

    # ---- 5. environment ----------------------------------------------------------------------
    step(4, total_steps, "Creating the load-shedding environment...")
    environment = LoadSheddingEnvironment(data)
    scenario_ids = config.SCENARIO_IDS if max_scenarios is None else config.SCENARIO_IDS[:max_scenarios]
    print(f"      {len(environment.areas)} areas, {environment.n_feeders} feeders, "
          f"hospital feeder = {config.HOSPITAL_FEEDER_ID} (always ON), {len(scenario_ids)} scenarios to run")

    # ---- 6/7. scheduling trials -------------------------------------------------------------------
    step(5, total_steps, "Running CSP Backtracking and Hill Climbing on every scenario...")
    report = EvaluationReport(environment, predictions, regression_metrics, scenario_ids)
    report.run_trials()

    # ---- 8. comparison ------------------------------------------------------------------------------
    step(6, total_steps, "Comparing CSP and Hill Climbing on identical scenarios...")
    report.compare_methods()

    step(7, total_steps, "Creating charts...")
    for path in report.create_charts():
        print(f"      saved {path.relative_to(config.PROJECT_ROOT)}")

    step(8, total_steps, "Saving schedules, metrics and reports...")
    report.save_outputs()
    summary_text = report.write_summary_text([
        f"Data validation: all {passed} checks passed. Existing CSV files were read only.",
    ])
    for path in (config.CSP_SCHEDULES_CSV, config.HC_SCHEDULES_CSV, config.INFEASIBLE_CSV,
                 config.REGRESSION_METRICS_CSV, config.SCHEDULING_METRICS_CSV, config.SCHEDULING_SUMMARY_CSV,
                 config.EXPERIMENT_SUMMARY_TXT):
        print(f"      saved {path.relative_to(config.PROJECT_ROOT)}")

    # ---- final summary ----------------------------------------------------------------------------------
    print("\n" + BANNER)
    print(" PROJECT RUN COMPLETED SUCCESSFULLY")
    print(BANNER)
    print(summary_text)
    print(f"Total run time: {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete AL-2002 load-shedding project.")
    parser.add_argument("--max-scenarios", type=int, default=None,
                        help="run scheduling on only the first N scenarios (default: all 50)")
    arguments = parser.parse_args()
    sys.exit(main(arguments.max_scenarios))
