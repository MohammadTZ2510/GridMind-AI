# AI-Based Electricity Demand Prediction and Fair Load-Shedding Scheduler

**AL-2002 Artificial Intelligence Laboratory — Fall 2026 — FAST-NUCES Karachi — Section BDS-5A**

> ⚠️ **ALL DATA IS SIMULATED.** It is not real data from K-Electric or any other electricity company, and the system is a *decision-support simulation*, not a controller for a real grid.

## Project description

The project simulates a small electricity distribution network (5 areas, 10 feeders, 24 hourly slots, 50 seeded scenarios). It

1. forecasts hourly feeder demand with **Regression** and compares it with two simple baselines,
2. uses **CSP Backtracking** to build a *valid* ON/OFF load-shedding schedule (hospital protection, protected hours, outage limits, supply capacity),
3. uses **Hill Climbing** to improve that schedule (less unmet demand, fairer outage hours),
4. compares CSP and Hill Climbing on **identical scenarios** and produces metrics and charts.

## Objectives and AI techniques

| Technique | Where | Implemented by |
|---|---|---|
| Regression (RandomForest, scikit-learn) | `src/demand_predictor.py` | scikit-learn model; evaluation + baselines are ours |
| Baselines: previous-hour and mean-demand forecast | `src/demand_predictor.py` | own code |
| CSP Backtracking | `src/csp_scheduler.py` | own implementation (no solver library) |
| Hill Climbing | `src/hill_climbing_scheduler.py` | own implementation |

## Dataset (created earlier by `dataset_generator.py`, seed 42)

| File | Rows | Content |
|---|---:|---|
| `data/feeders.csv` | 10 | static feeder data and outage rules (F001 = hospital) |
| `data/hourly_demand.csv` | 12,000 | demand per scenario × hour × feeder (target `actual_demand_mw`) |
| `data/hourly_supply.csv` | 1,200 | **network-wide** supply per scenario × hour |
| `data/weather_time.csv` | 1,200 | simulated weather / time flags |

`run_project.py` only **reads** these files. They are never regenerated or modified.

## How feeder decisions map to the proposal's area-level goal

The proposal talks about area-hour outage decisions. Our network has 2 feeders per area, so decisions are made per **feeder-hour** (needed to protect the hospital feeder F001 on its own) and then aggregated per **area**:

```
area outage hours = mean outage hours of the area's SHEDDABLE feeders
```

* Binary model: ON = full feeder demand served, OFF = full demand not served (no partial switching).
* The hospital feeder can never be switched off, so it is excluded from its area's mean. (Otherwise North Karachi would look "better treated" only because of the hospital.) Averaging per feeder also means no area is favoured for having more feeders.
* An area is never reported as "OFF" while a protected feeder in it stays ON.

## Hard constraints vs objective

**Hard constraints** (checked separately, never traded for a better score): hospital feeder always ON · protected hours ON · total outage hours ≤ limit · consecutive outage hours ≤ limit · served demand ≤ usable supply in every hour · binary decisions only.

**Objective (Hill Climbing, minimised):** `0.5·unmet + 0.3·fairness + 0.2·priority-weighted unmet`, each term normalised (see `src/hill_climbing_scheduler.py`).

**Fairness metric:** standard deviation of the 5 area outage-hour values (0 = perfectly equal).

## Project structure

```text
Project/
├── data/                      existing datasets (read only)
├── src/
│   ├── config.py              paths, seed, constants, algorithm settings
│   ├── data_loader.py         loading + validation of the 4 CSV files
│   ├── environment.py         LoadSheddingEnvironment (scenario, constraints, shortage, outage hours)
│   ├── demand_predictor.py    DemandPredictor (Regression + baselines, scenario-grouped CV)
│   ├── scheduler_base.py      Scheduler base class + ScheduleResult
│   ├── csp_scheduler.py       CSPScheduler (backtracking)
│   ├── hill_climbing_scheduler.py  HillClimbingScheduler
│   ├── evaluation.py          EvaluationReport (trials, comparison, charts, files)
│   └── visualization.py       Matplotlib / Seaborn charts
├── outputs/                   predictions/, schedules/, metrics/, charts/ (created automatically)
├── tests/test_project.py      automated tests
├── 01_data_exploration.ipynb  exploratory data analysis
├── dataset_generator.py       original data generator (unchanged)
├── run_project.py             main entry point
├── requirements.txt
└── README.md
```

## Environment setup (Windows, VS Code, Anaconda)

```powershell
conda activate load_shedding
pip install -r requirements.txt
```

(NumPy, Pandas, Matplotlib, Seaborn, scikit-learn and Jupyter are already installed in the `load_shedding` environment, so this normally changes nothing.)

## How to run

| Task | Command (run in the project root) |
|---|---|
| Data exploration | open `01_data_exploration.ipynb` in VS Code, select the `load_shedding` kernel, **Run All** |
| Complete project | `python run_project.py` |
| Quick demo (first 10 scenarios) | `python run_project.py --max-scenarios 10` |
| Tests | `python -m unittest discover -s tests -v` |

## Outputs

```text
outputs/predictions/demand_predictions.csv      actual, previous-hour, mean and regression forecasts (out-of-fold)
outputs/schedules/csp_schedules.csv             ON/OFF decision per scenario, hour and feeder (CSP)
outputs/schedules/hill_climbing_schedules.csv   same for Hill Climbing
outputs/schedules/infeasible_scenarios.csv      scenarios without a valid schedule, with the reason
outputs/metrics/regression_metrics.csv          MAE / RMSE per model (per fold and overall)
outputs/metrics/scheduling_metrics.csv          every metric per scenario and method
outputs/metrics/scheduling_summary.csv          mean / std / paired comparison CSP vs Hill Climbing
outputs/metrics/experiment_summary.txt          readable report of all results
outputs/charts/*.png                            actual_vs_predicted_demand, runtime, shortage, outage_hours_by_area
```

## Evaluation metrics

* **Regression:** MAE and RMSE for previous-hour baseline, mean-demand baseline (training data only) and RandomForest. Evaluation uses **scenario-grouped 5-fold cross-validation**, so no scenario is ever in both the training and test part. The out-of-fold predictions of all 50 scenarios are what the schedulers plan with.
* **Scheduling:** runtime, total and mean-per-hour unmet demand, over-shedding beyond the network shortage, outage hours by feeder and area, outage-hour imbalance, constraint violations, number of feasible / infeasible / unresolved scenarios. Means and standard deviations are reported over the same scenarios for both methods.

## Team

| Member | Roll no. | Responsibilities (from the proposal) |
|---|---|---|
| S. M. Nasar Ali (Leader) | 24K-2516 | CSP constraints, environment and integration, experiment runner, evaluation, repository coordination |
| Taha Zameer | 24K-2580 | Data generation and cleaning, regression baseline/model, Hill Climbing, metrics and charts |

## Limitations and assumptions

* **Simulated data only.** Results say nothing about a real grid. No external APIs, no real-time control.
* **Plans use forecast demand; results are measured with actual demand.** Because the forecast is not perfect, a plan that is valid for the forecast can slightly exceed usable supply in some hours when judged with actual demand. This is reported as `supply_overload_hours`. Hill Climbing removes over-shedding down to the forecast limit and therefore tends to have *more* such hours than the CSP — a real trade-off, not hidden.
* **One-step-ahead forecast:** the model uses the previous hour's (already observed) demand as a feature.
* **Infeasible scenarios** (no schedule can satisfy all hard constraints, e.g. supply far below what the allowed outages can remove) are detected, explained and recorded; Hill Climbing is skipped for them. Infeasibility can depend on the forecast, so the count may change if the model changes.
* **CSP search limits:** the backtracking search has a node and time limit (`config.py`) and a small restart portfolio of ordering strategies. A scenario that hits the limit would be reported as `LIMIT_REACHED` (feasibility unknown), never as a success.
* **Hill Climbing is a local search.** It stops at a local optimum and is not claimed to find the global optimum. Several seeded starts reduce, but do not remove, this risk.
* All-or-nothing feeders mean switching a feeder off usually removes more demand than the exact shortage.
