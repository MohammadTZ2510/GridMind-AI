# Project audit

## Source and preservation

Both supplied ZIP files contain identical file contents. The newer ZIP is the implementation base. Original input data, original saved output CSVs/charts, notebook, generator, proposal, tests and experiment entry point are retained. Original README is copied to docs/ORIGINAL_README.md. Only optional observation hooks were added to the two schedulers; their default behavior remains unchanged.

## Actual CSV schemas

### data/feeders.csv

10 rows × 13 columns.

`feeder_id`, `area`, `feeder_name`, `feeder_capacity_mw`, `priority`, `priority_weight`, `is_critical`, `critical_type`, `max_total_outage_hours`, `max_consecutive_outage_hours`, `protected_hours`, `base_demand_mw`, `demand_variation_pct`

### data/hourly_demand.csv

12,000 rows × 13 columns.

`scenario_id`, `hour`, `time_label`, `feeder_id`, `area`, `day_type`, `season`, `temperature_c`, `humidity_pct`, `previous_hour_demand_mw`, `actual_demand_mw`, `demand_category`, `feeder_capacity_mw`

### data/hourly_supply.csv

1,200 rows × 9 columns.

`scenario_id`, `hour`, `time_label`, `available_supply_mw`, `reserve_mw`, `usable_supply_mw`, `total_demand_mw`, `shortage_mw`, `supply_status`

### data/weather_time.csv

1,200 rows × 12 columns.

`scenario_id`, `hour`, `time_label`, `day_type`, `day_of_week`, `season`, `temperature_c`, `humidity_pct`, `is_weekend`, `is_peak_hour`, `is_morning_peak`, `is_evening_peak`

### outputs/metrics/regression_metrics.csv

18 rows × 7 columns.

`model`, `fold`, `n_rows`, `n_scenarios`, `MAE_MW`, `RMSE_MW`, `mae_improvement_vs_previous_hour_pct`

### outputs/metrics/scheduling_metrics.csv

99 rows × 35 columns.

`scenario_id`, `method`, `status`, `feasible`, `runtime_s`, `total_runtime_incl_csp_s`, `nodes`, `backtracks`, `restarts`, `iterations`, `planned_unmet_mwh`, `actual_unmet_mwh`, `mean_unmet_per_hour_mw`, `network_shortage_mwh`, `excess_outage_mwh`, `supply_overload_hours`, `total_outage_hours`, `area_outage_imbalance`, `hard_constraint_violations`, `objective_score`, `outage_hours_F001`, `outage_hours_F002`, `outage_hours_F003`, `outage_hours_F004`, `outage_hours_F005`, `outage_hours_F006`, `outage_hours_F007`, `outage_hours_F008`, `outage_hours_F009`, `outage_hours_F010`, `area_outage_North Karachi`, `area_outage_Gulshan`, `area_outage_Saddar`, `area_outage_Clifton`, `area_outage_Malir`

### outputs/metrics/scheduling_summary.csv

10 rows × 11 columns.

`metric`, `description`, `n_scenarios`, `csp_mean`, `csp_std`, `hc_mean`, `hc_std`, `hc_minus_csp_mean`, `hc_better_in`, `hc_equal_in`, `hc_worse_in`

### outputs/predictions/demand_predictions.csv

12,000 rows × 8 columns.

`scenario_id`, `hour`, `feeder_id`, `actual_demand_mw`, `previous_hour_baseline_mw`, `mean_baseline_mw`, `regression_prediction_mw`, `cv_fold`

### outputs/schedules/csp_schedules.csv

11,760 rows × 9 columns.

`scenario_id`, `hour`, `feeder_id`, `area`, `predicted_demand_mw`, `actual_demand_mw`, `decision`, `served_demand_mw`, `unmet_demand_mw`

### outputs/schedules/hill_climbing_schedules.csv

11,760 rows × 9 columns.

`scenario_id`, `hour`, `feeder_id`, `area`, `predicted_demand_mw`, `actual_demand_mw`, `decision`, `served_demand_mw`, `unmet_demand_mw`

### outputs/schedules/infeasible_scenarios.csv

1 rows × 8 columns.

`scenario_id`, `status`, `cause`, `reason`, `csp_nodes`, `csp_runtime_s`, `total_required_shed_mwh`, `hill_climbing`

## Findings addressed

- Flask was used but absent from requirements; it is now included.
- Original web routes read every CSV repeatedly. Cached data and mtime invalidation now support efficient API interactions.
- Empty JavaScript and analysis templates were placeholders; the new application implements the complete command center and dedicated analysis views.
- Original hourly interface labelled actual CSV shortage as forecast shortage. The new service computes forecast shortage from saved forecasts and reports actual shortage separately.
- Missing schedules and the infeasible S002 scenario require null/unavailable decisions, not OFF. Served/unmet values are also unavailable when decisions are missing.
- Saved execution summaries are not detailed search histories. The Algorithm Lab explicitly distinguishes saved aggregate metrics and fresh optional observed traces.
- Three.js positions are schematic; no geographical precision is claimed.
- Prediction input includes observed previous-hour demand: one-step-ahead forecasting, not a fully known future day.
- Scenario-grouped five-fold validation is retained; no model or algorithm is replaced with placeholders.

## Preservation verification

All 17 original files under data/ and outputs/ were compared byte-for-byte against Project(1).zip before packaging. None changed.