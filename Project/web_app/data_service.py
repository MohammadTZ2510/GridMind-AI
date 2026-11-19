"""Read-only, timestamp-aware CSV data service for the command centre.

Original CSV files remain the source of truth. Missing schedules are represented
by null decisions, never silently converted to an outage or an operational state.
"""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

DATASETS = {
    'feeders': (DATA_DIR / 'feeders.csv', 'Static feeder identity, capacity, priorities and outage rules.'),
    'hourly_demand': (DATA_DIR / 'hourly_demand.csv', 'Simulated actual demand for each scenario, hour and feeder.'),
    'hourly_supply': (DATA_DIR / 'hourly_supply.csv', 'Network supply and reserve; one shared value per scenario-hour.'),
    'weather_time': (DATA_DIR / 'weather_time.csv', 'Simulated weather, calendar and peak-hour flags.'),
    'demand_predictions': (OUTPUT_DIR / 'predictions/demand_predictions.csv', 'Out-of-fold regression predictions and two forecast baselines.'),
    'csp_schedules': (OUTPUT_DIR / 'schedules/csp_schedules.csv', 'Saved CSP decisions; served and unmet demand use actual demand.'),
    'hill_climbing_schedules': (OUTPUT_DIR / 'schedules/hill_climbing_schedules.csv', 'Saved Hill Climbing decisions; served and unmet demand use actual demand.'),
    'scheduling_metrics': (OUTPUT_DIR / 'metrics/scheduling_metrics.csv', 'Per-scenario algorithm execution and quality metrics.'),
    'scheduling_summary': (OUTPUT_DIR / 'metrics/scheduling_summary.csv', 'Paired comparison across feasible scenarios; not a selected-scenario statistic.'),
    'infeasible_scenarios': (OUTPUT_DIR / 'schedules/infeasible_scenarios.csv', 'Recorded infeasible or unresolved scenarios with actual reasons.'),
    'regression_metrics': (OUTPUT_DIR / 'metrics/regression_metrics.csv', 'Scenario-grouped cross-validation MAE and RMSE by fold and overall.'),
}
SCHEDULE_COLUMNS = ['scenario_id', 'hour', 'feeder_id', 'area', 'predicted_demand_mw',
                    'actual_demand_mw', 'decision', 'served_demand_mw', 'unmet_demand_mw']
COLUMN_DESCRIPTIONS = {
    'scenario_id': 'Seeded simulated daily scenario identifier.',
    'hour': 'Hourly slot, 0 through 23; each slot lasts one hour.',
    'feeder_id': 'Distribution feeder identifier, F001 through F010.',
    'area': 'Grid area served by the feeder.',
    'feeder_name': 'Descriptive feeder name.',
    'feeder_capacity_mw': 'Maximum feeder capacity in MW.',
    'priority': 'Integer service priority; higher values receive more protection.',
    'priority_weight': 'Weight used for priority-weighted unmet demand.',
    'is_critical': 'Whether the feeder must remain supplied at all hours.',
    'critical_type': 'Hospital, residential, commercial or industrial feeder category.',
    'max_total_outage_hours': 'Maximum permitted OFF slots across the day.',
    'max_consecutive_outage_hours': 'Maximum permitted consecutive OFF slots.',
    'protected_hours': 'Comma-separated hourly slots in which the feeder must remain ON.',
    'actual_demand_mw': 'Simulated observed demand, MW.',
    'regression_prediction_mw': 'Random Forest out-of-fold demand forecast, MW.',
    'predicted_demand_mw': 'Demand forecast used when the saved schedule was planned, MW.',
    'previous_hour_demand_mw': 'Previously observed demand for the same feeder, MW.',
    'previous_hour_baseline_mw': 'Forecast equal to the previous observed hourly demand, MW.',
    'mean_baseline_mw': 'Forecast from that feeder’s training-only mean demand, MW.',
    'cv_fold': 'Held-out scenario-grouped cross-validation fold, 1 through 5.',
    'decision': 'Saved binary scheduling decision, ON or OFF; missing means unavailable.',
    'served_demand_mw': 'Actual demand supplied according to the saved ON/OFF decision, MW.',
    'unmet_demand_mw': 'Actual demand not supplied according to the saved decision, MW.',
    'available_supply_mw': 'Total network generation available before reserve, MW.',
    'usable_supply_mw': 'Available network supply minus reserve, MW.',
    'reserve_mw': 'Supply held in reserve, MW.',
    'total_demand_mw': 'Sum of actual feeder demands in this network-hour, MW.',
    'shortage_mw': 'Max(0, actual network demand − usable supply), MW; not forecast shortage.',
    'temperature_c': 'Simulated temperature, degrees Celsius.',
    'humidity_pct': 'Simulated relative humidity, percent.',
    'runtime_s': 'Measured execution time in seconds; HC excludes its prerequisite CSP run.',
    'total_runtime_incl_csp_s': 'Execution time including the prerequisite CSP schedule, seconds.',
    'area_outage_imbalance': 'Population standard deviation of mean sheddable-feeder outage hours across areas.',
    'actual_unmet_mwh': 'Sum of actual OFF-feeder demand over one-hour slots, MWh/day.',
    'network_shortage_mwh': 'Sum of actual network shortage over one-hour slots, MWh/day.',
    'hard_constraint_violations': 'Constraint violations evaluated against planning forecasts.',
    'supply_overload_hours': 'Hours where actual served demand exceeded usable supply.',
    'objective_score': 'Normalized weighted planning score; lower is better.',
    'nodes': 'Actual CSP search nodes visited.',
    'backtracks': 'Actual CSP backtracks.',
    'iterations': 'Actual accepted HC improvements across starts.',
    'restarts': 'Recorded algorithm restart count.',
    'MAE_MW': 'Mean absolute forecasting error, MW.',
    'RMSE_MW': 'Root mean squared forecasting error, MW.',
}


def clean(value):
    """Convert NumPy scalars/non-finite missing values into strict JSON values."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


def records(frame):
    return json.loads(frame.to_json(orient='records', double_precision=10))


def canonical_method(method='CSP'):
    key = str(method).strip().lower().replace(' ', '').replace('_', '')
    if key in ('csp', 'backtracking', 'cspbacktracking'):
        return 'CSP'
    if key in ('hc', 'hillclimbing'):
        return 'HC'
    raise ValueError('Unknown algorithm. Choose CSP or HC.')


class DataStore:
    def __init__(self, fingerprint):
        self.fingerprint = fingerprint
        self.frames = {}
        self.scenarios = {}
        self.payloads = {}
        self.histories = {}
        self.lock = threading.RLock()
        optional = {'csp_schedules', 'hill_climbing_schedules', 'infeasible_scenarios'}
        for key, (path, _) in DATASETS.items():
            if path.exists():
                self.frames[key] = pd.read_csv(path, keep_default_na=(key != 'feeders'))
            elif key in optional:
                self.frames[key] = pd.DataFrame(columns=SCHEDULE_COLUMNS if key.endswith('_schedules') else ['scenario_id'])
            else:
                raise FileNotFoundError(f'Required project data is missing: {path.name}. Run python run_project.py first.')
        self.ids = sorted(self.frames['hourly_demand']['scenario_id'].unique().tolist())
        self.feeder_records = records(self.frames['feeders'])
        self.areas = list(dict.fromkeys(f['area'] for f in self.feeder_records))
        self._bootstrap = None

    def metadata(self):
        return [{
            'id': key, 'filename': path.name, 'rows': len(self.frames[key]),
            'columns': self.frames[key].columns.tolist(), 'column_count': len(self.frames[key].columns),
            'description': description, 'available': path.exists(),
            'column_descriptions': {c: COLUMN_DESCRIPTIONS.get(c, self._describe(c)) for c in self.frames[key].columns},
        } for key, (path, description) in DATASETS.items()]

    @staticmethod
    def _describe(column):
        if column.startswith('outage_hours_'):
            return 'Saved daily OFF-hour count for ' + column.removeprefix('outage_hours_') + '.'
        if column.startswith('area_outage_'):
            return 'Mean daily OFF hours of sheddable feeders in ' + column.removeprefix('area_outage_') + '.'
        if column.startswith(('csp_', 'hc_')):
            return column.replace('_', ' ').capitalize() + '; see dataset description for units.'
        return column.replace('_', ' ').capitalize() + ' as recorded in the source dataset.'

    def scenario(self, scenario_id):
        with self.lock:
            if scenario_id not in self.ids:
                raise KeyError('Unknown scenario: ' + scenario_id)
            if scenario_id in self.scenarios:
                return self.scenarios[scenario_id]
            f = self.frames
            demand = f['hourly_demand'].query('scenario_id == @scenario_id').copy()
            predictions = f['demand_predictions'].query('scenario_id == @scenario_id')
            prediction_columns = ['scenario_id', 'hour', 'feeder_id', 'regression_prediction_mw',
                                  'previous_hour_baseline_mw', 'mean_baseline_mw', 'cv_fold']
            joined = demand.merge(predictions[prediction_columns], on=['scenario_id', 'hour', 'feeder_id'], how='left', validate='one_to_one')
            meta = f['feeders'].drop(columns=['area', 'feeder_capacity_mw'])
            joined = joined.merge(meta, on='feeder_id', how='left', validate='many_to_one')
            weather = f['weather_time'].query('scenario_id == @scenario_id').sort_values('hour')
            joined = joined.merge(weather[['scenario_id', 'hour', 'day_of_week', 'is_weekend',
                                          'is_peak_hour', 'is_morning_peak', 'is_evening_peak']], on=['scenario_id', 'hour'], validate='many_to_one')
            supply = f['hourly_supply'].query('scenario_id == @scenario_id').sort_values('hour')
            schedules = {}
            for method, dataset in [('CSP', 'csp_schedules'), ('HC', 'hill_climbing_schedules')]:
                schedule = f[dataset].query('scenario_id == @scenario_id').copy()
                schedules[method] = schedule
                cols = ['scenario_id', 'hour', 'feeder_id', 'decision', 'served_demand_mw', 'unmet_demand_mw']
                renamed = {c: method.lower() + '_' + c for c in cols[3:]}
                joined = joined.merge(schedule[cols].rename(columns=renamed), on=cols[:3], how='left', validate='one_to_one')
            joined['different'] = (joined['csp_decision'].notna() & joined['hc_decision'].notna()
                                   & (joined['csp_decision'] != joined['hc_decision']))
            merged_records = records(joined.sort_values(['hour', 'feeder_id']))
            hourly = {h: [r for r in merged_records if r['hour'] == h] for h in range(24)}
            metric_rows = records(f['scheduling_metrics'].query('scenario_id == @scenario_id'))
            metrics = {'CSP': next((r for r in metric_rows if r['method'] == 'CSP'), None),
                       'HC': next((r for r in metric_rows if r['method'] == 'HillClimbing'), None)}
            inf = records(f['infeasible_scenarios'].query('scenario_id == @scenario_id'))
            infeasible = inf[0] if inf else None
            csp_metric = metrics['CSP']
            status = csp_metric['status'] if csp_metric else 'UNAVAILABLE'
            # Fairness is computed from actual saved decisions, excluding protected
            # non-sheddable feeders from each area's mean (same definition as core).
            fairness = {}
            area_records = [{'area': a} for a in self.areas]
            for method in ['CSP', 'HC']:
                schedule = schedules[method]
                complete = len(schedule) == len(demand) and schedule['decision'].isin(['ON', 'OFF']).all()
                feeder_hours = []
                for feeder in self.feeder_records:
                    sf = schedule[schedule['feeder_id'] == feeder['feeder_id']]
                    valid = len(sf) == 24 and sf['decision'].isin(['ON', 'OFF']).all()
                    feeder_hours.append({'feeder_id': feeder['feeder_id'], 'area': feeder['area'],
                                         'is_critical': feeder['is_critical'],
                                         'outage_hours': int((sf['decision'] == 'OFF').sum()) if valid else None,
                                         'unmet_mwh': float(sf['unmet_demand_mw'].sum()) if valid else None})
                mean_hours = []
                for ar in area_records:
                    members = [x for x in feeder_hours if x['area'] == ar['area']]
                    sheddable_ids = {x['feeder_id'] for x in self.feeder_records
                                    if x['area'] == ar['area'] and not x['is_critical'] and x['max_total_outage_hours'] > 0
                                    and len(str(x['protected_hours']).split(',')) < 24}
                    hours = [x['outage_hours'] for x in members if x['feeder_id'] in sheddable_ids]
                    mean = float(np.mean(hours)) if complete and hours else (0.0 if complete else None)
                    ar[method] = {'outage_hours': sum(x['outage_hours'] for x in members) if complete else None,
                                  'mean_outage_hours': mean,
                                  'unmet_mwh': sum(x['unmet_mwh'] for x in members) if complete else None}
                    mean_hours.append(mean)
                critical = [x for x in feeder_hours if x['is_critical']]
                fairness[method] = {'available': bool(complete), 'feeder_outage_hours': feeder_hours,
                                     'imbalance': float(np.std(mean_hours)) if complete else None,
                                     'critical_protected': all(x['outage_hours'] == 0 for x in critical) if complete else None}
            timeline = []
            supply_rows = {int(r['hour']): r for r in records(supply)}
            weather_rows = {int(r['hour']): r for r in records(weather)}
            for h in range(24):
                rows = hourly[h]
                row = {'hour': h, 'time_label': f'{h:02d}:00'}
                for col in ['actual_demand_mw', 'regression_prediction_mw', 'previous_hour_baseline_mw', 'mean_baseline_mw']:
                    vals = [r[col] for r in rows]
                    row[col] = sum(vals) if all(v is not None for v in vals) else None
                row['predicted_demand_mw'] = row['regression_prediction_mw']
                row.update(supply_rows.get(h, {}))
                # CSV shortage is based on ACTUAL demand; forecast shortage is separate.
                row['actual_shortage_mw'] = row.get('shortage_mw')
                forecast = row['predicted_demand_mw']
                usable = row.get('usable_supply_mw')
                row['forecast_shortage_mw'] = max(0, forecast - usable) if forecast is not None and usable is not None else None
                for method in ['csp', 'hc']:
                    available = len(rows) == 10 and all(r[method + '_decision'] in ('ON', 'OFF') for r in rows)
                    row[method + '_served_demand_mw'] = sum(r[method + '_served_demand_mw'] for r in rows) if available else None
                    row[method + '_unmet_demand_mw'] = sum(r[method + '_unmet_demand_mw'] for r in rows) if available else None
                timeline.append(row)
            model_metrics = []
            actual = joined['actual_demand_mw'].to_numpy()
            for model, col in [('Regression (RandomForest)', 'regression_prediction_mw'), ('Previous-hour baseline', 'previous_hour_baseline_mw'), ('Mean-demand baseline', 'mean_baseline_mw')]:
                err = actual - joined[col].to_numpy()
                model_metrics.append({'model': model, 'MAE_MW': float(np.mean(abs(err))), 'RMSE_MW': float(np.sqrt(np.mean(err ** 2)))})
            summary = {
                'scenario_id': scenario_id, 'status': status, 'feasible': status == 'FEASIBLE',
                'feeder_count': len(self.feeder_records), 'area_count': len(self.areas), 'hour_count': 24,
                'season': weather_rows[0]['season'], 'day_type': weather_rows[0]['day_type'],
                'temperature_c': float(weather['temperature_c'].mean()),
                'average_temperature': float(weather['temperature_c'].mean()),
                'average_humidity': float(weather['humidity_pct'].mean()),
                'minimum_temperature': float(weather['temperature_c'].min()),
                'maximum_temperature': float(weather['temperature_c'].max()),
                'total_demand_mwh': float(demand['actual_demand_mw'].sum()),
                'total_actual_demand_mwh': float(demand['actual_demand_mw'].sum()),
                'total_predicted_demand_mwh': float(joined['regression_prediction_mw'].sum()),
                'total_available_supply_mwh': float(supply['available_supply_mw'].sum()),
                'total_usable_supply_mwh': float(supply['usable_supply_mw'].sum()),
                'total_shortage_mwh': float(supply['shortage_mw'].sum()),
                'shortage_hours': int((supply['shortage_mw'] > 0).sum()),
                'peak_shortage_mw': float(supply['shortage_mw'].max()),
            }
            result = clean({'scenario_id': scenario_id, 'status': status, 'summary': summary, 'metrics': metrics,
                            'infeasible': infeasible, 'hourly': hourly, 'timeline': timeline,
                            'weather_rows': weather_rows, 'weather_history': records(weather),
                            'areas': area_records, 'fairness': fairness, 'forecast_metrics': model_metrics})
            self.scenarios[scenario_id] = result
            return result

    def bootstrap(self):
        with self.lock:
            if self._bootstrap is None:
                summaries = [self.scenario(s)['summary'] for s in self.ids]
                regression = self.frames['regression_metrics']
                self._bootstrap = clean({'app_name': 'GRIDMIND AI', 'simulated': True,
                    'scenarios': summaries, 'feeders': self.feeder_records, 'areas': self.areas,
                    'model_metrics': records(regression[regression['fold'].astype(str) == 'overall']),
                    'scheduling_summary': records(self.frames['scheduling_summary']),
                    'datasets': self.metadata(),
                    'notes': {'planning': 'Schedules use forecast demand. Saved served/unmet values use actual demand.',
                              'forecast': 'One-step-ahead evaluation uses observed previous-hour demand; not a deployable 24-hour-ahead forecast.',
                              'history': 'Original outputs contain aggregate execution metrics, not a full search trace.'}})
            return self._bootstrap

    def payload(self, scenario_id, hour=0, method='CSP'):
        method = canonical_method(method)
        if not isinstance(hour, int) or not 0 <= hour <= 23:
            raise ValueError('Hour must be an integer from 0 through 23.')
        key = (scenario_id, hour, method)
        with self.lock:
            if key in self.payloads:
                return self.payloads[key]
            s = self.scenario(scenario_id)
            history_key = (scenario_id, method)
            if history_key not in self.histories:
                history = []
                for h in range(24):
                    for r in s['hourly'][h]:
                        row = dict(r)
                        row['decision'] = r[method.lower() + '_decision']
                        row['schedule_decision'] = row['decision']
                        row['served_demand_mw'] = r[method.lower() + '_served_demand_mw']
                        row['unmet_demand_mw'] = r[method.lower() + '_unmet_demand_mw']
                        history.append(row)
                self.histories[history_key] = history
            history = self.histories[history_key]
            feeders = [r for r in history if r['hour'] == hour]
            available = len(feeders) == 10 and all(r['decision'] in ('ON', 'OFF') for r in feeders)
            comp_available = all(r['csp_decision'] in ('ON', 'OFF') and r['hc_decision'] in ('ON', 'OFF') for r in feeders)
            differences = [r['feeder_id'] for r in feeders if r['different']]
            network = s['timeline'][hour]
            totals = {c: network[c] for c in ['actual_demand_mw', 'predicted_demand_mw', 'available_supply_mw',
                       'usable_supply_mw', 'reserve_mw', 'forecast_shortage_mw', 'actual_shortage_mw']}
            totals.update({'served_demand_mw': network[method.lower() + '_served_demand_mw'],
                           'unmet_demand_mw': network[method.lower() + '_unmet_demand_mw'],
                           'on_count': sum(r['decision'] == 'ON' for r in feeders),
                           'off_count': sum(r['decision'] == 'OFF' for r in feeders),
                           'unknown_count': sum(r['decision'] not in ('ON', 'OFF') for r in feeders)})
            status = (s['metrics'][method] or {}).get('status', 'SKIPPED' if s['infeasible'] else 'UNAVAILABLE')
            result = {'scenario_id': scenario_id, 'hour': hour, 'method': method, 'status': status,
                      'scenario_status': s['status'], 'schedule_available': available,
                      'summary': s['summary'], 'totals': totals, 'weather': s['weather_rows'][hour],
                      'feeders': feeders, 'feeder_history': history, 'timeline': s['timeline'],
                      'weather_history': s['weather_history'], 'metrics': s['metrics'],
                      'infeasible': s['infeasible'], 'areas': s['areas'], 'fairness': s['fairness'],
                      'forecast_metrics': s['forecast_metrics'],
                      'comparison': {'available': comp_available, 'identical': not differences if comp_available else None,
                                     'different_feeders': differences}}
            # Keep at most the two active-method scenario histories per hour in
            # the payload cache; this is small for the fixed 50×24 project.
            self.payloads[key] = result
            return result


_store = None
_store_lock = threading.RLock()


def get_store():
    global _store
    stamp = tuple((key, path.stat().st_mtime_ns if path.exists() else None,
                   path.stat().st_size if path.exists() else None) for key, (path, _) in DATASETS.items())
    with _store_lock:
        if _store is None or _store.fingerprint != stamp:
            _store = DataStore(stamp)
        return _store


def load_all_data():
    f = get_store().frames
    return {'feeders': f['feeders'], 'demand': f['hourly_demand'], 'supply': f['hourly_supply'],
            'weather': f['weather_time'], 'predictions': f['demand_predictions'],
            'csp': f['csp_schedules'], 'hill_climbing': f['hill_climbing_schedules'],
            'metrics': f['scheduling_metrics'], 'infeasible': f['infeasible_scenarios']}


def get_scenario_ids():
    return get_store().ids


def get_scenario_overview(scenario_id):
    try:
        s = get_store().scenario(scenario_id)
        return {**s['summary'], 'metrics': list(s['metrics'].values()), 'weather_records': s['weather_history']}
    except KeyError:
        return None


def get_hour_details(scenario_id, hour, method='CSP'):
    try:
        return get_store().payload(scenario_id, int(hour), method)
    except KeyError:
        return None


def get_feeder_detail(scenario_id, hour, feeder_id, method='CSP'):
    p = get_hour_details(scenario_id, hour, method)
    return next((r for r in p['feeders'] if r['feeder_id'] == feeder_id), None) if p else None


def query_dataset(dataset, *, scenario='', hour=None, q='', sort='', direction='asc', page=1, page_size=30, area='', feeder=''):
    store = get_store()
    key = dataset.removesuffix('.csv')
    if key not in store.frames:
        raise KeyError('Unknown dataset.')
    frame = store.frames[key]
    if scenario and 'scenario_id' in frame:
        if scenario not in store.ids:
            raise KeyError('Unknown scenario: ' + scenario)
        frame = frame[frame['scenario_id'] == scenario]
    if hour is not None and 'hour' in frame:
        if not 0 <= hour <= 23:
            raise ValueError('Hour must be from 0 through 23.')
        frame = frame[frame['hour'] == hour]
    if area and 'area' in frame:
        frame = frame[frame['area'] == area]
    if feeder and 'feeder_id' in frame:
        frame = frame[frame['feeder_id'] == feeder]
    if q:
        mask = frame.astype(str).apply(lambda col: col.str.contains(q, case=False, regex=False, na=False)).any(axis=1)
        frame = frame[mask]
    if direction not in ('asc', 'desc'):
        raise ValueError('Sort direction must be asc or desc.')
    if sort:
        if sort not in frame:
            raise ValueError('Unknown sort column.')
        frame = frame.sort_values(sort, ascending=direction == 'asc', kind='stable', na_position='last')
    total = len(frame)
    if page < 1 or not 1 <= page_size <= 250:
        raise ValueError('Page must be positive and page_size must be between 1 and 250.')
    return key, frame, {'dataset': key, 'filename': DATASETS[key][0].name,
                        'columns': frame.columns.tolist(), 'total_rows': total, 'source_rows': len(store.frames[key]),
                        'page': page, 'page_size': page_size, 'pages': max(1, math.ceil(total / page_size)),
                        'total': total, 'total_pages': max(1, math.ceil(total / page_size)),
                        'descriptions': {c: COLUMN_DESCRIPTIONS.get(c, store._describe(c)) for c in frame.columns},
                        'column_descriptions': {c: COLUMN_DESCRIPTIONS.get(c, store._describe(c)) for c in frame.columns},
                        'rows': records(frame.iloc[(page - 1) * page_size:page * page_size])}
