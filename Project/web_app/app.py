"""GRIDMIND AI local Flask application. Original datasets are read-only."""
from __future__ import annotations

import os
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
from markupsafe import escape

try:
    from .data_service import canonical_method, clean, get_store, query_dataset
except ImportError:
    from data_service import canonical_method, clean, get_store, query_dataset

app = Flask(__name__)
app.json.sort_keys = False
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024


def integer_arg(name, default=None, low=None, high=None):
    raw = request.args.get(name)
    if raw is None or raw == '':
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be an integer.')
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f'{name} is outside its permitted range.')
    return value


def render_page(page='command', scenario_id=None, hour=None, feeder_id=None):
    scenario_id = scenario_id or request.args.get('scenario', 'S001')
    hour = hour if hour is not None else integer_arg('hour', 13, 0, 23)
    method = canonical_method(request.args.get('method', 'CSP'))
    store = get_store()
    if scenario_id not in store.ids:
        raise KeyError('Unknown scenario: ' + scenario_id)
    if not 0 <= hour <= 23:
        raise KeyError('Unknown hourly slot.')
    if feeder_id and feeder_id not in {f['feeder_id'] for f in store.feeder_records}:
        raise KeyError('Unknown feeder: ' + feeder_id)
    return render_template('command.html', page=page, initial_scenario=scenario_id,
                           initial_hour=hour, initial_method=method, initial_feeder=feeder_id,
                           scenario_id=scenario_id, hour=hour, feeder_id=feeder_id)


@app.get('/')
def dashboard():
    return render_page('command')


@app.get('/scenarios')
def scenarios():
    return render_page('scenarios')


@app.get('/scenario/<scenario_id>')
def scenario_detail(scenario_id):
    return render_page('command', scenario_id)


@app.get('/scenario/<scenario_id>/hour/<int:hour>')
def hourly_analysis(scenario_id, hour):
    return render_page('command', scenario_id, hour)


@app.get('/scenario/<scenario_id>/hour/<int:hour>/feeder/<feeder_id>')
def feeder_detail(scenario_id, hour, feeder_id):
    return render_page('feeder', scenario_id, hour, feeder_id)


@app.get('/demand')
def demand_intelligence():
    return render_page('demand')


@app.get('/weather')
def weather_intelligence():
    return render_page('weather')


@app.get('/fairness')
def fairness_analytics():
    return render_page('fairness')


@app.get('/algorithm-lab')
def algorithm_lab():
    return render_page('algorithm-lab')


@app.get('/data')
def data_explorer():
    return render_page('data')


@app.get('/api/bootstrap')
def bootstrap_api():
    return jsonify(get_store().bootstrap())


@app.get('/api/scenario/<scenario_id>')
def scenario_api(scenario_id):
    hour = integer_arg('hour', 13, 0, 23)
    method = canonical_method(request.args.get('method', 'CSP'))
    return jsonify(get_store().payload(scenario_id, hour, method))


@app.get('/api/scenario/<scenario_id>/hour/<int:hour>')
def hour_api(scenario_id, hour):
    return jsonify(get_store().payload(scenario_id, hour, canonical_method(request.args.get('method', 'CSP'))))


@app.get('/api/scenario/<scenario_id>/feeder/<feeder_id>')
def feeder_api(scenario_id, feeder_id):
    p = get_store().payload(scenario_id, integer_arg('hour', 13, 0, 23), request.args.get('method', 'CSP'))
    row = next((r for r in p['feeders'] if r['feeder_id'] == feeder_id), None)
    if row is None:
        raise KeyError('Unknown feeder: ' + feeder_id)
    return jsonify({'scenario_id': scenario_id, 'hour': p['hour'], 'method': p['method'], 'feeder': row,
                    'history': [r for r in p['feeder_history'] if r['feeder_id'] == feeder_id],
                    'weather_history': p['weather_history'], 'infeasible': p['infeasible']})


@app.get('/api/datasets')
def datasets_api():
    return jsonify({'datasets': get_store().metadata()})


def dataset_response(dataset, download=False):
    key, frame, payload = query_dataset(dataset,
        scenario=request.args.get('scenario', request.args.get('scenario_id', '')),
        hour=integer_arg('hour', None, 0, 23), q=request.args.get('q', '')[:250],
        sort=request.args.get('sort', ''), direction=request.args.get('direction', 'asc'),
        page=integer_arg('page', 1, 1), page_size=integer_arg('page_size', 30, 1, 250),
        area=request.args.get('area', ''), feeder=request.args.get('feeder', ''))
    if download or request.args.get('download', '').lower() in ('csv', '1', 'true'):
        return Response(frame.to_csv(index=False), mimetype='text/csv',
                        headers={'Content-Disposition': f'attachment; filename="{key}.csv"'})
    return jsonify(payload)


@app.get('/api/data/<dataset>')
def dataset_api(dataset):
    return dataset_response(dataset)


@app.get('/api/data/<dataset>/download')
def dataset_download(dataset):
    return dataset_response(dataset, download=True)


@app.post('/api/lab/run')
def run_lab_api():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ValueError('Provide a JSON object with scenario_id and method.')
    scenario_id = body.get('scenario_id', 'S001')
    if scenario_id not in get_store().ids:
        raise KeyError('Unknown scenario: ' + str(scenario_id))
    method = canonical_method(body.get('method', 'CSP'))
    try:
        from .lab_service import run_lab
    except ImportError:
        from lab_service import run_lab
    return jsonify(clean(run_lab(scenario_id, method)))


@app.get('/api/health')
def health_api():
    return jsonify({'status': 'ok', 'simulated': True, 'scenario_count': len(get_store().ids)})


@app.errorhandler(Exception)
def handle_error(error):
    if isinstance(error, KeyError):
        code, message = 404, str(error).strip("'")
    elif isinstance(error, ValueError):
        code, message = 400, str(error)
    elif isinstance(error, FileNotFoundError):
        code, message = 503, str(error)
    elif isinstance(error, RuntimeError):
        code, message = 409, str(error)
    elif isinstance(error, HTTPException):
        code, message = error.code, error.description
    else:
        app.logger.exception('Unhandled GRIDMIND error')
        code, message = 500, 'The request could not be completed. Check the local server log.'
    if request.path.startswith('/api/'):
        return jsonify({'error': message, 'status': code}), code
    return Response(f'<!doctype html><title>GRIDMIND AI</title><h1>GRIDMIND AI</h1><p>{escape(message)}</p><a href="/">Return to command centre</a>', status=code, mimetype='text/html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '5000')), debug=False, use_reloader=False)
