# GRIDMIND AI — Intelligent Electricity Grid Command Center

**Crimson Storm edition:** near-black translucent panels, crimson accents, animated rain and white lightning cores with a glowing red halo. The storm is decorative and independent of the simulated weather records. Use the top-right **Storm effects ON/OFF** control to pause it; reduced-motion preferences disable it by default. No external audio files or network calls are needed: optional sound is synthesized locally. Click **Sound OFF** to enable rain and strike-linked thunder, then adjust its volume slider. Sound starts muted on every page load, and pauses when storm effects are off or the tab is hidden.

A complete Flask / Three.js interface for the original AL-2002 demand prediction and fair load-shedding project. **All data is simulated.** This is educational decision support, with no connection to a physical electricity grid.

## Start on Windows

Extract the ZIP. Open a terminal in `Project` (the folder containing this README).

```powershell
conda activate load_shedding
python -m pip install -r requirements.txt
python web_app/app.py
```

Open **http://127.0.0.1:5000** in Edge, Chrome or Firefox. Keep the terminal open. Press Ctrl+C to stop. The Conda line is optional if you use another Python environment. Python 3.12 is recommended. After installing requirements, `START_GRIDMIND.bat` also starts the app.

Three.js, OrbitControls, GSAP and Plotly are packaged locally. No CDN, login or API key is needed; after Python dependencies are installed, the included data and interface work offline. Jupyter dependencies are included for the original notebook.

## Application views

- **Command Center:** power station → five area substations → ten real feeder IDs. Rotate, pan, zoom, hover and select. ON, OFF, critical, unavailable and algorithm-difference states are distinct. Positions are schematic, not actual geographic coordinates.
- **Scenarios:** all 50 scenarios, search, feasibility filtering, detailed scenario/hour/feeder navigation.
- **Demand Intelligence:** actual demand, Random Forest prediction, previous-hour and mean baselines; area/feeder selection, dynamic MAE/RMSE.
- **Weather:** original temperature, humidity, day type, season and peak flags; timelines, scatter plots and records. Association does not establish causation.
- **Fairness:** feeder outage histories, area outage comparison, unmet energy and imbalance. Fairness uses mean outage duration of sheddable feeders in each area, excluding the always-protected hospital.
- **Algorithm Lab:** saved aggregate execution metrics, explanations and optional fresh runs with real observed events. Original CSVs contain no detailed historical search trace. Fresh traces are capped at 600 events and marked if truncated. Fresh runs do not overwrite original results.
- **Data Explorer:** dataset dimensions, schema descriptions, search, filtering, sorting, pagination and filtered CSV downloads.

Scenario, hour and algorithm controls stay synchronized across views; the URL preserves selections. Comparison mode highlights different feeder decisions and explicitly reports identical decisions.

## Visual execution tracing

In Algorithm Lab, select a scenario and algorithm, then click **Run with tracing**. When the real run finishes, use **Play**, **Step**, **Back**, **Reset**, **End**, the speed selector or timeline slider.

The 10 × 24 board replays actual recorded states: green ON, red OFF, grey unassigned/unavailable, amber outline for the current change. CSP assignments are tentative until completion. A backtrack removes a tentative assignment. Hospital/protected and no-shortage slots are initialized from the real search state. Nothing is fabricated when a run has zero backtracks.

Hill Climbing records the actual schedule at each seeded start, then accepted changes. The score chart separates all five starts. The completed replay shows the best returned schedule, which can come from an earlier start. Raw JSON remains under collapsed **Technical details**.

This is recorded-event playback, not live streaming. The algorithm's recorded milliseconds are independent of playback speed. A truncated trace only shows the captured prefix and does not pretend to show the omitted execution. Infeasible/skipped runs keep the board unavailable.

The 3D network legend now lives in a separate wrapping footer, the header is outside the canvas, and only one reset control is shown. In-scene labels use collision suppression as the camera moves; full feeder details remain in the tooltip and inspector.

## Evaluation demo

1. Select S001 and inspect F001, the protected hospital feeder.
2. Change hours; compare CSP and Hill Climbing decisions.
3. Select S017, then S025, and show their different demand/weather/schedules.
4. Select S002: the saved run proved infeasible and Hill Climbing was skipped. Missing decisions are **unavailable**, never OFF or zero served energy.
5. Show demand errors, area fairness, downloadable CSVs, then run a fresh Algorithm Lab demonstration on S001.

## Original pipeline

The interface reads the included saved outputs; it does not retrain on every click.

```powershell
python run_project.py
```

This deliberately retrains the Random Forest and **replaces generated outputs**. It reads, and does not regenerate, original input CSVs. Exact results/timing may differ by dependency version, hardware and search budgets. This is not required to start the app.

```powershell
python run_project.py --max-scenarios 10
```

The subset command **replaces schedules with a subset**. Missing rows are shown as unavailable. Run the full experiment to restore all results. Do not run `dataset_generator.py` just to view the interface. The original EDA notebook remains `01_data_exploration.ipynb`.

## Algorithms and interpretation

1. **Random Forest regression:** predicts numeric feeder demand from time, weather, feeder identity, season/day type and observed previous-hour demand. One-hot encoding handles categories. Scenario-grouped five-fold cross-validation keeps complete scenarios outside their training fold. This is one-step-ahead evaluation, not an autonomous day-ahead forecast.
2. **CSP backtracking:** binary feeder-hour decisions, hospital/protected-hour rules, total/consecutive outage limits and forecast supply constraint. Search limits are distinguished from proven infeasibility.
3. **Hill Climbing:** valid CSP start, valid improving neighboring schedules and seeded perturbed restarts. The normalized objective weights are 0.5 unmet demand, 0.3 imbalance and 0.2 priority-weighted unmet demand. This is local optimization, without a global-optimality guarantee.

**Forecast shortage** = max(0, predicted demand − usable supply). **Actual network shortage** uses actual demand. **Unmet demand** is demand of feeders selected OFF. These are separate quantities. Forecast-valid schedules can overload supply under actual demand; the project tracks this separately.

MW is power; summing hourly MW over one-hour slots gives MWh energy. Total feeder outage hours sum feeder interruptions, not elapsed network hours. Overall model metrics come from the original metric file; filtered metrics use saved rounded forecasts and may differ slightly. Both algorithms use the same scenarios. Improving the combined objective does not guarantee every individual metric improves.

## Structure

- `data/`: original four datasets and metadata.
- `src/`: original AI algorithms with optional observer callbacks; logging is disabled by default.
- `outputs/`: original predictions, schedules, metrics and charts.
- `web_app/app.py`: Flask page, JSON and download routes.
- `web_app/data_service.py`: cached loading, joins and analytics.
- `web_app/lab_service.py`: isolated optional real demonstrations.
- `web_app/templates/command.html`: shared application shell.
- `web_app/static/`: styles, application JavaScript, Three.js network and local vendor dependencies.
- `tests/`: original AI tests and web/data integration tests.
- `docs/`: supplied proposal, original README, audit and actual testing report.

Existing page URLs remain supported through the new shell. Earlier templates remain for reference. Flask remains the backend. No React, database or Node build is required; package.json only supports development previews.

## Tests

```powershell
python -m unittest discover -s tests -v
```

See `docs/TESTING_REPORT.md` for verified coverage and any limits. Trace callbacks only observe events; rules, objective, ordering and seeded random draws are preserved. Logging overhead naturally changes runtime measurements.

## Troubleshooting

- Missing Flask: run `python -m pip install -r requirements.txt` using the same Python as the app.
- Port busy: PowerShell `$env:PORT="5001"`, start again and open http://127.0.0.1:5001.
- No WebGL: the network uses an interactive 2D fallback with identical data. Hardware acceleration improves the 3D experience.
- Missing schedules: read the infeasibility warning or check whether outputs contain a full run. Missing is not OFF.
- After reproducing outputs: wait for completion, then reload. CSV caches check file modification times.

Original team responsibilities and academic details are preserved in `docs/ORIGINAL_README.md` and the supplied proposal. Third-party notices are in `web_app/static/vendor/`.


## Cinematic storm sound and lighting update

Rain uses a quiet stereo wash with occasional drops tapping on glass. Thunder sits above the rain in the mix. After the first flash, strikes recur every 5.5 seconds. Thunder combines a sharp wideband crack, irregular secondary fractures and a 6.5–8 second rolling stereo tail. A peak limiter replaces the earlier heavy compression. Enable Sound at the top and adjust the existing slider. All sound is synthesized locally, so no downloads or API keys are required.

Lightning retains its white core and red halo, with finer branching, short return-stroke pulses, broad cloud illumination, a lower red bounce and strike-linked rain glints. These are stylized 2D lighting cues for reflection and refraction, not physical optical simulation. Storm OFF and reduced-motion controls remain available.
