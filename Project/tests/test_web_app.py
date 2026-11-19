"""Data-grounded integration checks for the complete GRIDMIND interface.

Run from the project root: python -m unittest discover -s tests -p test_web_app.py -v
The exhaustive test exercises every scenario/hour/method combination without
regenerating input data or overwriting saved experiments.
"""
from __future__ import annotations

import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web_app"))


def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in (ROOT / "data", ROOT / "outputs")
            for p in folder.rglob("*") if p.is_file()}


class TestWebApplication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.started = time.perf_counter()
        cls.app_module = importlib.import_module("web_app.app")
        cls.app_module.app.config.update(TESTING=True)
        cls.client = cls.app_module.app.test_client()
        cls.before = source_hashes()
        cls.demand = pd.read_csv(ROOT / "data/hourly_demand.csv")
        cls.weather = pd.read_csv(ROOT / "data/weather_time.csv")
        cls.supply = pd.read_csv(ROOT / "data/hourly_supply.csv")
        cls.predictions = pd.read_csv(ROOT / "outputs/predictions/demand_predictions.csv")
        cls.csp = pd.read_csv(ROOT / "outputs/schedules/csp_schedules.csv")
        cls.hc = pd.read_csv(ROOT / "outputs/schedules/hill_climbing_schedules.csv")
        cls.context_count = 0
        cls.boot = cls.get_json("/api/bootstrap")

    @classmethod
    def get_json(cls, url):
        result = cls.client.get(url)
        if result.status_code != 200:
            raise AssertionError(f"{url}: HTTP {result.status_code}, {result.get_data(as_text=True)[:300]}")
        # The API must emit JSON null, never the nonstandard NaN/Infinity tokens.
        body = result.get_data(as_text=True)
        return json.loads(body, parse_constant=lambda value: (_ for _ in ()).throw(AssertionError(value)))

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "before"):
            if source_hashes() != cls.before:
                raise AssertionError("Web access or laboratory mutated original data/outputs")
            print(f"\nGRIDMIND verified {cls.context_count} scenario-hour-method API contexts in "
                  f"{time.perf_counter() - cls.started:.2f}s; input/output hashes unchanged.")

    def test_01_bootstrap_and_datasets(self):
        self.assertEqual([s["scenario_id"] for s in self.boot["scenarios"]],
                         [f"S{i:03}" for i in range(1, 51)])
        self.assertEqual(len(self.boot["feeders"]), 10)
        self.assertEqual(len(self.boot["areas"]), 5)
        self.assertGreaterEqual(len(self.boot["datasets"]), 10)
        self.assertEqual(len(self.boot["model_metrics"]), 3)
        self.assertEqual(sum(s["status"] == "INFEASIBLE" for s in self.boot["scenarios"]), 1)

    def test_02_all_2400_scenario_hour_method_contexts(self):
        demand = self.demand.set_index(["scenario_id", "hour", "feeder_id"])
        predicted = self.predictions.set_index(["scenario_id", "hour", "feeder_id"])
        supply = self.supply.set_index(["scenario_id", "hour"])
        weather = self.weather.set_index(["scenario_id", "hour"])
        schedules = {"CSP": self.csp.set_index(["scenario_id", "hour", "feeder_id"]),
                     "HC": self.hc.set_index(["scenario_id", "hour", "feeder_id"])}
        for sid in [f"S{i:03}" for i in range(1, 51)]:
            for hour in range(24):
                for method in ("CSP", "HC"):
                    with self.subTest(scenario=sid, hour=hour, method=method):
                        obj = self.get_json(f"/api/scenario/{sid}?hour={hour}&method={method}")
                        self.__class__.context_count += 1
                        self.assertEqual((obj["scenario_id"], obj["hour"], obj["method"]), (sid, hour, method))
                        self.assertEqual(len(obj["feeders"]), 10)
                        self.assertEqual(len(obj["timeline"]), 24)
                        self.assertEqual(len(obj["weather_history"]), 24)
                        self.assertEqual(len(obj["feeder_history"]), 240)
                        self.assertEqual(len(obj["areas"]), 5)
                        expected_demand = demand.loc[(sid, hour), "actual_demand_mw"].sum()
                        expected_forecast = predicted.loc[(sid, hour), "regression_prediction_mw"].sum()
                        totals = obj["totals"]
                        self.assertAlmostEqual(totals["actual_demand_mw"], expected_demand, places=5)
                        self.assertAlmostEqual(totals["predicted_demand_mw"], expected_forecast, places=5)
                        self.assertAlmostEqual(totals["usable_supply_mw"], supply.loc[(sid,hour), "usable_supply_mw"], places=5)
                        self.assertAlmostEqual(totals["forecast_shortage_mw"], max(0, expected_forecast - totals["usable_supply_mw"]), places=5)
                        self.assertAlmostEqual(obj["weather"]["temperature_c"], weather.loc[(sid,hour), "temperature_c"], places=5)
                        self.assertAlmostEqual(obj["weather"]["humidity_pct"], weather.loc[(sid,hour), "humidity_pct"], places=5)
                        for feeder in obj["feeders"]:
                            key = (sid, hour, feeder["feeder_id"])
                            self.assertAlmostEqual(feeder["actual_demand_mw"], demand.loc[key, "actual_demand_mw"], places=5)
                            self.assertAlmostEqual(feeder["regression_prediction_mw"], predicted.loc[key, "regression_prediction_mw"], places=5)
                            if key in schedules[method].index:
                                expected = schedules[method].loc[key]
                                self.assertEqual(feeder["decision"], expected["decision"])
                                self.assertAlmostEqual(feeder["served_demand_mw"], expected["served_demand_mw"], places=5)
                                self.assertAlmostEqual(feeder["unmet_demand_mw"], expected["unmet_demand_mw"], places=5)
                            else:
                                self.assertIsNone(feeder["decision"])
                                self.assertIsNone(feeder["served_demand_mw"])
                                self.assertIsNone(feeder["unmet_demand_mw"])
                        if sid == "S002":
                            self.assertFalse(obj["schedule_available"])
                            self.assertEqual(totals["unknown_count"], 10)
                            self.assertIsNotNone(obj["infeasible"])
                        else:
                            self.assertTrue(obj["schedule_available"])
                            self.assertEqual(totals["on_count"] + totals["off_count"], 10)
                            self.assertEqual(next(f for f in obj["feeders"] if f["feeder_id"]=="F001")["decision"], "ON")

    def test_03_page_routes(self):
        for path in ("/", "/scenarios", "/scenario/S001", "/scenario/S002", "/scenario/S017/hour/19",
                     "/scenario/S025/hour/13/feeder/F003", "/scenario/S050", "/demand", "/weather",
                     "/fairness", "/algorithm-lab", "/data"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn(b"GRIDMIND", response.data)
        for path in ("/api/scenario/S999", "/api/scenario/S001?hour=24", "/api/scenario/S001?hour=-1",
                     "/api/scenario/S001?hour=abc", "/api/scenario/S001?method=INVALID",
                     "/scenario/S999", "/scenario/S001/hour/24", "/scenario/S001/hour/1/feeder/F999",
                     "/api/data/not-a-dataset"):
            with self.subTest(path=path):
                self.assertIn(self.client.get(path).status_code, (400,404))

    def test_04_dataset_downloads_search_sort_and_pagination(self):
        for dataset in self.boot["datasets"]:
            identifier = dataset["id"]
            with self.subTest(dataset=identifier):
                data = self.get_json(f"/api/data/{identifier}?page=1&page_size=7")
                self.assertLessEqual(len(data["rows"]), 7)
                self.assertEqual(data["columns"], dataset["columns"])
                response = self.client.get(f"/api/data/{identifier}/download")
                self.assertEqual(response.status_code, 200)
                downloaded = pd.read_csv(io.BytesIO(response.data), keep_default_na=False)
                self.assertEqual(len(downloaded), dataset["rows"])
                self.assertEqual(downloaded.columns.tolist(), dataset["columns"])
        ds = next(d["id"] for d in self.boot["datasets"] if d["filename"]=="hourly_demand.csv")
        selected = self.get_json(f"/api/data/{ds}?scenario=S017&hour=19&page_size=100")
        self.assertEqual(len(selected["rows"]), 10)
        self.assertTrue(all(r["scenario_id"]=="S017" and r["hour"]==19 for r in selected["rows"]))
        sorted_data = self.get_json(f"/api/data/{ds}?scenario=S017&hour=19&sort=actual_demand_mw&direction=desc&page_size=100")
        values = [r["actual_demand_mw"] for r in sorted_data["rows"]]
        self.assertEqual(values, sorted(values,reverse=True))
        searched = self.get_json(f"/api/data/{ds}?scenario=S017&hour=19&q=F001")
        self.assertEqual(len(searched["rows"]), 1)
        self.assertEqual(searched["rows"][0]["feeder_id"], "F001")
        page1 = self.get_json(f"/api/data/{ds}?page=1&page_size=7")
        page2 = self.get_json(f"/api/data/{ds}?page=2&page_size=7")
        self.assertNotEqual(page1["rows"], page2["rows"])

    def test_05_missing_schedule_data_is_unknown_not_off(self):
        from web_app.data_service import DataStore, get_store
        # An isolated in-memory store simulates damaged/missing output rows.
        # The original files and shared cache are left untouched.
        isolated = DataStore(get_store().fingerprint)
        frame = isolated.frames["csp_schedules"]
        missing = ((frame["scenario_id"] == "S017") & (frame["hour"] == 19)
                   & (frame["feeder_id"] == "F003"))
        isolated.frames["csp_schedules"] = frame[~missing].copy()
        with patch.object(self.app_module, "get_store", return_value=isolated):
            obj = self.get_json("/api/scenario/S017?hour=19&method=CSP")
            row = next(f for f in obj["feeders"] if f["feeder_id"] == "F003")
            self.assertIsNone(row["decision"])
            self.assertIsNone(row["served_demand_mw"])
            self.assertIsNone(row["unmet_demand_mw"])
            self.assertFalse(obj["schedule_available"])
            self.assertEqual(obj["totals"]["unknown_count"], 1)
            self.assertIsNone(obj["totals"]["served_demand_mw"])
            self.assertIsNone(obj["totals"]["unmet_demand_mw"])
            self.assertFalse(obj["comparison"]["available"])
            self.assertIsNone(obj["comparison"]["identical"])
            self.assertFalse(obj["fairness"]["CSP"]["available"])
            self.assertTrue(obj["fairness"]["HC"]["available"])

    def test_06_optional_observers_preserve_algorithm_outputs(self):
        from src.data_loader import DataLoader
        from src.environment import LoadSheddingEnvironment
        from src.csp_scheduler import CSPScheduler
        from src.hill_climbing_scheduler import HillClimbingScheduler
        env = LoadSheddingEnvironment(DataLoader().load())
        env.set_predictions(self.predictions)
        env.load_scenario("S001")
        baseline = CSPScheduler(env).generate_schedule()
        events = []
        observed = CSPScheduler(env, event_callback=events.append).generate_schedule()
        np.testing.assert_array_equal(baseline.schedule, observed.schedule)
        self.assertTrue(any(e["event"]=="assign" for e in events))
        plain_hc = HillClimbingScheduler(env, initial_schedule=baseline.schedule).generate_schedule()
        hc_events = []
        logged_hc = HillClimbingScheduler(env, initial_schedule=baseline.schedule, event_callback=hc_events.append).generate_schedule()
        np.testing.assert_array_equal(plain_hc.schedule, logged_hc.schedule)
        self.assertEqual(plain_hc.details["final_score"], logged_hc.details["final_score"])
        self.assertTrue(any(e["event"]=="accepted_move" for e in hc_events))
        self.assertEqual(env.get_violations(logged_hc.schedule,"planning"), [])

    def test_07_live_lab_records_real_events(self):
        for method in ("CSP", "HC"):
            with self.subTest(method=method):
                response = self.client.post("/api/lab/run", json={"scenario_id":"S001", "method":method})
                self.assertEqual(response.status_code, 200)
                obj = response.get_json()
                self.assertEqual(obj["status"], "FEASIBLE")
                self.assertTrue(obj["trace"])
                self.assertLessEqual(len(obj["trace"]), 600)
                self.assertEqual(obj["summary"]["constraint_violations"], 0)
                self.assertTrue(all(e["step"]==i+1 for i,e in enumerate(obj["trace"])))
        skipped = self.client.post("/api/lab/run", json={"scenario_id":"S002", "method":"HC"})
        self.assertEqual(skipped.status_code,200)
        self.assertEqual(skipped.get_json()["status"],"INFEASIBLE")
        self.assertIn("skipped", [e["event"] for e in skipped.get_json()["trace"]])


    def test_08_area_fairness_forecast_metrics_and_comparison(self):
        feeder_meta = pd.read_csv(ROOT / "data/feeders.csv", keep_default_na=False)
        sheddable = feeder_meta[(~feeder_meta["is_critical"]) & (feeder_meta["max_total_outage_hours"] > 0)]
        for sid in [f"S{i:03}" for i in range(1,51)]:
            obj = self.get_json(f"/api/scenario/{sid}?hour=13&method=CSP")
            source = self.predictions[self.predictions["scenario_id"] == sid]
            columns = {"Regression (RandomForest)":"regression_prediction_mw",
                       "Previous-hour baseline":"previous_hour_baseline_mw", "Mean-demand baseline":"mean_baseline_mw"}
            for metric in obj["forecast_metrics"]:
                error = source["actual_demand_mw"].to_numpy() - source[columns[metric["model"]]].to_numpy()
                self.assertAlmostEqual(metric["MAE_MW"], float(np.mean(abs(error))), places=6)
                self.assertAlmostEqual(metric["RMSE_MW"], float(np.sqrt(np.mean(error**2))), places=6)
            for method, schedule in [("CSP",self.csp),("HC",self.hc)]:
                sf = schedule[schedule["scenario_id"] == sid]
                if sf.empty:
                    self.assertFalse(obj["fairness"][method]["available"])
                    self.assertIsNone(obj["fairness"][method]["imbalance"])
                    continue
                feeder_hours = sf.assign(outage=sf["decision"].eq("OFF").astype(int)).groupby("feeder_id")["outage"].sum()
                means = []
                for area in obj["areas"]:
                    af = sf[sf["area"] == area["area"]]
                    ids = sheddable.loc[sheddable["area"] == area["area"],"feeder_id"].tolist()
                    expected_mean = float(feeder_hours.loc[ids].mean())
                    means.append(expected_mean)
                    self.assertAlmostEqual(area[method]["mean_outage_hours"], expected_mean, places=7)
                    self.assertEqual(area[method]["outage_hours"], int(af["decision"].eq("OFF").sum()))
                    self.assertAlmostEqual(area[method]["unmet_mwh"], float(af["unmet_demand_mw"].sum()), places=5)
                self.assertAlmostEqual(obj["fairness"][method]["imbalance"], float(np.std(means)), places=7)
                self.assertTrue(obj["fairness"][method]["critical_protected"])
            csp_hour = self.csp[(self.csp["scenario_id"]==sid) & (self.csp["hour"]==13)].set_index("feeder_id")
            hc_hour = self.hc[(self.hc["scenario_id"]==sid) & (self.hc["hour"]==13)].set_index("feeder_id")
            if len(csp_hour)==10 and len(hc_hour)==10:
                expected_diff = csp_hour.index[csp_hour["decision"] != hc_hour["decision"]].tolist()
                self.assertEqual(sorted(obj["comparison"]["different_feeders"]), sorted(expected_diff))
                self.assertEqual(obj["comparison"]["identical"], not expected_diff)
            else:
                self.assertFalse(obj["comparison"]["available"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
