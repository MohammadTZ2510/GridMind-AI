"""
tests/test_project.py
=====================
Tests for the AL-2002 load-shedding project. They use the REAL project code and the
real (simulated) CSV files. Run from the project root with either:

    python -m unittest discover -s tests -v
    python -m pytest tests -v            (only if pytest is installed)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.csp_scheduler import CSPScheduler  # noqa: E402
from src.data_loader import DataLoader, parse_protected_hours  # noqa: E402
from src.demand_predictor import DemandPredictor  # noqa: E402
from src.environment import LoadSheddingEnvironment, longest_off_run  # noqa: E402
from src.evaluation import EvaluationReport  # noqa: E402
from src.hill_climbing_scheduler import HillClimbingScheduler  # noqa: E402
from src.scheduler_base import STATUS_FEASIBLE, STATUS_INFEASIBLE  # noqa: E402

ON, OFF = config.ON, config.OFF


class ProjectTestCase(unittest.TestCase):
    """Loads the data once and trains a small, fast forecast model for all tests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.data = DataLoader().load()
        cls.predictor = DemandPredictor(n_estimators=30)        # small forest = fast tests
        cls.predictions, cls.regression_metrics = cls.predictor.cross_validate_by_scenario(
            cls.data.demand, cls.data.weather)
        cls.env = LoadSheddingEnvironment(cls.data)
        cls.env.set_predictions(cls.predictions)

    def idx(self, feeder_id: str) -> int:
        return self.env.feeder_index[feeder_id]

    def all_on(self) -> np.ndarray:
        return np.ones((self.env.n_feeders, self.env.n_hours), dtype=int)

    def scenario_where_all_on_is_valid(self) -> str:
        """A scenario with no supply shortage (planning demand) so that 'everything ON' is a valid schedule."""
        for scenario_id in config.SCENARIO_IDS:
            self.env.load_scenario(scenario_id)
            if self.env.is_valid(self.all_on(), "planning"):
                return scenario_id
        self.fail("No scenario without shortage was found")

    def feasible_results(self, count: int):
        """CSP results for the first `count` feasible scenarios."""
        found = []
        for scenario_id in config.SCENARIO_IDS:
            self.env.load_scenario(scenario_id)
            result = CSPScheduler(self.env).generate_schedule()
            if result.is_feasible and result.details.get("total_required_shed_mwh", 0) > 0:
                found.append((scenario_id, result))
            if len(found) == count:
                break
        return found


class TestData(ProjectTestCase):
    def test_1_dataset_loading_and_row_counts(self):
        self.assertEqual(len(self.data.feeders), 10)
        self.assertEqual(len(self.data.demand), 12000)
        self.assertEqual(len(self.data.supply), 1200)
        self.assertEqual(len(self.data.weather), 1200)
        self.assertTrue(all(line.startswith("[PASS]") for line in self.data.validation_log))

    def test_2_feeder_count(self):
        self.assertEqual(self.env.n_feeders, config.NUM_FEEDERS)
        self.assertEqual(len(self.env.areas), config.NUM_AREAS)
        for area in self.env.areas:
            self.assertEqual(self.env.feeder_area.count(area), config.FEEDERS_PER_AREA)

    def test_3_scenario_count(self):
        self.assertEqual(len(self.data.scenario_ids), 50)
        self.assertEqual(self.data.scenario_ids, config.SCENARIO_IDS)
        self.env.load_scenario("S001")
        self.assertEqual(self.env.actual_demand.shape, (10, 24))

    def test_protected_hours_parser(self):
        self.assertEqual(parse_protected_hours(""), frozenset())
        self.assertEqual(parse_protected_hours("08,09,10"), frozenset({8, 9, 10}))


class TestHardConstraints(ProjectTestCase):
    """is_valid() must accept a good schedule and reject every kind of violation."""

    def setUp(self) -> None:
        self.scenario = self.scenario_where_all_on_is_valid()
        self.env.load_scenario(self.scenario)

    def test_baseline_schedule_is_valid(self):
        self.assertEqual(self.env.get_violations(self.all_on(), "planning"), [])

    def test_4_hospital_protection(self):
        schedule = self.all_on()
        schedule[self.idx(config.HOSPITAL_FEEDER_ID), 3] = OFF
        violations = self.env.get_violations(schedule)
        self.assertFalse(self.env.is_valid(schedule))
        self.assertTrue(any("hospital" in v for v in violations))

    def test_5_protected_hours(self):
        # F004 is protected at hour 12 (see feeders.csv)
        self.assertIn(12, self.env.protected_hours[self.idx("F004")])
        schedule = self.all_on()
        schedule[self.idx("F004"), 12] = OFF
        self.assertFalse(self.env.is_valid(schedule))

    def test_6_max_total_outage_hours(self):
        feeder = self.idx("F006")                       # limit: 6 outage hours
        self.assertEqual(self.env.max_total_outage[feeder], 6)
        schedule = self.all_on()
        schedule[feeder, [0, 2, 4, 6, 8, 10, 12]] = OFF   # 7 separate hours (no consecutive problem)
        violations = self.env.get_violations(schedule)
        self.assertTrue(any("outage hours" in v for v in violations))
        schedule[feeder, 12] = ON                         # exactly 6 -> allowed
        self.assertEqual(self.env.get_violations(schedule), [])

    def test_7_max_consecutive_outage_hours(self):
        feeder = self.idx("F006")                       # limit: 2 consecutive hours
        self.assertEqual(self.env.max_consecutive_outage[feeder], 2)
        schedule = self.all_on()
        schedule[feeder, [5, 6, 7]] = OFF
        self.assertTrue(any("consecutive" in v for v in self.env.get_violations(schedule)))
        schedule[feeder, 7] = ON
        self.assertEqual(self.env.get_violations(schedule), [])
        self.assertEqual(longest_off_run([1, 0, 0, 0, 1, 0]), 3)

    def test_8_supply_capacity(self):
        # Find a scenario where everything ON exceeds the usable supply in some hour.
        for scenario_id in config.SCENARIO_IDS:
            self.env.load_scenario(scenario_id)
            if (self.env.planning_demand.sum(axis=0) > self.env.usable_supply + 1e-6).any():
                break
        violations = self.env.get_violations(self.all_on())
        self.assertTrue(any("usable supply" in v for v in violations))

    def test_9_binary_values_only(self):
        for bad_value in (2, -1):
            schedule = self.all_on()
            schedule[1, 1] = bad_value
            self.assertFalse(self.env.is_valid(schedule))
        wrong_shape = np.ones((10, 23), dtype=int)
        self.assertFalse(self.env.is_valid(wrong_shape))
        for _, result in self.feasible_results(5):
            self.assertTrue(np.isin(result.schedule, (0, 1)).all())


class TestForecasting(ProjectTestCase):
    def test_10_predictions_non_negative_and_within_capacity(self):
        p = self.predictions
        self.assertEqual(len(p), 12000)
        self.assertTrue((p["regression_prediction_mw"] >= 0).all())
        capacity = p["feeder_id"].map(self.data.feeders.set_index("feeder_id")["feeder_capacity_mw"])
        self.assertTrue((p["regression_prediction_mw"] <= capacity + 1e-9).all())
        for column in ["scenario_id", "hour", "feeder_id", "actual_demand_mw", "previous_hour_baseline_mw",
                       "mean_baseline_mw", "regression_prediction_mw"]:
            self.assertIn(column, p.columns)

    def test_no_scenario_leakage_and_target_not_a_feature(self):
        features, _ = self.predictor.prepare_features(self.data.demand, self.data.weather)
        self.assertNotIn("actual_demand_mw", features.columns)
        # every scenario is predicted exactly once (out-of-fold), by a model trained on the other folds
        self.assertEqual(self.predictions.groupby("scenario_id")["cv_fold"].nunique().max(), 1)

    def test_regression_beats_baselines(self):
        overall = self.regression_metrics[self.regression_metrics["fold"] == "overall"].set_index("model")["MAE_MW"]
        self.assertLess(overall["Regression (RandomForest)"], overall["Previous-hour baseline"])
        self.assertLess(overall["Regression (RandomForest)"], overall["Mean-demand baseline"])


class TestSchedulers(ProjectTestCase):
    def test_11_schedule_validity_csp_and_hill_climbing(self):
        for scenario_id, csp in self.feasible_results(6):
            self.env.load_scenario(scenario_id)
            self.assertEqual(csp.status, STATUS_FEASIBLE)
            self.assertEqual(self.env.get_violations(csp.schedule, "planning"), [])
            hc = HillClimbingScheduler(self.env, initial_schedule=csp.schedule).generate_schedule()
            self.assertEqual(self.env.get_violations(hc.schedule, "planning"), [])
            self.assertTrue((hc.schedule[self.idx(config.HOSPITAL_FEEDER_ID)] == ON).all())
            self.assertLessEqual(hc.details["final_score"], hc.details["initial_score"] + 1e-12)

    def test_12_shortage_calculation(self):
        scenario_id, csp = self.feasible_results(1)[0]
        self.env.load_scenario(scenario_id)
        for basis in ("actual", "planning"):
            demand = self.env.demand_for(basis)
            report = self.env.calculate_shortage(csp.schedule, basis)
            expected_served = (csp.schedule * demand).sum(axis=0)
            expected_unmet = ((1 - csp.schedule) * demand).sum(axis=0)
            np.testing.assert_allclose(report["served_demand_mw"], expected_served)
            np.testing.assert_allclose(report["unmet_demand_mw"], expected_unmet)
            np.testing.assert_allclose(report["served_demand_mw"] + report["unmet_demand_mw"], demand.sum(axis=0))
        # network shortage (actual basis) must equal shortage_mw stored in hourly_supply.csv
        supply = self.data.supply[self.data.supply["scenario_id"] == scenario_id].sort_values("hour")
        report = self.env.calculate_shortage(csp.schedule, "actual")
        np.testing.assert_allclose(report["network_shortage_mw"], supply["shortage_mw"].to_numpy(), atol=1e-3)
        # a valid planning schedule can never leave planned unmet demand below the network shortage
        planned = self.env.calculate_shortage(csp.schedule, "planning")
        self.assertTrue((planned["unmet_demand_mw"] + 1e-6 >= planned["network_shortage_mw"]).all())

    def test_13_hill_climbing_never_accepts_invalid_schedule(self):
        checked = 0
        for scenario_id, csp in self.feasible_results(5):
            self.env.load_scenario(scenario_id)
            hc = HillClimbingScheduler(self.env, initial_schedule=csp.schedule, record_history=True)
            hc.generate_schedule()
            self.assertGreater(len(hc.accepted_history), 0)
            for accepted in hc.accepted_history:
                self.assertEqual(self.env.get_violations(accepted, "planning"), [])
                checked += 1
        self.assertGreater(checked, 0)

    def test_hill_climbing_score_decreases_with_each_accepted_move(self):
        scenario_id, csp = self.feasible_results(1)[0]
        self.env.load_scenario(scenario_id)
        hc = HillClimbingScheduler(self.env, initial_schedule=csp.schedule, num_restarts=1, record_history=True)
        hc.generate_schedule()
        scores = [hc.score_schedule(csp.schedule)] + [hc.score_schedule(s) for s in hc.accepted_history]
        self.assertTrue(all(b < a for a, b in zip(scores, scores[1:])))

    def test_hill_climbing_rejects_invalid_start(self):
        self.env.load_scenario(self.scenario_where_all_on_is_valid())
        bad = self.all_on()
        bad[self.idx(config.HOSPITAL_FEEDER_ID), 0] = OFF
        with self.assertRaises(ValueError):
            HillClimbingScheduler(self.env, initial_schedule=bad).generate_schedule()

    def test_infeasible_scenario_is_reported_not_faked(self):
        self.env.load_scenario("S002", use_predictions=False)    # supply far below what can be shed
        result = CSPScheduler(self.env).generate_schedule()
        self.assertEqual(result.status, STATUS_INFEASIBLE)
        self.assertIsNone(result.schedule)
        self.assertIn("supply", result.cause)
        self.assertTrue(result.reason)
        hc = HillClimbingScheduler(self.env).generate_schedule()
        self.assertEqual(hc.status, STATUS_INFEASIBLE)
        self.assertIsNone(hc.schedule)

    def test_fairness_metric_and_area_aggregation(self):
        self.env.load_scenario("S001")
        schedule = self.all_on()
        self.assertEqual(self.env.outage_imbalance(schedule), 0.0)
        schedule[self.idx("F003"), 5] = OFF               # one outage hour in Gulshan only
        area = self.env.get_area_outage_hours(schedule)
        self.assertEqual(area["Gulshan"], 0.5)            # mean over its 2 sheddable feeders
        self.assertEqual(self.env.get_area_outage_hours(schedule, per_feeder=False)["Gulshan"], 1.0)
        self.assertGreater(self.env.outage_imbalance(schedule), 0.0)


class TestReproducibility(ProjectTestCase):
    def test_14_reproducibility_with_seed_42(self):
        self.assertEqual(config.RANDOM_SEED, 42)
        # forecasts
        again, _ = DemandPredictor(n_estimators=30).cross_validate_by_scenario(self.data.demand, self.data.weather)
        pd.testing.assert_frame_equal(self.predictions, again)
        # CSP + Hill Climbing
        scenario_id, first = self.feasible_results(1)[0]
        self.env.load_scenario(scenario_id)
        second = CSPScheduler(self.env).generate_schedule()
        np.testing.assert_array_equal(first.schedule, second.schedule)
        hc1 = HillClimbingScheduler(self.env, initial_schedule=first.schedule).generate_schedule()
        hc2 = HillClimbingScheduler(self.env, initial_schedule=first.schedule).generate_schedule()
        np.testing.assert_array_equal(hc1.schedule, hc2.schedule)


class TestEvaluation(ProjectTestCase):
    def test_run_trials_and_compare_methods(self):
        report = EvaluationReport(self.env, self.predictions, self.regression_metrics,
                                  scenario_ids=["S001", "S002", "S003"], log=lambda message: None)
        results = report.run_trials()
        summary = report.compare_methods()
        self.assertEqual(set(results["scenario_id"]), {"S001", "S002", "S003"})
        # S002 is infeasible: reported, not hidden
        csp_s002 = results[(results["scenario_id"] == "S002") & (results["method"] == "CSP")].iloc[0]
        self.assertFalse(csp_s002["feasible"])
        self.assertEqual(len(report.infeasible_rows), 1)
        self.assertTrue((results.loc[results["feasible"], "hard_constraint_violations"] == 0).all())
        self.assertEqual(int(summary["n_scenarios"].iloc[0]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
