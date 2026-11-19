"""
demand_predictor.py
===================
`DemandPredictor` - forecasts hourly feeder demand with Regression and compares
it with the two simple baselines from the proposal.

Model choice: RandomForestRegressor
-----------------------------------
Demand depends NON-LINEARLY on the hour of the day (morning/evening peaks), and
the shape differs per feeder type (residential, commercial, industrial, hospital).
A random forest handles these interactions automatically, needs little tuning and
is robust to noise. (A plain linear model was tried during development and was
clearly worse.) It is trained with random_state=42, so results are reproducible.

Features (only information available when the forecast is made)
---------------------------------------------------------------
hour, hour_sin, hour_cos, previous_hour_demand_mw, temperature_c, humidity_pct,
is_peak_hour, day_type, season, feeder_id.
This is a ONE-STEP-AHEAD forecast: the previous hour's demand has already been
observed. `actual_demand_mw` (the target) and future hours are never used as inputs.

Fair evaluation without leakage
-------------------------------
Rows from the same scenario are strongly related (same weather, same day). A random
row split would leak information, so we use scenario-grouped cross-validation
(GroupKFold): each of the 50 scenarios is predicted by a model that never saw ANY
hour of that scenario. The result is an out-of-fold forecast for all 50 scenarios,
which the schedulers then use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import config


class DemandPredictor:
    """Regression demand forecaster plus baselines and scenario-grouped evaluation."""

    def __init__(self, n_estimators: int = config.MODEL_N_ESTIMATORS,
                 random_state: int = config.RANDOM_SEED, n_jobs: int = -1) -> None:
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.categorical_features = list(config.CATEGORICAL_FEATURES)
        self.numeric_features = list(config.NUMERIC_FEATURES)
        self.model: Pipeline | None = None

    # ------------------------------------------------------------------ model
    def _build_pipeline(self) -> Pipeline:
        """One-hot encode the categorical columns, pass numeric columns through, then regress."""
        encoder = ColumnTransformer(
            transformers=[("categorical", OneHotEncoder(handle_unknown="ignore"), self.categorical_features)],
            remainder="passthrough",   # numeric columns are used unchanged
        )
        forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=config.MODEL_MIN_SAMPLES_LEAF,
            max_features=config.MODEL_MAX_FEATURES,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        return Pipeline([("encode", encoder), ("regressor", forest)])

    # --------------------------------------------------------------- features
    def prepare_features(self, demand: pd.DataFrame, weather: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Build the model input table X and the target y (actual_demand_mw).

        The `is_peak_hour` flag comes from weather_time.csv (joined on scenario + hour).
        Row order of `demand` is preserved, so X, y and `demand` line up row by row.
        """
        peak = weather[["scenario_id", "hour", "is_peak_hour"]]
        data = demand.merge(peak, on=["scenario_id", "hour"], how="left", validate="many_to_one")
        if data["is_peak_hour"].isna().any():
            raise ValueError("Some demand rows have no matching weather_time record")
        data["is_peak_hour"] = data["is_peak_hour"].astype(int)
        data["hour_sin"] = np.sin(2.0 * np.pi * data["hour"] / config.HOURS_PER_SCENARIO)
        data["hour_cos"] = np.cos(2.0 * np.pi * data["hour"] / config.HOURS_PER_SCENARIO)

        features = data[self.categorical_features + self.numeric_features].copy()
        for column in self.categorical_features:
            features[column] = features[column].astype(str)
        target = data[config.TARGET_COLUMN].astype(float)
        return features, target

    # ------------------------------------------------------- train / predict
    def train(self, features: pd.DataFrame, target: pd.Series) -> "DemandPredictor":
        """Fit the regression pipeline."""
        self.model = self._build_pipeline()
        self.model.fit(features, target)
        return self

    def predict(self, features: pd.DataFrame, capacity: pd.Series | np.ndarray | None = None) -> np.ndarray:
        """Predict demand in MW. Output is never negative and (if capacity is given) never above capacity."""
        if self.model is None:
            raise RuntimeError("Model is not trained yet - call train() first")
        predictions = np.maximum(self.model.predict(features), 0.0)
        if capacity is not None:
            predictions = np.minimum(predictions, np.asarray(capacity, dtype=float))
        return predictions

    # ------------------------------------------------------------- baselines
    @staticmethod
    def baseline_previous_hour(demand: pd.DataFrame) -> np.ndarray:
        """Baseline 1: forecast = the previous hour's demand."""
        return demand["previous_hour_demand_mw"].to_numpy(dtype=float)

    @staticmethod
    def baseline_mean_demand(train_demand: pd.DataFrame, target_demand: pd.DataFrame) -> np.ndarray:
        """Baseline 2: forecast = the feeder's mean demand, computed from TRAINING data only."""
        feeder_means = train_demand.groupby("feeder_id")[config.TARGET_COLUMN].mean()
        return target_demand["feeder_id"].map(feeder_means).to_numpy(dtype=float)

    # ------------------------------------------------------------- evaluation
    @staticmethod
    def evaluate(y_true, y_pred) -> dict[str, float]:
        """MAE = mean(|error|);  RMSE = sqrt(mean(error^2)).  Both in MW."""
        return {
            "MAE_MW": float(mean_absolute_error(y_true, y_pred)),
            "RMSE_MW": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        }

    def cross_validate_by_scenario(self, demand: pd.DataFrame, weather: pd.DataFrame,
                                   n_splits: int = config.N_CV_FOLDS) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Scenario-grouped K-fold evaluation.

        Returns
        -------
        predictions : one row per scenario-hour-feeder with actual demand, both baselines and the
                      out-of-fold regression prediction (columns required by the project).
        metrics     : MAE/RMSE for each model, per fold and overall.
        """
        demand = demand.reset_index(drop=True)
        features, target = self.prepare_features(demand, weather)
        groups = demand["scenario_id"]

        previous_hour = np.zeros(len(demand))
        mean_baseline = np.zeros(len(demand))
        regression = np.zeros(len(demand))
        fold_of_row = np.zeros(len(demand), dtype=int)
        metric_rows = []

        splitter = GroupKFold(n_splits=n_splits)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(features, target, groups), start=1):
            if set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]):
                raise RuntimeError("Scenario leakage: a scenario appears in both train and test")
            self.train(features.iloc[train_idx], target.iloc[train_idx])
            test_rows = demand.iloc[test_idx]

            previous_hour[test_idx] = self.baseline_previous_hour(test_rows)
            mean_baseline[test_idx] = self.baseline_mean_demand(demand.iloc[train_idx], test_rows)
            regression[test_idx] = self.predict(features.iloc[test_idx], test_rows["feeder_capacity_mw"])
            fold_of_row[test_idx] = fold

            for model_name, values in (("Previous-hour baseline", previous_hour),
                                       ("Mean-demand baseline", mean_baseline),
                                       ("Regression (RandomForest)", regression)):
                row = {"model": model_name, "fold": str(fold), "n_rows": len(test_idx),
                       "n_scenarios": int(groups.iloc[test_idx].nunique())}
                row.update(self.evaluate(target.iloc[test_idx], values[test_idx]))
                metric_rows.append(row)

        # Overall (all 50 scenarios, every row predicted exactly once out-of-fold)
        for model_name, values in (("Previous-hour baseline", previous_hour),
                                   ("Mean-demand baseline", mean_baseline),
                                   ("Regression (RandomForest)", regression)):
            row = {"model": model_name, "fold": "overall", "n_rows": len(demand),
                   "n_scenarios": int(groups.nunique())}
            row.update(self.evaluate(target, values))
            metric_rows.append(row)

        predictions = pd.DataFrame({
            "scenario_id": demand["scenario_id"],
            "hour": demand["hour"],
            "feeder_id": demand["feeder_id"],
            "actual_demand_mw": demand["actual_demand_mw"].round(3),
            "previous_hour_baseline_mw": np.round(previous_hour, 3),
            "mean_baseline_mw": np.round(mean_baseline, 3),
            "regression_prediction_mw": np.round(regression, 3),
            "cv_fold": fold_of_row,
        })
        metrics = pd.DataFrame(metric_rows)
        overall = metrics[metrics["fold"] == "overall"].set_index("model")["MAE_MW"]
        metrics["mae_improvement_vs_previous_hour_pct"] = np.where(
            metrics["fold"] == "overall",
            100.0 * (overall["Previous-hour baseline"] - metrics["MAE_MW"]) / overall["Previous-hour baseline"],
            np.nan)
        return predictions, metrics
