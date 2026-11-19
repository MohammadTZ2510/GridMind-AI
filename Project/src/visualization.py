"""
visualization.py
================
Creates the charts required by the proposal with Matplotlib and Seaborn.
Every chart is drawn from real experiment output (DataFrames) - nothing is typed in by hand.

Charts
------
1. actual_vs_predicted_demand.png        : actual vs regression vs baselines
2. csp_vs_hill_climbing_runtime.png      : runtime comparison
3. csp_vs_hill_climbing_shortage.png     : unmet demand comparison
4. outage_hours_by_area.png              : fairness - outage hours per area
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # draw to files only (no pop-up windows, works everywhere)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

CSP_COLOR = "#1f77b4"
HC_COLOR = "#ff7f0e"
DPI = 200

sns.set_theme(style="whitegrid", context="notebook")


def _finish(figure: plt.Figure, path: Path) -> Path:
    """Save a figure as a high-quality PNG and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_actual_vs_predicted(predictions: pd.DataFrame, path: Path,
                             example_scenario: str, example_feeder: str) -> Path:
    """Left: network demand by hour averaged over all scenarios. Right: one scenario and feeder."""
    columns = {
        "actual_demand_mw": ("Actual demand", "black", "-"),
        "regression_prediction_mw": ("Regression prediction", HC_COLOR, "-"),
        "previous_hour_baseline_mw": ("Previous-hour baseline", "#2ca02c", "--"),
        "mean_baseline_mw": ("Mean-demand baseline", "#9467bd", ":"),
    }
    figure, (left, right) = plt.subplots(1, 2, figsize=(15, 5.5))

    # Left panel: total network demand per scenario-hour, then the mean over the 50 scenarios.
    network = predictions.groupby(["scenario_id", "hour"])[list(columns)].sum().reset_index()
    by_hour = network.groupby("hour")[list(columns)].mean()
    for column, (label, color, style) in columns.items():
        left.plot(by_hour.index, by_hour[column], label=label, color=color, linestyle=style,
                  linewidth=2.5 if column == "actual_demand_mw" else 1.8)
    left.set_title(f"Network demand by hour, mean of {network['scenario_id'].nunique()} scenarios")
    left.set_xlabel("Hour of day")
    left.set_ylabel("Total demand (MW)")
    left.set_xticks(range(0, 24, 2))
    left.legend()

    # Right panel: a single feeder in a single scenario (out-of-fold forecast).
    example = predictions[(predictions["scenario_id"] == example_scenario)
                          & (predictions["feeder_id"] == example_feeder)].sort_values("hour")
    for column, (label, color, style) in columns.items():
        right.plot(example["hour"], example[column], label=label, color=color, linestyle=style,
                   linewidth=2.5 if column == "actual_demand_mw" else 1.8,
                   marker="o" if column in ("actual_demand_mw", "regression_prediction_mw") else None,
                   markersize=4)
    right.set_title(f"Scenario {example_scenario}, feeder {example_feeder}: hourly demand")
    right.set_xlabel("Hour of day")
    right.set_ylabel("Feeder demand (MW)")
    right.set_xticks(range(0, 24, 2))
    right.legend()

    figure.suptitle("Actual vs Predicted Hourly Demand (SIMULATED DATA)", fontsize=14, fontweight="bold")
    figure.tight_layout()
    return _finish(figure, path)


def _bar_with_error(ax, labels, means, stds, colors, value_format: str) -> None:
    """Bars with standard-deviation whiskers and a value label above each bar."""
    positions = np.arange(len(labels))
    ax.bar(positions, means, yerr=stds, capsize=6, color=colors, edgecolor="black", alpha=0.9)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    top = max(m + s for m, s in zip(means, stds)) if len(means) else 1.0
    for x, m, sd in zip(positions, means, stds):
        ax.text(x, m + sd + 0.02 * top, value_format.format(m), ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, top * 1.12)


def plot_runtime(results: pd.DataFrame, path: Path) -> Path:
    """Mean runtime per scenario (with standard deviation) for the scheduling methods."""
    csp = results[(results["method"] == "CSP") & results["feasible"]]
    hc = results[(results["method"] == "HillClimbing") & results["feasible"]]
    labels = ["CSP\nBacktracking", "Hill Climbing\n(improvement step only)", "Hill Climbing\n(CSP start + improvement)"]
    values = [csp["runtime_s"], hc["runtime_s"], hc["total_runtime_incl_csp_s"]]
    means = [v.mean() * 1000 for v in values]
    stds = [v.std(ddof=0) * 1000 for v in values]
    figure, ax = plt.subplots(figsize=(8.5, 5.5))
    _bar_with_error(ax, labels, means, stds, [CSP_COLOR, HC_COLOR, "#d62728"], "{:.1f} ms")
    medians = [v.median() * 1000 for v in values]
    ax.scatter(np.arange(3), medians, marker="D", s=70, color="white", edgecolor="black", zorder=5,
               label="Median (typical scenario): " + ", ".join(f"{m:.1f} ms" for m in medians))
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylabel("Runtime per scenario (milliseconds)")
    ax.set_title(f"CSP vs Hill Climbing: runtime\n(mean ± std over {len(csp)} feasible scenarios)")
    return _finish(figure, path)


def plot_shortage(results: pd.DataFrame, path: Path) -> Path:
    """Unmet demand of CSP and Hill Climbing next to the unavoidable network shortage."""
    csp = results[(results["method"] == "CSP") & results["feasible"]]
    hc = results[(results["method"] == "HillClimbing") & results["feasible"]]
    labels = ["Network shortage\n(minimum to shed)", "CSP\nunmet demand", "Hill Climbing\nunmet demand"]
    values = [csp["network_shortage_mwh"], csp["actual_unmet_mwh"], hc["actual_unmet_mwh"]]
    means = [v.mean() for v in values]
    stds = [v.std(ddof=0) for v in values]
    figure, ax = plt.subplots(figsize=(8.5, 5.5))
    _bar_with_error(ax, labels, means, stds, ["#7f7f7f", CSP_COLOR, HC_COLOR], "{:.1f} MWh")
    ax.set_ylabel("Demand not served per day (MWh)")
    ax.set_title(f"CSP vs Hill Climbing: unmet demand per scenario\n"
                 f"(actual demand, mean ± std over {len(csp)} feasible scenarios)")
    return _finish(figure, path)


def plot_outage_by_area(results: pd.DataFrame, areas: list[str], path: Path) -> Path:
    """Left: outage hours per area (mean over scenarios). Right: spread of the imbalance metric."""
    feasible = results[results["feasible"]]
    area_columns = [f"area_outage_{a}" for a in areas]
    long = feasible.melt(id_vars=["scenario_id", "method"], value_vars=area_columns,
                         var_name="area", value_name="outage_hours")
    long["area"] = long["area"].str.replace("area_outage_", "", regex=False)
    long["method"] = long["method"].replace({"HillClimbing": "Hill Climbing"})

    figure, (left, right) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [2, 1]})
    sns.barplot(data=long, x="area", y="outage_hours", hue="method", order=areas,
                hue_order=["CSP", "Hill Climbing"], palette=[CSP_COLOR, HC_COLOR],
                errorbar="sd", capsize=0.1, edgecolor="black", ax=left)
    left.set_title("Outage hours by area (mean ± std over scenarios)")
    left.set_xlabel("Area")
    left.set_ylabel("Outage hours per sheddable feeder (hours/day)")
    left.set_ylim(0, left.get_ylim()[1] * 1.2)           # head-room so the legend never covers the bars
    left.legend(title="Method", loc="upper center", ncol=2)

    imbalance = feasible.copy()
    imbalance["method"] = imbalance["method"].replace({"HillClimbing": "Hill Climbing"})
    sns.boxplot(data=imbalance, x="method", y="area_outage_imbalance", order=["CSP", "Hill Climbing"],
                hue="method", palette=[CSP_COLOR, HC_COLOR], legend=False, ax=right)
    right.set_title("Fairness: outage-hour imbalance\n(std of area outage hours, lower = fairer)")
    right.set_xlabel("Method")
    right.set_ylabel("Imbalance (hours)")

    figure.suptitle("Fair Distribution of Outage Hours by Area (SIMULATED DATA)", fontsize=14, fontweight="bold")
    figure.tight_layout()
    return _finish(figure, path)
