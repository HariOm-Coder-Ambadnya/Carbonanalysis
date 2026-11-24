
"""
CARBON AND COST-AWARE SERVERLESS LOAD BALANCING
Dataset-based graph and image generator.

Run from the project root:
    py generate_dataset_figures.py

Generated images are saved in:
    figures/

Expected dataset locations:
    Coding/FinalWorkloadDataset/final_workload_dataset.csv
    Coding/CarbonAnalysis/workload_carbon_dataset.csv
    Coding/CarbonAnalysis/carbon_intensity_2021_window.csv
    Coding/EnergyModel/results/time_aware_energy_summary.csv
    Coding/EnergyModel/results/time_aware_energy_estimates.csv
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

WORKLOAD_FILE = (
    ROOT / "Coding" / "FinalWorkloadDataset"
    / "final_workload_dataset.csv"
)

MERGED_FILE = (
    ROOT / "Coding" / "CarbonAnalysis"
    / "workload_carbon_dataset.csv"
)

CARBON_FILE = (
    ROOT / "Coding" / "CarbonAnalysis"
    / "carbon_intensity_2021_window.csv"
)

ENERGY_SUMMARY_FILE = (
    ROOT / "Coding" / "EnergyModel" / "results"
    / "time_aware_energy_summary.csv"
)

ENERGY_ESTIMATES_FILE = (
    ROOT / "Coding" / "EnergyModel" / "results"
    / "time_aware_energy_estimates.csv"
)

LOAD_BALANCING_FOLDER = (
    ROOT / "Coding" / "LoadBalancing" / "results"
)

OUTPUT_FOLDER = ROOT / "figures"
OUTPUT_FOLDER.mkdir(exist_ok=True)


# ============================================================
# 2. COMMON FUNCTIONS
# ============================================================

def normalize(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def load_dataset(path):
    if not path.exists():
        print(f"[NOT FOUND] {path}")
        return None

    try:
        df = pd.read_csv(path)

        print(f"\n[LOADED] {path.name}")
        print(f"Rows: {len(df):,}")
        print("Columns:", list(df.columns))

        return df

    except Exception as error:
        print(f"[ERROR] {path}: {error}")
        return None


def find_column(df, candidates, contains=()):
    """
    Find columns even when the CSV uses slightly different names.
    """

    if df is None:
        return None

    columns = {
        normalize(column): column
        for column in df.columns
    }

    for candidate in candidates:
        key = normalize(candidate)

        if key in columns:
            return columns[key]

    for column in df.columns:
        key = normalize(column)

        if any(normalize(word) in key for word in contains):
            return column

    return None


def get_time_column(df):
    return find_column(
        df,
        [
            "timestamp",
            "datetime",
            "time",
            "minute_timestamp",
            "start_time",
            "interval_start",
            "utc_timestamp",
            "minute"
        ],
        contains=["timestamp", "datetime"]
    )


def parse_time(df, column):
    values = df[column]

    if pd.api.types.is_numeric_dtype(values):
        values = pd.to_numeric(values, errors="coerce")

        median = values.dropna().median()

        unit = "s" if median < 10**11 else "ms"

        return pd.to_datetime(
            values,
            unit=unit,
            utc=True,
            errors="coerce"
        )

    return pd.to_datetime(
        values,
        utc=True,
        errors="coerce"
    )


def save_graph(fig, filename):
    path = OUTPUT_FOLDER / filename

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"[GRAPH CREATED] {filename}")


def numeric_column(df, column):
    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 3. WORKLOAD ANALYSIS GRAPHS
# ============================================================

def generate_workload_graphs():

    print("\n========== WORKLOAD ANALYSIS ==========")

    df = load_dataset(WORKLOAD_FILE)

    if df is None:
        return

    time_col = get_time_column(df)

    invocation_col = find_column(
        df,
        [
            "invocations",
            "invocation_count",
            "request_count",
            "total_invocations",
            "invocations_per_minute",
            "calls"
        ],
        contains=["invocation"]
    )

    duration_col = find_column(
        df,
        [
            "total_execution_seconds",
            "execution_seconds",
            "total_duration",
            "duration_sum"
        ],
        contains=["executionseconds", "durationsum"]
    )

    average_duration_col = find_column(
        df,
        [
            "avg_duration",
            "average_duration",
            "mean_duration"
        ],
        contains=["avgduration", "meanduration"]
    )

    # --------------------------------------------------------
    # GRAPH 1: Invocation distribution
    # --------------------------------------------------------

    if invocation_col:

        values = numeric_column(
            df, invocation_col
        ).dropna()

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(
            values,
            bins=60,
            edgecolor="black"
        )

        ax.set_title(
            "Distribution of Serverless Invocations"
        )

        ax.set_xlabel("Invocations per minute")
        ax.set_ylabel("Number of intervals")

        ax.grid(axis="y", alpha=0.3)

        save_graph(
            fig,
            "workload_invocation_distribution.png"
        )

    else:
        print("[SKIP] Invocation column not found.")

    # --------------------------------------------------------
    # GRAPH 2: Workload timeline
    # --------------------------------------------------------

    if time_col and invocation_col:

        temp = pd.DataFrame({
            "time": parse_time(df, time_col),
            "invocations": numeric_column(
                df, invocation_col
            )
        }).dropna()

        temp = temp.sort_values("time")

        hourly = (
            temp.set_index("time")["invocations"]
            .resample("1h")
            .sum()
        )

        fig, ax = plt.subplots(figsize=(13, 6))

        ax.plot(
            hourly.index,
            hourly.values,
            linewidth=0.9
        )

        ax.set_title(
            "Serverless Workload Timeline"
        )

        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Invocations per hour")

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%d %b")
        )

        ax.grid(True, alpha=0.3)

        fig.autofmt_xdate()

        save_graph(
            fig,
            "workload_timeline.png"
        )

    # --------------------------------------------------------
    # GRAPH 3: Hourly workload pattern
    # --------------------------------------------------------

    if time_col and invocation_col:

        temp = pd.DataFrame({
            "time": parse_time(df, time_col),
            "invocations": numeric_column(
                df, invocation_col
            )
        }).dropna()

        temp["hour"] = temp["time"].dt.hour

        hourly_pattern = (
            temp.groupby("hour")["invocations"]
            .mean()
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            hourly_pattern.index,
            hourly_pattern.values,
            marker="o"
        )

        ax.set_xticks(range(24))

        ax.set_title(
            "Average Workload Pattern by Hour"
        )

        ax.set_xlabel("Hour of day (UTC)")
        ax.set_ylabel("Average invocations per minute")

        ax.grid(True, alpha=0.3)

        save_graph(
            fig,
            "workload_hourly_pattern.png"
        )

    # --------------------------------------------------------
    # GRAPH 4: Execution duration distribution
    # --------------------------------------------------------

    selected_duration = (
        duration_col
        if duration_col
        else average_duration_col
    )

    if selected_duration:

        values = numeric_column(
            df, selected_duration
        ).dropna()

        values = values[values >= 0]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(
            values,
            bins=60,
            edgecolor="black"
        )

        ax.set_title(
            "Distribution of Execution Duration"
        )

        ax.set_xlabel(selected_duration)
        ax.set_ylabel("Number of observations")

        ax.grid(axis="y", alpha=0.3)

        save_graph(
            fig,
            "workload_duration_distribution.png"
        )


# ============================================================
# 4. CARBON INTENSITY GRAPHS
# ============================================================

def generate_carbon_graphs():

    print("\n========== CARBON ANALYSIS ==========")

    df = load_dataset(CARBON_FILE)

    if df is None:
        df = load_dataset(MERGED_FILE)

    if df is None:
        return

    time_col = get_time_column(df)

    carbon_col = find_column(
        df,
        [
            "carbon_intensity",
            "carbon_intensity_gco2eq_kwh",
            "lifecycle_carbon_intensity",
            "lifecycle_gco2eq_kwh",
            "intensity"
        ],
        contains=["carbonintensity", "lifecycleintensity"]
    )

    if not time_col or not carbon_col:

        print("[SKIP] Carbon timestamp/intensity columns not found.")
        return

    temp = pd.DataFrame({
        "time": parse_time(df, time_col),
        "carbon": numeric_column(df, carbon_col)
    }).dropna()

    temp = temp.sort_values("time")

    # --------------------------------------------------------
    # GRAPH 5: Carbon intensity timeline
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(
        temp["time"],
        temp["carbon"],
        linewidth=1
    )

    ax.set_title(
        "Historical Regional Carbon Intensity"
    )

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Carbon intensity (gCO2eq/kWh)")

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%d %b")
    )

    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()

    save_graph(
        fig,
        "carbon_intensity_timeline.png"
    )

    # --------------------------------------------------------
    # GRAPH 6: Carbon intensity distribution
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        temp["carbon"],
        bins=35,
        edgecolor="black"
    )

    ax.set_title(
        "Distribution of Historical Carbon Intensity"
    )

    ax.set_xlabel("Carbon intensity (gCO2eq/kWh)")
    ax.set_ylabel("Number of observations")

    ax.grid(axis="y", alpha=0.3)

    save_graph(
        fig,
        "carbon_intensity_distribution.png"
    )

    # --------------------------------------------------------
    # GRAPH 7: Workload vs carbon intensity
    # --------------------------------------------------------

    merged = load_dataset(MERGED_FILE)

    if merged is None:
        return

    merged_time = get_time_column(merged)

    merged_carbon = find_column(
        merged,
        [
            "carbon_intensity",
            "carbon_intensity_gco2eq_kwh",
            "lifecycle_carbon_intensity",
            "lifecycle_gco2eq_kwh"
        ],
        contains=["carbonintensity", "lifecycleintensity"]
    )

    merged_workload = find_column(
        merged,
        [
            "invocations",
            "invocation_count",
            "invocations_per_minute"
        ],
        contains=["invocation"]
    )

    if not all([
        merged_time,
        merged_carbon,
        merged_workload
    ]):
        print("[SKIP] Merged workload/carbon columns unavailable.")
        return

    temp = pd.DataFrame({
        "time": parse_time(merged, merged_time),
        "carbon": numeric_column(merged, merged_carbon),
        "workload": numeric_column(merged, merged_workload)
    }).dropna()

    temp = temp.sort_values("time")

    fig, ax1 = plt.subplots(figsize=(13, 6))

    ax1.plot(
        temp["time"],
        temp["workload"],
        linewidth=0.8,
        label="Workload"
    )

    ax1.set_xlabel("Time (UTC)")
    ax1.set_ylabel("Invocations per minute")

    ax2 = ax1.twinx()

    ax2.plot(
        temp["time"],
        temp["carbon"],
        linewidth=0.8,
        label="Carbon intensity"
    )

    ax2.set_ylabel(
        "Carbon intensity (gCO2eq/kWh)"
    )

    ax1.set_title(
        "Workload and Carbon Intensity Relationship"
    )

    ax1.grid(True, alpha=0.25)

    fig.autofmt_xdate()

    save_graph(
        fig,
        "workload_vs_carbon_timeline.png"
    )


# ============================================================
# 5. ENERGY MODEL GRAPHS
# ============================================================

def generate_energy_graphs():

    print("\n========== ENERGY MODEL ==========")

    df = load_dataset(ENERGY_SUMMARY_FILE)

    if df is None:
        df = load_dataset(ENERGY_ESTIMATES_FILE)

    if df is None:
        return

    scenario_col = find_column(
        df,
        [
            "scenario",
            "capacity_scenario",
            "configuration",
            "model"
        ],
        contains=["scenario", "configuration"]
    )

    energy_col = find_column(
        df,
        [
            "energy_kwh",
            "total_energy_kwh",
            "estimated_energy_kwh",
            "energy_consumption_kwh",
            "total_kwh"
        ],
        contains=["energykwh", "totalkwh"]
    )

    if not energy_col:
        energy_col = find_column(
            df, [], contains=["energy"]
        )

    if not scenario_col:
        scenario_col = find_column(
            df, [], contains=["scenario", "capacity"]
        )

    if not energy_col or not scenario_col:

        print("[SKIP] Energy/scenario columns not identified.")
        return

    temp = df[
        [scenario_col, energy_col]
    ].copy()

    temp[energy_col] = pd.to_numeric(
        temp[energy_col],
        errors="coerce"
    )

    temp = temp.dropna()

    if temp.empty:
        return

    # --------------------------------------------------------
    # GRAPH 8: Energy comparison
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        temp[scenario_col].astype(str),
        temp[energy_col]
    )

    ax.set_title(
        "Time-Aware Energy Consumption by Capacity Scenario"
    )

    ax.set_xlabel("Capacity scenario")
    ax.set_ylabel("Estimated energy (kWh)")

    ax.grid(axis="y", alpha=0.3)

    save_graph(
        fig,
        "energy_strategy_comparison.png"
    )


# ============================================================
# 6. LOAD BALANCING RESULT DISCOVERY
# ============================================================

def inspect_load_balancing_results():

    print("\n========== LOAD BALANCING RESULT FILES ==========")

    if not LOAD_BALANCING_FOLDER.exists():
        print("[NOT FOUND] LoadBalancing results folder.")
        return

    csv_files = list(
        LOAD_BALANCING_FOLDER.rglob("*.csv")
    )

    if not csv_files:
        print("No CSV files found in LoadBalancing/results.")
        return

    for path in csv_files:
        try:
            df = pd.read_csv(path, nrows=5)

            print("\nFile:", path.relative_to(ROOT))
            print("Columns:", list(df.columns))

        except Exception as error:
            print("Could not inspect:", path.name, error)


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 65)
    print("CARBON AND COST-AWARE SERVERLESS LOAD BALANCING")
    print("DATASET-BASED REPORT FIGURE GENERATOR")
    print("=" * 65)

    print("\nProject root:", ROOT)
    print("Output folder:", OUTPUT_FOLDER)

    generate_workload_graphs()

    generate_carbon_graphs()

    generate_energy_graphs()

    inspect_load_balancing_results()

    print("\n")
    print("=" * 65)
    print("GRAPH GENERATION FINISHED")
    print("=" * 65)

    print("\nGenerated images are inside:")
    print(OUTPUT_FOLDER)

    print("\nCheck the terminal for skipped graphs or missing columns.")


if __name__ == "__main__":
    main()
