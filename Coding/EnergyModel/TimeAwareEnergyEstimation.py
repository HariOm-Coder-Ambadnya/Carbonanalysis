
import os
import math
import numpy as np
import pandas as pd

from energy_model_config import SCENARIOS

# Original Azure Functions invocation trace
INPUT_FILE = (
    "AzureFunctionsInvocationTraceForTwoWeeksJan2021/"
    "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
)

OUTPUT_DIR = "Coding/EnergyModel/results"

TOTAL_MINUTES = 20160
INTERVAL_SECONDS = 60
TOTAL_SECONDS = TOTAL_MINUTES * INTERVAL_SECONDS


def load_invocations():

    print("Loading original invocation trace...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    required = [
        "app",
        "func",
        "end_timestamp",
        "duration"
    ]

    if not all(col in df.columns for col in required):
        raise ValueError("Unexpected invocation trace columns.")

    df["end_timestamp"] = pd.to_numeric(
        df["end_timestamp"], errors="coerce"
    )

    df["duration"] = pd.to_numeric(
        df["duration"], errors="coerce"
    )

    if df[["end_timestamp", "duration"]].isnull().any().any():
        raise ValueError("Invalid timestamps or durations found.")

    if (df["duration"] < 0).any():
        raise ValueError("Negative invocation duration found.")

    # Derive invocation start timestamp.
    df["start_timestamp"] = (
        df["end_timestamp"] - df["duration"]
    )

    # Positive-duration invocations only.
    # Zero-duration invocations consume no modelled execution time.
    df = df[df["duration"] > 0].copy()

    # Keep events inside the reconstructed 14-day observation window.
    df = df[
        (df["start_timestamp"] >= 0)
        & (df["end_timestamp"] <= TOTAL_SECONDS)
    ].copy()

    print(f"Positive-duration invocations: {len(df)}")

    return df


def create_events(df):

    print("Generating start and completion events...")

    starts = df["start_timestamp"].to_numpy()
    ends = df["end_timestamp"].to_numpy()

    event_times = np.concatenate([starts, ends])

    event_changes = np.concatenate([
        np.ones(len(starts), dtype=np.int64),
        -np.ones(len(ends), dtype=np.int64)
    ])

    events = pd.DataFrame({
        "time": event_times,
        "change": event_changes
    })

    # Combine events at identical timestamps.
    events = (
        events.groupby("time", as_index=False)["change"]
        .sum()
        .sort_values("time")
        .reset_index(drop=True)
    )

    print(f"Unique event timestamps: {len(events)}")

    return events


def calculate_scenario(events, scenario_name, config):

    capacity = config["concurrency_capacity"]
    idle_power = config["idle_power_w"]
    peak_power = config["peak_power_w"]

    if capacity <= 0:
        raise ValueError("Invalid concurrency capacity.")

    if idle_power < 0 or peak_power < idle_power:
        raise ValueError("Invalid power parameters.")

    # Energy accumulated in every minute, in Wh.
    minute_energy_wh = np.zeros(TOTAL_MINUTES)

    # Track the highest number of instances per minute.
    minute_max_instances = np.zeros(
        TOTAL_MINUTES, dtype=np.int64
    )

    active_invocations = 0
    previous_time = 0.0

    for row in events.itertuples(index=False):

        current_time = float(row.time)

        # Integrate the state before applying events at current_time.
        if current_time > previous_time:

            if active_invocations > 0:

                instance_count = math.ceil(
                    active_invocations / capacity
                )

                utilisation = min(
                    active_invocations /
                    (instance_count * capacity),
                    1.0
                )

                power_per_instance = (
                    idle_power
                    + utilisation * (peak_power - idle_power)
                )

                total_power = (
                    instance_count * power_per_instance
                )

                # Split energy across minute boundaries.
                segment_start = previous_time

                while segment_start < current_time:

                    minute_index = int(
                        segment_start // INTERVAL_SECONDS
                    )

                    minute_end = min(
                        (minute_index + 1) * INTERVAL_SECONDS,
                        current_time
                    )

                    segment_duration = (
                        minute_end - segment_start
                    )

                    energy_wh = (
                        total_power * segment_duration / 3600
                    )

                    minute_energy_wh[minute_index] += energy_wh

                    minute_max_instances[minute_index] = max(
                        minute_max_instances[minute_index],
                        instance_count
                    )

                    segment_start = minute_end

        # Apply all changes at the current timestamp.
        active_invocations += int(row.change)

        if active_invocations < 0:
            raise ValueError(
                f"Negative concurrency at {current_time}"
            )

        previous_time = current_time

    if active_invocations != 0:
        raise ValueError(
            "Unclosed invocations remain after event processing."
        )

    # Convert Wh to kWh.
    minute_energy_kwh = minute_energy_wh / 1000

    return {
        f"{scenario_name}_energy_wh": minute_energy_wh,
        f"{scenario_name}_energy_kwh": minute_energy_kwh,
        f"{scenario_name}_maximum_instances": minute_max_instances
    }


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    invocations = load_invocations()

    events = create_events(invocations)

    results = pd.DataFrame({
        "minute": np.arange(TOTAL_MINUTES)
    })

    print("\nRunning time-aware energy model...")

    for scenario_name, config in SCENARIOS.items():

        print(f"Processing {scenario_name}...")

        scenario_results = calculate_scenario(
            events,
            scenario_name,
            config
        )

        for column, values in scenario_results.items():
            results[column] = values

    # Merge carbon intensity from the validated dataset.
    carbon_file = (
        "Coding/CarbonAnalysis/workload_carbon_dataset.csv"
    )

    carbon_df = pd.read_csv(carbon_file)

    if len(carbon_df) != TOTAL_MINUTES:
        raise ValueError("Unexpected carbon dataset size.")

    results["carbon_intensity_lifecycle"] = (
        carbon_df["carbon_intensity_lifecycle"]
    )

    results["carbon_intensity_direct"] = (
        carbon_df["carbon_intensity_direct"]
    )

    # Calculate emissions for every scenario.
    for scenario_name in SCENARIOS:

        results[
            f"{scenario_name}_carbon_lifecycle_g"
        ] = (
            results[f"{scenario_name}_energy_kwh"]
            * results["carbon_intensity_lifecycle"]
        )

        results[
            f"{scenario_name}_carbon_direct_g"
        ] = (
            results[f"{scenario_name}_energy_kwh"]
            * results["carbon_intensity_direct"]
        )

    # Save detailed minute-level results.
    detailed_path = os.path.join(
        OUTPUT_DIR,
        "time_aware_energy_estimates.csv"
    )

    results.to_csv(detailed_path, index=False)

    # Create summary.
    summary_rows = []

    for scenario_name in SCENARIOS:

        summary_rows.append({

            "scenario": scenario_name,

            "total_energy_kwh": results[
                f"{scenario_name}_energy_kwh"
            ].sum(),

            "total_lifecycle_carbon_kg": (
                results[
                    f"{scenario_name}_carbon_lifecycle_g"
                ].sum() / 1000
            ),

            "total_direct_carbon_kg": (
                results[
                    f"{scenario_name}_carbon_direct_g"
                ].sum() / 1000
            ),

            "maximum_instances": results[
                f"{scenario_name}_maximum_instances"
            ].max()

        })

    summary = pd.DataFrame(summary_rows)

    summary_path = os.path.join(
        OUTPUT_DIR,
        "time_aware_energy_summary.csv"
    )

    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("TIME-AWARE ENERGY ESTIMATION COMPLETED")
    print("=" * 65)

    print(f"Processed minutes: {len(results)}")

    print("\nScenario summary:")
    print(summary.to_string(index=False))

    print("\nDetailed output:")
    print(os.path.abspath(detailed_path))

    print("\nSummary output:")
    print(os.path.abspath(summary_path))

    print(
        "\nNOTE: This is a simulated provisioning model. "
        "It does not reproduce Azure's actual autoscaling."
    )


if __name__ == "__main__":
    main()
