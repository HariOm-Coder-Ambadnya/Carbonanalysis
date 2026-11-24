
from pathlib import Path
import math
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SCHEDULE_DIR = (
    ROOT / "Coding" / "LoadBalancing" / "results"
    / "forecast_assisted"
)

CARBON_FILE = (
    ROOT / "Coding" / "CarbonAnalysis"
    / "workload_carbon_dataset.csv"
)

OUTPUT_DIR = SCHEDULE_DIR / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEADLINES = [15, 30, 60]

# Existing simulated energy model assumptions
CAPACITY_PER_INSTANCE = 20
IDLE_POWER_WATTS = 44
PEAK_POWER_WATTS = 125

# Azure Consumption-plan pricing assumptions
MEMORY_GB = 1.0
EXECUTION_PRICE_PER_MILLION = 0.20
GB_SECOND_PRICE = 0.000016

MIN_BILLABLE_SECONDS = 0.100
BILLING_ROUNDING_SECONDS = 0.001


# ============================================================
# LOAD CARBON DATA
# ============================================================

def load_carbon_data():

    carbon = pd.read_csv(CARBON_FILE)

    required = ["minute", "carbon_intensity_lifecycle"]

    for column in required:
        if column not in carbon.columns:
            raise ValueError(
                f"Missing carbon column: {column}. "
                f"Available: {carbon.columns.tolist()}"
            )

    carbon = carbon.sort_values("minute").reset_index(drop=True)

    if carbon["minute"].duplicated().any():
        raise ValueError("Duplicate minute values in carbon dataset.")

    carbon_values = carbon["carbon_intensity_lifecycle"].to_numpy(
        dtype=float
    )

    minute_values = carbon["minute"].to_numpy(dtype=int)

    if not np.array_equal(
        minute_values,
        np.arange(len(minute_values))
    ):
        raise ValueError(
            "Carbon minute timeline must be continuous from zero."
        )

    if not np.isfinite(carbon_values).all():
        raise ValueError("Carbon dataset contains invalid values.")

    return carbon_values


# ============================================================
# ENERGY AND CARBON CALCULATION
# ============================================================

def calculate_energy_and_carbon(schedule, carbon_values):

    starts = schedule["scheduled_start_timestamp"].to_numpy(
        dtype=float
    )

    durations = schedule["duration"].to_numpy(dtype=float)

    valid = (
        np.isfinite(starts)
        & np.isfinite(durations)
        & (durations > 0)
    )

    starts = starts[valid]
    durations = durations[valid]

    ends = starts + durations

    # Event: (timestamp, concurrency change)
    events = []

    for start, end in zip(starts, ends):
        events.append((start, 1))
        events.append((end, -1))

    events.sort(key=lambda item: item[0])

    active = 0
    previous_time = None

    total_energy_kwh = 0.0
    total_carbon_kg = 0.0
    maximum_concurrency = 0
    maximum_instances = 0

    index = 0

    while index < len(events):

        current_time = events[index][0]

        # Integrate the interval before processing current events.
        if previous_time is not None and current_time > previous_time:

            interval_start = previous_time
            interval_end = current_time

            while interval_start < interval_end and active > 0:

                minute = int(interval_start // 60)

                if minute < 0 or minute >= len(carbon_values):
                    raise ValueError(
                        f"Scheduled minute {minute} is outside "
                        "the carbon timeline."
                    )

                next_minute_boundary = (minute + 1) * 60

                segment_end = min(
                    interval_end,
                    next_minute_boundary
                )

                segment_seconds = segment_end - interval_start

                instances = math.ceil(
                    active / CAPACITY_PER_INSTANCE
                )

                utilisation = active / (
                    instances * CAPACITY_PER_INSTANCE
                )

                power_watts = instances * (
                    IDLE_POWER_WATTS
                    + utilisation * (
                        PEAK_POWER_WATTS - IDLE_POWER_WATTS
                    )
                )

                energy_kwh = (
                    power_watts * segment_seconds / 3600
                ) / 1000

                carbon_kg = (
                    energy_kwh * carbon_values[minute] / 1000
                )

                total_energy_kwh += energy_kwh
                total_carbon_kg += carbon_kg

                interval_start = segment_end

        # Process all events at the same timestamp.
        while (
            index < len(events)
            and events[index][0] == current_time
        ):
            active += events[index][1]
            index += 1

        if active < 0:
            raise ValueError("Invalid concurrency event sequence.")

        maximum_concurrency = max(
            maximum_concurrency, active
        )

        maximum_instances = max(
            maximum_instances,
            math.ceil(active / CAPACITY_PER_INSTANCE)
        )

        previous_time = current_time

    return {
        "energy_kwh": total_energy_kwh,
        "carbon_emissions_kg": total_carbon_kg,
        "maximum_concurrency": maximum_concurrency,
        "maximum_instances": maximum_instances
    }


# ============================================================
# AZURE COST ESTIMATION
# ============================================================

def calculate_azure_cost(schedule):

    durations = schedule["duration"].to_numpy(dtype=float)

    durations = durations[
        np.isfinite(durations) & (durations >= 0)
    ]

    # Round execution duration upward to the nearest millisecond.
    rounded = (
        np.ceil(durations / BILLING_ROUNDING_SECONDS)
        * BILLING_ROUNDING_SECONDS
    )

    billable = np.maximum(
        rounded,
        MIN_BILLABLE_SECONDS
    )

    invocation_count = len(billable)

    total_gb_seconds = np.sum(
        billable * MEMORY_GB
    )

    execution_cost = (
        invocation_count / 1_000_000
    ) * EXECUTION_PRICE_PER_MILLION

    resource_cost = (
        total_gb_seconds * GB_SECOND_PRICE
    )

    return {
        "invocation_count": invocation_count,
        "total_gb_seconds": total_gb_seconds,
        "execution_cost_usd": execution_cost,
        "resource_cost_usd": resource_cost,
        "total_cost_usd": execution_cost + resource_cost
    }


# ============================================================
# EVALUATE ONE STRATEGY
# ============================================================

def evaluate_strategy(name, schedule, carbon_values):

    energy = calculate_energy_and_carbon(
        schedule, carbon_values
    )

    cost = calculate_azure_cost(schedule)

    return {
        "strategy": name,
        **energy,
        **cost
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nSCHEDULED STRATEGY EVALUATION")
    print("=" * 45)

    carbon_values = load_carbon_data()

    results = []

    for deadline in DEADLINES:

        filename = (
            f"forecast_assisted_schedule_{deadline}min.csv"
        )

        path = SCHEDULE_DIR / filename

        if not path.exists():
            raise FileNotFoundError(path)

        print(f"\nEvaluating {deadline}-minute scenario...")

        schedule = pd.read_csv(path)

        required_columns = [
            "original_start_timestamp",
            "scheduled_start_timestamp",
            "duration"
        ]

        for column in required_columns:
            if column not in schedule.columns:
                raise ValueError(
                    f"Missing column {column} in {filename}"
                )

        # Original execution: no scheduling delay.
        baseline = schedule.copy()

        baseline["scheduled_start_timestamp"] = (
            baseline["original_start_timestamp"]
        )

        baseline_result = evaluate_strategy(
            f"Baseline_{deadline}min",
            baseline,
            carbon_values
        )

        proposed_result = evaluate_strategy(
            f"Forecast_Assisted_{deadline}min",
            schedule,
            carbon_values
        )

        # Calculate relative changes against the corresponding baseline.
        for metric in [
            "energy_kwh",
            "carbon_emissions_kg",
            "total_cost_usd"
        ]:

            original = baseline_result[metric]
            scheduled = proposed_result[metric]

            change = (
                (original - scheduled) / original * 100
                if original != 0 else 0
            )

            proposed_result[
                f"{metric}_reduction_percent"
            ] = change

        results.append(baseline_result)
        results.append(proposed_result)

        print(
            f"Energy: {baseline_result['energy_kwh']:.4f}"
            f" -> {proposed_result['energy_kwh']:.4f} kWh"
        )

        print(
            "Carbon: "
            f"{baseline_result['carbon_emissions_kg']:.4f}"
            f" -> {proposed_result['carbon_emissions_kg']:.4f} kg"
        )

        print(
            "Cost: "
            f"${baseline_result['total_cost_usd']:.4f}"
            f" -> ${proposed_result['total_cost_usd']:.4f}"
        )

    result_df = pd.DataFrame(results)

    output_file = OUTPUT_DIR / "strategy_evaluation.csv"

    result_df.to_csv(output_file, index=False)

    print("\nEVALUATION COMPLETED")
    print("=" * 45)

    display_columns = [
        "strategy",
        "energy_kwh",
        "carbon_emissions_kg",
        "total_cost_usd",
        "maximum_concurrency",
        "maximum_instances"
    ]

    print(
        result_df[display_columns].to_string(index=False)
    )

    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
