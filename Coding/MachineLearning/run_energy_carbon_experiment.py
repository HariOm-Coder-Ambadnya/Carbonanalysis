
from pathlib import Path
import sys
import json
import time
import traceback
import math

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MACHINE_LEARNING_DIR = PROJECT_ROOT / "Coding" / "MachineLearning"
LOAD_BALANCING_DIR = PROJECT_ROOT / "Coding" / "LoadBalancing"

SCENARIOS_DIR = MACHINE_LEARNING_DIR / "scenarios"
RESULTS_DIR = MACHINE_LEARNING_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "energy_carbon_evaluation_with_multi_objective.csv"

sys.path.insert(0, str(LOAD_BALANCING_DIR))


# ============================================================
# 2. IMPORT EXISTING SCHEDULERS
# ============================================================

from baseline_scheduler import schedule_baseline
from carbon_aware_scheduler import schedule_carbon_aware
from capacity_constrained_scheduler import schedule_jobs
from exact_capacity_scheduler import run_scheduler

from multi_objective_scheduler import (
    calculate_minute_energy_metrics,
    run_multi_objective_scheduler
)

# ============================================================
# 3. EXPERIMENT CONFIGURATION
# ============================================================

DEADLINES = [15, 30, 60]

STRATEGIES = [
    "BASELINE",
    "CARBON_AWARE",
    "APPROXIMATE_CAPACITY",
    "EXACT_CAPACITY",
    "MULTI_OBJECTIVE"
]

CARBON_COLUMN = "carbon_intensity_lifecycle"

REQUIRED_WORKLOAD_COLUMNS = [
    "app",
    "func",
    "start_timestamp",
    "end_timestamp",
    "duration",
    "workload_type"
]

REQUIRED_CARBON_COLUMNS = [
    "minute",
    CARBON_COLUMN
]


# ============================================================
# 4. ENERGY MODEL ASSUMPTIONS
# ============================================================

CAPACITY_PER_INSTANCE = 20

IDLE_POWER_WATTS = 44.0
PEAK_POWER_WATTS = 125.0

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
WATT_SECONDS_PER_KWH = 3_600_000


# ============================================================
# 5. AZURE GROSS COST ASSUMPTIONS
# ============================================================

MEMORY_MB = 1024

EXECUTION_PRICE_PER_MILLION = 0.20
GB_SECOND_PRICE = 0.000016

MINIMUM_BILLABLE_DURATION_SECONDS = 0.1
DURATION_ROUNDING_MILLISECONDS = 1

MEMORY_INCREMENT_MB = 128
MAXIMUM_BILLABLE_MEMORY_MB = 1536

# This evaluator calculates gross cost.
# It does not subtract monthly free grants.


# ============================================================
# 6. LOAD SCENARIO
# ============================================================

def load_scenario(scenario_dir):

    workload_file = scenario_dir / "classified_workload.csv"
    carbon_file = scenario_dir / "carbon_data.csv"
    metadata_file = scenario_dir / "metadata.json"

    for file_path in [
        workload_file,
        carbon_file,
        metadata_file
    ]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing scenario file: {file_path}"
            )

    workload = pd.read_csv(workload_file)
    carbon_df = pd.read_csv(carbon_file)

    with open(metadata_file, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    missing_workload = [
        col for col in REQUIRED_WORKLOAD_COLUMNS
        if col not in workload.columns
    ]

    missing_carbon = [
        col for col in REQUIRED_CARBON_COLUMNS
        if col not in carbon_df.columns
    ]

    if missing_workload:
        raise ValueError(
            f"Missing workload columns: {missing_workload}"
        )

    if missing_carbon:
        raise ValueError(
            f"Missing carbon columns: {missing_carbon}"
        )

    workload = workload.reset_index(drop=True)

    carbon_df = (
        carbon_df
        .sort_values("minute")
        .reset_index(drop=True)
    )

    expected_minutes = np.arange(len(carbon_df))

    if not np.array_equal(
        carbon_df["minute"].to_numpy(),
        expected_minutes
    ):
        raise ValueError(
            "Carbon timeline is not continuous."
        )

    # Convert workload timestamps and durations to numeric values.
    for column in [
        "start_timestamp",
        "end_timestamp",
        "duration"
    ]:
        workload[column] = pd.to_numeric(
            workload[column],
            errors="raise"
        )

    # Validate duration consistency.
    calculated_duration = (
        workload["end_timestamp"]
        - workload["start_timestamp"]
    )

    if not np.allclose(
        calculated_duration,
        workload["duration"],
        rtol=0,
        atol=1e-6
    ):
        raise ValueError(
            "Workload duration does not match end_timestamp - start_timestamp."
        )

    carbon_values = pd.to_numeric(
        carbon_df[CARBON_COLUMN],
        errors="coerce"
    ).to_numpy(dtype=float)

    if workload.empty:
        raise ValueError("Workload is empty.")

    if not np.isfinite(
        workload[
            ["start_timestamp", "end_timestamp", "duration"]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Invalid workload timestamps or durations."
        )

    if not np.isfinite(carbon_values).all():
        raise ValueError(
            "Invalid carbon intensity values."
        )

    if (workload["duration"] < 0).any():
        raise ValueError(
            "Negative execution duration detected."
        )

    if (workload["start_timestamp"] < 0).any():
        raise ValueError(
            "Negative workload start timestamp detected."
        )

    return workload, carbon_values, carbon_df, metadata

# ============================================================
# 7. RUN ONE SCHEDULER
# ============================================================

def unpack_scheduler_result(result):

    if isinstance(result, tuple):

        if len(result) < 2:
            raise ValueError(
                "Unexpected scheduler tuple."
            )

        return result[0], result[1]

    return result, {}


def execute_scheduler(
    strategy,
    workload,
    carbon_values,
    deadline,
    minute_metrics=None
):

    if strategy == "BASELINE":

        result = schedule_baseline(
            workload.copy()
        )

        return result, {}

    if strategy == "CARBON_AWARE":

        result = schedule_carbon_aware(
            workload.copy(),
            carbon_values,
            deadline
        )

        return unpack_scheduler_result(result)

    if strategy == "APPROXIMATE_CAPACITY":

        result = schedule_jobs(
            workload.copy(),
            carbon_values,
            deadline
        )

        return unpack_scheduler_result(result)

    if strategy == "EXACT_CAPACITY":

        result = run_scheduler(
            workload.copy(),
            carbon_values,
            deadline
        )

        return unpack_scheduler_result(result)

    if strategy == "MULTI_OBJECTIVE":

        if minute_metrics is None:
            raise ValueError(
                "Minute metrics are required for MULTI_OBJECTIVE."
            )

        result = run_multi_objective_scheduler(
            workload.copy(),
            minute_metrics.copy(),
            deadline
        )

        return result, {}

    raise ValueError(
        f"Unknown strategy: {strategy}"
    )

# ============================================================
# 8. VALIDATE SCHEDULE
# ============================================================

def validate_schedule(
    schedule,
    original_workload,
    deadline_minutes
):

    if not isinstance(schedule, pd.DataFrame):
        raise TypeError(
            "Scheduler did not return a DataFrame."
        )

    if len(schedule) != len(original_workload):
        raise ValueError(
            "Invocation count changed."
        )

    required_columns = [
        "scheduled_start_timestamp",
        "duration",
        "workload_type"
    ]

    for column in required_columns:
        if column not in schedule.columns:
            raise ValueError(
                f"Missing schedule column: {column}"
            )

    scheduled_starts = pd.to_numeric(
        schedule["scheduled_start_timestamp"],
        errors="coerce"
    ).to_numpy(dtype=float)

    scheduled_durations = pd.to_numeric(
        schedule["duration"],
        errors="coerce"
    ).to_numpy(dtype=float)

    original_starts = original_workload[
        "start_timestamp"
    ].to_numpy(dtype=float)

    original_durations = original_workload[
        "duration"
    ].to_numpy(dtype=float)

    if not np.isfinite(scheduled_starts).all():
        raise ValueError(
            "Invalid scheduled start timestamps."
        )

    if not np.isfinite(scheduled_durations).all():
        raise ValueError(
            "Invalid scheduled durations."
        )

    waiting_times = scheduled_starts - original_starts

    if (waiting_times < -1e-6).any():
        raise ValueError(
            "A job was scheduled before its arrival."
        )

    waiting_times = np.maximum(waiting_times, 0)

    maximum_wait = deadline_minutes * 60

    violations = waiting_times > maximum_wait + 1e-6

    if violations.any():
        raise ValueError(
            f"{int(violations.sum())} deadline violations."
        )

    if not np.allclose(
        scheduled_durations,
        original_durations,
        rtol=0,
        atol=1e-6
    ):
        raise ValueError(
            "Execution durations changed."
        )

    immediate_mask = (
        original_workload["workload_type"]
        .astype(str)
        .str.lower()
        .eq("immediate")
        .to_numpy()
    )

    if (
        waiting_times[immediate_mask] > 1e-6
    ).any():
        raise ValueError(
            "Immediate jobs were delayed."
        )

    scheduled_ends = (
        scheduled_starts + scheduled_durations
    )

    return {
        "invocations": int(len(schedule)),
        "delayed_jobs": int(
            (waiting_times > 1e-6).sum()
        ),
        "average_wait_seconds": float(
            waiting_times.mean()
        ),
        "maximum_wait_seconds": float(
            waiting_times.max()
        ),
        "deadline_violations": int(
            violations.sum()
        ),
        "maximum_scheduled_end_seconds": float(
            scheduled_ends.max()
        )
    }


# ============================================================
# 9. ESTIMATE ENERGY AND CARBON
# ============================================================

def evaluate_energy_and_carbon(
    schedule,
    carbon_values
):
    """
    Estimates concurrency at one-second resolution.

    Each invocation occupies integer seconds from:
        floor(scheduled_start)
    through:
        ceil(scheduled_end) - 1

    This is a one-second discretized estimate, not
    continuous-time measured energy consumption.
    """

    starts = pd.to_numeric(
        schedule["scheduled_start_timestamp"],
        errors="raise"
    ).to_numpy(dtype=float)

    durations = pd.to_numeric(
        schedule["duration"],
        errors="raise"
    ).to_numpy(dtype=float)

    ends = starts + durations

    if not np.isfinite(starts).all():
        raise ValueError("Invalid scheduled starts.")

    if not np.isfinite(ends).all():
        raise ValueError("Invalid scheduled ends.")

    if (starts < 0).any():
        raise ValueError("Negative scheduled start.")

    positive_duration = durations > 0

    start_seconds = np.floor(
        starts[positive_duration]
    ).astype(np.int64)

    end_seconds = np.ceil(
        ends[positive_duration]
    ).astype(np.int64)

    # Ignore zero-duration jobs for concurrency.
    valid_intervals = end_seconds > start_seconds

    start_seconds = start_seconds[valid_intervals]
    end_seconds = end_seconds[valid_intervals]

    if len(start_seconds) == 0:
        return {
            "estimated_energy_kwh": 0.0,
            "estimated_peak_concurrency": 0,
            "estimated_carbon_g_partial": 0.0,
            "estimated_carbon_gco2e": 0.0,
            "carbon_coverage_complete": True,
            "carbon_coverage_energy_percent": 100.0,
            "carbon_coverage_seconds": 0,
            "required_coverage_seconds": 0
        }

    horizon_seconds = int(end_seconds.max())

    # Difference-array approach:
    # +1 at each start, -1 at each end.
    start_events = np.bincount(
        start_seconds,
        minlength=horizon_seconds + 1
    )

    end_events = np.bincount(
        end_seconds,
        minlength=horizon_seconds + 1
    )

    concurrency_changes = (
        start_events - end_events
    )

    concurrency = np.cumsum(
        concurrency_changes[:-1],
        dtype=np.int64
    )

    if (concurrency < 0).any():
        raise ValueError(
            "Negative concurrency detected."
        )

    peak_concurrency = int(
        concurrency.max()
    )

    # Estimate number of instances.
    instances = np.ceil(
        concurrency / CAPACITY_PER_INSTANCE
    )

    # Avoid division by zero when no jobs are running.
    total_capacity = (
        instances * CAPACITY_PER_INSTANCE
    )

    utilization = np.divide(
        concurrency,
        total_capacity,
        out=np.zeros_like(
            concurrency,
            dtype=float
        ),
        where=total_capacity > 0
    )

    # Linear idle-to-peak power model.
    power_watts = instances * (
        IDLE_POWER_WATTS
        + utilization * (
            PEAK_POWER_WATTS - IDLE_POWER_WATTS
        )
    )

    energy_kwh_per_second = (
        power_watts / WATT_SECONDS_PER_KWH
    )

    total_energy_kwh = float(
        energy_kwh_per_second.sum()
    )

    # Carbon intensity is available once per minute.
    carbon_coverage_seconds = (
        len(carbon_values) * SECONDS_PER_MINUTE
    )

    covered_seconds = min(
        horizon_seconds,
        carbon_coverage_seconds
    )

    carbon_intensity_per_second = np.repeat(
        carbon_values,
        SECONDS_PER_MINUTE
    )

    energy_in_covered_period = (
        energy_kwh_per_second[:covered_seconds]
    )

    carbon_in_covered_period = (
        carbon_intensity_per_second[:covered_seconds]
    )

    partial_carbon_g = float(
        np.sum(
            energy_in_covered_period
            * carbon_in_covered_period
        )
    )

    covered_energy_kwh = float(
        energy_in_covered_period.sum()
    )

    if total_energy_kwh > 0:
        energy_coverage_percent = (
            covered_energy_kwh / total_energy_kwh
        ) * 100
    else:
        energy_coverage_percent = 100.0

    coverage_complete = (
        horizon_seconds <= carbon_coverage_seconds
    )

    # Do not report partial emissions as total emissions.
    total_carbon_g = (
        partial_carbon_g
        if coverage_complete
        else np.nan
    )

    return {
        "estimated_energy_kwh": total_energy_kwh,
        "estimated_peak_concurrency": peak_concurrency,
        "estimated_carbon_g_partial": partial_carbon_g,
        "estimated_carbon_gco2e": total_carbon_g,
        "carbon_coverage_complete": bool(
            coverage_complete
        ),
        "carbon_coverage_energy_percent": float(
            energy_coverage_percent
        ),
        "carbon_coverage_seconds": int(
            covered_seconds
        ),
        "required_coverage_seconds": int(
            horizon_seconds
        )
    }


# ============================================================
# 10. CALCULATE GROSS AZURE COST
# ============================================================

def calculate_gross_cost(schedule):

    durations = pd.to_numeric(
        schedule["duration"],
        errors="raise"
    ).to_numpy(dtype=float)

    if not np.isfinite(durations).all():
        raise ValueError(
            "Invalid durations for cost calculation."
        )

    if (durations < 0).any():
        raise ValueError(
            "Negative duration in cost calculation."
        )

    # Minimum billable duration.
    billable_durations = np.maximum(
        durations,
        MINIMUM_BILLABLE_DURATION_SECONDS
    )

    # Round up to the next millisecond.
    billable_durations = (
        np.ceil(
            billable_durations * 1000
        ) / 1000
    )

    # Round memory up to 128 MB increments.
    billable_memory_mb = (
        math.ceil(
            MEMORY_MB / MEMORY_INCREMENT_MB
        ) * MEMORY_INCREMENT_MB
    )

    billable_memory_mb = min(
        billable_memory_mb,
        MAXIMUM_BILLABLE_MEMORY_MB
    )

    memory_gb = billable_memory_mb / 1024

    execution_cost = (
        len(schedule)
        * EXECUTION_PRICE_PER_MILLION
        / 1_000_000
    )

    resource_cost = (
        billable_durations.sum()
        * memory_gb
        * GB_SECOND_PRICE
    )

    total_cost = (
        execution_cost + resource_cost
    )

    return {
        "gross_execution_cost_usd": float(
            execution_cost
        ),
        "gross_resource_cost_usd": float(
            resource_cost
        ),
        "gross_total_cost_usd": float(
            total_cost
        ),
        "billable_memory_mb": int(
            billable_memory_mb
        )
    }


# ============================================================
# 11. RUN EXPERIMENT
# ============================================================

def run_experiment():

    scenario_dirs = sorted(
        [
            path for path in SCENARIOS_DIR.iterdir()
            if path.is_dir()
            and path.name.startswith("scenario_")
        ],
        key=lambda path: int(
            path.name.split("_")[-1]
        )
    )

    # # TEMPORARY TEST: run only scenario_001
    # scenario_dirs = scenario_dirs[:1]

    if not scenario_dirs:
        raise FileNotFoundError(
            f"No scenarios found: {SCENARIOS_DIR}"
        )

    total_runs = (
        len(scenario_dirs)
        * len(DEADLINES)
        * len(STRATEGIES)
    )

    completed = 0
    successful = 0
    failed = 0

    all_results = []

    print("\n" + "=" * 75)
    print("ENERGY AND CARBON EVALUATION")
    print("=" * 75)

    print("Scenarios:", len(scenario_dirs))
    print("Deadlines:", DEADLINES)
    print("Strategies:", STRATEGIES)
    print("Expected executions:", total_runs)

    print("\nEnergy assumptions:")
    print("Capacity per instance:", CAPACITY_PER_INSTANCE)
    print("Idle power:", IDLE_POWER_WATTS, "W")
    print("Peak power:", PEAK_POWER_WATTS, "W")

    print("\nOutput:", OUTPUT_FILE)

    experiment_start = time.time()

    for scenario_index, scenario_dir in enumerate(
        scenario_dirs,
        start=1
    ):

        scenario_id = scenario_dir.name

        print("\n" + "-" * 75)
        print(
            f"SCENARIO {scenario_index}/{len(scenario_dirs)}: "
            f"{scenario_id}"
        )
        print("-" * 75)

        try:

            workload, carbon_values, carbon_df, metadata = load_scenario(
                scenario_dir
            )

            # Calculate minute-level energy metrics once per scenario.
            minute_metrics = calculate_minute_energy_metrics(
                carbon_df.copy()
            )

            print("Jobs:", len(workload))
            print("Carbon records:", len(carbon_values))

        except Exception as error:

            print("SCENARIO LOAD FAILED:", error)

            for deadline in DEADLINES:
                for strategy in STRATEGIES:

                    completed += 1
                    failed += 1

                    all_results.append({
                        "scenario_id": scenario_id,
                        "deadline_minutes": deadline,
                        "strategy": strategy,
                        "status": "FAILED",
                        "error": str(error)
                    })

            pd.DataFrame(all_results).to_csv(
                OUTPUT_FILE,
                index=False
            )

            continue

        for deadline in DEADLINES:

            print(f"\nDeadline: {deadline} minutes")

            for strategy in STRATEGIES:

                completed += 1

                print(
                    f"\n[{completed}/{total_runs}] "
                    f"Running {strategy}"
                )

                run_start = time.time()

                try:

                    schedule, scheduler_summary = (
                        execute_scheduler(
                            strategy,
                            workload,
                            carbon_values,
                            deadline,
                            minute_metrics=minute_metrics
                        )
                    )

                    scheduling_metrics = validate_schedule(
                        schedule,
                        workload,
                        deadline
                    )

                    energy_carbon_metrics = (
                        evaluate_energy_and_carbon(
                            schedule,
                            carbon_values
                        )
                    )

                    cost_metrics = calculate_gross_cost(
                        schedule
                    )

                    row = {
                        "scenario_id": scenario_id,
                        "scenario_type": metadata.get(
                            "scenario_type"
                        ),
                        "window_start_minute": metadata.get(
                            "window_start_minute"
                        ),
                        "window_end_minute": metadata.get(
                            "window_end_minute"
                        ),
                        "deadline_minutes": deadline,
                        "strategy": strategy,
                        "status": "SUCCESS",
                        **scheduling_metrics,
                        **energy_carbon_metrics,
                        **cost_metrics,
                        "evaluation_time_seconds": (
                            time.time() - run_start
                        )
                    }

                    if isinstance(scheduler_summary, dict):

                        row["scheduler_peak_concurrency"] = (
                            scheduler_summary.get(
                                "maximum_one_second_slot_concurrency"
                            )
                        )

                        row["scheduler_capacity_limit"] = (
                            scheduler_summary.get(
                                "capacity_limit"
                            )
                        )

                    all_results.append(row)
                    successful += 1

                    carbon_status = (
                        "COMPLETE"
                        if row["carbon_coverage_complete"]
                        else "INCOMPLETE"
                    )

                    print(
                        "SUCCESS | "
                        f"Energy: "
                        f"{row['estimated_energy_kwh']:.6f} kWh | "
                        f"Carbon: {carbon_status} | "
                        f"Cost: "
                        f"${row['gross_total_cost_usd']:.6f}"
                    )

                except Exception as error:

                    failed += 1

                    error_message = (
                        f"{type(error).__name__}: {error}"
                    )

                    print("FAILED:", error_message)

                    all_results.append({
                        "scenario_id": scenario_id,
                        "deadline_minutes": deadline,
                        "strategy": strategy,
                        "status": "FAILED",
                        "error": error_message,
                        "evaluation_time_seconds": (
                            time.time() - run_start
                        )
                    })

                    traceback.print_exc()

                # Save progress after each execution.
                pd.DataFrame(all_results).to_csv(
                    OUTPUT_FILE,
                    index=False
                )

    # ========================================================
    # 12. FINAL SUMMARY
    # ========================================================

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    elapsed = time.time() - experiment_start

    print("\n" + "=" * 75)
    print("ENERGY AND CARBON EXPERIMENT COMPLETED")
    print("=" * 75)

    print("Expected executions:", total_runs)
    print("Completed:", completed)
    print("Successful:", successful)
    print("Failed:", failed)

    print(
        "Total time:",
        round(elapsed / 60, 2),
        "minutes"
    )

    print("\nResults saved:")
    print(OUTPUT_FILE)

    successful_df = results_df[
        results_df["status"] == "SUCCESS"
    ]

    if not successful_df.empty:

        print("\nAverage energy by strategy:")

        print(
            successful_df
            .groupby("strategy")[
                "estimated_energy_kwh"
            ]
            .mean()
            .to_string()
        )

        print("\nCarbon coverage by strategy:")

        print(
            successful_df
            .groupby("strategy")[
                "carbon_coverage_complete"
            ]
            .mean()
            .mul(100)
            .to_string()
        )

        print("\nAverage gross cost by strategy:")

        print(
            successful_df
            .groupby("strategy")[
                "gross_total_cost_usd"
            ]
            .mean()
            .to_string()
        )

    print("\n" + "=" * 75)


# ============================================================
# 13. MAIN
# ============================================================

if __name__ == "__main__":

    try:
        run_experiment()

    except KeyboardInterrupt:

        print(
            "\nExperiment interrupted. "
            "Previously saved progress is retained."
        )

        sys.exit(1)

    except Exception as error:

        print("\nEXPERIMENT FAILED:", error)

        traceback.print_exc()

        sys.exit(1)
