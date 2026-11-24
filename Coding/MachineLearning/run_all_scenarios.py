
from pathlib import Path
import sys
import json
import traceback
import time

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

sys.path.insert(0, str(LOAD_BALANCING_DIR))


# ============================================================
# 2. IMPORT SCHEDULERS
# ============================================================

from baseline_scheduler import schedule_baseline
from carbon_aware_scheduler import schedule_carbon_aware
from capacity_constrained_scheduler import schedule_jobs
from exact_capacity_scheduler import run_scheduler


# ============================================================
# 3. CONFIGURATION
# ============================================================

DEADLINES = [15, 30, 60]

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
# 4. LOAD SCENARIO
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

    for col in [
        "start_timestamp",
        "end_timestamp",
        "duration"
    ]:
        workload[col] = pd.to_numeric(
            workload[col],
            errors="raise"
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
            "Workload contains invalid numeric values."
        )

    if not np.isfinite(carbon_values).all():
        raise ValueError(
            "Carbon data contains invalid values."
        )

    if (workload["duration"] < 0).any():
        raise ValueError(
            "Negative execution duration detected."
        )

    return workload, carbon_df, carbon_values, metadata


# ============================================================
# 5. NORMALIZE SCHEDULER RETURN VALUE
# ============================================================

def unpack_scheduler_result(result):

    if isinstance(result, tuple):

        if len(result) < 2:
            raise ValueError(
                "Unexpected scheduler tuple."
            )

        return result[0], result[1]

    return result, {}


# ============================================================
# 6. VALIDATE SCHEDULE
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
            f"Invocation count mismatch: "
            f"{len(schedule)} != {len(original_workload)}"
        )

    required = [
        "scheduled_start_timestamp",
        "duration",
        "workload_type"
    ]

    missing = [
        col for col in required
        if col not in schedule.columns
    ]

    if missing:
        raise ValueError(
            f"Missing schedule columns: {missing}"
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
            "Jobs scheduled before original arrival."
        )

    waiting_times = np.maximum(waiting_times, 0)

    max_wait = deadline_minutes * 60

    violations = waiting_times > max_wait + 1e-6

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

    delayed_mask = waiting_times > 1e-6

    scheduled_ends = (
        scheduled_starts + scheduled_durations
    )

    metrics = {
        "invocations": int(len(schedule)),
        "delayed_jobs": int(delayed_mask.sum()),
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

    return metrics


# ============================================================
# 7. RUN ONE SCHEDULER
# ============================================================

def execute_scheduler(
    strategy,
    workload,
    carbon_values,
    deadline
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

        return result, {}

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

    raise ValueError(
        f"Unknown strategy: {strategy}"
    )


# ============================================================
# 8. RUN ALL EXPERIMENTS
# ============================================================

def run_all_experiments():

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

    if not scenario_dirs:
        raise FileNotFoundError(
            f"No scenarios found in {SCENARIOS_DIR}"
        )

    strategies = [
        "BASELINE",
        "CARBON_AWARE",
        "APPROXIMATE_CAPACITY",
        "EXACT_CAPACITY"
    ]

    total_runs = (
        len(scenario_dirs)
        * len(DEADLINES)
        * len(strategies)
    )

    completed_runs = 0
    successful_runs = 0
    failed_runs = 0

    all_results = []

    print("\n" + "=" * 75)
    print("FULL SERVERLESS SCHEDULER EXPERIMENT")
    print("=" * 75)

    print("Scenarios:", len(scenario_dirs))
    print("Deadlines:", DEADLINES)
    print("Strategies:", strategies)
    print("Expected executions:", total_runs)

    start_time = time.time()

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

            (
                workload,
                carbon_df,
                carbon_values,
                metadata
            ) = load_scenario(scenario_dir)

            print("Jobs:", len(workload))
            print("Carbon records:", len(carbon_values))

        except Exception as error:

            print("SCENARIO LOAD FAILED:", error)

            for deadline in DEADLINES:
                for strategy in strategies:

                    completed_runs += 1
                    failed_runs += 1

                    all_results.append({
                        "scenario_id": scenario_id,
                        "deadline_minutes": deadline,
                        "strategy": strategy,
                        "status": "FAILED",
                        "error": str(error)
                    })

            continue

        for deadline in DEADLINES:

            print(
                f"\nDeadline: {deadline} minutes"
            )

            for strategy in strategies:

                completed_runs += 1

                print(
                    f"\n[{completed_runs}/{total_runs}] "
                    f"Running {strategy}..."
                )

                run_start = time.time()

                try:

                    schedule, summary = execute_scheduler(
                        strategy,
                        workload,
                        carbon_values,
                        deadline
                    )

                    metrics = validate_schedule(
                        schedule,
                        workload,
                        deadline
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
                        "execution_time_seconds": (
                            time.time() - run_start
                        ),
                        **metrics
                    }

                    if isinstance(summary, dict):

                        row["peak_concurrency"] = (
                            summary.get(
                                "maximum_one_second_slot_concurrency"
                            )
                        )

                        row["capacity_limit"] = (
                            summary.get("capacity_limit")
                        )

                        row["scheduler_delayed_percentage"] = (
                            summary.get("delayed_percentage")
                        )

                    all_results.append(row)

                    successful_runs += 1

                    print(
                        "SUCCESS | "
                        f"Delayed: {metrics['delayed_jobs']} | "
                        f"Average wait: "
                        f"{metrics['average_wait_seconds']:.2f}s | "
                        f"Time: "
                        f"{row['execution_time_seconds']:.2f}s"
                    )

                except Exception as error:

                    failed_runs += 1

                    error_message = (
                        f"{type(error).__name__}: {error}"
                    )

                    print("FAILED:", error_message)

                    all_results.append({
                        "scenario_id": scenario_id,
                        "deadline_minutes": deadline,
                        "strategy": strategy,
                        "status": "FAILED",
                        "execution_time_seconds": (
                            time.time() - run_start
                        ),
                        "error": error_message
                    })

                    traceback.print_exc()

                # Save progress after every execution.
                progress_df = pd.DataFrame(all_results)

                progress_file = (
                    RESULTS_DIR
                    / "all_scenarios_scheduler_results.csv"
                )

                progress_df.to_csv(
                    progress_file,
                    index=False
                )

    # ========================================================
    # 9. FINAL RESULTS
    # ========================================================

    results_df = pd.DataFrame(all_results)

    output_file = (
        RESULTS_DIR
        / "all_scenarios_scheduler_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 75)
    print("EXPERIMENT COMPLETED")
    print("=" * 75)

    print("Total executions:", total_runs)
    print("Completed:", completed_runs)
    print("Successful:", successful_runs)
    print("Failed:", failed_runs)

    print(
        "Total time:",
        round(elapsed / 60, 2),
        "minutes"
    )

    print("\nResults saved:")
    print(output_file)

    if not results_df.empty:

        print("\nExecution status:")
        print(
            results_df["status"].value_counts()
        )

        successful_df = results_df[
            results_df["status"] == "SUCCESS"
        ]

        if not successful_df.empty:

            print("\nAverage waiting time by strategy:")

            print(
                successful_df
                .groupby("strategy")[
                    "average_wait_seconds"
                ]
                .mean()
                .sort_values()
                .to_string()
            )

    print("\n" + "=" * 75)


# ============================================================
# 10. MAIN
# ============================================================

if __name__ == "__main__":

    try:
        run_all_experiments()

    except KeyboardInterrupt:

        print(
            "\nExperiment interrupted by user. "
            "Previously saved progress is retained."
        )

        sys.exit(1)

    except Exception as error:

        print("\nEXPERIMENT FAILED:", error)

        traceback.print_exc()

        sys.exit(1)
