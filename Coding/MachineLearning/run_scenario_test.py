
from pathlib import Path
import sys
import json
import traceback

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

# Make existing scheduler modules importable.
sys.path.insert(0, str(LOAD_BALANCING_DIR))


# ============================================================
# 2. IMPORT EXISTING SCHEDULERS
# ============================================================

try:
    from baseline_scheduler import schedule_baseline

    from carbon_aware_scheduler import (
        schedule_carbon_aware
    )

    from capacity_constrained_scheduler import (
        schedule_jobs
    )

    from exact_capacity_scheduler import (
        run_scheduler
    )

except ImportError as error:
    print("\nERROR: Could not import scheduler modules.")
    print("Check that all scheduler files are inside:")
    print(LOAD_BALANCING_DIR)
    raise error


# ============================================================
# 3. CONFIGURATION
# ============================================================

SCENARIO_ID = "scenario_001"

# First test only one deadline.
DEADLINE_MINUTES = 15

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
# 4. LOAD SCENARIO DATA
# ============================================================

def load_scenario(scenario_id):

    print("\nLoading scenario:", scenario_id)

    scenario_dir = SCENARIOS_DIR / scenario_id

    workload_file = (
        scenario_dir / "classified_workload.csv"
    )

    carbon_file = (
        scenario_dir / "carbon_data.csv"
    )

    metadata_file = (
        scenario_dir / "metadata.json"
    )

    # Check files.
    for file_path in [
        workload_file,
        carbon_file,
        metadata_file
    ]:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required scenario file not found: {file_path}"
            )

    # Load workload.
    workload = pd.read_csv(workload_file)

    # Load carbon data.
    carbon_df = pd.read_csv(carbon_file)

    # Load metadata.
    with open(metadata_file, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    # Validate workload columns.
    missing_workload = [
        column
        for column in REQUIRED_WORKLOAD_COLUMNS
        if column not in workload.columns
    ]

    if missing_workload:
        raise ValueError(
            f"Missing workload columns: {missing_workload}"
        )

    # Validate carbon columns.
    missing_carbon = [
        column
        for column in REQUIRED_CARBON_COLUMNS
        if column not in carbon_df.columns
    ]

    if missing_carbon:
        raise ValueError(
            f"Missing carbon columns: {missing_carbon}"
        )

    # Sort carbon timeline.
    carbon_df = carbon_df.sort_values(
        "minute"
    ).reset_index(drop=True)

    # Check continuous minute timeline.
    expected_minutes = np.arange(len(carbon_df))

    actual_minutes = carbon_df["minute"].to_numpy()

    if not np.array_equal(
        actual_minutes,
        expected_minutes
    ):
        raise ValueError(
            "Carbon timeline is not continuous."
        )

    # Extract carbon intensity values.
    carbon_values = carbon_df[
        CARBON_COLUMN
    ].to_numpy(dtype=float)

    if len(carbon_values) == 0:
        raise ValueError("Carbon timeline is empty.")

    if not np.isfinite(carbon_values).all():
        raise ValueError(
            "Carbon intensity contains missing or invalid values."
        )

    # Check workload values.
    if workload.empty:
        raise ValueError("Workload dataset is empty.")

    numeric_columns = [
        "start_timestamp",
        "end_timestamp",
        "duration"
    ]

    for column in numeric_columns:

        workload[column] = pd.to_numeric(
            workload[column],
            errors="raise"
        )

    if workload[numeric_columns].isna().any().any():
        raise ValueError(
            "Workload contains missing timestamp or duration values."
        )

    if (workload["duration"] < 0).any():
        raise ValueError(
            "Workload contains negative durations."
        )

    print("\nScenario loaded successfully.")

    print("Scenario ID:", scenario_id)
    print("Total jobs:", len(workload))
    print("Carbon records:", len(carbon_df))
    print(
        "Deadline:",
        DEADLINE_MINUTES,
        "minutes"
    )

    return (
        workload,
        carbon_df,
        carbon_values,
        metadata
    )


# ============================================================
# 5. VALIDATE SCHEDULER OUTPUT
# ============================================================

def validate_schedule(
    schedule,
    original_workload,
    deadline_minutes
):

    if not isinstance(schedule, pd.DataFrame):
        raise TypeError(
            "Scheduler did not return a pandas DataFrame."
        )

    if schedule.empty:
        raise ValueError(
            "Scheduler returned an empty schedule."
        )

    if len(schedule) != len(original_workload):
        raise ValueError(
            "Invocation count changed. "
            f"Original: {len(original_workload)}, "
            f"Scheduled: {len(schedule)}"
        )

    required_columns = [
        "scheduled_start_timestamp",
        "duration",
        "workload_type"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in schedule.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing scheduled columns: {missing_columns}"
        )

    scheduled_starts = pd.to_numeric(
        schedule["scheduled_start_timestamp"],
        errors="coerce"
    )

    scheduled_durations = pd.to_numeric(
        schedule["duration"],
        errors="coerce"
    )

    original_starts = pd.to_numeric(
        original_workload["start_timestamp"],
        errors="coerce"
    )

    original_durations = pd.to_numeric(
        original_workload["duration"],
        errors="coerce"
    )

    if scheduled_starts.isna().any():
        raise ValueError(
            "Scheduled start timestamps contain invalid values."
        )

    if scheduled_durations.isna().any():
        raise ValueError(
            "Scheduled durations contain invalid values."
        )

    # Calculate waiting time.
    waiting_times = (
        scheduled_starts.to_numpy()
        - original_starts.to_numpy()
    )

    # No job should be scheduled before its arrival.
    if (waiting_times < -1e-6).any():
        raise ValueError(
            "Some jobs were scheduled before their original arrival."
        )

    waiting_times = np.maximum(waiting_times, 0)

    # Check deadline.
    maximum_allowed_wait = deadline_minutes * 60

    if (waiting_times > maximum_allowed_wait + 1e-6).any():
        raise ValueError(
            "Some jobs exceeded the configured deadline."
        )

    # Check execution durations are unchanged.
    if not np.allclose(
        scheduled_durations.to_numpy(),
        original_durations.to_numpy(),
        rtol=0,
        atol=1e-6
    ):
        raise ValueError(
            "Execution durations changed during scheduling."
        )

    # Immediate jobs must not be delayed.
    immediate_mask = (
        original_workload["workload_type"]
        .astype(str)
        .str.lower()
        .eq("immediate")
        .to_numpy()
    )

    if immediate_mask.any():

        immediate_waiting = waiting_times[
            immediate_mask
        ]

        if (immediate_waiting > 1e-6).any():
            raise ValueError(
                "Immediate jobs were delayed."
            )

    # Calculate summary metrics.
    delayed_mask = waiting_times > 1e-6

    metrics = {
        "invocations": int(len(schedule)),

        "delayed_jobs": int(
            delayed_mask.sum()
        ),

        "average_wait_seconds": float(
            waiting_times.mean()
        ),

        "maximum_wait_seconds": float(
            waiting_times.max()
        ),

        "deadline_violations": 0
    }

    return metrics


# ============================================================
# 6. RUN SCHEDULER TESTS
# ============================================================

def run_test():

    print("\n" + "=" * 65)
    print("SERVERLESS SCHEDULER SCENARIO TEST")
    print("=" * 65)

    # --------------------------------------------------------
    # Load scenario
    # --------------------------------------------------------

    (
        workload,
        carbon_df,
        carbon_values,
        metadata
    ) = load_scenario(SCENARIO_ID)

    results = []

    # --------------------------------------------------------
    # A. BASELINE SCHEDULER
    # --------------------------------------------------------

    print("\n[1/4] Running Baseline Scheduler...")

    baseline_schedule = schedule_baseline(
        workload.copy()
    )

    baseline_metrics = validate_schedule(
        baseline_schedule,
        workload,
        DEADLINE_MINUTES
    )

    results.append({
        "strategy": "BASELINE",
        **baseline_metrics
    })

    print("Baseline completed.")


    # --------------------------------------------------------
    # B. CARBON-AWARE SCHEDULER
    # --------------------------------------------------------

    print("\n[2/4] Running Carbon-Aware Scheduler...")

    carbon_schedule = schedule_carbon_aware(
        workload.copy(),
        carbon_values,
        DEADLINE_MINUTES
    )

    carbon_metrics = validate_schedule(
        carbon_schedule,
        workload,
        DEADLINE_MINUTES
    )

    results.append({
        "strategy": "CARBON_AWARE",
        **carbon_metrics
    })

    print("Carbon-aware scheduling completed.")


    # --------------------------------------------------------
    # C. APPROXIMATE CAPACITY-CONSTRAINED SCHEDULER
    # --------------------------------------------------------

    print(
        "\n[3/4] Running Approximate Capacity-Constrained Scheduler..."
    )

    capacity_result = schedule_jobs(
        workload.copy(),
        carbon_values,
        DEADLINE_MINUTES
    )

    # This scheduler returns (schedule, summary).
    if isinstance(capacity_result, tuple):

        capacity_schedule = capacity_result[0]
        capacity_summary = capacity_result[1]

    else:

        capacity_schedule = capacity_result
        capacity_summary = {}

    capacity_metrics = validate_schedule(
        capacity_schedule,
        workload,
        DEADLINE_MINUTES
    )

    results.append({
        "strategy": "APPROXIMATE_CAPACITY",
        **capacity_metrics
    })

    print("Approximate capacity scheduling completed.")


    # --------------------------------------------------------
    # D. EXACT CAPACITY-CONSTRAINED SCHEDULER
    # --------------------------------------------------------

    print(
        "\n[4/4] Running Exact Capacity-Constrained Scheduler..."
    )

    exact_result = run_scheduler(
        workload.copy(),
        carbon_values,
        DEADLINE_MINUTES
    )

    # Exact scheduler returns (schedule, summary).
    if isinstance(exact_result, tuple):

        exact_schedule = exact_result[0]
        exact_summary = exact_result[1]

    else:

        exact_schedule = exact_result
        exact_summary = {}

    exact_metrics = validate_schedule(
        exact_schedule,
        workload,
        DEADLINE_MINUTES
    )

    exact_result_row = {
        "strategy": "EXACT_CAPACITY",
        **exact_metrics
    }

    # Include exact capacity information when available.
    if exact_summary:

        exact_result_row["peak_concurrency"] = (
            exact_summary.get(
                "maximum_one_second_slot_concurrency"
            )
        )

        exact_result_row["capacity_limit"] = (
            exact_summary.get("capacity_limit")
        )

        exact_result_row["exact_delayed_percentage"] = (
            exact_summary.get("delayed_percentage")
        )

    results.append(exact_result_row)

    print("Exact capacity scheduling completed.")


    # ========================================================
    # 7. DISPLAY RESULTS
    # ========================================================

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("SCHEDULER VALIDATION RESULTS")
    print("=" * 65)

    print(
        results_df.to_string(index=False)
    )


    # ========================================================
    # 8. SAVE RESULTS
    # ========================================================

    output_file = (
        RESULTS_DIR
        / f"{SCENARIO_ID}_scheduler_test.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\nResults saved successfully:")
    print(output_file)

    print("\nScenario metadata:")
    print(
        json.dumps(
            metadata,
            indent=4
        )
    )

    print("\n" + "=" * 65)
    print("ALL FOUR SCHEDULER TESTS COMPLETED SUCCESSFULLY")
    print("=" * 65)


# ============================================================
# 9. MAIN
# ============================================================

if __name__ == "__main__":

    try:

        run_test()

    except Exception as error:

        print("\n" + "=" * 65)
        print("SCENARIO TEST FAILED")
        print("=" * 65)

        print("Error:", error)

        traceback.print_exc()

        sys.exit(1)
