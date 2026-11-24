
import numpy as np
import pandas as pd

from datetime import datetime, timedelta, timezone

from config import (
    OUTPUT_DIR,
    INPUT_FILE,
    DEADLINE_SCENARIOS_MINUTES,
    CARBON_COLUMN
)


# --------------------------------------------------
# 1. LOAD CLASSIFIED WORKLOAD
# --------------------------------------------------

def load_classified_workload():

    input_file = OUTPUT_DIR / "classified_workload.csv"

    print("\nLoading classified workload...")

    if not input_file.exists():
        raise FileNotFoundError(
            f"Classified workload not found: {input_file}"
        )

    df = pd.read_csv(input_file)

    required_columns = [
        "app",
        "func",
        "end_timestamp",
        "duration",
        "start_timestamp",
        "workload_type"
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Missing workload column: {column}"
            )

    print(f"Loaded invocations: {len(df):,}")

    return df


# --------------------------------------------------
# 2. LOAD HOURLY CARBON DATA
# --------------------------------------------------

def load_carbon_data():

    print("\nLoading workload-carbon dataset...")

    carbon_df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "minute",
        CARBON_COLUMN
    ]

    for column in required_columns:

        if column not in carbon_df.columns:
            raise ValueError(
                f"Missing carbon column: {column}"
            )

    carbon_df = carbon_df[
        ["minute", CARBON_COLUMN]
    ].copy()

    carbon_df["minute"] = pd.to_numeric(
        carbon_df["minute"],
        errors="raise"
    )

    carbon_df[CARBON_COLUMN] = pd.to_numeric(
        carbon_df[CARBON_COLUMN],
        errors="raise"
    )

    if carbon_df.isnull().any().any():
        raise ValueError(
            "Missing values in carbon dataset."
        )

    if carbon_df["minute"].duplicated().any():
        raise ValueError(
            "Duplicate minute values in carbon dataset."
        )

    carbon_df = carbon_df.sort_values("minute")

    expected_minutes = np.arange(len(carbon_df))

    if not np.array_equal(
        carbon_df["minute"].to_numpy(),
        expected_minutes
    ):
        raise ValueError(
            "Carbon dataset must contain continuous minute indices "
            "starting from zero."
        )

    carbon_values = carbon_df[
        CARBON_COLUMN
    ].to_numpy(dtype=float)

    if not np.isfinite(carbon_values).all():
        raise ValueError(
            "Invalid carbon-intensity values found."
        )

    print(f"Carbon records loaded: {len(carbon_df):,}")

    print(
        f"Mean lifecycle carbon intensity: "
        f"{carbon_values.mean():.2f} gCO2e/kWh"
    )

    return carbon_values


# --------------------------------------------------
# 3. CARBON-AWARE SCHEDULING
# --------------------------------------------------

def schedule_carbon_aware(
    df,
    carbon_values,
    deadline_minutes
):

    print(
        f"\nRunning carbon-aware scheduling "
        f"with {deadline_minutes}-minute deadline..."
    )

    scheduled = df.copy()

    start_times = scheduled[
        "start_timestamp"
    ].to_numpy(dtype=float)

    durations = scheduled[
        "duration"
    ].to_numpy(dtype=float)

    workload_types = scheduled[
        "workload_type"
    ].to_numpy()

    total_carbon_minutes = len(carbon_values)

    # Convert original start timestamps into minute indices.
    original_minutes = np.floor(
        start_times / 60
    ).astype(np.int64)

    if (
        (original_minutes < 0).any()
        or (original_minutes >= total_carbon_minutes).any()
    ):
        raise ValueError(
            "Some invocation timestamps fall outside the "
            "available carbon-data window."
        )

    # Carbon intensity at each invocation's original minute.
    original_carbon = carbon_values[
        original_minutes
    ]

    # Default: execute at the original start time.
    best_start_times = start_times.copy()

    best_carbon = original_carbon.copy()

    # Only delay-tolerant jobs are eligible for rescheduling.
    tolerant_mask = (
        workload_types == "DELAY_TOLERANT"
    )

    # Examine future minute boundaries within the deadline.
    # Each candidate is compared with the original carbon intensity.
    for offset in range(1, deadline_minutes + 1):

        candidate_minutes = (
            original_minutes + offset
        )

        valid_mask = (
            tolerant_mask
            & (candidate_minutes < total_carbon_minutes)
        )

        candidate_indices = np.flatnonzero(
            valid_mask
        )

        if len(candidate_indices) == 0:
            continue

        candidate_start_times = (
            candidate_minutes[candidate_indices] * 60.0
        )

        waiting_times = (
            candidate_start_times
            - start_times[candidate_indices]
        )

        # Enforce the exact deadline, not just minute count.
        within_deadline = (
            (waiting_times > 0)
            & (waiting_times <= deadline_minutes * 60)
        )

        candidate_indices = candidate_indices[
            within_deadline
        ]

        if len(candidate_indices) == 0:
            continue

        candidate_minutes_valid = candidate_minutes[
            candidate_indices
        ]

        candidate_carbon = carbon_values[
            candidate_minutes_valid
        ]

        # Update only when carbon intensity is strictly lower.
        # Since offsets are checked in ascending order,
        # equal-carbon candidates retain the earlier schedule.
        improvement_mask = (
            candidate_carbon
            < best_carbon[candidate_indices]
        )

        improved_indices = candidate_indices[
            improvement_mask
        ]

        if len(improved_indices) == 0:
            continue

        best_start_times[improved_indices] = (
            candidate_minutes[improved_indices] * 60.0
        )

        best_carbon[improved_indices] = (
            candidate_carbon[improvement_mask]
        )

    # Calculate waiting time.
    waiting_times = (
        best_start_times - start_times
    )

    # Create scheduling output.
    scheduled["original_carbon_intensity"] = (
        original_carbon
    )

    scheduled["scheduled_carbon_intensity"] = (
        best_carbon
    )

    scheduled["scheduled_start_timestamp"] = (
        best_start_times
    )

    scheduled["scheduled_end_timestamp"] = (
        best_start_times + durations
    )

    scheduled["waiting_time_seconds"] = (
        waiting_times
    )

    scheduled["deadline_minutes"] = (
        deadline_minutes
    )

    scheduled["scheduling_strategy"] = (
        "CARBON_AWARE"
    )

    # Record whether the invocation was actually delayed.
    scheduled["was_delayed"] = (
        waiting_times > 0
    )

    return scheduled


# --------------------------------------------------
# 4. VALIDATE CARBON-AWARE SCHEDULE
# --------------------------------------------------

def validate_schedule(
    original,
    scheduled,
    deadline_minutes
):

    print("\nVALIDATING CARBON-AWARE SCHEDULE")
    print("================================")

    # Check invocation count.
    assert len(original) == len(scheduled), (
        "Invocation count changed."
    )

    # Immediate jobs must not be delayed.
    immediate_mask = (
        scheduled["workload_type"] == "IMMEDIATE"
    )

    assert (
        scheduled.loc[
            immediate_mask,
            "waiting_time_seconds"
        ] == 0
    ).all(), (
        "Immediate workload was delayed."
    )

    # Waiting times must be non-negative.
    assert (
        scheduled["waiting_time_seconds"] >= 0
    ).all(), (
        "Negative waiting time found."
    )

    # No invocation may exceed its deadline.
    assert (
        scheduled["waiting_time_seconds"]
        <= deadline_minutes * 60 + 1e-9
    ).all(), (
        "Deadline violation found."
    )

    # Execution durations must remain unchanged.
    calculated_durations = (
        scheduled["scheduled_end_timestamp"]
        - scheduled["scheduled_start_timestamp"]
    )

    assert np.allclose(
        calculated_durations,
        scheduled["duration"]
    ), (
        "Execution durations changed."
    )

    # Scheduled carbon must not be higher than original carbon.
    assert (
        scheduled["scheduled_carbon_intensity"]
        <= scheduled["original_carbon_intensity"] + 1e-9
    ).all(), (
        "Carbon intensity increased."
    )

    # Validate category preservation.
    assert (
        scheduled["workload_type"].equals(
            original["workload_type"]
        )
    ), (
        "Workload categories changed."
    )

    # Validate that original timestamps are preserved.
    assert np.allclose(
        scheduled["start_timestamp"],
        original["start_timestamp"]
    ), (
        "Original start timestamps changed."
    )

    print("PASS: All invocations preserved.")
    print("PASS: Immediate jobs were not delayed.")
    print("PASS: Waiting times are non-negative.")
    print("PASS: All jobs meet their deadlines.")
    print("PASS: Execution durations preserved.")
    print("PASS: Carbon intensity did not increase.")
    print("PASS: Workload categories preserved.")
    print("PASS: Original timestamps preserved.")

    print("\nAll carbon-aware validation checks passed.")


# --------------------------------------------------
# 5. GENERATE SCHEDULING SUMMARY
# --------------------------------------------------

def generate_summary(scheduled):

    summary = (
        scheduled.groupby("workload_type")
        .agg(
            invocation_count=("duration", "count"),

            average_duration=("duration", "mean"),

            total_execution_seconds=("duration", "sum"),

            average_waiting_seconds=(
                "waiting_time_seconds", "mean"
            ),

            maximum_waiting_seconds=(
                "waiting_time_seconds", "max"
            ),

            delayed_invocations=(
                "was_delayed", "sum"
            ),

            average_original_carbon=(
                "original_carbon_intensity", "mean"
            ),

            average_scheduled_carbon=(
                "scheduled_carbon_intensity", "mean"
            )
        )
        .reset_index()
    )

    summary["percentage"] = (
        summary["invocation_count"]
        / len(scheduled)
        * 100
    )

    return summary


# --------------------------------------------------
# 6. SAVE RESULTS
# --------------------------------------------------

def save_results(
    scheduled,
    summary,
    deadline_minutes
):

    output_file = (
        OUTPUT_DIR
        / f"carbon_aware_schedule_{deadline_minutes}min.csv"
    )

    summary_file = (
        OUTPUT_DIR
        / f"carbon_aware_summary_{deadline_minutes}min.csv"
    )

    scheduled.to_csv(
        output_file,
        index=False
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print("\nRESULTS SAVED")

    print(f"Schedule: {output_file}")

    print(f"Summary: {summary_file}")


# --------------------------------------------------
# 7. MAIN EXECUTION
# --------------------------------------------------

def main():

    original = load_classified_workload()

    carbon_values = load_carbon_data()

    for deadline in DEADLINE_SCENARIOS_MINUTES:

        scheduled = schedule_carbon_aware(
            original,
            carbon_values,
            deadline
        )

        validate_schedule(
            original,
            scheduled,
            deadline
        )

        summary = generate_summary(
            scheduled
        )

        print(
            f"\nCARBON-AWARE SUMMARY: "
            f"{deadline}-MINUTE DEADLINE"
        )

        print(
            summary.to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}"
            )
        )

        save_results(
            scheduled,
            summary,
            deadline
        )

        del scheduled
        del summary

    print("\nALL CARBON-AWARE SCHEDULING SCENARIOS COMPLETED!")


if __name__ == "__main__":
    main()
