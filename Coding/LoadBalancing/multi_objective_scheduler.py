
import gc
import numpy as np
import pandas as pd

from config import (
    OUTPUT_DIR,
    INPUT_FILE,
    DEADLINE_SCENARIOS_MINUTES,
    CARBON_COLUMN,
    CONCURRENCY_CAPACITY,
    IDLE_POWER_W,
    PEAK_POWER_W,
    MEMORY_ALLOCATION_MB,
    EXECUTION_PRICE_PER_MILLION,
    PRICE_PER_GB_SECOND,
    ENERGY_WEIGHT,
    CARBON_WEIGHT,
    COST_WEIGHT,
    INTERVAL_SECONDS
)


# ==========================================================
# 1. LOAD CLASSIFIED WORKLOAD
# ==========================================================

def load_classified_workload():

    path = OUTPUT_DIR / "classified_workload.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Classified workload not found: {path}"
        )

    print("\nLoading classified workload...")

    df = pd.read_csv(path)

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

    if df.empty:
        raise ValueError("Workload dataset is empty.")

    numeric_columns = [
        "end_timestamp",
        "duration",
        "start_timestamp"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        )

        if not np.isfinite(
            df[column].to_numpy(dtype=float)
        ).all():

            raise ValueError(
                f"Invalid numeric values in {column}."
            )

    if (df["duration"] < 0).any():
        raise ValueError("Negative execution duration found.")

    valid_categories = {
        "IMMEDIATE",
        "DELAY_TOLERANT"
    }

    if not set(df["workload_type"].unique()).issubset(
        valid_categories
    ):
        raise ValueError("Invalid workload category.")

    print(f"Total invocations: {len(df):,}")

    return df


# ==========================================================
# 2. LOAD MINUTE-LEVEL DATA
# ==========================================================

def load_minute_metrics():

    print("\nLoading minute-level energy and carbon data...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "minute",
        "total_execution_seconds",
        "average_concurrency",
        "maximum_concurrency",
        CARBON_COLUMN
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Missing input column: {column}"
            )

    df = df[required_columns].copy()

    for column in required_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        )

    df = df.sort_values(
        "minute"
    ).reset_index(drop=True)

    if df.isnull().any().any():
        raise ValueError(
            "Missing values found in minute-level dataset."
        )

    expected_minutes = np.arange(len(df))

    if not np.array_equal(
        df["minute"].to_numpy(),
        expected_minutes
    ):
        raise ValueError(
            "Minute indices must be continuous and start at zero."
        )

    numeric_values = df[
        required_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Non-finite values found in minute-level dataset."
        )

    if (
        (df["total_execution_seconds"] < 0).any()
        or (df["average_concurrency"] < 0).any()
        or (df["maximum_concurrency"] < 0).any()
        or (df[CARBON_COLUMN] < 0).any()
    ):
        raise ValueError(
            "Negative workload or carbon values found."
        )

    print(f"Minute records: {len(df):,}")

    return df


# ==========================================================
# 3. NORMALIZATION
# ==========================================================

def min_max_normalize(values):

    values = np.asarray(
        values,
        dtype=float
    )

    finite_mask = np.isfinite(values)

    if not finite_mask.any():
        raise ValueError(
            "No finite values available for normalization."
        )

    minimum = np.min(
        values[finite_mask]
    )

    maximum = np.max(
        values[finite_mask]
    )

    result = np.zeros_like(
        values,
        dtype=float
    )

    if maximum == minimum:

        result[finite_mask] = 0.0

    else:

        result[finite_mask] = (
            values[finite_mask] - minimum
        ) / (maximum - minimum)

    # Undefined values receive a neutral normalized score.
    result[~finite_mask] = 0.5

    return result


# ==========================================================
# 4. ESTIMATE MINUTE-LEVEL ENERGY
# ==========================================================

def calculate_minute_energy_metrics(minute_df):

    print("\nEstimating minute-level energy...")

    average_concurrency = minute_df[
        "average_concurrency"
    ].to_numpy(dtype=float)

    maximum_concurrency = minute_df[
        "maximum_concurrency"
    ].to_numpy(dtype=float)

    execution_seconds = minute_df[
        "total_execution_seconds"
    ].to_numpy(dtype=float)

    # Estimate required instances.
    instance_count = np.ceil(
        maximum_concurrency / CONCURRENCY_CAPACITY
    ).astype(int)

    # Calculate total concurrency capacity.
    total_capacity = (
        instance_count * CONCURRENCY_CAPACITY
    )

    # Estimate utilization.
    utilization = np.divide(
        average_concurrency,
        total_capacity,
        out=np.zeros_like(average_concurrency),
        where=total_capacity > 0
    )

    utilization = np.clip(
        utilization,
        0,
        1
    )

    # Estimate power consumption.
    power_w = np.where(
        instance_count > 0,

        instance_count * (
            IDLE_POWER_W
            + utilization * (
                PEAK_POWER_W - IDLE_POWER_W
            )
        ),

        0.0
    )

    # Energy consumed during one minute.
    energy_kwh = (
        power_w
        * INTERVAL_SECONDS
        / 3_600_000
    )

    # Energy per execution second.
    energy_intensity = np.divide(
        energy_kwh,
        execution_seconds,
        out=np.full_like(
            energy_kwh,
            np.nan
        ),
        where=execution_seconds > 0
    )

    # ------------------------------------------------------
    # FIX: HANDLE UNDEFINED ENERGY INTENSITY
    # ------------------------------------------------------

    valid_energy = (
        np.isfinite(energy_intensity)
        & (energy_intensity >= 0)
    )

    if not valid_energy.any():
        raise ValueError(
            "No valid energy-intensity values available."
        )

    median_energy_intensity = np.median(
        energy_intensity[valid_energy]
    )

    undefined_count = (
        ~valid_energy
    ).sum()

    energy_intensity[~valid_energy] = (
        median_energy_intensity
    )

    print(
        f"Valid energy-intensity minutes: "
        f"{valid_energy.sum():,}"
    )

    print(
        f"Undefined energy-intensity minutes: "
        f"{undefined_count:,}"
    )

    print(
        f"Median energy-intensity fallback: "
        f"{median_energy_intensity:.10f} kWh/sec"
    )

    minute_df["estimated_instances"] = (
        instance_count
    )

    minute_df["estimated_power_w"] = (
        power_w
    )

    minute_df["estimated_energy_kwh"] = (
        energy_kwh
    )

    minute_df["energy_per_execution_second"] = (
        energy_intensity
    )

    if not np.isfinite(
        minute_df[
            "energy_per_execution_second"
        ].to_numpy(dtype=float)
    ).all():

        raise ValueError(
            "Energy-intensity calculation still contains "
            "non-finite values."
        )

    print(
        "PASS: Minute-level energy metrics calculated."
    )

    return minute_df


# ==========================================================
# 5. CALCULATE INVOCATION COST
# ==========================================================

def calculate_invocation_cost(durations):

    # Minimum billable execution duration.
    billable_duration = np.maximum(
        durations,
        0.100
    )

    # Round upward to nearest millisecond.
    billable_duration = (
        np.ceil(
            billable_duration * 1000
        ) / 1000
    )

    # Round memory to nearest 128 MB.
    billable_memory_mb = (
        np.ceil(
            MEMORY_ALLOCATION_MB / 128
        ) * 128
    )

    billable_memory_mb = min(
        billable_memory_mb,
        1536
    )

    # Calculate GB-seconds.
    gb_seconds = (
        billable_memory_mb / 1024
    ) * billable_duration

    # Execution charge.
    execution_cost = (
        EXECUTION_PRICE_PER_MILLION
        / 1_000_000
    )

    # Resource charge.
    resource_cost = (
        gb_seconds * PRICE_PER_GB_SECOND
    )

    total_cost = (
        execution_cost + resource_cost
    )

    return total_cost


# ==========================================================
# 6. CALCULATE OBJECTIVE SCORES
# ==========================================================

def calculate_objective_scores(
    energy_values,
    carbon_values,
    cost_values
):

    energy_normalized = min_max_normalize(
        energy_values
    )

    carbon_normalized = min_max_normalize(
        carbon_values
    )

    cost_normalized = min_max_normalize(
        cost_values
    )

    score = (
        ENERGY_WEIGHT * energy_normalized
        + CARBON_WEIGHT * carbon_normalized
        + COST_WEIGHT * cost_normalized
    )

    return (
        score,
        energy_normalized,
        carbon_normalized,
        cost_normalized
    )


# ==========================================================
# 7. MULTI-OBJECTIVE SCHEDULER
# ==========================================================

def run_multi_objective_scheduler(
    workload,
    minute_metrics,
    deadline_minutes
):

    print(
        f"\nRunning multi-objective scheduling: "
        f"{deadline_minutes}-minute deadline"
    )

    start_times = workload[
        "start_timestamp"
    ].to_numpy(dtype=float)

    durations = workload[
        "duration"
    ].to_numpy(dtype=float)

    workload_types = workload[
        "workload_type"
    ].to_numpy()

    carbon = minute_metrics[
        CARBON_COLUMN
    ].to_numpy(dtype=float)

    energy_intensity = minute_metrics[
        "energy_per_execution_second"
    ].to_numpy(dtype=float)

    number_of_minutes = len(minute_metrics)

    if number_of_minutes == 0:
        raise ValueError("Minute-level metrics are empty.")

    if not np.isfinite(start_times).all():
        raise ValueError("Invalid workload start timestamps.")

    if not np.isfinite(durations).all() or (durations < 0).any():
        raise ValueError("Invalid workload durations.")

    if not np.isfinite(carbon).all() or (carbon < 0).any():
        raise ValueError("Invalid carbon intensity values.")

    if (
        not np.isfinite(energy_intensity).all()
        or (energy_intensity < 0).any()
    ):
        raise ValueError("Invalid energy intensity values.")

    # Convert original start times to minute indices.
    original_minutes = np.floor(
        start_times / 60
    ).astype(np.int64)

    if (
        (original_minutes < 0).any()
        or (original_minutes >= number_of_minutes).any()
    ):
        raise ValueError(
            "Invocation falls outside available timeline."
        )

    # Calculate invocation-level cost.
    invocation_cost = calculate_invocation_cost(
        durations
    )

    # Normalize energy and carbon across candidate minutes.
    energy_normalized = min_max_normalize(
        energy_intensity
    )

    carbon_normalized = min_max_normalize(
        carbon
    )

    # Cost is invocation-specific and time-independent.
    cost_normalized = min_max_normalize(
        invocation_cost
    )

    # Original objective components.
    original_energy = energy_intensity[
        original_minutes
    ]

    original_carbon = carbon[
        original_minutes
    ]

    original_energy_norm = energy_normalized[
        original_minutes
    ]

    original_carbon_norm = carbon_normalized[
        original_minutes
    ]

    original_cost_norm = cost_normalized

    original_score = (
        ENERGY_WEIGHT * original_energy_norm
        + CARBON_WEIGHT * original_carbon_norm
        + COST_WEIGHT * original_cost_norm
    )

    # Initially, every invocation executes at its original time.
    best_start = start_times.copy()

    best_energy = original_energy.copy()

    best_carbon = original_carbon.copy()

    best_score = original_score.copy()

    # Only delay-tolerant workload can be rescheduled.
    tolerant_mask = (
        workload_types == "DELAY_TOLERANT"
    )

    # ------------------------------------------------------
    # Examine all eligible future minute boundaries.
    # ------------------------------------------------------

    for offset in range(
        1,
        deadline_minutes + 1
    ):

        candidate_minutes = (
            original_minutes + offset
        )

        valid_mask = (
            tolerant_mask
            & (candidate_minutes < number_of_minutes)
        )

        candidate_indices = np.flatnonzero(
            valid_mask
        )

        if len(candidate_indices) == 0:
            continue

        candidate_start = (
            candidate_minutes[candidate_indices] * 60.0
        )

        waiting_times = (
            candidate_start
            - start_times[candidate_indices]
        )

        within_deadline = (
            (waiting_times > 0)
            & (
                waiting_times
                <= deadline_minutes * 60 + 1e-9
            )
        )

        candidate_indices = candidate_indices[
            within_deadline
        ]

        if len(candidate_indices) == 0:
            continue

        selected_minutes = candidate_minutes[
            candidate_indices
        ]

        candidate_energy_norm = (
            energy_normalized[selected_minutes]
        )

        candidate_carbon_norm = (
            carbon_normalized[selected_minutes]
        )

        candidate_cost_norm = (
            cost_normalized[candidate_indices]
        )

        candidate_energy = (
            energy_intensity[selected_minutes]
        )

        candidate_carbon = (
            carbon[selected_minutes]
        )

        candidate_score = (
            ENERGY_WEIGHT * candidate_energy_norm
            + CARBON_WEIGHT * candidate_carbon_norm
            + COST_WEIGHT * candidate_cost_norm
        )

        # Select only if the combined score improves.
        improvement = (
            candidate_score
            < best_score[candidate_indices] - 1e-12
        )

        improved_indices = candidate_indices[
            improvement
        ]

        if len(improved_indices) == 0:
            continue

        best_score[improved_indices] = (
            candidate_score[improvement]
        )

        best_start[improved_indices] = (
            selected_minutes[improvement] * 60.0
        )

        best_energy[improved_indices] = (
            candidate_energy[improvement]
        )

        best_carbon[improved_indices] = (
            candidate_carbon[improvement]
        )

    # ------------------------------------------------------
    # Prepare output
    # ------------------------------------------------------

    result = workload.copy()

    result["original_carbon_intensity"] = (
        original_carbon
    )

    result["scheduled_carbon_intensity"] = (
        best_carbon
    )

    result["original_energy_intensity"] = (
        original_energy
    )

    result["scheduled_energy_intensity"] = (
        best_energy
    )

    result["estimated_cost_usd"] = (
        invocation_cost
    )

    result["scheduled_start_timestamp"] = (
        best_start
    )

    result["scheduled_end_timestamp"] = (
        best_start + durations
    )

    result["waiting_time_seconds"] = (
        best_start - start_times
    )

    result["was_delayed"] = (
        result["waiting_time_seconds"] > 0
    )

    result["original_objective_score"] = (
        original_score
    )

    result["scheduled_objective_score"] = (
        best_score
    )

    result["deadline_minutes"] = (
        deadline_minutes
    )

    result["scheduling_strategy"] = (
        "MULTI_OBJECTIVE"
    )

    print(
        f"Scheduled invocations: {len(result):,}"
    )

    return result


# ==========================================================
# 8. VALIDATE SCHEDULE
# ==========================================================

def validate_schedule(
    original,
    scheduled,
    deadline_minutes
):

    print(
        "\nVALIDATING MULTI-OBJECTIVE SCHEDULE"
    )

    print(
        "==================================="
    )

    checks = {}

    checks["Invocation count preserved"] = (
        len(original) == len(scheduled)
    )

    checks["Original timestamps preserved"] = (
        np.allclose(
            scheduled["start_timestamp"],
            original["start_timestamp"]
        )
    )

    checks["Waiting times non-negative"] = (
        scheduled["waiting_time_seconds"] >= 0
    ).all()

    checks["Deadline compliance"] = (
        scheduled["waiting_time_seconds"]
        <= deadline_minutes * 60 + 1e-9
    ).all()

    immediate_mask = (
        scheduled["workload_type"] == "IMMEDIATE"
    )

    checks["Immediate jobs not delayed"] = (
        scheduled.loc[
            immediate_mask,
            "waiting_time_seconds"
        ] == 0
    ).all()

    calculated_duration = (
        scheduled["scheduled_end_timestamp"]
        - scheduled["scheduled_start_timestamp"]
    )

    checks["Execution durations preserved"] = (
        np.allclose(
            calculated_duration,
            scheduled["duration"]
        )
    )

    checks["Objective score did not increase"] = (
        scheduled["scheduled_objective_score"]
        <= scheduled["original_objective_score"] + 1e-9
    ).all()

    checks["Estimated costs are valid"] = (
        np.isfinite(
            scheduled["estimated_cost_usd"]
        ).all()
        and
        (scheduled["estimated_cost_usd"] >= 0).all()
    )

    checks["Carbon values are valid"] = (
        np.isfinite(
            scheduled["scheduled_carbon_intensity"]
        ).all()
        and
        (scheduled["scheduled_carbon_intensity"] >= 0).all()
    )

    checks["Energy values are valid"] = (
        np.isfinite(
            scheduled["scheduled_energy_intensity"]
        ).all()
        and
        (scheduled["scheduled_energy_intensity"] >= 0).all()
    )

    checks["Objective scores are finite"] = (
        np.isfinite(
            scheduled["scheduled_objective_score"]
        ).all()
        and
        np.isfinite(
            scheduled["original_objective_score"]
        ).all()
    )

    checks["Workload categories preserved"] = (
        scheduled["workload_type"].equals(
            original["workload_type"]
        )
    )

    failed_checks = []

    for name, passed in checks.items():

        if passed:

            print(f"PASS: {name}")

        else:

            print(f"FAIL: {name}")

            failed_checks.append(name)

    if failed_checks:

        raise AssertionError(
            "Validation failed: "
            + ", ".join(failed_checks)
        )

    print(
        "\nAll multi-objective validation checks passed."
    )


# ==========================================================
# 9. GENERATE SUMMARY
# ==========================================================

def generate_summary(scheduled):

    summary = (
        scheduled.groupby("workload_type")
        .agg(

            invocation_count=(
                "duration",
                "count"
            ),

            delayed_invocations=(
                "was_delayed",
                "sum"
            ),

            average_waiting_seconds=(
                "waiting_time_seconds",
                "mean"
            ),

            maximum_waiting_seconds=(
                "waiting_time_seconds",
                "max"
            ),

            average_original_carbon=(
                "original_carbon_intensity",
                "mean"
            ),

            average_scheduled_carbon=(
                "scheduled_carbon_intensity",
                "mean"
            ),

            average_original_energy_intensity=(
                "original_energy_intensity",
                "mean"
            ),

            average_scheduled_energy_intensity=(
                "scheduled_energy_intensity",
                "mean"
            ),

            total_estimated_cost_usd=(
                "estimated_cost_usd",
                "sum"
            ),

            average_original_objective=(
                "original_objective_score",
                "mean"
            ),

            average_scheduled_objective=(
                "scheduled_objective_score",
                "mean"
            )
        )
        .reset_index()
    )

    summary["percentage"] = (
        summary["invocation_count"]
        / len(scheduled)
        * 100
    )

    summary["delayed_percentage"] = (
        summary["delayed_invocations"]
        / summary["invocation_count"]
        * 100
    )

    return summary


# ==========================================================
# 10. SAVE RESULTS
# ==========================================================

def save_results(
    scheduled,
    summary,
    deadline_minutes
):

    output_file = (
        OUTPUT_DIR
        / f"multi_objective_schedule_{deadline_minutes}min.csv"
    )

    summary_file = (
        OUTPUT_DIR
        / f"multi_objective_summary_{deadline_minutes}min.csv"
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

    print(
        f"Schedule: {output_file}"
    )

    print(
        f"Summary: {summary_file}"
    )


# ==========================================================
# 11. MAIN EXECUTION
# ==========================================================

def main():

    # Validate objective weights.
    weights = np.array(
        [
            ENERGY_WEIGHT,
            CARBON_WEIGHT,
            COST_WEIGHT
        ],
        dtype=float
    )

    if (
        (weights < 0).any()
        or not np.isclose(weights.sum(), 1.0)
    ):
        raise ValueError(
            "Objective weights must be non-negative "
            "and sum to 1."
        )

    print(
        "\nMULTI-OBJECTIVE SCHEDULER"
    )

    print(
        "========================="
    )

    print(
        f"Energy weight: {ENERGY_WEIGHT:.2f}"
    )

    print(
        f"Carbon weight: {CARBON_WEIGHT:.2f}"
    )

    print(
        f"Cost weight: {COST_WEIGHT:.2f}"
    )

    print(
        "\nNOTE: Azure Consumption execution cost is "
        "time-independent in the current model."
    )

    workload = load_classified_workload()

    minute_metrics = load_minute_metrics()

    minute_metrics = calculate_minute_energy_metrics(
        minute_metrics
    )

    for deadline in DEADLINE_SCENARIOS_MINUTES:

        scheduled = run_multi_objective_scheduler(
            workload,
            minute_metrics,
            deadline
        )

        validate_schedule(
            workload,
            scheduled,
            deadline
        )

        summary = generate_summary(
            scheduled
        )

        print(
            f"\nMULTI-OBJECTIVE SUMMARY: "
            f"{deadline}-MINUTE DEADLINE"
        )

        print(
            summary.to_string(
                index=False,
                float_format=lambda value: f"{value:.6f}"
            )
        )

        save_results(
            scheduled,
            summary,
            deadline
        )

        del scheduled
        del summary

        gc.collect()

    print(
        "\nALL MULTI-OBJECTIVE SCENARIOS COMPLETED!"
    )


if __name__ == "__main__":

    main()
