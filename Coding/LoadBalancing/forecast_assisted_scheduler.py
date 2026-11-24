
from pathlib import Path
import math

import numpy as np
import pandas as pd


# =========================================================
# 1. PATHS AND CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLASSIFIED_FILE = (
    PROJECT_ROOT
    / "Coding"
    / "LoadBalancing"
    / "results"
    / "classified_workload.csv"
)

CARBON_WORKLOAD_FILE = (
    PROJECT_ROOT
    / "Coding"
    / "CarbonAnalysis"
    / "workload_carbon_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "Coding"
    / "LoadBalancing"
    / "results"
    / "forecast_assisted"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INTERVAL_SECONDS = 60

DEADLINE_SCENARIOS = [15, 30, 60]

CONCURRENCY_CAPACITY = 20

CARBON_COLUMN = "carbon_intensity_lifecycle"

IMMEDIATE_LABEL = "IMMEDIATE"
DELAY_TOLERANT_LABEL = "DELAY_TOLERANT"


# =========================================================
# 2. LOAD DATA
# =========================================================

def load_data():

    if not CLASSIFIED_FILE.exists():
        raise FileNotFoundError(CLASSIFIED_FILE)

    if not CARBON_WORKLOAD_FILE.exists():
        raise FileNotFoundError(CARBON_WORKLOAD_FILE)

    jobs = pd.read_csv(CLASSIFIED_FILE)

    workload = pd.read_csv(CARBON_WORKLOAD_FILE)

    required_job_columns = [
        "start_timestamp",
        "duration",
    ]

    for column in required_job_columns:
        if column not in jobs.columns:
            raise ValueError(
                f"Missing job column: {column}"
            )

    category_candidates = [
        "workload_category",
        "workload_type",
        "workload_class",
        "category",
    ]

    category_column = next(
        (
            column
            for column in category_candidates
            if column in jobs.columns
        ),
        None,
    )

    if category_column is None:
        raise ValueError(
            "Could not find workload category column. "
            f"Available columns: {list(jobs.columns)}"
        )

    required_workload_columns = [
        "minute",
        "invocation_count",
        "maximum_concurrency",
        CARBON_COLUMN,
    ]

    for column in required_workload_columns:
        if column not in workload.columns:
            raise ValueError(
                f"Missing workload column: {column}"
            )

    jobs["start_timestamp"] = pd.to_numeric(
        jobs["start_timestamp"],
        errors="raise",
    )

    jobs["duration"] = pd.to_numeric(
        jobs["duration"],
        errors="raise",
    )

    workload = workload.sort_values(
        "minute"
    ).reset_index(drop=True)

    jobs[category_column] = (
        jobs[category_column]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )

    workload["minute"] = pd.to_numeric(
        workload["minute"],
        errors="raise",
    ).astype(int)

    if jobs["start_timestamp"].isna().any():
        raise ValueError("Missing job start timestamps.")

    if jobs["duration"].isna().any():
        raise ValueError("Missing job durations.")

    if (jobs["duration"] < 0).any():
        raise ValueError("Negative durations found.")

    if workload["minute"].duplicated().any():
        raise ValueError("Duplicate workload minutes found.")

    expected_minutes = np.arange(len(workload))

    if not np.array_equal(
        workload["minute"].to_numpy(),
        expected_minutes,
    ):
        raise ValueError(
            "Workload minute index must be continuous "
            "and start from zero."
        )

    jobs["workload_category"] = jobs[category_column]

    print("\nDATA LOADED")
    print("-----------")
    print(f"Total jobs: {len(jobs):,}")
    print(f"Workload minutes: {len(workload):,}")
    print(f"Category column: {category_column}")

    return jobs, workload


# =========================================================
# 3. CREATE ONE-MINUTE-AHEAD FORECASTS
# =========================================================

def create_forecasts(workload):

    forecast = workload[
        [
            "minute",
            "invocation_count",
            "maximum_concurrency",
        ]
    ].copy()

    # -----------------------------------------------------
    # Invocation count:
    # Validated 5-minute moving average.
    # -----------------------------------------------------

    rolling_prediction = (
        forecast["invocation_count"]
        .rolling(
            window=5,
            min_periods=5,
        )
        .mean()
    )

    # At minute t, forecast the invocation count at t+1.
    # The first four observations have no five-minute
    # history, so persistence is used as a fallback.

    invocation_forecast = rolling_prediction.to_numpy()

    actual_invocations = (
        forecast["invocation_count"].to_numpy(dtype=float)
    )

    invocation_forecast = np.where(
        np.isfinite(invocation_forecast),
        invocation_forecast,
        actual_invocations,
    )

    # -----------------------------------------------------
    # Maximum concurrency:
    # Validated persistence baseline.
    # -----------------------------------------------------

    actual_concurrency = (
        forecast["maximum_concurrency"]
        .to_numpy(dtype=float)
    )

    # Forecast for minute t+1 uses observed concurrency at t.
    concurrency_forecast = np.empty(
        len(forecast),
        dtype=float,
    )

    concurrency_forecast[0] = actual_concurrency[0]

    concurrency_forecast[1:] = (
        actual_concurrency[:-1]
    )

    forecast["predicted_invocation_count_next_minute"] = (
        np.maximum(invocation_forecast, 0)
    )

    forecast["predicted_maximum_concurrency_next_minute"] = (
        np.maximum(concurrency_forecast, 0)
    )

    forecast["predicted_required_instances"] = (
        np.ceil(
            forecast[
                "predicted_maximum_concurrency_next_minute"
            ]
            / CONCURRENCY_CAPACITY
        )
        .astype(int)
    )

    # Forecast at minute k is based on observations
    # available at minute k-1.
    forecast["forecast_for_minute"] = (
        forecast["minute"] + 1
    )

    forecast.to_csv(
        OUTPUT_DIR / "workload_forecasts.csv",
        index=False,
    )

    print("\nFORECASTS CREATED")
    print("-----------------")
    print("Invocation count: 5-minute moving average")
    print("Maximum concurrency: persistence")
    print("Forecast horizon: 1 minute")
    print(
        "Capacity per simulated instance:",
        CONCURRENCY_CAPACITY,
    )

    return forecast


# =========================================================
# 4. CREATE CARBON-BASED SCHEDULE
# =========================================================

def schedule_jobs(
    jobs,
    workload,
    deadline_minutes,
):

    carbon = workload[
        CARBON_COLUMN
    ].to_numpy(dtype=float)

    number_of_minutes = len(carbon)

    if not np.isfinite(carbon).all():
        raise ValueError(
            "Missing or invalid carbon intensity values."
        )

    starts = jobs[
        "start_timestamp"
    ].to_numpy(dtype=float)

    durations = jobs[
        "duration"
    ].to_numpy(dtype=float)

    original_minutes = np.floor(
        starts / INTERVAL_SECONDS
    ).astype(int)

    if (
        original_minutes.min() < 0
        or original_minutes.max() >= number_of_minutes
    ):
        raise ValueError(
            "Some jobs are outside the carbon dataset timeline."
        )

    categories = jobs[
        "workload_category"
    ].to_numpy()

    immediate_mask = (
        categories == IMMEDIATE_LABEL
    )

    tolerant_mask = (
        categories == DELAY_TOLERANT_LABEL
    )

    if not np.all(
        immediate_mask | tolerant_mask
    ):
        unknown = np.unique(
            categories[~(immediate_mask | tolerant_mask)]
        )

        raise ValueError(
            f"Unexpected workload categories: {unknown}"
        )

    # Initially, all jobs remain at original timestamps.
    scheduled_starts = starts.copy()

    # Record selected execution minute for each job.
    scheduled_minutes = original_minutes.copy()

    # -----------------------------------------------------
    # Group delay-tolerant jobs by original minute.
    # This avoids iterating over all 594,285 jobs.
    # -----------------------------------------------------

    tolerant_original_minutes = np.unique(
        original_minutes[tolerant_mask]
    )

    delayed_groups = 0

    for original_minute in tolerant_original_minutes:

        deadline_minute = min(
            original_minute + deadline_minutes,
            number_of_minutes - 1,
        )

        candidate_minutes = np.arange(
            original_minute,
            deadline_minute + 1,
        )

        candidate_carbon = carbon[
            candidate_minutes
        ]

        best_position = int(
            np.argmin(candidate_carbon)
        )

        best_minute = int(
            candidate_minutes[best_position]
        )

        original_carbon = carbon[
            original_minute
        ]

        best_carbon = carbon[
            best_minute
        ]

        # Shift only when carbon intensity is strictly lower.
        if best_carbon < original_carbon:

            selected_mask = (
                tolerant_mask
                & (original_minutes == original_minute)
            )

            scheduled_minutes[selected_mask] = (
                best_minute
            )

            scheduled_starts[selected_mask] = (
                best_minute * INTERVAL_SECONDS
            )

            delayed_groups += 1

    waiting_seconds = (
        scheduled_starts - starts
    )

    scheduled_ends = (
        scheduled_starts + durations
    )

    # -----------------------------------------------------
    # Build result.
    # -----------------------------------------------------

    result = jobs.copy()

    result["original_start_timestamp"] = starts

    result["scheduled_start_timestamp"] = (
        scheduled_starts
    )

    result["scheduled_end_timestamp"] = (
        scheduled_ends
    )

    result["original_minute"] = original_minutes

    result["scheduled_minute"] = scheduled_minutes

    result["waiting_time_seconds"] = waiting_seconds

    result["original_carbon_intensity"] = (
        carbon[original_minutes]
    )

    result["scheduled_carbon_intensity"] = (
        carbon[scheduled_minutes]
    )

    # Forecasted capacity requirement at the selected minute.
    actual_concurrency = (
        workload["maximum_concurrency"]
        .to_numpy(dtype=float)
    )

    predicted_concurrency = np.empty(
        number_of_minutes,
        dtype=float,
    )

    predicted_concurrency[0] = (
        actual_concurrency[0]
    )

    predicted_concurrency[1:] = (
        actual_concurrency[:-1]
    )

    result["forecast_maximum_concurrency"] = (
        predicted_concurrency[scheduled_minutes]
    )

    result["forecast_required_instances"] = (
        np.ceil(
            result["forecast_maximum_concurrency"]
            / CONCURRENCY_CAPACITY
        )
        .astype(int)
    )

    # This is an advisory forecast, not a hard capacity limit.
    result["scheduling_strategy"] = (
        f"FORECAST_ASSISTED_{deadline_minutes}MIN"
    )

    return result, delayed_groups


# =========================================================
# 5. EVALUATE SCHEDULE
# =========================================================

def evaluate_schedule(
    result,
    deadline_minutes,
    delayed_groups,
):

    immediate = result[
        result["workload_category"] == IMMEDIATE_LABEL
    ]

    tolerant = result[
        result["workload_category"] == DELAY_TOLERANT_LABEL
    ]

    delayed = tolerant[
        tolerant["waiting_time_seconds"] > 0
    ]

    original_average_carbon = (
        tolerant["original_carbon_intensity"].mean()
    )

    scheduled_average_carbon = (
        tolerant["scheduled_carbon_intensity"].mean()
    )

    if original_average_carbon > 0:

        carbon_intensity_change_percent = (
            (
                original_average_carbon
                - scheduled_average_carbon
            )
            / original_average_carbon
        ) * 100

    else:
        carbon_intensity_change_percent = 0.0

    summary = {
        "deadline_minutes": deadline_minutes,
        "total_invocations": len(result),
        "immediate_invocations": len(immediate),
        "delay_tolerant_invocations": len(tolerant),
        "delayed_invocations": len(delayed),
        "delayed_percentage": (
            len(delayed) / len(tolerant) * 100
            if len(tolerant) else 0
        ),
        "delayed_original_minute_groups": delayed_groups,
        "average_waiting_seconds": (
            tolerant["waiting_time_seconds"].mean()
            if len(tolerant) else 0
        ),
        "maximum_waiting_seconds": (
            result["waiting_time_seconds"].max()
        ),
        "average_original_carbon_intensity": (
            original_average_carbon
        ),
        "average_scheduled_carbon_intensity": (
            scheduled_average_carbon
        ),
        "carbon_intensity_change_percent": (
            carbon_intensity_change_percent
        ),
        "average_forecast_required_instances": (
            result["forecast_required_instances"].mean()
        ),
        "maximum_forecast_required_instances": (
            result["forecast_required_instances"].max()
        ),
    }

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    assert len(result) > 0

    assert (
        result["waiting_time_seconds"] >= 0
    ).all()

    assert (
        result["scheduled_start_timestamp"]
        >= result["original_start_timestamp"]
    ).all()

    assert (
        result["scheduled_end_timestamp"]
        - result["scheduled_start_timestamp"]
        - result["duration"]
    ).abs().max() < 1e-8

    assert (
        result["waiting_time_seconds"]
        <= deadline_minutes * 60 + 1e-8
    ).all()

    # Immediate jobs must never be delayed.
    assert (
        immediate["waiting_time_seconds"] == 0
    ).all()

    # Carbon intensity should never increase for shifted jobs.
    assert (
        result.loc[
            result["waiting_time_seconds"] > 0,
            "scheduled_carbon_intensity",
        ]
        <
        result.loc[
            result["waiting_time_seconds"] > 0,
            "original_carbon_intensity",
        ]
    ).all()

    return summary


# =========================================================
# 6. MAIN
# =========================================================

def main():

    print("\nFORECAST-ASSISTED LOAD BALANCING")
    print("================================")

    jobs, workload = load_data()

    forecasts = create_forecasts(workload)

    summaries = []

    for deadline in DEADLINE_SCENARIOS:

        print(f"\nRUNNING {deadline}-MINUTE SCENARIO")
        print("-" * 45)

        result, delayed_groups = schedule_jobs(
            jobs,
            workload,
            deadline,
        )

        summary = evaluate_schedule(
            result,
            deadline,
            delayed_groups,
        )

        summaries.append(summary)

        output_file = (
            OUTPUT_DIR
            / f"forecast_assisted_schedule_{deadline}min.csv"
        )

        result.to_csv(
            output_file,
            index=False,
        )

        print(f"Total jobs: {summary['total_invocations']:,}")
        print(f"Delayed jobs: {summary['delayed_invocations']:,}")
        print(
            f"Average waiting: "
            f"{summary['average_waiting_seconds']:.2f} seconds"
        )
        print(
            f"Average carbon intensity: "
            f"{summary['average_original_carbon_intensity']:.2f}"
            f" -> "
            f"{summary['average_scheduled_carbon_intensity']:.2f}"
        )
        print(
            f"Carbon intensity change: "
            f"{summary['carbon_intensity_change_percent']:.4f}%"
        )

        print(f"Saved: {output_file}")

        print("\nPASS: Schedule validation completed.")

    # -----------------------------------------------------
    # SAVE SUMMARY
    # -----------------------------------------------------

    summary_df = pd.DataFrame(summaries)

    summary_file = (
        OUTPUT_DIR
        / "forecast_assisted_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    print("\nALL SCENARIOS COMPLETED")
    print("=======================")

    print(
        summary_df.to_string(index=False)
    )

    print("\nOutput directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
