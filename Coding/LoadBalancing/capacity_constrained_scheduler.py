
from pathlib import Path
import math
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = (
    ROOT / "Coding" / "LoadBalancing" / "results"
)

INPUT_FILE = RESULT_DIR / "classified_workload.csv"

CARBON_FILE = (
    ROOT / "Coding" / "CarbonAnalysis"
    / "workload_carbon_dataset.csv"
)

OUTPUT_DIR = RESULT_DIR / "capacity_constrained"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEADLINES = [15, 30, 60]

CAPACITY_PER_INSTANCE = 20

# Experimental ceiling derived from observed baseline peak.
MAX_CONCURRENCY = 3027

# One-minute execution-work budget, in execution-seconds.
CAPACITY_BUDGET = MAX_CONCURRENCY * 60


# ============================================================
# LOAD INPUTS
# ============================================================

def load_inputs():

    jobs = pd.read_csv(INPUT_FILE)
    carbon = pd.read_csv(CARBON_FILE)

    required_job_columns = [
        "start_timestamp",
        "duration",
        "workload_type"
    ]

    for col in required_job_columns:
        if col not in jobs.columns:
            raise ValueError(
                f"Missing job column: {col}"
            )

    if "carbon_intensity_lifecycle" not in carbon.columns:
        raise ValueError(
            "Missing carbon_intensity_lifecycle column."
        )

    jobs["start_timestamp"] = pd.to_numeric(
        jobs["start_timestamp"], errors="coerce"
    )

    jobs["duration"] = pd.to_numeric(
        jobs["duration"], errors="coerce"
    )

    if jobs[["start_timestamp", "duration"]].isna().any().any():
        raise ValueError("Invalid job timestamps or durations.")

    jobs["workload_type"] = (
        jobs["workload_type"].astype(str).str.upper()
    )

    valid_types = {"IMMEDIATE", "DELAY_TOLERANT"}

    if not set(jobs["workload_type"].unique()).issubset(valid_types):
        raise ValueError(
            "Unexpected workload_type values: "
            f"{jobs['workload_type'].unique()}"
        )

    carbon = carbon.sort_values("minute").reset_index(drop=True)

    carbon_values = carbon[
        "carbon_intensity_lifecycle"
    ].to_numpy(dtype=float)

    if not np.isfinite(carbon_values).all():
        raise ValueError("Invalid carbon intensity values.")

    return jobs, carbon_values


# ============================================================
# EXECUTION-WORK ACCOUNTING
# ============================================================

def add_execution_work(load, start, duration):
    """
    Assign each invocation's execution seconds to its
    scheduled start minute.

    This is a per-minute execution-work approximation.
    It is not an exact instantaneous concurrency model.
    """

    if duration <= 0:
        return

    minute = int(start // 60)

    if 0 <= minute < len(load):
        load[minute] += duration


def can_accept(load, minute, duration):

    if duration <= 0:
        return True

    if minute < 0 or minute >= len(load):
        return False

    return (
        load[minute] + duration
        <= CAPACITY_BUDGET
    )


# ============================================================
# CREATE CAPACITY-CONSTRAINED SCHEDULE
# ============================================================

def schedule_jobs(jobs, carbon_values, deadline):

    jobs = jobs.copy()

    jobs["original_start_timestamp"] = (
        jobs["start_timestamp"]
    )

    jobs["scheduled_start_timestamp"] = (
        jobs["start_timestamp"]
    )

    jobs["original_minute"] = (
        jobs["start_timestamp"] // 60
    ).astype(int)

    jobs["scheduled_minute"] = jobs["original_minute"]

    jobs["waiting_time_seconds"] = 0.0

    jobs["original_carbon_intensity"] = [
        carbon_values[m]
        for m in jobs["original_minute"]
    ]

    jobs["scheduled_carbon_intensity"] = (
        jobs["original_carbon_intensity"]
    )

    # Track execution work already assigned to each minute.
    minute_load = np.zeros(
        len(carbon_values),
        dtype=float
    )

    immediate_mask = (
        jobs["workload_type"] == "IMMEDIATE"
    )

    tolerant_mask = (
        jobs["workload_type"] == "DELAY_TOLERANT"
    )

    # Immediate jobs are never moved.
    immediate = jobs.loc[immediate_mask]

    for start, duration in zip(
        immediate["scheduled_start_timestamp"],
        immediate["duration"]
    ):
        add_execution_work(
            minute_load, start, duration
        )

    # Process delay-tolerant jobs in original arrival order.
    tolerant_indices = jobs.index[tolerant_mask]

    tolerant = jobs.loc[tolerant_indices].sort_values(
        "original_start_timestamp"
    )

    delayed_count = 0

    for idx, row in tolerant.iterrows():

        original_start = float(
            row["original_start_timestamp"]
        )

        duration = float(row["duration"])

        original_minute = int(
            original_start // 60
        )

        original_carbon = carbon_values[original_minute]

        # Candidate start times preserve the original
        # second offset within the minute.
        original_offset = original_start % 60

        first_candidate = original_minute
        last_candidate = min(
            original_minute + deadline,
            len(carbon_values) - 1
        )

        candidates = list(
            range(first_candidate, last_candidate + 1)
        )

        # Prefer lower-carbon candidates, but only accept
        # a candidate if its execution-work budget permits it.
        candidates.sort(
            key=lambda m: (carbon_values[m], m)
        )

        selected_minute = original_minute

        for candidate_minute in candidates:

            candidate_start = (
                candidate_minute * 60 + original_offset
            )

            waiting = candidate_start - original_start

            if waiting < 0 or waiting > deadline * 60:
                continue

            # Do not move to a higher-carbon minute.
            if (
                carbon_values[candidate_minute]
                > original_carbon
            ):
                continue

            if can_accept(
                minute_load,
                candidate_minute,
                duration
            ):
                selected_minute = candidate_minute
                break

        selected_start = (
            selected_minute * 60 + original_offset
        )

        add_execution_work(
            minute_load,
            selected_start,
            duration
        )

        jobs.at[
            idx, "scheduled_start_timestamp"
        ] = selected_start

        jobs.at[
            idx, "scheduled_minute"
        ] = selected_minute

        jobs.at[
            idx, "waiting_time_seconds"
        ] = selected_start - original_start

        jobs.at[
            idx, "scheduled_carbon_intensity"
        ] = carbon_values[selected_minute]

        if selected_start > original_start:
            delayed_count += 1

    jobs["scheduled_end_timestamp"] = (
        jobs["scheduled_start_timestamp"]
        + jobs["duration"]
    )

    jobs["scheduling_strategy"] = (
        f"CAPACITY_CONSTRAINED_{deadline}MIN"
    )

    # Per-minute execution-work equivalent concurrency.
    average_equivalent = minute_load / 60

    summary = {
        "deadline_minutes": deadline,
        "total_invocations": len(jobs),
        "immediate_invocations": int(immediate_mask.sum()),
        "delay_tolerant_invocations": int(tolerant_mask.sum()),
        "delayed_invocations": delayed_count,
        "delayed_percentage": (
            delayed_count / max(int(tolerant_mask.sum()), 1)
        ) * 100,
        "average_waiting_seconds": jobs.loc[
            tolerant_mask, "waiting_time_seconds"
        ].mean(),
        "maximum_waiting_seconds": jobs[
            "waiting_time_seconds"
        ].max(),
        "average_original_carbon_intensity": (
            jobs["original_carbon_intensity"].mean()
        ),
        "average_scheduled_carbon_intensity": (
            jobs["scheduled_carbon_intensity"].mean()
        ),
        "execution_work_peak_equivalent_concurrency": (
            average_equivalent.max()
        ),
        "execution_work_budget": MAX_CONCURRENCY
    }

    return jobs, summary


# ============================================================
# VALIDATION
# ============================================================

def validate_schedule(schedule, deadline):

    immediate = schedule[
        schedule["workload_type"] == "IMMEDIATE"
    ]

    tolerant = schedule[
        schedule["workload_type"] == "DELAY_TOLERANT"
    ]

    assert (
        schedule["scheduled_start_timestamp"]
        >= schedule["original_start_timestamp"]
    ).all()

    assert (
        schedule["waiting_time_seconds"] <= deadline * 60 + 0.001
    ).all()

    assert np.allclose(
        schedule["scheduled_end_timestamp"],
        schedule["scheduled_start_timestamp"]
        + schedule["duration"]
    )

    assert np.allclose(
        immediate["scheduled_start_timestamp"],
        immediate["original_start_timestamp"]
    )

    assert (
        tolerant["scheduled_carbon_intensity"]
        <= tolerant["original_carbon_intensity"] + 1e-9
    ).all()

    print("PASS: Schedule validation completed.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nCAPACITY-CONSTRAINED CARBON-AWARE SCHEDULER")
    print("=" * 55)

    jobs, carbon_values = load_inputs()

    print(f"Total jobs: {len(jobs):,}")
    print(f"Carbon timeline: {len(carbon_values):,} minutes")
    print(f"Experimental concurrency ceiling: {MAX_CONCURRENCY:,}")
    print(f"Per-minute execution-work budget: {CAPACITY_BUDGET:,.0f} seconds")

    summaries = []

    for deadline in DEADLINES:

        print(f"\nRUNNING {deadline}-MINUTE SCENARIO")
        print("-" * 45)

        schedule, summary = schedule_jobs(
            jobs,
            carbon_values,
            deadline
        )

        validate_schedule(schedule, deadline)

        output_file = (
            OUTPUT_DIR
            / f"capacity_constrained_schedule_{deadline}min.csv"
        )

        schedule.to_csv(output_file, index=False)

        summaries.append(summary)

        print(
            f"Delayed jobs: {summary['delayed_invocations']:,}"
        )

        print(
            "Average waiting: "
            f"{summary['average_waiting_seconds']:.2f} seconds"
        )

        print(
            "Average carbon intensity: "
            f"{summary['average_original_carbon_intensity']:.2f}"
            " -> "
            f"{summary['average_scheduled_carbon_intensity']:.2f}"
        )

        print(
            "Peak execution-work equivalent concurrency: "
            f"{summary['execution_work_peak_equivalent_concurrency']:.2f}"
        )

        print(f"Saved: {output_file}")

    summary_df = pd.DataFrame(summaries)

    summary_file = (
        OUTPUT_DIR / "capacity_constrained_summary.csv"
    )

    summary_df.to_csv(summary_file, index=False)

    print("\nALL SCENARIOS COMPLETED")
    print("=" * 55)

    print(summary_df.to_string(index=False))
    print(f"\nSaved: {summary_file}")


if __name__ == "__main__":
    main()
