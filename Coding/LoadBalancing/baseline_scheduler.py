
import pandas as pd
import numpy as np

from config import OUTPUT_DIR


# --------------------------------------------------
# 1. LOAD CLASSIFIED WORKLOAD
# --------------------------------------------------

def load_classified_workload():

    input_file = OUTPUT_DIR / "classified_workload.csv"

    print("Loading classified workload...")

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
                f"Missing required column: {column}"
            )

    print(f"Loaded invocations: {len(df):,}")

    return df


# --------------------------------------------------
# 2. EXECUTE BASELINE SCHEDULING
# --------------------------------------------------

def schedule_baseline(df):

    print("\nExecuting baseline scheduling...")

    scheduled = df.copy()

    # Baseline executes every invocation at its original
    # calculated start time. No intentional postponement.
    scheduled["scheduled_start_timestamp"] = (
        scheduled["start_timestamp"]
    )

    scheduled["scheduled_end_timestamp"] = (
        scheduled["scheduled_start_timestamp"]
        + scheduled["duration"]
    )

    # No intentional waiting
    scheduled["waiting_time_seconds"] = 0.0

    # Record the scheduling strategy
    scheduled["scheduling_strategy"] = "BASELINE"

    return scheduled


# --------------------------------------------------
# 3. VALIDATE BASELINE SCHEDULE
# --------------------------------------------------

def validate_baseline(scheduled):

    print("\nVALIDATING BASELINE SCHEDULE")
    print("============================")

    # Every invocation must be preserved
    assert len(scheduled) == 1_980_951, (
        "Invocation count has changed."
    )

    # No missing scheduled timestamps
    assert scheduled[
        "scheduled_start_timestamp"
    ].notna().all(), (
        "Missing scheduled start timestamps."
    )

    # Original start times must be preserved
    assert np.allclose(
        scheduled["scheduled_start_timestamp"],
        scheduled["start_timestamp"]
    ), "Baseline changed original start times."

    # Waiting time must be zero
    assert (
        scheduled["waiting_time_seconds"] == 0
    ).all(), "Baseline contains unexpected waiting time."

    # Execution duration must be preserved
    assert np.allclose(
        scheduled["scheduled_end_timestamp"]
        - scheduled["scheduled_start_timestamp"],
        scheduled["duration"]
    ), "Execution duration has changed."

    # Workload categories must be preserved
    assert (
        scheduled["workload_type"].notna().all()
    ), "Workload categories are missing."

    print("PASS: All original invocations preserved.")
    print("PASS: Original start times preserved.")
    print("PASS: No intentional scheduling delay.")
    print("PASS: Execution durations preserved.")
    print("PASS: Workload categories preserved.")

    print("\nAll baseline validation checks passed.")


# --------------------------------------------------
# 4. GENERATE BASELINE SUMMARY
# --------------------------------------------------

def generate_baseline_summary(scheduled):

    summary = (
        scheduled.groupby("workload_type")
        .agg(
            invocation_count=("duration", "count"),
            average_duration=("duration", "mean"),
            total_execution_seconds=("duration", "sum"),
            average_waiting_seconds=(
                "waiting_time_seconds", "mean"
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
# 5. SAVE BASELINE RESULTS
# --------------------------------------------------

def save_results(scheduled, summary):

    output_file = (
        OUTPUT_DIR / "baseline_schedule.csv"
    )

    summary_file = (
        OUTPUT_DIR / "baseline_schedule_summary.csv"
    )

    scheduled.to_csv(
        output_file,
        index=False
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print("\nBASELINE RESULTS SAVED")

    print(f"Schedule: {output_file}")
    print(f"Summary: {summary_file}")


# --------------------------------------------------
# 6. MAIN EXECUTION
# --------------------------------------------------

def main():

    df = load_classified_workload()

    scheduled = schedule_baseline(df)

    validate_baseline(scheduled)

    summary = generate_baseline_summary(scheduled)

    print("\nBASELINE SCHEDULING SUMMARY")
    print("===========================")

    print(summary.to_string(index=False))

    save_results(scheduled, summary)

    print("\nBASELINE SCHEDULING COMPLETED!")


if __name__ == "__main__":
    main()
