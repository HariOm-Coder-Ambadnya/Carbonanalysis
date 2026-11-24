
from pathlib import Path
from collections import defaultdict
import math
import sqlite3

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

CLASSIFIED_FILE = (
    ROOT / "Coding/LoadBalancing/results/classified_workload.csv"
)

SCHEDULE_DIR = (
    ROOT / "Coding/LoadBalancing/results/exact_capacity"
)

OUTPUT_DIR = SCHEDULE_DIR / "final_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_ROWS = 1_980_951
CAPACITY_LIMIT = 3027
CHUNK_SIZE = 100_000

SCENARIOS = {
    "15min": 15 * 60,
    "30min": 30 * 60,
    "60min": 60 * 60,
}

IDENTITY_COLUMNS = [
    "app",
    "func",
    "end_timestamp",
    "duration",
    "start_timestamp",
    "workload_type",
]


def load_csv(path, columns=None):
    return pd.read_csv(path, usecols=columns)


def add_to_database(connection, dataframe, table):
    columns = IDENTITY_COLUMNS

    records = dataframe[columns].itertuples(
        index=False, name=None
    )

    placeholders = ",".join(["?"] * len(columns))

    connection.executemany(
        f"INSERT INTO {table} VALUES ({placeholders})",
        records,
    )


def create_reference_database():
    db_path = OUTPUT_DIR / "reference_validation.sqlite"

    if db_path.exists():
        db_path.unlink()

    connection = sqlite3.connect(db_path)

    column_sql = """
        app TEXT,
        func TEXT,
        end_timestamp REAL,
        duration REAL,
        start_timestamp REAL,
        workload_type TEXT
    """

    connection.execute(
        f"CREATE TABLE reference_data ({column_sql})"
    )

    connection.execute(
        f"CREATE TABLE schedule_data ({column_sql})"
    )

    print("\nBuilding reference record counts...")

    for chunk in pd.read_csv(
        CLASSIFIED_FILE,
        chunksize=CHUNK_SIZE
    ):
        add_to_database(
            connection, chunk, "reference_data"
        )

    connection.execute("""
        CREATE INDEX reference_identity
        ON reference_data (
            app, func, end_timestamp, duration,
            start_timestamp, workload_type
        )
    """)

    connection.commit()

    return connection


def compare_record_multiplicity(connection, schedule_file):
    connection.execute("DELETE FROM schedule_data")

    for chunk in pd.read_csv(
        schedule_file,
        chunksize=CHUNK_SIZE
    ):
        add_to_database(
            connection, chunk, "schedule_data"
        )

    connection.commit()

    missing = connection.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT
                app, func, end_timestamp, duration,
                start_timestamp, workload_type,
                COUNT(*) AS total
            FROM reference_data
            GROUP BY
                app, func, end_timestamp, duration,
                start_timestamp, workload_type

            EXCEPT

            SELECT
                app, func, end_timestamp, duration,
                start_timestamp, workload_type,
                COUNT(*) AS total
            FROM schedule_data
            GROUP BY
                app, func, end_timestamp, duration,
                start_timestamp, workload_type
        )
    """).fetchone()[0]

    extra = connection.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT
                app, func, end_timestamp, duration,
                start_timestamp, workload_type,
                COUNT(*) AS total
            FROM schedule_data
            GROUP BY
                app, func, end_timestamp, duration,
                start_timestamp, workload_type

            EXCEPT

            SELECT
                app, func, end_timestamp, duration,
                start_timestamp, workload_type,
                COUNT(*) AS total
            FROM reference_data
            GROUP BY
                app, func, end_timestamp, duration,
                start_timestamp, workload_type
        )
    """).fetchone()[0]

    return missing, extra


def validate_schedule(label, deadline_seconds, connection):
    schedule_file = (
        SCHEDULE_DIR /
        f"exact_capacity_schedule_{label}.csv"
    )

    print("\n" + "=" * 65)
    print(f"VALIDATING {label.upper()} SCHEDULE")
    print("=" * 65)

    if not schedule_file.exists():
        print(f"FAIL: File not found: {schedule_file}")
        return None

    row_count = 0
    immediate_changed = 0
    deadline_violations = 0
    duration_changed = 0
    negative_wait = 0
    invalid_intervals = 0

    slot_events = defaultdict(int)

    for chunk in pd.read_csv(
        schedule_file,
        chunksize=CHUNK_SIZE
    ):
        row_count += len(chunk)

        original = chunk["original_start_timestamp"].to_numpy(
            dtype=np.float64
        )

        scheduled = chunk["scheduled_start_timestamp"].to_numpy(
            dtype=np.float64
        )

        end = chunk["scheduled_end_timestamp"].to_numpy(
            dtype=np.float64
        )

        duration = chunk["duration"].to_numpy(
            dtype=np.float64
        )

        waiting = chunk["waiting_time_seconds"].to_numpy(
            dtype=np.float64
        )

        types = chunk["workload_type"].astype(str).to_numpy()

        immediate_mask = types == "immediate"
        tolerant_mask = types == "delay_tolerant"

        immediate_changed += int(
            np.count_nonzero(
                ~np.isclose(
                    original[immediate_mask],
                    scheduled[immediate_mask],
                    atol=1e-6
                )
            )
        )

        deadline_violations += int(
            np.count_nonzero(
                waiting[tolerant_mask] >
                deadline_seconds + 1e-6
            )
        )

        duration_changed += int(
            np.count_nonzero(
                ~np.isclose(
                    end - scheduled,
                    duration,
                    atol=1e-5
                )
            )
        )

        negative_wait += int(
            np.count_nonzero(waiting < -1e-6)
        )

        invalid_intervals += int(
            np.count_nonzero(
                (duration < 0) |
                (end < scheduled - 1e-6)
            )
        )

        # Conservative one-second slot reservation:
        # [floor(start), ceil(end))
        positive = duration > 0

        starts = np.floor(scheduled[positive]).astype(np.int64)
        ends = np.ceil(end[positive] - 1e-10).astype(np.int64)

        for start_slot, end_slot in zip(starts, ends):
            if end_slot <= start_slot:
                end_slot = start_slot + 1

            slot_events[int(start_slot)] += 1
            slot_events[int(end_slot)] -= 1

    missing, extra = compare_record_multiplicity(
        connection, schedule_file
    )

    # Compute maximum one-second slot occupancy.
    current = 0
    maximum = 0

    for slot in sorted(slot_events):
        current += slot_events[slot]
        maximum = max(maximum, current)

    checks = {
        "Expected row count": row_count == EXPECTED_ROWS,
        "Original records preserved": missing == 0 and extra == 0,
        "Immediate start times unchanged": immediate_changed == 0,
        "Delay-tolerant deadlines respected": deadline_violations == 0,
        "Execution durations preserved": duration_changed == 0,
        "No negative waiting times": negative_wait == 0,
        "Valid execution intervals": invalid_intervals == 0,
        "One-second-slot capacity": maximum <= CAPACITY_LIMIT,
    }

    print(f"Rows: {row_count:,}")
    print(f"Missing record groups: {missing}")
    print(f"Extra record groups: {extra}")
    print(f"Changed immediate starts: {immediate_changed}")
    print(f"Deadline violations: {deadline_violations}")
    print(f"Changed durations: {duration_changed}")
    print(f"Negative waiting times: {negative_wait}")
    print(f"Invalid intervals: {invalid_intervals}")
    print(f"Maximum one-second slot occupancy: {maximum:,}")
    print(f"Capacity ceiling: {CAPACITY_LIMIT:,}")

    print("\nCHECK RESULTS")

    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")

    passed_count = sum(checks.values())

    print(f"\nResult: {passed_count}/{len(checks)} checks passed")

    return {
        "scenario": label,
        "rows": row_count,
        "missing_record_groups": missing,
        "extra_record_groups": extra,
        "immediate_start_changes": immediate_changed,
        "deadline_violations": deadline_violations,
        "duration_changes": duration_changed,
        "negative_waiting_times": negative_wait,
        "invalid_intervals": invalid_intervals,
        "max_one_second_slot_occupancy": maximum,
        "capacity_limit": CAPACITY_LIMIT,
        "checks_passed": passed_count,
        "total_checks": len(checks),
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def main():
    print("FINAL SERVERLESS SCHEDULER VALIDATION")
    print("=" * 65)

    connection = create_reference_database()

    results = []

    try:
        for label, deadline in SCENARIOS.items():
            result = validate_schedule(
                label, deadline, connection
            )

            if result:
                results.append(result)

    finally:
        connection.close()

    if results:
        report = pd.DataFrame(results)

        report_path = (
            OUTPUT_DIR / "final_validation_report.csv"
        )

        report.to_csv(report_path, index=False)

        print("\n" + "=" * 65)
        print("FINAL VALIDATION SUMMARY")
        print("=" * 65)

        print(
            report[
                [
                    "scenario",
                    "rows",
                    "max_one_second_slot_occupancy",
                    "checks_passed",
                    "total_checks",
                    "status",
                ]
            ].to_string(index=False)
        )

        print(f"\nReport saved to: {report_path}")

        if all(result["status"] == "PASS" for result in results):
            print("\nALL SCHEDULES PASSED VALIDATION.")
        else:
            print("\nSOME VALIDATION CHECKS FAILED.")


if __name__ == "__main__":
    main()
