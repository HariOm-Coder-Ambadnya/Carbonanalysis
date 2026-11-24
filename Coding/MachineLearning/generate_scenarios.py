
from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WORKLOAD_FILE = (
    PROJECT_ROOT
    / "Coding"
    / "LoadBalancing"
    / "results"
    / "classified_workload.csv"
)

CARBON_FILE = (
    PROJECT_ROOT
    / "Coding"
    / "CarbonAnalysis"
    / "workload_carbon_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "Coding"
    / "MachineLearning"
    / "scenarios"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SCENARIO CONFIGURATION
# ============================================================

SCENARIO_DURATION_MINUTES = 360  # Six hours

# Keep enough carbon timeline for the maximum scheduling deadline.
MAX_DEADLINE_MINUTES = 60

# A scenario is created only if it contains workload records.
MINIMUM_JOBS_PER_SCENARIO = 1


# ============================================================
# 3. LOAD ORIGINAL DATA
# ============================================================

def load_data():

    if not WORKLOAD_FILE.exists():
        raise FileNotFoundError(
            f"Workload file not found: {WORKLOAD_FILE}"
        )

    if not CARBON_FILE.exists():
        raise FileNotFoundError(
            f"Carbon file not found: {CARBON_FILE}"
        )

    print("Loading original workload dataset...")

    workload = pd.read_csv(WORKLOAD_FILE)

    print("Loading original carbon dataset...")

    carbon = pd.read_csv(CARBON_FILE)

    required_workload = [
        "app",
        "func",
        "start_timestamp",
        "end_timestamp",
        "duration",
        "workload_type",
    ]

    required_carbon = [
        "minute",
        "carbon_intensity_lifecycle",
    ]

    missing_workload = [
        col for col in required_workload
        if col not in workload.columns
    ]

    missing_carbon = [
        col for col in required_carbon
        if col not in carbon.columns
    ]

    if missing_workload:
        raise ValueError(
            f"Missing workload columns: {missing_workload}"
        )

    if missing_carbon:
        raise ValueError(
            f"Missing carbon columns: {missing_carbon}"
        )

    workload["start_timestamp"] = pd.to_numeric(
        workload["start_timestamp"],
        errors="raise",
    )

    workload["duration"] = pd.to_numeric(
        workload["duration"],
        errors="raise",
    )

    carbon["minute"] = pd.to_numeric(
        carbon["minute"],
        errors="raise",
    )

    carbon["carbon_intensity_lifecycle"] = pd.to_numeric(
        carbon["carbon_intensity_lifecycle"],
        errors="raise",
    )

    workload = workload.sort_values(
        "start_timestamp"
    ).reset_index(drop=True)

    carbon = carbon.sort_values(
        "minute"
    ).reset_index(drop=True)

    if workload[
        ["start_timestamp", "duration"]
    ].isna().any().any():
        raise ValueError("Invalid workload timestamps or durations.")

    if (workload["start_timestamp"] < 0).any():
        raise ValueError("Negative workload timestamps found.")

    if (workload["duration"] < 0).any():
        raise ValueError("Negative workload durations found.")

    expected_minutes = np.arange(len(carbon))

    if not np.array_equal(
        carbon["minute"].to_numpy(dtype=int),
        expected_minutes,
    ):
        raise ValueError(
            "Carbon timeline must be continuous from minute zero."
        )

    if not np.isfinite(
        carbon["carbon_intensity_lifecycle"].to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid carbon intensity values.")

    print("\nORIGINAL DATA LOADED")
    print("----------------------------")
    print(f"Workload records: {len(workload):,}")
    print(f"Carbon records:   {len(carbon):,}")
    print(
        f"Workload duration: "
        f"{workload['start_timestamp'].min():,.0f} to "
        f"{workload['start_timestamp'].max():,.0f} seconds"
    )

    return workload, carbon


# ============================================================
# 4. GENERATE SCENARIOS
# ============================================================

def generate_scenarios(workload, carbon):

    carbon_values = carbon[
        "carbon_intensity_lifecycle"
    ].to_numpy(dtype=float)

    total_carbon_minutes = len(carbon_values)

    scenario_duration_seconds = (
        SCENARIO_DURATION_MINUTES * 60
    )

    lookahead_seconds = MAX_DEADLINE_MINUTES * 60

    scenario_count = 0
    total_scenario_jobs = 0

    summaries = []

    # Each window begins at a six-hour boundary.
    for start_minute in range(
        0,
        total_carbon_minutes,
        SCENARIO_DURATION_MINUTES,
    ):

        start_seconds = start_minute * 60

        end_seconds = (
            start_seconds + scenario_duration_seconds
        )

        # Select jobs arriving inside this scenario window.
        mask = (
            (workload["start_timestamp"] >= start_seconds)
            & (workload["start_timestamp"] < end_seconds)
        )

        scenario_jobs = workload.loc[mask].copy()

        if len(scenario_jobs) < MINIMUM_JOBS_PER_SCENARIO:
            continue

        # Carbon data includes the scheduling lookahead.
        carbon_end_minute = min(
            start_minute
            + SCENARIO_DURATION_MINUTES
            + MAX_DEADLINE_MINUTES,
            total_carbon_minutes,
        )

        scenario_carbon = carbon.iloc[
            start_minute:carbon_end_minute
        ].copy()

        # Rebase workload timestamps to scenario time zero.
        scenario_jobs["original_start_timestamp"] = (
            scenario_jobs["start_timestamp"]
        )

        scenario_jobs["start_timestamp"] = (
            scenario_jobs["start_timestamp"]
            - start_seconds
        )

        scenario_jobs["end_timestamp"] = (
            scenario_jobs["end_timestamp"]
            - start_seconds
        )

        # Rebase carbon minute indices to zero.
        scenario_carbon["original_minute"] = (
            scenario_carbon["minute"]
        )

        scenario_carbon["minute"] = (
            scenario_carbon["minute"] - start_minute
        )

        scenario_id = f"scenario_{scenario_count + 1:03d}"

        scenario_dir = OUTPUT_DIR / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)

        workload_output = scenario_dir / "classified_workload.csv"
        carbon_output = scenario_dir / "carbon_data.csv"
        metadata_output = scenario_dir / "metadata.json"

        scenario_jobs.to_csv(
            workload_output,
            index=False,
        )

        scenario_carbon.to_csv(
            carbon_output,
            index=False,
        )

        metadata = {
            "scenario_id": scenario_id,
            "scenario_type": "original_trace_window",
            "window_start_minute": int(start_minute),
            "window_end_minute": int(
                start_minute + SCENARIO_DURATION_MINUTES
            ),
            "window_duration_minutes": SCENARIO_DURATION_MINUTES,
            "carbon_lookahead_minutes": MAX_DEADLINE_MINUTES,
            "job_count": int(len(scenario_jobs)),
            "immediate_jobs": int(
                (
                    scenario_jobs["workload_type"].str.upper()
                    == "IMMEDIATE"
                ).sum()
            ),
            "delay_tolerant_jobs": int(
                (
                    scenario_jobs["workload_type"].str.upper()
                    == "DELAY_TOLERANT"
                ).sum()
            ),
            "mean_duration_seconds": float(
                scenario_jobs["duration"].mean()
            ),
            "mean_carbon_intensity": float(
                scenario_carbon[
                    "carbon_intensity_lifecycle"
                ].mean()
            ),
            "deadlines_minutes": [15, 30, 60],
        }

        with metadata_output.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metadata, file, indent=4)

        scenario_count += 1
        total_scenario_jobs += len(scenario_jobs)

        summaries.append(metadata)

        print(
            f"{scenario_id}: "
            f"{len(scenario_jobs):,} jobs | "
            f"{metadata['immediate_jobs']:,} immediate | "
            f"{metadata['delay_tolerant_jobs']:,} delay-tolerant"
        )

    # Save scenario-level summary.
    summary_file = OUTPUT_DIR / "scenario_summary.csv"

    pd.DataFrame(summaries).to_csv(
        summary_file,
        index=False,
    )

    print("\nSCENARIO GENERATION COMPLETED")
    print("--------------------------------")
    print(f"Scenarios generated: {scenario_count}")
    print(f"Total scenario job records: {total_scenario_jobs:,}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Summary: {summary_file}")


# ============================================================
# 5. MAIN
# ============================================================

def main():

    workload, carbon = load_data()

    generate_scenarios(
        workload,
        carbon,
    )


if __name__ == "__main__":
    main()
