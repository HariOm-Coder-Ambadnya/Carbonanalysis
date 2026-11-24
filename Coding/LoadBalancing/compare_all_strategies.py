
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RESULTS = ROOT / "Coding" / "LoadBalancing" / "results"
CARBON_FILE = (
    ROOT / "Coding" / "CarbonAnalysis"
    / "workload_carbon_dataset.csv"
)

OUTPUT_DIR = RESULTS / "consolidated_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEADLINES = [15, 30, 60]

CAPACITY_PER_INSTANCE = 20
IDLE_POWER_WATTS = 44
PEAK_POWER_WATTS = 125

CAPACITY_CEILING = 3027

MEMORY_GB = 1.0
EXECUTION_PRICE_PER_MILLION = 0.20
GB_SECOND_PRICE = 0.000016
MIN_BILLABLE_SECONDS = 0.100
BILLING_ROUNDING_SECONDS = 0.001


# ============================================================
# STRATEGY FILES
# ============================================================

STRATEGIES = {
    "Baseline": (
        RESULTS,
        "baseline_schedule.csv",
        "Not applicable"
    ),
    "Carbon_Aware": (
        RESULTS,
        "carbon_aware_schedule_{deadline}min.csv",
        "Not enforced"
    ),
    "Multi_Objective": (
        RESULTS,
        "multi_objective_schedule_{deadline}min.csv",
        "Not enforced"
    ),
    "Forecast_Assisted": (
        RESULTS / "forecast_assisted",
        "forecast_assisted_schedule_{deadline}min.csv",
        "Advisory only"
    ),
    "Approx_Capacity": (
        RESULTS / "capacity_constrained",
        "capacity_constrained_schedule_{deadline}min.csv",
        "Approximate"
    ),
    "One_Second_Slot_Capacity": (
        RESULTS / "exact_capacity",
        "exact_capacity_schedule_{deadline}min.csv",
        "One-second-slot model"
    )
}


# ============================================================
# LOAD CARBON DATA
# ============================================================

def load_carbon_data():
    carbon = pd.read_csv(
        CARBON_FILE,
        usecols=["minute", "carbon_intensity_lifecycle"]
    ).sort_values("minute")

    minutes = carbon["minute"].to_numpy(dtype=int)
    values = carbon[
        "carbon_intensity_lifecycle"
    ].to_numpy(dtype=float)

    if not np.array_equal(minutes, np.arange(len(minutes))):
        raise ValueError("Carbon minute timeline is not continuous.")

    if not np.isfinite(values).all():
        raise ValueError("Invalid carbon intensity values.")

    return values


# ============================================================
# READ ONLY REQUIRED SCHEDULE COLUMNS
# ============================================================

def load_schedule(path):
    header = pd.read_csv(path, nrows=0).columns.tolist()

    required = {
        "duration",
        "scheduled_start_timestamp"
    }

    original_column = (
        "original_start_timestamp"
        if "original_start_timestamp" in header
        else "start_timestamp"
    )

    required.add(original_column)

    optional = {
        "workload_type",
        "waiting_time_seconds"
    }

    selected = [
        column for column in header
        if column in required or column in optional
    ]

    missing = required.difference(selected)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}"
        )

    return pd.read_csv(path, usecols=selected), original_column


# ============================================================
# FAST EVENT-BASED ENERGY AND CARBON MODEL
# ============================================================

def calculate_energy_and_carbon(schedule, carbon_values):

    starts = schedule[
        "scheduled_start_timestamp"
    ].to_numpy(dtype=float)

    durations = schedule["duration"].to_numpy(dtype=float)

    valid = (
        np.isfinite(starts)
        & np.isfinite(durations)
        & (durations > 0)
    )

    starts = starts[valid]
    durations = durations[valid]

    if len(starts) == 0:
        return {
            "energy_kwh": 0.0,
            "carbon_emissions_kg": 0.0,
            "maximum_concurrency": 0,
            "maximum_instances": 0
        }

    ends = starts + durations

    event_times = np.concatenate([starts, ends])
    event_changes = np.concatenate([
        np.ones(len(starts), dtype=np.int32),
        -np.ones(len(ends), dtype=np.int32)
    ])

    order = np.argsort(event_times, kind="stable")
    event_times = event_times[order]
    event_changes = event_changes[order]

    unique_times, first_indices = np.unique(
        event_times,
        return_index=True
    )

    net_changes = np.add.reduceat(
        event_changes,
        first_indices
    )

    cumulative_active = np.concatenate([
        np.array([0], dtype=np.int64),
        np.cumsum(net_changes, dtype=np.int64)
    ])

    maximum_concurrency = int(cumulative_active.max())

    # Add minute boundaries so each segment uses one carbon value.
    timeline_end = len(carbon_values) * 60

    if starts.min() < 0 or ends.max() > timeline_end:
        raise ValueError(
            "Schedule extends outside the carbon timeline."
        )

    minute_boundaries = np.arange(
        0,
        timeline_end + 60,
        60,
        dtype=float
    )

    breakpoints = np.unique(np.concatenate([
        unique_times,
        minute_boundaries
    ]))

    event_positions = np.searchsorted(
        unique_times,
        breakpoints[:-1],
        side="right"
    )

    active = cumulative_active[event_positions]

    seconds = np.diff(breakpoints)

    minutes = (
        breakpoints[:-1] // 60
    ).astype(int)

    valid_segments = (
        (active > 0)
        & (seconds > 0)
        & (minutes >= 0)
        & (minutes < len(carbon_values))
    )

    active = active[valid_segments]
    seconds = seconds[valid_segments]
    minutes = minutes[valid_segments]

    instances = (
        active + CAPACITY_PER_INSTANCE - 1
    ) // CAPACITY_PER_INSTANCE

    utilisation = active / (
        instances * CAPACITY_PER_INSTANCE
    )

    power = instances * (
        IDLE_POWER_WATTS
        + utilisation * (
            PEAK_POWER_WATTS - IDLE_POWER_WATTS
        )
    )

    segment_energy = (
        power * seconds / 3600
    ) / 1000

    energy_kwh = float(segment_energy.sum())

    carbon_kg = float(
        np.sum(
            segment_energy
            * carbon_values[minutes] / 1000
        )
    )

    maximum_instances = math.ceil(
        maximum_concurrency / CAPACITY_PER_INSTANCE
    )

    return {
        "energy_kwh": energy_kwh,
        "carbon_emissions_kg": carbon_kg,
        "maximum_concurrency": maximum_concurrency,
        "maximum_instances": maximum_instances
    }


# ============================================================
# AZURE COST ESTIMATION
# ============================================================

def calculate_cost(schedule):

    durations = schedule["duration"].to_numpy(dtype=float)

    durations = durations[
        np.isfinite(durations) & (durations >= 0)
    ]

    rounded = (
        np.ceil(
            durations / BILLING_ROUNDING_SECONDS
        ) * BILLING_ROUNDING_SECONDS
    )

    billable = np.maximum(
        rounded,
        MIN_BILLABLE_SECONDS
    )

    count = len(billable)
    gb_seconds = float(billable.sum() * MEMORY_GB)

    execution_cost = (
        count / 1_000_000
    ) * EXECUTION_PRICE_PER_MILLION

    resource_cost = gb_seconds * GB_SECOND_PRICE

    return {
        "invocation_count": count,
        "total_gb_seconds": gb_seconds,
        "execution_cost_usd": execution_cost,
        "resource_cost_usd": resource_cost,
        "total_cost_usd": execution_cost + resource_cost
    }


# ============================================================
# WAITING TIME AND DEADLINE METRICS
# ============================================================

def calculate_scheduling_metrics(
    schedule,
    original_column,
    deadline
):
    original = schedule[
        original_column
    ].to_numpy(dtype=float)

    scheduled = schedule[
        "scheduled_start_timestamp"
    ].to_numpy(dtype=float)

    waiting = scheduled - original

    if "waiting_time_seconds" in schedule.columns:
        supplied_wait = schedule[
            "waiting_time_seconds"
        ].to_numpy(dtype=float)

        valid_supplied = np.isfinite(supplied_wait)

        waiting[valid_supplied] = supplied_wait[valid_supplied]

    waiting = np.maximum(waiting, 0)

    delayed = waiting > 1e-6

    if "workload_type" in schedule.columns:
        workload_types = (
            schedule["workload_type"]
            .fillna("")
            .astype(str)
            .str.upper()
            .to_numpy()
        )

        tolerant = workload_types == "DELAY_TOLERANT"
        immediate = workload_types == "IMMEDIATE"
    else:
        tolerant = np.ones(len(waiting), dtype=bool)
        immediate = np.zeros(len(waiting), dtype=bool)

    tolerant_count = int(tolerant.sum())
    delayed_count = int((delayed & tolerant).sum())

    compliant = (
        waiting[tolerant] <= deadline * 60 + 1e-6
    )

    deadline_compliance = (
        float(compliant.mean() * 100)
        if tolerant_count else 100.0
    )

    return {
        "delayed_invocations": int(delayed.sum()),
        "delayed_tolerant_invocations": delayed_count,
        "average_waiting_seconds": float(
            waiting[tolerant].mean()
        ) if tolerant_count else 0.0,
        "maximum_waiting_seconds": float(
            waiting[tolerant].max()
        ) if tolerant_count else 0.0,
        "deadline_compliance_percent": deadline_compliance,
        "immediate_invocations_delayed": int(
            (delayed & immediate).sum()
        )
    }


# ============================================================
# EVALUATE ONE SCHEDULE
# ============================================================

def evaluate_one(
    strategy,
    deadline,
    path,
    capacity_status,
    carbon_values
):

    print(f"  Reading {path.name} ...", flush=True)

    schedule, original_column = load_schedule(path)

    energy = calculate_energy_and_carbon(
        schedule,
        carbon_values
    )

    cost = calculate_cost(schedule)

    timing = calculate_scheduling_metrics(
        schedule,
        original_column,
        deadline
    )

    peak = energy["maximum_concurrency"]

    if strategy == "Baseline":
        capacity_compliance = "Baseline reference"
    elif capacity_status in (
        "Approximate",
        "One-second-slot model"
    ):
        capacity_compliance = (
            "PASS" if peak <= CAPACITY_CEILING else "FAIL"
        )
    else:
        capacity_compliance = "Not enforced"

    return {
        "strategy": strategy,
        "deadline_minutes": deadline,
        **energy,
        **cost,
        **timing,
        "capacity_ceiling": CAPACITY_CEILING,
        "capacity_model": capacity_status,
        "capacity_compliance": capacity_compliance
    }


# ============================================================
# MAIN COMPARISON
# ============================================================

def main():

    print("\nCONSOLIDATED LOAD-BALANCING EVALUATION")
    print("=" * 60)

    carbon_values = load_carbon_data()

    results = []

    # Baseline is common to all deadline scenarios.
    baseline_path = RESULTS / "baseline_schedule.csv"

    baseline_rows = {}

    print("\nEvaluating baseline...")

    for deadline in DEADLINES:
        baseline_rows[deadline] = evaluate_one(
            "Baseline",
            deadline,
            baseline_path,
            "Not applicable",
            carbon_values
        )

    results.extend(baseline_rows.values())

    # Evaluate the five scheduling approaches.
    for strategy, (
        directory,
        filename_pattern,
        capacity_status
    ) in STRATEGIES.items():

        if strategy == "Baseline":
            continue

        for deadline in DEADLINES:

            filename = filename_pattern.format(
                deadline=deadline
            )

            path = directory / filename

            if not path.exists():
                raise FileNotFoundError(path)

            print(
                f"\nEvaluating {strategy}, "
                f"{deadline}-minute window..."
            )

            row = evaluate_one(
                strategy,
                deadline,
                path,
                capacity_status,
                carbon_values
            )

            baseline = baseline_rows[deadline]

            for metric in [
                "energy_kwh",
                "carbon_emissions_kg",
                "total_cost_usd"
            ]:
                original = baseline[metric]
                current = row[metric]

                row[f"{metric}_reduction_percent"] = (
                    (original - current) / original * 100
                    if original else 0.0
                )

            results.append(row)

    df = pd.DataFrame(results)

    output_csv = (
        OUTPUT_DIR / "consolidated_strategy_comparison.csv"
    )

    df.to_csv(output_csv, index=False)

    # --------------------------------------------------------
    # BEST STRATEGY SUMMARY
    # --------------------------------------------------------

    proposed = df[df["strategy"] != "Baseline"].copy()

    best = (
        proposed.sort_values("carbon_emissions_kg")
        .groupby("deadline_minutes", as_index=False)
        .first()
    )

    best.to_csv(
        OUTPUT_DIR / "best_strategy_by_deadline.csv",
        index=False
    )

    # --------------------------------------------------------
    # GRAPH 1: ENERGY
    # --------------------------------------------------------

    pivot_energy = df.pivot(
        index="strategy",
        columns="deadline_minutes",
        values="energy_kwh"
    )

    pivot_energy.plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.ylabel("Energy (kWh)")
    plt.xlabel("Scheduling strategy")
    plt.title("Energy Consumption Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "energy_comparison.png",
        dpi=200
    )
    plt.close()

    # --------------------------------------------------------
    # GRAPH 2: CARBON
    # --------------------------------------------------------

    pivot_carbon = df.pivot(
        index="strategy",
        columns="deadline_minutes",
        values="carbon_emissions_kg"
    )

    pivot_carbon.plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.ylabel("Carbon emissions (kg CO2eq)")
    plt.xlabel("Scheduling strategy")
    plt.title("Carbon Emissions Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "carbon_comparison.png",
        dpi=200
    )
    plt.close()

    # --------------------------------------------------------
    # GRAPH 3: AVERAGE WAITING TIME
    # --------------------------------------------------------

    pivot_wait = df.pivot(
        index="strategy",
        columns="deadline_minutes",
        values="average_waiting_seconds"
    )

    pivot_wait.plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.ylabel("Average waiting time (seconds)")
    plt.xlabel("Scheduling strategy")
    plt.title("Average Waiting Time Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "waiting_time_comparison.png",
        dpi=200
    )
    plt.close()

    # --------------------------------------------------------
    # GRAPH 4: CARBON REDUCTION
    # --------------------------------------------------------

    reduction_column = "carbon_emissions_kg_reduction_percent"

    reduction = proposed.pivot(
        index="strategy",
        columns="deadline_minutes",
        values=reduction_column
    )

    reduction.plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.ylabel("Carbon reduction (%)")
    plt.xlabel("Scheduling strategy")
    plt.title("Carbon Reduction Compared With Baseline")
    plt.axhline(0, linewidth=0.8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "carbon_reduction_comparison.png",
        dpi=200
    )
    plt.close()

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    columns = [
        "strategy",
        "deadline_minutes",
        "energy_kwh",
        "carbon_emissions_kg",
        "total_cost_usd",
        "delayed_tolerant_invocations",
        "average_waiting_seconds",
        "deadline_compliance_percent",
        "maximum_concurrency",
        "capacity_compliance"
    ]

    print("\n\nFINAL COMPARISON")
    print("=" * 90)

    print(
        df[columns].round(4).to_string(index=False)
    )

    print("\nSaved results to:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")
    print("- consolidated_strategy_comparison.csv")
    print("- best_strategy_by_deadline.csv")
    print("- energy_comparison.png")
    print("- carbon_comparison.png")
    print("- waiting_time_comparison.png")
    print("- carbon_reduction_comparison.png")


if __name__ == "__main__":
    main()
