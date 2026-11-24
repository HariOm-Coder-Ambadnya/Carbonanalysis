
import math
import pandas as pd

from pathlib import Path

from azure_cost_config import (
    INPUT_FILE,
    OUTPUT_DIR,
    TRACE_START_UTC,
    COST_CURRENCY,
    EXECUTION_PRICE_PER_MILLION,
    PRICE_PER_GB_SECOND,
    FREE_EXECUTIONS_PER_MONTH,
    FREE_GB_SECONDS_PER_MONTH,
    MEMORY_SCENARIOS_MB,
    MEMORY_ROUNDING_MB,
    MAX_BILLABLE_MEMORY_MB,
    MIN_EXECUTION_DURATION_SECONDS,
    DURATION_ROUNDING_SECONDS,
    CALCULATE_FREE_GRANT_SCENARIO,
)


def round_up(value, unit):
    """Round a positive value up to the next billing unit."""
    return math.ceil(value / unit) * unit


def calculate_billable_duration(duration):
    """
    Apply the configured billing assumptions:
    - Minimum duration: 100 ms
    - Duration rounding: 1 ms
    """
    duration = max(
        float(duration),
        MIN_EXECUTION_DURATION_SECONDS
    )

    return round_up(
        duration,
        DURATION_ROUNDING_SECONDS
    )


def calculate_billable_memory(memory_mb):
    """Round memory up to the next 128 MB increment."""
    if memory_mb <= 0:
        raise ValueError("Memory allocation must be positive.")

    rounded_memory = round_up(
        memory_mb,
        MEMORY_ROUNDING_MB
    )

    return min(
        rounded_memory,
        MAX_BILLABLE_MEMORY_MB
    )


def load_invocation_trace():
    print("Loading original Azure Functions trace...")

    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(
            f"Input trace not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "app",
        "func",
        "end_timestamp",
        "duration",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print(f"Original records: {len(df):,}")

    # Preserve original records, including duplicates and zero durations.
    df = df.copy()

    df["end_timestamp"] = pd.to_numeric(
        df["end_timestamp"],
        errors="raise"
    )

    df["duration"] = pd.to_numeric(
        df["duration"],
        errors="raise"
    )

    if df[["end_timestamp", "duration"]].isna().any().any():
        raise ValueError("Missing timestamp or duration values found.")

    if (df["duration"] < 0).any():
        raise ValueError("Negative execution durations found.")

    if (df["end_timestamp"] < 0).any():
        raise ValueError("Negative completion timestamps found.")

    trace_start = pd.Timestamp(
        TRACE_START_UTC,
        tz="UTC"
    )

    df["end_datetime_utc"] = (
        trace_start
        + pd.to_timedelta(
            df["end_timestamp"],
            unit="s"
        )
    )

    # Group by completion month for illustrative monthly grants.
    df["billing_month"] = (
        df["end_datetime_utc"]
        .dt.strftime("%Y-%m")
    )

    # Minute-level aggregation key.
    df["minute"] = (
        df["end_timestamp"] // 60
    ).astype(int)

    return df


def calculate_scenario(df, memory_mb):
    print(f"Processing memory scenario: {memory_mb} MB")

    result = df[
        [
            "minute",
            "billing_month",
            "end_datetime_utc",
            "duration",
        ]
    ].copy()

    result["assumed_memory_mb"] = memory_mb

    result["billable_memory_mb"] = (
        result["assumed_memory_mb"]
        .apply(calculate_billable_memory)
    )

    result["billable_duration_seconds"] = (
        result["duration"]
        .apply(calculate_billable_duration)
    )

    # GB-seconds = memory in GB * billable duration.
    result["gb_seconds"] = (
        result["billable_memory_mb"] / 1024
    ) * result["billable_duration_seconds"]

    # Per-invocation execution charge.
    result["execution_charge_usd"] = (
        EXECUTION_PRICE_PER_MILLION / 1_000_000
    )

    # Per-invocation resource charge.
    result["resource_charge_usd"] = (
        result["gb_seconds"] * PRICE_PER_GB_SECOND
    )

    result["gross_cost_usd"] = (
        result["execution_charge_usd"]
        + result["resource_charge_usd"]
    )

    return result


def create_minute_summary(result, memory_mb):
    """Aggregate gross estimated costs by minute."""

    summary = (
        result.groupby("minute")
        .agg(
            invocation_count=("minute", "size"),
            total_gb_seconds=("gb_seconds", "sum"),
            total_execution_charge_usd=(
                "execution_charge_usd", "sum"
            ),
            total_resource_charge_usd=(
                "resource_charge_usd", "sum"
            ),
            total_gross_cost_usd=(
                "gross_cost_usd", "sum"
            ),
        )
        .reset_index()
    )

    summary["memory_mb"] = memory_mb

    return summary


def create_monthly_summary(result, memory_mb):
    """
    Apply monthly free grants separately to each calendar month.

    This is illustrative and assumes this trace is the only
    workload consuming the subscription's free grants.
    """

    monthly = (
        result.groupby("billing_month")
        .agg(
            invocation_count=("billing_month", "size"),
            total_gb_seconds=("gb_seconds", "sum"),
            gross_execution_charge_usd=(
                "execution_charge_usd", "sum"
            ),
            gross_resource_charge_usd=(
                "resource_charge_usd", "sum"
            ),
            gross_total_cost_usd=(
                "gross_cost_usd", "sum"
            ),
        )
        .reset_index()
    )

    # Apply grants only when enabled in configuration.
    if CALCULATE_FREE_GRANT_SCENARIO:
        free_executions = FREE_EXECUTIONS_PER_MONTH
        free_gb_seconds = FREE_GB_SECONDS_PER_MONTH
    else:
        free_executions = 0
        free_gb_seconds = 0

    monthly["free_execution_grant"] = free_executions
    monthly["free_gb_second_grant"] = free_gb_seconds

    monthly["billable_executions_after_grant"] = (
        monthly["invocation_count"] - free_executions
    ).clip(lower=0)

    monthly["billable_gb_seconds_after_grant"] = (
        monthly["total_gb_seconds"] - free_gb_seconds
    ).clip(lower=0)

    monthly["net_execution_charge_usd"] = (
        monthly["billable_executions_after_grant"]
        * EXECUTION_PRICE_PER_MILLION
        / 1_000_000
    )

    monthly["net_resource_charge_usd"] = (
        monthly["billable_gb_seconds_after_grant"]
        * PRICE_PER_GB_SECOND
    )

    monthly["net_total_cost_usd"] = (
        monthly["net_execution_charge_usd"]
        + monthly["net_resource_charge_usd"]
    )

    monthly["memory_mb"] = memory_mb

    return monthly


def create_scenario_summary(
    result,
    monthly_summary,
    memory_mb
):
    gross_execution_cost = (
        result["execution_charge_usd"].sum()
    )

    gross_resource_cost = (
        result["resource_charge_usd"].sum()
    )

    gross_total_cost = (
        result["gross_cost_usd"].sum()
    )

    net_total_cost = (
        monthly_summary["net_total_cost_usd"].sum()
    )

    return {
        "memory_mb": memory_mb,
        "total_invocations": len(result),
        "total_gb_seconds": result["gb_seconds"].sum(),
        "gross_execution_cost_usd": gross_execution_cost,
        "gross_resource_cost_usd": gross_resource_cost,
        "gross_total_cost_usd": gross_total_cost,
        "net_cost_after_illustrative_grants_usd": net_total_cost,
        "currency": COST_CURRENCY,
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_invocation_trace()

    scenario_summaries = []
    all_minute_summaries = []
    all_monthly_summaries = []

    for memory_mb in MEMORY_SCENARIOS_MB:

        result = calculate_scenario(
            df,
            memory_mb
        )

        minute_summary = create_minute_summary(
            result,
            memory_mb
        )

        monthly_summary = create_monthly_summary(
            result,
            memory_mb
        )

        scenario_summary = create_scenario_summary(
            result,
            monthly_summary,
            memory_mb
        )

        scenario_summaries.append(scenario_summary)
        all_minute_summaries.append(minute_summary)
        all_monthly_summaries.append(monthly_summary)

        # Save invocation-level details.
        result.to_csv(
            OUTPUT_DIR / f"cost_details_{memory_mb}MB.csv",
            index=False
        )

    final_summary = pd.DataFrame(
        scenario_summaries
    )

    minute_output = pd.concat(
        all_minute_summaries,
        ignore_index=True
    )

    monthly_output = pd.concat(
        all_monthly_summaries,
        ignore_index=True
    )

    # Save all reports.
    final_summary.to_csv(
        OUTPUT_DIR / "azure_cost_summary.csv",
        index=False
    )

    minute_output.to_csv(
        OUTPUT_DIR / "azure_cost_minute_summary.csv",
        index=False
    )

    monthly_output.to_csv(
        OUTPUT_DIR / "azure_cost_monthly_summary.csv",
        index=False
    )

    print("\nAZURE FUNCTIONS COST ESTIMATION COMPLETED")
    print("=" * 65)

    print(f"Total invocations: {len(df):,}")
    print(f"Memory scenarios: {MEMORY_SCENARIOS_MB}")
    print(f"Currency: {COST_CURRENCY}")
    print(
        "Free grant scenario:",
        CALCULATE_FREE_GRANT_SCENARIO
    )

    print("\nSCENARIO SUMMARY")
    print("=" * 65)

    print(
        final_summary.to_string(index=False)
    )

    print("\nMONTHLY BREAKDOWN")
    print("=" * 65)

    print(
        monthly_output[
            [
                "billing_month",
                "memory_mb",
                "invocation_count",
                "total_gb_seconds",
                "gross_total_cost_usd",
                "net_total_cost_usd",
            ]
        ].to_string(index=False)
    )

    print("\nSaved files:")
    print(OUTPUT_DIR / "azure_cost_summary.csv")
    print(OUTPUT_DIR / "azure_cost_minute_summary.csv")
    print(OUTPUT_DIR / "azure_cost_monthly_summary.csv")

    print("\nCost estimation finished successfully.")


if __name__ == "__main__":
    main()
