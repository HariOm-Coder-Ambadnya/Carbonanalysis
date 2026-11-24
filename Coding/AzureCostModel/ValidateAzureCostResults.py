
import pandas as pd

from azure_cost_config import (
    OUTPUT_DIR,
    MEMORY_SCENARIOS_MB,
    PRICE_PER_GB_SECOND,
    EXECUTION_PRICE_PER_MILLION,
    MIN_EXECUTION_DURATION_SECONDS,
)


def check(condition, message):
    if condition:
        print(f"PASS: {message}")
        return True

    print(f"FAIL: {message}")
    return False


def main():
    print("Loading Azure cost estimation results...\n")

    summary = pd.read_csv(
        OUTPUT_DIR / "azure_cost_summary.csv"
    )

    monthly = pd.read_csv(
        OUTPUT_DIR / "azure_cost_monthly_summary.csv"
    )

    total_checks = 0
    passed_checks = 0

    def validate(condition, message):
        nonlocal total_checks, passed_checks

        total_checks += 1

        if check(condition, message):
            passed_checks += 1

    # Validate overall summary.
    validate(
        len(summary) == len(MEMORY_SCENARIOS_MB),
        "All memory scenarios are present"
    )

    validate(
        summary["memory_mb"].tolist() == MEMORY_SCENARIOS_MB,
        "Memory scenarios match configuration"
    )

    expected_invocations = 1_980_951

    validate(
        (summary["total_invocations"] == expected_invocations).all(),
        "All original invocations are included"
    )

    # Validate every memory scenario.
    for memory_mb in MEMORY_SCENARIOS_MB:

        print(f"\nValidating {memory_mb} MB scenario...")
        details = pd.read_csv(
            OUTPUT_DIR / f"cost_details_{memory_mb}MB.csv"
        )

        scenario = summary[
            summary["memory_mb"] == memory_mb
        ].iloc[0]

        validate(
            len(details) == expected_invocations,
            f"{memory_mb} MB: correct invocation count"
        )

        validate(
            not details.isna().any().any(),
            f"{memory_mb} MB: no missing values"
        )

        validate(
            (details["billable_duration_seconds"] >=
             MIN_EXECUTION_DURATION_SECONDS).all(),
            f"{memory_mb} MB: minimum billable duration respected"
        )

        validate(
            (details["billable_memory_mb"] > 0).all(),
            f"{memory_mb} MB: positive billable memory"
        )

        validate(
            (details["gb_seconds"] >= 0).all(),
            f"{memory_mb} MB: non-negative GB-seconds"
        )

        validate(
            (details["gross_cost_usd"] >= 0).all(),
            f"{memory_mb} MB: non-negative gross cost"
        )

        expected_gb_seconds = (
            details["billable_memory_mb"] / 1024
        ) * details["billable_duration_seconds"]

        validate(
            (
                details["gb_seconds"]
                - expected_gb_seconds
            ).abs().max() < 1e-8,
            f"{memory_mb} MB: GB-seconds formula"
        )

        expected_execution_charge = (
            EXECUTION_PRICE_PER_MILLION / 1_000_000
        )

        validate(
            (
                details["execution_charge_usd"]
                - expected_execution_charge
            ).abs().max() < 1e-10,
            f"{memory_mb} MB: execution charge formula"
        )

        expected_resource_charge = (
            details["gb_seconds"] * PRICE_PER_GB_SECOND
        )

        validate(
            (
                details["resource_charge_usd"]
                - expected_resource_charge
            ).abs().max() < 1e-8,
            f"{memory_mb} MB: resource charge formula"
        )

        validate(
            abs(
                details["gross_cost_usd"].sum()
                - scenario["gross_total_cost_usd"]
            ) < 1e-6,
            f"{memory_mb} MB: gross total matches summary"
        )

        scenario_monthly = monthly[
            monthly["memory_mb"] == memory_mb
        ]

        validate(
            scenario_monthly["invocation_count"].sum()
            == expected_invocations,
            f"{memory_mb} MB: monthly counts match total"
        )

        validate(
            abs(
                scenario_monthly["gross_total_cost_usd"].sum()
                - scenario["gross_total_cost_usd"]
            ) < 1e-6,
            f"{memory_mb} MB: monthly gross cost matches total"
        )

    # Check expected proportional cost behaviour.
    costs = summary.set_index("memory_mb")

    validate(
        costs.loc[1024, "gross_total_cost_usd"]
        > costs.loc[512, "gross_total_cost_usd"],
        "Higher memory produces higher gross cost"
    )

    validate(
        costs.loc[1536, "gross_total_cost_usd"]
        > costs.loc[1024, "gross_total_cost_usd"],
        "1.5 GB scenario costs more than 1 GB"
    )

    print("\n" + "=" * 65)
    print("AZURE COST MODEL VALIDATION SUMMARY")
    print("=" * 65)

    print(f"Total checks: {total_checks}")
    print(f"Passed: {passed_checks}")
    print(f"Failed: {total_checks - passed_checks}")

    if passed_checks == total_checks:
        print("\nAll cost model validation checks passed.")
    else:
        print("\nSome checks failed. Review the errors above.")


if __name__ == "__main__":
    main()
