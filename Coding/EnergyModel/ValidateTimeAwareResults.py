
import numpy as np
import pandas as pd

DETAIL_FILE = (
    "Coding/EnergyModel/results/time_aware_energy_estimates.csv"
)

WORKLOAD_FILE = (
    "Coding/CarbonAnalysis/workload_carbon_dataset.csv"
)

SCENARIOS = {
    "capacity_10": 10,
    "capacity_20": 20,
    "capacity_50": 50
}


def main():

    print("Loading time-aware results...")

    df = pd.read_csv(DETAIL_FILE)
    workload = pd.read_csv(WORKLOAD_FILE)

    checks = []

    checks.append((
        "Expected 20,160 records",
        len(df) == 20160
    ))

    checks.append((
        "No duplicate minute records",
        not df["minute"].duplicated().any()
    ))

    checks.append((
        "All minute indices are present",
        np.array_equal(
            np.sort(df["minute"].to_numpy()),
            np.arange(20160)
        )
    ))

    checks.append((
        "No missing energy values",
        not df.isnull().any().any()
    ))

    # Validate that carbon data matches the original merged dataset.
    checks.append((
        "Carbon dataset has matching row count",
        len(workload) == len(df)
    ))

    if len(workload) == len(df):

        merged = df.merge(
            workload[
                [
                    "minute",
                    "carbon_intensity_lifecycle",
                    "carbon_intensity_direct",
                    "maximum_concurrency"
                ]
            ],
            on="minute",
            how="left",
            validate="one_to_one",
            suffixes=("", "_source")
        )

        checks.append((
            "Every minute has matching workload data",
            merged["maximum_concurrency"].notna().all()
        ))

        for scenario, capacity in SCENARIOS.items():

            energy = merged[f"{scenario}_energy_kwh"]

            lifecycle = merged[
                f"{scenario}_carbon_lifecycle_g"
            ]

            direct = merged[
                f"{scenario}_carbon_direct_g"
            ]

            instances = merged[
                f"{scenario}_maximum_instances"
            ]

            expected_instances = np.ceil(
                merged["maximum_concurrency"] / capacity
            )

            checks.append((
                f"{scenario}: energy is non-negative",
                (energy >= 0).all()
            ))

            checks.append((
                f"{scenario}: carbon emissions are non-negative",
                (lifecycle >= 0).all()
                and (direct >= 0).all()
            ))

            checks.append((
                f"{scenario}: instance counts are non-negative",
                (instances >= 0).all()
            ))

            checks.append((
                f"{scenario}: instance count matches concurrency",
                np.array_equal(
                    instances.to_numpy(),
                    expected_instances.to_numpy()
                )
            ))

            expected_lifecycle = (
                energy
                * merged["carbon_intensity_lifecycle"]
            )

            expected_direct = (
                energy
                * merged["carbon_intensity_direct"]
            )

            checks.append((
                f"{scenario}: lifecycle emissions formula",
                np.allclose(lifecycle, expected_lifecycle)
            ))

            checks.append((
                f"{scenario}: direct emissions formula",
                np.allclose(direct, expected_direct)
            ))

            # No active concurrency means no modelled energy.
            idle = merged["maximum_concurrency"] == 0

            checks.append((
                f"{scenario}: idle intervals have zero energy",
                (energy[idle] == 0).all()
            ))

    print("\nTIME-AWARE VALIDATION")
    print("=" * 65)

    for description, passed in checks:

        print(
            f"{'PASS' if passed else 'FAIL'}: {description}"
        )

    failed = [
        description
        for description, passed in checks
        if not passed
    ]

    print("\nSUMMARY")
    print("=" * 65)
    print(f"Total checks: {len(checks)}")
    print(f"Passed: {len(checks) - len(failed)}")
    print(f"Failed: {len(failed)}")

    if failed:
        raise AssertionError(
            "Time-aware validation failed."
        )

    print("\nAll time-aware validation checks passed.")


if __name__ == "__main__":
    main()
