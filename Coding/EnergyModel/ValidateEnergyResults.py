
import os
import numpy as np
import pandas as pd

DETAIL_FILE = (
    "Coding/EnergyModel/results/energy_carbon_estimates.csv"
)

SUMMARY_FILE = (
    "Coding/EnergyModel/results/energy_carbon_summary.csv"
)

SCENARIOS = {
    "capacity_10": 10,
    "capacity_20": 20,
    "capacity_50": 50
}

def main():

    print("Loading energy estimation results...")

    df = pd.read_csv(DETAIL_FILE)
    summary = pd.read_csv(SUMMARY_FILE)

    checks = []

    # 1. Verify dataset size.
    checks.append((
        "Expected 20,160 records",
        len(df) == 20160
    ))

    # 2. Verify minute uniqueness.
    checks.append((
        "No duplicate minute records",
        not df["minute"].duplicated().any()
    ))

    for scenario, capacity in SCENARIOS.items():

        energy = df[f"{scenario}_energy_kwh"]
        power = df[f"{scenario}_power_w"]
        instances = df[f"{scenario}_instance_count"]
        utilisation = df[f"{scenario}_utilisation_proxy"]

        lifecycle = df[f"{scenario}_carbon_lifecycle_g"]
        direct = df[f"{scenario}_carbon_direct_g"]

        # 3. No negative values.
        checks.append((
            f"{scenario}: non-negative energy",
            (energy >= 0).all()
        ))

        checks.append((
            f"{scenario}: non-negative power",
            (power >= 0).all()
        ))

        checks.append((
            f"{scenario}: non-negative instance counts",
            (instances >= 0).all()
        ))

        # 4. Utilisation proxy must be between 0 and 1.
        checks.append((
            f"{scenario}: utilisation between 0 and 1",
            ((utilisation >= 0) & (utilisation <= 1)).all()
        ))

        # 5. Verify energy = power x 60 / 3,600,000.
        expected_energy = power * 60 / 3600000

        checks.append((
            f"{scenario}: energy formula correct",
            np.allclose(energy, expected_energy)
        ))

        # 6. Verify carbon calculation.
        expected_lifecycle = (
            energy * df["carbon_intensity_lifecycle"]
        )

        expected_direct = (
            energy * df["carbon_intensity_direct"]
        )

        checks.append((
            f"{scenario}: lifecycle carbon formula correct",
            np.allclose(lifecycle, expected_lifecycle)
        ))

        checks.append((
            f"{scenario}: direct carbon formula correct",
            np.allclose(direct, expected_direct)
        ))

        # 7. Verify instance count based on maximum concurrency.
        expected_instances = np.ceil(
            df["maximum_concurrency"] / capacity
        )

        checks.append((
            f"{scenario}: instance count correct",
            np.array_equal(
                instances.to_numpy(),
                expected_instances.to_numpy()
            )
        ))

        # 8. Idle intervals should have zero energy.
        idle_mask = df["maximum_concurrency"] == 0

        checks.append((
            f"{scenario}: idle intervals consume zero modelled energy",
            (energy[idle_mask] == 0).all()
        ))

    print("\nVALIDATION RESULTS")
    print("=" * 65)

    for description, passed in checks:

        status = "PASS" if passed else "FAIL"

        print(f"{status}: {description}")

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
            "Validation failed. Review the checks above."
        )

    print("\nAll validation checks passed.")

    print("\nSaved scenario summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
