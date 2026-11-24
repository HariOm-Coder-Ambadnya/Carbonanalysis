
import os
import math
import pandas as pd

from energy_model_config import (
    INPUT_FILE,
    OUTPUT_DIR,
    SCENARIOS,
    INTERVAL_SECONDS,
    PRIMARY_CARBON_COLUMN,
    SECONDARY_CARBON_COLUMN
)


def load_dataset():
    """Load and validate the merged workload-carbon dataset."""

    print("Loading workload-carbon dataset...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "minute",
        "invocation_count",
        "average_concurrency",
        "maximum_concurrency",
        PRIMARY_CARBON_COLUMN,
        SECONDARY_CARBON_COLUMN
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    numeric_columns = [
        "minute",
        "invocation_count",
        "average_concurrency",
        "maximum_concurrency",
        PRIMARY_CARBON_COLUMN,
        SECONDARY_CARBON_COLUMN
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    if df[numeric_columns].isnull().any().any():
        raise ValueError(
            "Missing or invalid values found in required columns."
        )

    if (df["average_concurrency"] < 0).any():
        raise ValueError("Negative average concurrency found.")

    if (df["maximum_concurrency"] < 0).any():
        raise ValueError("Negative maximum concurrency found.")

    if (df[PRIMARY_CARBON_COLUMN] < 0).any():
        raise ValueError("Negative lifecycle carbon intensity found.")

    if (df[SECONDARY_CARBON_COLUMN] < 0).any():
        raise ValueError("Negative direct carbon intensity found.")

    if (df["average_concurrency"] >
            df["maximum_concurrency"] + 0.000001).any():
        raise ValueError(
            "Average concurrency exceeds maximum concurrency."
        )

    if df["minute"].duplicated().any():
        raise ValueError("Duplicate minute records found.")

    df = df.sort_values("minute").reset_index(drop=True)

    print(f"Dataset loaded: {len(df)} records")

    return df


def estimate_energy(df, scenario_name, config):
    """
    Estimate energy for one scenario.

    Instance count is based on maximum concurrency.
    Utilisation is a concurrency-capacity proxy, NOT measured CPU usage.

    The model assumes the provisioned instances remain available
    for the entire one-minute interval.
    """

    capacity = config["concurrency_capacity"]
    idle_power = config["idle_power_w"]
    peak_power = config["peak_power_w"]

    if capacity <= 0:
        raise ValueError("Concurrency capacity must be positive.")

    if idle_power < 0 or peak_power < idle_power:
        raise ValueError("Invalid power configuration.")

    result = df.copy()

    instance_counts = []
    utilisation_values = []
    total_power_values = []
    energy_wh_values = []
    energy_kwh_values = []

    for _, row in result.iterrows():

        average_concurrency = row["average_concurrency"]
        maximum_concurrency = row["maximum_concurrency"]

        # No active workload means no provisioned instances
        # in this simplified scale-to-zero simulation.
        if maximum_concurrency == 0:
            instance_count = 0
            utilisation = 0.0
            total_power = 0.0
            energy_wh = 0.0

        else:
            # Required instances based on peak concurrency.
            instance_count = math.ceil(
                maximum_concurrency / capacity
            )

            # Concurrency-based capacity utilisation proxy.
            total_capacity = instance_count * capacity

            utilisation = min(
                average_concurrency / total_capacity,
                1.0
            )

            # Simplified linear power model.
            power_per_instance = (
                idle_power
                + utilisation * (peak_power - idle_power)
            )

            total_power = (
                instance_count * power_per_instance
            )

            # Energy = Power x Time.
            energy_wh = (
                total_power * INTERVAL_SECONDS / 3600
            )

        energy_kwh = energy_wh / 1000

        instance_counts.append(instance_count)
        utilisation_values.append(utilisation)
        total_power_values.append(total_power)
        energy_wh_values.append(energy_wh)
        energy_kwh_values.append(energy_kwh)

    result[f"{scenario_name}_instance_count"] = instance_counts

    result[f"{scenario_name}_utilisation_proxy"] = (
        utilisation_values
    )

    result[f"{scenario_name}_power_w"] = total_power_values

    result[f"{scenario_name}_energy_wh"] = energy_wh_values

    result[f"{scenario_name}_energy_kwh"] = energy_kwh_values

    # Carbon emissions using lifecycle carbon intensity.
    result[f"{scenario_name}_carbon_lifecycle_g"] = (
        result[f"{scenario_name}_energy_kwh"]
        * result[PRIMARY_CARBON_COLUMN]
    )

    # Carbon emissions using direct carbon intensity.
    result[f"{scenario_name}_carbon_direct_g"] = (
        result[f"{scenario_name}_energy_kwh"]
        * result[SECONDARY_CARBON_COLUMN]
    )

    return result


def create_summary(results):
    """Create one summary row for each scenario."""

    summary_rows = []

    for scenario_name in SCENARIOS:

        summary_rows.append({
            "scenario": scenario_name,

            "total_energy_kwh":
                results[f"{scenario_name}_energy_kwh"].sum(),

            "average_power_w":
                results[f"{scenario_name}_power_w"].mean(),

            "peak_power_w":
                results[f"{scenario_name}_power_w"].max(),

            "total_lifecycle_carbon_g":
                results[
                    f"{scenario_name}_carbon_lifecycle_g"
                ].sum(),

            "total_direct_carbon_g":
                results[
                    f"{scenario_name}_carbon_direct_g"
                ].sum(),

            "maximum_instances":
                results[
                    f"{scenario_name}_instance_count"
                ].max(),

            "average_instances":
                results[
                    f"{scenario_name}_instance_count"
                ].mean()
        })

    return pd.DataFrame(summary_rows)


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_dataset()

    print("\nStarting energy estimation...")

    results = df.copy()

    for scenario_name, config in SCENARIOS.items():

        print(f"Processing scenario: {scenario_name}")

        results = estimate_energy(
            results,
            scenario_name,
            config
        )

    # Save minute-level energy and carbon estimates.
    detailed_file = os.path.join(
        OUTPUT_DIR,
        "energy_carbon_estimates.csv"
    )

    results.to_csv(detailed_file, index=False)

    # Generate scenario-level summary.
    summary = create_summary(results)

    summary_file = os.path.join(
        OUTPUT_DIR,
        "energy_carbon_summary.csv"
    )

    summary.to_csv(summary_file, index=False)

    print("\n" + "=" * 60)
    print("ENERGY ESTIMATION COMPLETED")
    print("=" * 60)

    print(f"Processed intervals: {len(results)}")

    print("\nScenario Summary:")
    print(summary.to_string(index=False))

    print("\nSaved detailed results:")
    print(os.path.abspath(detailed_file))

    print("\nSaved scenario summary:")
    print(os.path.abspath(summary_file))

    print("\nIMPORTANT:")
    print(
        "These are model-based estimates using assumed "
        "power and concurrency parameters."
    )
    print(
        "Concurrency utilisation is not measured CPU utilisation."
    )


if __name__ == "__main__":
    main()
