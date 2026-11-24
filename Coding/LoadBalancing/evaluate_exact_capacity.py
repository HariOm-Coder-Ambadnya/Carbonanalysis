
from pathlib import Path
import pandas as pd

from evaluate_scheduled_strategies import (
    load_carbon_data,
    evaluate_strategy
)


ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    ROOT / "Coding" / "LoadBalancing"
    / "results" / "exact_capacity"
)

OUTPUT_DIR = INPUT_DIR / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEADLINES = [15, 30, 60]


def main():

    print("\nEXACT-CAPACITY STRATEGY EVALUATION")
    print("=" * 50)

    carbon_values = load_carbon_data()

    results = []

    for deadline in DEADLINES:

        filename = (
            f"exact_capacity_schedule_{deadline}min.csv"
        )

        schedule = pd.read_csv(INPUT_DIR / filename)

        # Original execution-time baseline.
        baseline = schedule.copy()

        baseline["scheduled_start_timestamp"] = (
            baseline["original_start_timestamp"]
        )

        baseline_result = evaluate_strategy(
            f"Baseline_{deadline}min",
            baseline,
            carbon_values
        )

        proposed_result = evaluate_strategy(
            f"Exact_Capacity_{deadline}min",
            schedule,
            carbon_values
        )

        for metric in [
            "energy_kwh",
            "carbon_emissions_kg",
            "total_cost_usd"
        ]:

            original = baseline_result[metric]
            scheduled = proposed_result[metric]

            reduction = (
                (original - scheduled) / original * 100
                if original else 0
            )

            proposed_result[
                f"{metric}_reduction_percent"
            ] = reduction

        results.extend([
            baseline_result,
            proposed_result
        ])

        print(f"\n{deadline}-MINUTE SCENARIO")

        print(
            f"Energy: "
            f"{baseline_result['energy_kwh']:.4f} -> "
            f"{proposed_result['energy_kwh']:.4f} kWh"
        )

        print(
            f"Carbon: "
            f"{baseline_result['carbon_emissions_kg']:.4f} -> "
            f"{proposed_result['carbon_emissions_kg']:.4f} kg"
        )

        print(
            f"Cost: "
            f"${baseline_result['total_cost_usd']:.4f} -> "
            f"${proposed_result['total_cost_usd']:.4f}"
        )

    result_df = pd.DataFrame(results)

    output_file = (
        OUTPUT_DIR / "exact_capacity_evaluation.csv"
    )

    result_df.to_csv(output_file, index=False)

    print("\nEVALUATION COMPLETED")
    print("=" * 50)

    columns = [
        "strategy",
        "energy_kwh",
        "carbon_emissions_kg",
        "total_cost_usd",
        "maximum_concurrency",
        "maximum_instances"
    ]

    print(result_df[columns].to_string(index=False))

    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
