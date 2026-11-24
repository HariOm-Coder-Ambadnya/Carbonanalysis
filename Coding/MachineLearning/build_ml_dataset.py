
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SCENARIOS_DIR = (
    ROOT / "Coding" / "MachineLearning" / "scenarios"
)

RESULTS_FILE = (
    ROOT
    / "Coding"
    / "MachineLearning"
    / "results"
    / "energy_carbon_evaluation_with_multi_objective.csv"
)

OUTPUT_DIR = ROOT / "Coding" / "MachineLearning" / "results"

FEATURES_FILE = OUTPUT_DIR / "ml_scenario_dataset.csv"
LABELS_FILE = OUTPUT_DIR / "ml_scenario_labels.csv"
FINAL_FILE = OUTPUT_DIR / "ml_training_dataset.csv"


def safe_mean(series):
    return float(series.mean()) if len(series) else 0.0


def safe_std(series):
    return float(series.std()) if len(series) > 1 else 0.0


def safe_quantile(series, quantile):
    return float(series.quantile(quantile)) if len(series) else 0.0


def extract_scenario_features(scenario_dir):

    workload_path = scenario_dir / "classified_workload.csv"
    carbon_path = scenario_dir / "carbon_data.csv"
    metadata_path = scenario_dir / "metadata.json"

    workload = pd.read_csv(workload_path)
    carbon = pd.read_csv(carbon_path)

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    required_workload = {
        "app",
        "func",
        "start_timestamp",
        "end_timestamp",
        "duration",
        "workload_type",
    }

    required_carbon = {
        "minute",
        "carbon_intensity_lifecycle",
    }

    if not required_workload.issubset(workload.columns):
        raise ValueError(
            f"Missing workload columns in {scenario_dir.name}: "
            f"{required_workload - set(workload.columns)}"
        )

    if not required_carbon.issubset(carbon.columns):
        raise ValueError(
            f"Missing carbon columns in {scenario_dir.name}: "
            f"{required_carbon - set(carbon.columns)}"
        )

    workload["duration"] = pd.to_numeric(
        workload["duration"], errors="coerce"
    )

    workload["start_timestamp"] = pd.to_numeric(
        workload["start_timestamp"], errors="coerce"
    )

    workload["end_timestamp"] = pd.to_numeric(
        workload["end_timestamp"], errors="coerce"
    )

    carbon["carbon_intensity_lifecycle"] = pd.to_numeric(
        carbon["carbon_intensity_lifecycle"], errors="coerce"
    )

    workload = workload.dropna(
        subset=["duration", "start_timestamp", "end_timestamp"]
    ).copy()

    carbon_values = carbon[
        "carbon_intensity_lifecycle"
    ].dropna()

    if workload.empty or carbon_values.empty:
        raise ValueError(
            f"Empty workload or carbon data in {scenario_dir.name}"
        )

    # --------------------------------------------------
    # Basic workload features
    # --------------------------------------------------

    total_jobs = len(workload)

    workload_types = (
        workload["workload_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    type_counts = workload_types.value_counts()

    immediate_jobs = sum(
        count
        for name, count in type_counts.items()
        if "immediate" in name
    )

    delayed_jobs = sum(
        count
        for name, count in type_counts.items()
        if "delay" in name or "tolerant" in name
    )

    durations = workload["duration"]
    arrivals = workload["start_timestamp"]

    # --------------------------------------------------
    # Duration distribution features
    # --------------------------------------------------

    total_execution_seconds = float(durations.sum())

    duration_mean = safe_mean(durations)

    long_job_ratio_10s = float((durations >= 10).mean())
    long_job_ratio_30s = float((durations >= 30).mean())
    long_job_ratio_60s = float((durations >= 60).mean())

    # --------------------------------------------------
    # Arrival distribution across six one-hour bins
    # --------------------------------------------------

    arrival_minute = arrivals / 60.0

    arrival_bins = pd.cut(
        arrival_minute,
        bins=[-np.inf, 60, 120, 180, 240, 300, np.inf],
        labels=False,
        include_lowest=True,
    )

    arrival_distribution = (
        arrival_bins.value_counts(normalize=True)
        .reindex(range(6), fill_value=0)
    )

    # --------------------------------------------------
    # Function identity features
    # --------------------------------------------------

    unique_function_pairs = (
        workload[["app", "func"]]
        .drop_duplicates()
        .shape[0]
    )

    # --------------------------------------------------
    # Construct feature dictionary
    # --------------------------------------------------

    features = {
        # Scenario metadata
        "scenario_id": scenario_dir.name,
        "scenario_order": int(scenario_dir.name.split("_")[-1]),
        "scenario_type": metadata.get("scenario_type"),

        # Workload size and composition
        "total_jobs": total_jobs,
        "unique_applications": workload["app"].nunique(),
        "unique_functions": workload["func"].nunique(),
        "unique_function_pairs": unique_function_pairs,

        "immediate_job_ratio": immediate_jobs / total_jobs,
        "delay_tolerant_job_ratio": delayed_jobs / total_jobs,

        # Existing duration features
        "duration_mean": duration_mean,
        "duration_median": float(durations.median()),
        "duration_std": safe_std(durations),
        "duration_max": float(durations.max()),

        # New duration percentile features
        "duration_p75": safe_quantile(durations, 0.75),
        "duration_p90": safe_quantile(durations, 0.90),
        "duration_p95": safe_quantile(durations, 0.95),
        "duration_p99": safe_quantile(durations, 0.99),

        # New long-running job features
        "long_job_ratio_10s": long_job_ratio_10s,
        "long_job_ratio_30s": long_job_ratio_30s,
        "long_job_ratio_60s": long_job_ratio_60s,

        # New execution volume features
        "total_execution_seconds": total_execution_seconds,
        "execution_time_per_job": (
            total_execution_seconds / total_jobs
        ),

        # Arrival features
        "arrival_mean_seconds": safe_mean(arrivals),
        "arrival_std_seconds": safe_std(arrivals),

        # Carbon features
        "carbon_mean": safe_mean(carbon_values),
        "carbon_median": float(carbon_values.median()),
        "carbon_std": safe_std(carbon_values),
        "carbon_min": float(carbon_values.min()),
        "carbon_max": float(carbon_values.max()),
    }

    for hour in range(6):
        features[f"arrival_ratio_hour_{hour + 1}"] = float(
            arrival_distribution.loc[hour]
        )

    return features


def create_winning_labels():

    results = pd.read_csv(RESULTS_FILE)

    results = results[
        results["status"] == "SUCCESS"
    ].copy()

    expected_strategies = 5
    expected_deadlines = 3

    expected_per_scenario = (
        expected_strategies * expected_deadlines
    )

    counts = results.groupby("scenario_id").size()

    if not (counts == expected_per_scenario).all():
        raise ValueError(
            "Some scenarios do not have exactly 15 successful "
            "strategy/deadline evaluations."
        )

    # --------------------------------------------------
    # Normalize energy and carbon within each scenario
    # --------------------------------------------------

    metrics = [
        "estimated_energy_kwh",
        "estimated_carbon_gco2e",
    ]

    for metric in metrics:

        minimum = results.groupby("scenario_id")[metric].transform("min")
        maximum = results.groupby("scenario_id")[metric].transform("max")

        difference = maximum - minimum

        results[f"{metric}_normalized"] = (
            (results[metric] - minimum)
            / difference.where(difference != 0, 1)
        )

    # Cost is intentionally excluded because it is invariant
    # across scheduling configurations in the current model.

    results["combined_score"] = (
        0.5 * results["estimated_energy_kwh_normalized"]
        + 0.5 * results["estimated_carbon_gco2e_normalized"]
    )

    # --------------------------------------------------
    # Deterministic winner selection
    # --------------------------------------------------

    winners = (
        results.sort_values(
            [
                "scenario_id",
                "combined_score",
                "estimated_energy_kwh",
                "estimated_carbon_gco2e",
                "strategy",
                "deadline_minutes",
            ],
            kind="stable",
        )
        .groupby("scenario_id", as_index=False)
        .first()
    )

    labels = winners[
        [
            "scenario_id",
            "strategy",
            "deadline_minutes",
            "estimated_energy_kwh",
            "estimated_carbon_gco2e",
            "combined_score",
        ]
    ].copy()

    return labels


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scenario_dirs = sorted(
        [
            path
            for path in SCENARIOS_DIR.iterdir()
            if path.is_dir()
            and path.name.startswith("scenario_")
        ],
        key=lambda path: int(path.name.split("_")[-1]),
    )

    print("Extracting scenario features...")

    feature_rows = []

    for index, scenario_dir in enumerate(
        scenario_dirs, start=1
    ):

        print(
            f"[{index}/{len(scenario_dirs)}] "
            f"{scenario_dir.name}"
        )

        feature_rows.append(
            extract_scenario_features(scenario_dir)
        )

    features = pd.DataFrame(feature_rows)

    print("\nGenerating winning labels...")

    labels = create_winning_labels()

    dataset = features.merge(
        labels,
        on="scenario_id",
        how="inner",
        validate="one_to_one",
    )

    if len(dataset) != len(scenario_dirs):
        raise ValueError(
            "Feature/label mismatch: "
            f"{len(dataset)} rows for {len(scenario_dirs)} scenarios."
        )

    dataset = dataset.sort_values(
        "scenario_order"
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Dataset validation
    # --------------------------------------------------

    if features.isnull().any().any():
        raise ValueError("Missing values found in feature dataset.")

    if dataset["scenario_id"].duplicated().any():
        raise ValueError("Duplicate scenario IDs found.")

    # --------------------------------------------------
    # Save datasets
    # --------------------------------------------------

    features.to_csv(FEATURES_FILE, index=False)
    labels.to_csv(LABELS_FILE, index=False)
    dataset.to_csv(FINAL_FILE, index=False)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("ML DATASET CREATED")
    print("=" * 65)

    print("Scenario feature rows:", len(features))
    print("Winning labels:", len(labels))
    print("Final dataset rows:", len(dataset))
    print("Total columns:", len(dataset.columns))
    print("Feature columns:", len(features.columns) - 3)

    print("\nNew workload features:")
    new_features = [
        "duration_p75",
        "duration_p90",
        "duration_p95",
        "duration_p99",
        "long_job_ratio_10s",
        "long_job_ratio_30s",
        "long_job_ratio_60s",
        "total_execution_seconds",
        "execution_time_per_job",
        "unique_function_pairs",
    ]

    for feature in new_features:
        print(f"{feature}: added")

    print("\nWinning strategy distribution:")
    print(dataset["strategy"].value_counts().to_string())

    print("\nWinning strategy/deadline distribution:")
    print(
        dataset.groupby(
            ["strategy", "deadline_minutes"]
        ).size().sort_values(ascending=False).to_string()
    )

    print("\nFiles saved:")
    print(FEATURES_FILE)
    print(LABELS_FILE)
    print(FINAL_FILE)


if __name__ == "__main__":
    main()
