
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    ROOT
    / "Coding"
    / "MachineLearning"
    / "results"
    / "ml_scenario_dataset.csv"
)

EVALUATION_FILE = (
    ROOT
    / "Coding"
    / "MachineLearning"
    / "results"
    / "energy_carbon_evaluation_with_multi_objective.csv"
)

MODEL_DIR = ROOT / "Coding" / "MachineLearning" / "models"
RESULT_DIR = ROOT / "Coding" / "MachineLearning" / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TEST_SCENARIOS = 11
RANDOM_STATE = 42

EXPECTED_CONFIGURATIONS = 15

# Previous model results, retained for comparison.
PREVIOUS_RESULTS = {
    "energy": {
        "mae": 0.415024,
        "rmse": 1.122100,
        "r2": 0.073400,
    },
    "carbon": {
        "mae": 308.090011,
        "rmse": 840.576408,
        "r2": 0.021991,
    },
}


def load_dataset():

    features = pd.read_csv(FEATURE_FILE)
    evaluations = pd.read_csv(EVALUATION_FILE)

    excluded = {
        "scenario_id",
        "scenario_order",
        "scenario_type",
        "strategy",
        "deadline_minutes",
    }

    feature_columns = [
        column
        for column in features.columns
        if column not in excluded
    ]

    if not feature_columns:
        raise ValueError("No scenario feature columns found.")

    required = {
        "scenario_id",
        "strategy",
        "deadline_minutes",
        "estimated_energy_kwh",
        "estimated_carbon_gco2e",
        "gross_total_cost_usd",
        "status",
        "carbon_coverage_complete",
    }

    missing = required - set(evaluations.columns)

    if missing:
        raise ValueError(
            f"Missing evaluation columns: {missing}"
        )

    # Use only successful evaluations with complete carbon coverage.
    evaluations = evaluations[
        (evaluations["status"] == "SUCCESS")
        & (evaluations["carbon_coverage_complete"] == True)
    ].copy()

    # Validate the number of configurations per scenario.
    configuration_counts = (
        evaluations.groupby("scenario_id").size()
    )

    invalid = configuration_counts[
        configuration_counts != EXPECTED_CONFIGURATIONS
    ]

    if not invalid.empty:
        raise ValueError(
            "Expected exactly 15 configurations per scenario.\n"
            f"Invalid scenarios:\n{invalid}"
        )

    dataset = evaluations.merge(
        features[
            ["scenario_id", "scenario_order"] + feature_columns
        ],
        on="scenario_id",
        how="inner",
        validate="many_to_one",
    )

    dataset = dataset.dropna(
        subset=feature_columns
        + [
            "estimated_energy_kwh",
            "estimated_carbon_gco2e",
        ]
    )

    if dataset["scenario_id"].nunique() != 56:
        raise ValueError(
            "Expected 56 scenarios after merging, found "
            f"{dataset['scenario_id'].nunique()}."
        )

    if dataset[feature_columns].select_dtypes(
        exclude=np.number
    ).shape[1] > 0:
        raise ValueError(
            "Non-numeric scenario features found. "
            "Check feature generation before training."
        )

    return dataset, feature_columns


def build_pipeline(feature_columns):

    categorical_columns = [
        "strategy",
        "deadline_minutes",
    ]

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                feature_columns,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )


def evaluate_model(name, model, X_test, y_test):

    predicted = model.predict(X_test)

    mae = mean_absolute_error(y_test, predicted)
    rmse = np.sqrt(mean_squared_error(y_test, predicted))
    r2 = r2_score(y_test, predicted)

    previous = PREVIOUS_RESULTS[name]

    print(f"\n{name.upper()} MODEL RESULTS")
    print("-" * 55)

    print(f"Previous MAE : {previous['mae']:.6f}")
    print(f"New MAE      : {mae:.6f}")

    print(f"Previous RMSE: {previous['rmse']:.6f}")
    print(f"New RMSE     : {rmse:.6f}")

    print(f"Previous R2  : {previous['r2']:.6f}")
    print(f"New R2       : {r2:.6f}")

    print("\nChange in metrics:")

    print(
        "MAE change : "
        f"{mae - previous['mae']:+.6f}"
    )

    print(
        "RMSE change: "
        f"{rmse - previous['rmse']:+.6f}"
    )

    print(
        "R2 change  : "
        f"{r2 - previous['r2']:+.6f}"
    )

    return predicted, {
        "model": name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def calculate_actual_objective(group):

    energy_min = group["estimated_energy_kwh"].min()
    energy_max = group["estimated_energy_kwh"].max()

    carbon_min = group["estimated_carbon_gco2e"].min()
    carbon_max = group["estimated_carbon_gco2e"].max()

    energy_range = energy_max - energy_min
    carbon_range = carbon_max - carbon_min

    if energy_range > 0:
        energy_score = (
            group["estimated_energy_kwh"] - energy_min
        ) / energy_range
    else:
        energy_score = 0.0

    if carbon_range > 0:
        carbon_score = (
            group["estimated_carbon_gco2e"] - carbon_min
        ) / carbon_range
    else:
        carbon_score = 0.0

    return 0.5 * energy_score + 0.5 * carbon_score



def generate_recommendations(predictions):

    recommendations = []

    for scenario_id, group in predictions.groupby("scenario_id"):

        group = group.copy()

        # Normalize predicted energy.
        energy_range = (
            group["predicted_energy"].max()
            - group["predicted_energy"].min()
        )

        if energy_range > 0:
            group["energy_score"] = (
                group["predicted_energy"]
                - group["predicted_energy"].min()
            ) / energy_range
        else:
            group["energy_score"] = 0.0

        # Normalize predicted carbon.
        carbon_range = (
            group["predicted_carbon"].max()
            - group["predicted_carbon"].min()
        )

        if carbon_range > 0:
            group["carbon_score"] = (
                group["predicted_carbon"]
                - group["predicted_carbon"].min()
            ) / carbon_range
        else:
            group["carbon_score"] = 0.0

        # ML predicted objective score.
        group["predicted_objective_score"] = (
            0.5 * group["energy_score"]
            + 0.5 * group["carbon_score"]
        )

        # Actual objective scores are used only for evaluation.
        group["actual_objective_score"] = (
            calculate_actual_objective(group)
        )

        # 1. ML-selected configuration.
        selected = group.sort_values(
            [
                "predicted_objective_score",
                "strategy",
                "deadline_minutes",
            ],
            kind="stable",
        ).iloc[0]

        # 2. Actual best configuration (oracle reference).
        actual_best = group.sort_values(
            [
                "actual_objective_score",
                "strategy",
                "deadline_minutes",
            ],
            kind="stable",
        ).iloc[0]

        # 3. Fixed multi-objective configuration.
        fixed_candidates = group[
            (group["strategy"] == "MULTI_OBJECTIVE")
            & (group["deadline_minutes"] == 15)
        ]

        if len(fixed_candidates) != 1:
            raise ValueError(
                f"Expected exactly one MULTI_OBJECTIVE/15 configuration "
                f"for {scenario_id}, found {len(fixed_candidates)}"
            )

        fixed = fixed_candidates.iloc[0]

        selected_actual_score = float(
            selected["actual_objective_score"]
        )

        fixed_actual_score = float(
            fixed["actual_objective_score"]
        )

        actual_best_score = float(
            actual_best["actual_objective_score"]
        )

        recommendations.append({

            "scenario_id": scenario_id,

            # ML recommendation.
            "recommended_strategy": selected["strategy"],
            "recommended_deadline_minutes": int(
                selected["deadline_minutes"]
            ),

            "predicted_energy_kwh": selected["predicted_energy"],
            "actual_energy_kwh": selected["estimated_energy_kwh"],

            "predicted_carbon_gco2e": selected["predicted_carbon"],
            "actual_carbon_gco2e": selected["estimated_carbon_gco2e"],

            "actual_gross_cost_usd": selected["gross_total_cost_usd"],

            "actual_objective_score": selected_actual_score,

            "actual_objective_regret": (
                selected_actual_score - actual_best_score
            ),

            # Fixed multi-objective baseline.
            "fixed_strategy": fixed["strategy"],
            "fixed_deadline_minutes": int(
                fixed["deadline_minutes"]
            ),
            "fixed_actual_energy_kwh": fixed["estimated_energy_kwh"],
            "fixed_actual_carbon_gco2e": (
                fixed["estimated_carbon_gco2e"]
            ),
            "fixed_actual_cost_usd": fixed["gross_total_cost_usd"],
            "fixed_actual_objective_score": fixed_actual_score,
            "fixed_objective_regret": (
                fixed_actual_score - actual_best_score
            ),

            # Oracle reference.
            "actual_best_strategy": actual_best["strategy"],
            "actual_best_deadline_minutes": int(
                actual_best["deadline_minutes"]
            ),

            "recommendation_matches_actual_best": (
                selected["strategy"] == actual_best["strategy"]
                and int(selected["deadline_minutes"])
                == int(actual_best["deadline_minutes"])
            ),

        })

    return pd.DataFrame(recommendations)


def create_scenario_error_report(predictions):

    report = predictions.copy()

    report["energy_absolute_error"] = (
        report["estimated_energy_kwh"]
        - report["predicted_energy"]
    ).abs()

    report["carbon_absolute_error"] = (
        report["estimated_carbon_gco2e"]
        - report["predicted_carbon"]
    ).abs()

    scenario_report = (
        report.groupby(
            ["scenario_id", "scenario_order"],
            as_index=False,
        )
        .agg(
            energy_mae=("energy_absolute_error", "mean"),
            carbon_mae=("carbon_absolute_error", "mean"),
            actual_mean_energy=("estimated_energy_kwh", "mean"),
            predicted_mean_energy=("predicted_energy", "mean"),
            actual_mean_carbon=("estimated_carbon_gco2e", "mean"),
            predicted_mean_carbon=("predicted_carbon", "mean"),
        )
        .sort_values("scenario_order")
    )

    return scenario_report


def main():

    print("=" * 70)
    print("CARBON AND COST-AWARE LOAD BALANCING")
    print("ML OUTCOME PREDICTION MODEL TRAINING")
    print("=" * 70)

    data, feature_columns = load_dataset()

    print("\nDataset shape:", data.shape)
    print("Unique scenarios:", data["scenario_id"].nunique())
    print("Scenario features:", len(feature_columns))
    print("Evaluation configurations:", len(data))

    # --------------------------------------------------
    # Chronological scenario-level split
    # --------------------------------------------------

    scenario_orders = sorted(
        data["scenario_order"].unique()
    )

    if len(scenario_orders) <= TEST_SCENARIOS:
        raise ValueError(
            "Not enough scenarios for chronological splitting."
        )

    test_orders = set(scenario_orders[-TEST_SCENARIOS:])

    train = data[
        ~data["scenario_order"].isin(test_orders)
    ].copy()

    test = data[
        data["scenario_order"].isin(test_orders)
    ].copy()

    train_scenarios = set(train["scenario_id"].unique())
    test_scenarios = set(test["scenario_id"].unique())

    if train_scenarios.intersection(test_scenarios):
        raise ValueError("Scenario leakage detected!")

    print("\nCHRONOLOGICAL SPLIT")
    print("-" * 45)
    print("Training scenarios:", len(train_scenarios))
    print("Testing scenarios:", len(test_scenarios))
    print("Training records:", len(train))
    print("Testing records:", len(test))

    print(
        "Training range:",
        min(train["scenario_order"]),
        "to",
        max(train["scenario_order"]),
    )

    print(
        "Testing range:",
        min(test["scenario_order"]),
        "to",
        max(test["scenario_order"]),
    )

    input_columns = feature_columns + [
        "strategy",
        "deadline_minutes",
    ]

    X_train = train[input_columns]
    X_test = test[input_columns]

    # --------------------------------------------------
    # Train energy and carbon models
    # --------------------------------------------------

    targets = {
        "energy": "estimated_energy_kwh",
        "carbon": "estimated_carbon_gco2e",
    }

    predictions = test[
        [
            "scenario_id",
            "scenario_order",
            "strategy",
            "deadline_minutes",
            "estimated_energy_kwh",
            "estimated_carbon_gco2e",
            "gross_total_cost_usd",
        ]
    ].copy()

    model_metrics = []

    for name, target_column in targets.items():

        print(f"\nTraining {name} prediction model...")

        y_train = train[target_column]
        y_test = test[target_column]

        model = build_pipeline(feature_columns)

        model.fit(X_train, y_train)

        predicted, metrics = evaluate_model(
            name,
            model,
            X_test,
            y_test,
        )

        predictions[f"predicted_{name}"] = predicted

        model_metrics.append(metrics)

        model_path = (
            MODEL_DIR / f"{name}_prediction_model.joblib"
        )

        joblib.dump(model, model_path)

        print("Saved:", model_path)

    # --------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------

    recommendation_df = generate_recommendations(
        predictions
    )

    # --------------------------------------------------
    # Per-scenario prediction errors
    # --------------------------------------------------

    scenario_error_df = create_scenario_error_report(
        predictions
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    prediction_path = (
        RESULT_DIR / "ml_outcome_predictions.csv"
    )

    recommendation_path = (
        RESULT_DIR / "ml_recommendation_test_results.csv"
    )

    scenario_error_path = (
        RESULT_DIR / "ml_scenario_error_report.csv"
    )

    metrics_path = (
        RESULT_DIR / "ml_model_metrics.csv"
    )

    predictions.to_csv(prediction_path, index=False)

    recommendation_df.to_csv(
        recommendation_path,
        index=False,
    )

    scenario_error_df.to_csv(
        scenario_error_path,
        index=False,
    )

    pd.DataFrame(model_metrics).to_csv(
        metrics_path,
        index=False,
    )

    # --------------------------------------------------
    # Display final results
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("RECOMMENDATION RESULTS")
    print("=" * 70)

    print(
        recommendation_df.to_string(index=False)
    )

    print("\nRecommended strategy distribution:")

    print(
        recommendation_df["recommended_strategy"]
        .value_counts()
        .to_string()
    )

    print("\nRecommendations matching actual best:")

    print(
        recommendation_df[
            "recommendation_matches_actual_best"
        ].value_counts().to_string()
    )

    print("\nMean actual objective regret:")

    print(
        recommendation_df["actual_objective_regret"].mean()
    )

    print("\nPER-SCENARIO PREDICTION ERRORS")
    print("=" * 70)

    print(scenario_error_df.to_string(index=False))

    print("\nSCENARIO 052 ERROR")

    scenario_052 = scenario_error_df[
        scenario_error_df["scenario_id"] == "scenario_052"
    ]

    if not scenario_052.empty:
        print(scenario_052.to_string(index=False))

    print("\nSaved files:")
    print(prediction_path)
    print(recommendation_path)
    print(scenario_error_path)
    print(metrics_path)

    print("\nOUTCOME MODEL TRAINING COMPLETED")


if __name__ == "__main__":
    main()
