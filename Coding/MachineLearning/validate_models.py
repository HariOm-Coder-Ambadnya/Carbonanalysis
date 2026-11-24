
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =========================================================
# 1. PATHS AND CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "Coding"
    / "CarbonAnalysis"
    / "workload_carbon_dataset.csv"
)

ML_DIR = PROJECT_ROOT / "Coding" / "MachineLearning"
MODEL_DIR = ML_DIR / "models"
RESULT_DIR = ML_DIR / "results"
GRAPH_DIR = RESULT_DIR / "validation_graphs"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.80

TARGETS = {
    "invocation_count": {
        "target_column": "target_invocation_count",
        "source_column": "invocation_count",
        "model_file": "best_target_invocation_count.joblib",
    },
    "maximum_concurrency": {
        "target_column": "target_maximum_concurrency",
        "source_column": "maximum_concurrency",
        "model_file": "best_target_maximum_concurrency.joblib",
    },
}

LAG_INTERVALS = [1, 2, 5, 10, 15, 30, 60]
ROLLING_WINDOWS = [5, 15, 30, 60]


# =========================================================
# 2. LOAD DATA AND MODEL METADATA
# =========================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    metadata_file = MODEL_DIR / "model_metadata.json"

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {metadata_file}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "timestamp_utc",
        "invocation_count",
        "maximum_concurrency",
        "average_concurrency",
        "total_execution_seconds",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce",
    )

    if df["timestamp_utc"].isna().any():
        raise ValueError(
            "Invalid timestamps found in the dataset."
        )

    df = (
        df.sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    if df["timestamp_utc"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps found. "
            "The dataset must contain one row per minute."
        )

    with open(metadata_file, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    print("\nVALIDATION DATA LOADED")
    print("----------------------")
    print(f"Dataset rows: {len(df):,}")
    print(
        f"First timestamp: {df['timestamp_utc'].iloc[0]}"
    )
    print(
        f"Last timestamp: {df['timestamp_utc'].iloc[-1]}"
    )

    return df, metadata


# =========================================================
# 3. RECREATE THE EXACT TRAINING FEATURES
# =========================================================

def create_features(df):

    data = df.copy()

    # Calendar features
    data["hour"] = data["timestamp_utc"].dt.hour
    data["day_of_week"] = (
        data["timestamp_utc"].dt.day_of_week
    )
    data["day_of_month"] = data["timestamp_utc"].dt.day
    data["minute_of_hour"] = (
        data["timestamp_utc"].dt.minute
    )

    # Cyclical time features
    data["hour_sin"] = np.sin(
        2 * np.pi * data["hour"] / 24
    )

    data["hour_cos"] = np.cos(
        2 * np.pi * data["hour"] / 24
    )

    data["day_sin"] = np.sin(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["day_cos"] = np.cos(
        2 * np.pi * data["day_of_week"] / 7
    )

    source_columns = [
        "invocation_count",
        "average_concurrency",
        "maximum_concurrency",
        "total_execution_seconds",
    ]

    for col in source_columns:

        if col not in data.columns:
            continue

        # Historical lag features
        for lag in LAG_INTERVALS:

            data[f"{col}_lag_{lag}"] = (
                data[col].shift(lag)
            )

        # Rolling features use observations through t-1.
        # The current value at t is also available separately
        # as a regular feature in the training dataset.

        past_values = data[col].shift(1)

        for window in ROLLING_WINDOWS:

            data[f"{col}_rolling_mean_{window}"] = (
                past_values
                .rolling(
                    window=window,
                    min_periods=window,
                )
                .mean()
            )

            data[f"{col}_rolling_std_{window}"] = (
                past_values
                .rolling(
                    window=window,
                    min_periods=window,
                )
                .std()
            )

    # One-minute-ahead targets
    data["target_invocation_count"] = (
        data["invocation_count"].shift(-1)
    )

    data["target_maximum_concurrency"] = (
        data["maximum_concurrency"].shift(-1)
    )

    data["forecast_timestamp"] = (
        data["timestamp_utc"]
        + pd.Timedelta(minutes=1)
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return data


# =========================================================
# 4. METRIC CALCULATION
# =========================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    # Invocation counts and concurrency cannot be negative.
    predicted = np.maximum(predicted, 0)

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    r2 = r2_score(
        actual,
        predicted,
    )

    total_actual = np.sum(np.abs(actual))

    if total_actual > 0:

        wape = (
            np.sum(np.abs(actual - predicted))
            / total_actual
        ) * 100

    else:
        wape = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "WAPE_percent": wape,
    }


# =========================================================
# 5. BASELINE PREDICTIONS
# =========================================================

def create_baseline_predictions(data, source_column):

    # The current row represents minute t.
    # The target is minute t+1.
    #
    # Therefore, all baseline methods use information
    # available through minute t.

    persistence = data[source_column]

    five_minute_average = (
        data[source_column]
        .rolling(
            window=5,
            min_periods=5,
        )
        .mean()
    )

    fifteen_minute_average = (
        data[source_column]
        .rolling(
            window=15,
            min_periods=15,
        )
        .mean()
    )

    return {
        "Persistence": persistence,
        "MovingAverage_5min": five_minute_average,
        "MovingAverage_15min": fifteen_minute_average,
    }


# =========================================================
# 6. VALIDATE ONE TARGET
# =========================================================

def validate_target(
    data,
    feature_columns,
    target_name,
    settings,
):

    target_column = settings["target_column"]
    source_column = settings["source_column"]

    model_path = MODEL_DIR / settings["model_file"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    # Remove only rows without a target.
    usable = data.dropna(
        subset=[target_column]
    ).copy()

    # Chronological 80:20 split.
    split_index = int(
        len(usable) * TRAIN_RATIO
    )

    train_data = usable.iloc[:split_index].copy()
    test_data = usable.iloc[split_index:].copy()

    X_test = test_data[feature_columns]

    y_train = train_data[target_column]
    y_test = test_data[target_column]

    print(f"\nTARGET: {target_name}")
    print("=" * 60)
    print(f"Training rows: {len(train_data):,}")
    print(f"Testing rows: {len(test_data):,}")

    # Load trained model.
    model = joblib.load(model_path)

    ml_predictions = np.maximum(
        model.predict(X_test),
        0,
    )

    predictions = {
        "ML_Best_Model": ml_predictions,
    }

    # Generate baseline forecasts.
    baselines = create_baseline_predictions(
        data,
        source_column,
    )

    for baseline_name, baseline_values in baselines.items():

        predictions[baseline_name] = (
            baseline_values
            .iloc[test_data.index]
            .to_numpy(dtype=float)
        )

    # -----------------------------------------------------
    # FAIR COMPARISON
    # -----------------------------------------------------

    # Evaluate every method on exactly the same rows.
    # This prevents a method from being evaluated on
    # a different subset of the test data.

    common_valid = np.ones(
        len(test_data),
        dtype=bool,
    )

    for predicted in predictions.values():

        common_valid &= np.isfinite(predicted)

    actual_values = y_test.to_numpy(dtype=float)

    common_valid &= np.isfinite(actual_values)

    if common_valid.sum() == 0:
        raise ValueError(
            f"No common valid test rows for {target_name}."
        )

    actual_valid = actual_values[common_valid]

    comparison = []

    for model_name, predicted in predictions.items():

        predicted_valid = predicted[common_valid]

        metrics = calculate_metrics(
            actual_valid,
            predicted_valid,
        )

        comparison.append({
            "target": target_name,
            "model": model_name,
            "evaluated_rows": len(actual_valid),
            **metrics,
        })

    comparison_df = pd.DataFrame(comparison)

    comparison_df = (
        comparison_df.sort_values("RMSE")
        .reset_index(drop=True)
    )

    print("\nMODEL VALIDATION RESULTS")
    print("------------------------")
    print(comparison_df.to_string(index=False))

    # -----------------------------------------------------
    # SAVE PREDICTIONS
    # -----------------------------------------------------

    prediction_df = pd.DataFrame({
        "timestamp_utc": (
            test_data["timestamp_utc"].to_numpy()
        ),
        "forecast_timestamp": (
            test_data["forecast_timestamp"].to_numpy()
        ),
        "actual": actual_values,
    })

    for model_name, predicted in predictions.items():

        prediction_df[model_name] = predicted

    prediction_file = (
        RESULT_DIR
        / f"validation_predictions_{target_name}.csv"
    )

    prediction_df.to_csv(
        prediction_file,
        index=False,
    )

    # -----------------------------------------------------
    # HIGH-WORKLOAD SPIKE EVALUATION
    # -----------------------------------------------------

    # Define spikes using the 95th percentile of
    # the training target values only.

    spike_threshold = y_train.quantile(0.95)

    spike_mask = (
        actual_values >= spike_threshold
    )

    spike_results = []

    for model_name, predicted in predictions.items():

        valid = (
            spike_mask
            & common_valid
            & np.isfinite(predicted)
        )

        if valid.sum() == 0:
            continue

        metrics = calculate_metrics(
            actual_values[valid],
            predicted[valid],
        )

        spike_results.append({
            "target": target_name,
            "model": model_name,
            "training_p95_threshold": spike_threshold,
            "spike_rows": int(valid.sum()),
            **metrics,
        })

    spike_df = pd.DataFrame(spike_results)

    print("\nHIGH-WORKLOAD EVALUATION")
    print("------------------------")
    print(
        f"Training P95 threshold: {spike_threshold:.4f}"
    )
    print(
        f"Test spike observations: {int(spike_mask.sum())}"
    )

    if not spike_df.empty:
        print(spike_df.to_string(index=False))

    # -----------------------------------------------------
    # GRAPH 1: ACTUAL VS ML PREDICTIONS
    # -----------------------------------------------------

    plot_rows = min(
        500,
        len(prediction_df),
    )

    plt.figure(figsize=(14, 6))

    plt.plot(
        prediction_df["forecast_timestamp"].iloc[:plot_rows],
        prediction_df["actual"].iloc[:plot_rows],
        label="Actual",
        linewidth=1.5,
    )

    plt.plot(
        prediction_df["forecast_timestamp"].iloc[:plot_rows],
        prediction_df["ML_Best_Model"].iloc[:plot_rows],
        label="ML Best Model",
        linewidth=1,
    )

    plt.title(
        f"Actual vs ML Predictions - {target_name}"
    )

    plt.xlabel("Forecast timestamp")
    plt.ylabel(target_name)
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR
        / f"{target_name}_actual_vs_predicted.png",
        dpi=300,
    )

    plt.close()

    # -----------------------------------------------------
    # GRAPH 2: MODEL RMSE COMPARISON
    # -----------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.bar(
        comparison_df["model"],
        comparison_df["RMSE"],
    )

    plt.title(
        f"RMSE Comparison - {target_name}"
    )

    plt.xlabel("Forecasting method")
    plt.ylabel("RMSE (lower is better)")
    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR
        / f"{target_name}_rmse_comparison.png",
        dpi=300,
    )

    plt.close()

    # -----------------------------------------------------
    # GRAPH 3: ACTUAL VS PREDICTED SCATTER
    # -----------------------------------------------------

    plt.figure(figsize=(7, 7))

    plt.scatter(
        prediction_df["actual"],
        prediction_df["ML_Best_Model"],
        alpha=0.35,
        s=12,
    )

    maximum = max(
        prediction_df["actual"].max(),
        prediction_df["ML_Best_Model"].max(),
    )

    plt.plot(
        [0, maximum],
        [0, maximum],
        linestyle="--",
    )

    plt.title(
        f"Actual vs Predicted - {target_name}"
    )

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR
        / f"{target_name}_scatter.png",
        dpi=300,
    )

    plt.close()

    print(f"\nSaved predictions: {prediction_file}")

    return comparison_df, spike_df


# =========================================================
# 7. MAIN
# =========================================================

def main():

    print("\nMODEL VALIDATION AND BASELINE COMPARISON")
    print("========================================")

    df, metadata = load_data()

    data = create_features(df)

    feature_columns = metadata["feature_columns"]

    missing_features = [
        col
        for col in feature_columns
        if col not in data.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing training features: {missing_features}"
        )

    comparison_results = []
    spike_results = []

    for target_name, settings in TARGETS.items():

        comparison_df, spike_df = validate_target(
            data,
            feature_columns,
            target_name,
            settings,
        )

        comparison_results.append(comparison_df)
        spike_results.append(spike_df)

    # -----------------------------------------------------
    # SAVE COMBINED RESULTS
    # -----------------------------------------------------

    all_comparisons = pd.concat(
        comparison_results,
        ignore_index=True,
    )

    all_spikes = pd.concat(
        spike_results,
        ignore_index=True,
    )

    all_comparisons.to_csv(
        RESULT_DIR / "baseline_model_comparison.csv",
        index=False,
    )

    all_spikes.to_csv(
        RESULT_DIR / "high_workload_evaluation.csv",
        index=False,
    )

    print("\nVALIDATION COMPLETED")
    print("====================")

    print("\nFINAL MODEL COMPARISON")
    print("----------------------")
    print(
        all_comparisons.to_string(index=False)
    )

    print("\nHIGH-WORKLOAD COMPARISON")
    print("-----------------------")
    print(
        all_spikes.to_string(index=False)
    )

    print("\nResults directory:")
    print(RESULT_DIR)

    print("\nGraphs directory:")
    print(GRAPH_DIR)


if __name__ == "__main__":
    main()
