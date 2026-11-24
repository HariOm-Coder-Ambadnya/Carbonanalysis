
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. PATHS AND CONFIGURATION
# ---------------------------------------------------------

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

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.80
RANDOM_STATE = 42

TARGETS = {
    "invocation_count": "target_invocation_count",
    "maximum_concurrency": "target_maximum_concurrency",
}

LAG_INTERVALS = [1, 2, 5, 10, 15, 30, 60]
ROLLING_WINDOWS = [5, 15, 30, 60]


# ---------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------

def load_dataset():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "minute",
        "timestamp_utc",
        "invocation_count",
        "average_concurrency",
        "maximum_concurrency",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="raise"
    )

    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    if df["timestamp_utc"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    if len(df) < 500:
        raise ValueError(
            "Dataset is too small for the configured experiment."
        )

    print("\nDATASET LOADED")
    print("-------------------------")
    print(f"Rows: {len(df):,}")
    print(f"First timestamp: {df['timestamp_utc'].min()}")
    print(f"Last timestamp:  {df['timestamp_utc'].max()}")

    return df


# ---------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------

def create_features(df):

    data = df.copy()

    # Calendar features
    data["hour"] = data["timestamp_utc"].dt.hour
    data["day_of_week"] = data["timestamp_utc"].dt.dayofweek
    data["day_of_month"] = data["timestamp_utc"].dt.day
    data["minute_of_hour"] = data["timestamp_utc"].dt.minute

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

        # Past observations only
        for lag in LAG_INTERVALS:
            data[f"{col}_lag_{lag}"] = data[col].shift(lag)

        # Shift before rolling to prevent current/future leakage
        past_values = data[col].shift(1)

        for window in ROLLING_WINDOWS:

            data[f"{col}_rolling_mean_{window}"] = (
                past_values
                .rolling(window=window, min_periods=window)
                .mean()
            )

            data[f"{col}_rolling_std_{window}"] = (
                past_values
                .rolling(window=window, min_periods=window)
                .std()
            )

    # Next-minute targets
    data["target_invocation_count"] = (
        data["invocation_count"].shift(-1)
    )

    data["target_maximum_concurrency"] = (
        data["maximum_concurrency"].shift(-1)
    )

    # Timestamp corresponding to the prediction target
    data["forecast_timestamp"] = (
        data["timestamp_utc"] + pd.Timedelta(minutes=1)
    )

    data = data.replace([np.inf, -np.inf], np.nan)

    return data


# ---------------------------------------------------------
# 4. MODEL DEFINITIONS
# ---------------------------------------------------------

def create_models():

    linear_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])

    random_forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            RandomForestRegressor(
                n_estimators=120,
                max_depth=12,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    gradient_boosting = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            HistGradientBoostingRegressor(
                max_iter=150,
                learning_rate=0.08,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    return {
        "LinearRegression": linear_model,
        "RandomForest": random_forest,
        "HistGradientBoosting": gradient_boosting,
    }


# ---------------------------------------------------------
# 5. METRICS
# ---------------------------------------------------------

def calculate_metrics(actual, predicted):

    predicted = np.maximum(predicted, 0)

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# ---------------------------------------------------------
# 6. TRAIN AND COMPARE MODELS
# ---------------------------------------------------------

def train_and_compare(data, feature_columns, target_column):

    usable = data.dropna(
        subset=[target_column]
    ).copy()

    split_index = int(len(usable) * TRAIN_RATIO)

    train_data = usable.iloc[:split_index].copy()
    test_data = usable.iloc[split_index:].copy()

    X_train = train_data[feature_columns]
    y_train = train_data[target_column]

    X_test = test_data[feature_columns]
    y_test = test_data[target_column]

    print(f"\nTARGET: {target_column}")
    print("-------------------------")
    print(f"Training rows: {len(train_data):,}")
    print(f"Testing rows:  {len(test_data):,}")

    models = create_models()

    comparison = []
    fitted_models = {}

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        predictions = np.maximum(predictions, 0)

        metrics = calculate_metrics(
            y_test,
            predictions
        )

        comparison.append({
            "target": target_column,
            "model": model_name,
            **metrics,
        })

        fitted_models[model_name] = model

        print(
            f"MAE:  {metrics['MAE']:.4f}\n"
            f"RMSE: {metrics['RMSE']:.4f}\n"
            f"R2:   {metrics['R2']:.4f}"
        )

    comparison_df = pd.DataFrame(comparison)

    comparison_df = comparison_df.sort_values(
        "RMSE"
    ).reset_index(drop=True)

    best_name = comparison_df.iloc[0]["model"]
    best_model = fitted_models[best_name]

    print(f"\nBest model for {target_column}: {best_name}")

    # Save the best model
    model_path = MODEL_DIR / f"best_{target_column}.joblib"

    joblib.dump(best_model, model_path)

    # Save test predictions
    test_predictions = test_data[
        ["timestamp_utc", "forecast_timestamp", target_column]
    ].copy()

    test_predictions["predicted_value"] = np.maximum(
        best_model.predict(X_test),
        0
    )

    test_predictions = test_predictions.rename(
        columns={
            target_column: "actual_value"
        }
    )

    prediction_path = (
        RESULT_DIR / f"test_predictions_{target_column}.csv"
    )

    test_predictions.to_csv(
        prediction_path,
        index=False
    )

    return (
        comparison_df,
        best_name,
        model_path,
        test_predictions,
    )


# ---------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------

def main():

    print("\nSERVERLESS WORKLOAD ML TRAINING")
    print("================================")

    df = load_dataset()

    data = create_features(df)

    excluded_columns = {
        "timestamp_utc",
        "forecast_timestamp",
        "minute",
        "hour_timestamp_utc",
        "zone_name",
        "zone_id",
        "data_source",
        "data_estimated",
        "data_estimation_method",
        "carbon_intensity_direct",
        "carbon_intensity_lifecycle",
        "target_invocation_count",
        "target_maximum_concurrency",
    }

    feature_columns = [
        col for col in data.columns
        if col not in excluded_columns
        and pd.api.types.is_numeric_dtype(data[col])
    ]

    if not feature_columns:
        raise ValueError("No numeric features available.")

    # Remove features that are entirely missing
    feature_columns = [
        col for col in feature_columns
        if data[col].notna().any()
    ]

    print(f"\nFeature count: {len(feature_columns)}")

    all_comparisons = []
    best_models = {}
    all_test_predictions = {}

    for target_name, target_column in TARGETS.items():

        (
            comparison,
            best_name,
            model_path,
            predictions,
        ) = train_and_compare(
            data,
            feature_columns,
            target_column
        )

        all_comparisons.append(comparison)

        best_models[target_name] = {
            "model_name": best_name,
            "model_path": str(model_path),
        }

        all_test_predictions[target_name] = predictions

    # Save combined comparison
    comparison_df = pd.concat(
        all_comparisons,
        ignore_index=True
    )

    comparison_path = (
        RESULT_DIR / "model_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    # Save test forecasts aligned by forecast timestamp
    invocation_predictions = (
        all_test_predictions["invocation_count"]
        .rename(columns={
            "actual_value": "actual_invocation_count",
            "predicted_value": "predicted_invocation_count",
        })
    )

    concurrency_predictions = (
        all_test_predictions["maximum_concurrency"]
        .rename(columns={
            "actual_value": "actual_maximum_concurrency",
            "predicted_value": "predicted_maximum_concurrency",
        })
    )

    forecast_df = invocation_predictions.merge(
        concurrency_predictions[
            [
                "forecast_timestamp",
                "actual_maximum_concurrency",
                "predicted_maximum_concurrency",
            ]
        ],
        on="forecast_timestamp",
        how="inner",
    )

    forecast_path = (
        RESULT_DIR / "test_workload_forecasts.csv"
    )

    forecast_df.to_csv(
        forecast_path,
        index=False
    )

    # Save feature metadata
    metadata = {
        "prediction_horizon": "1 minute",
        "training_ratio": TRAIN_RATIO,
        "random_state": RANDOM_STATE,
        "feature_columns": feature_columns,
        "best_models": best_models,
        "targets": TARGETS,
    }

    metadata_path = MODEL_DIR / "model_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nTRAINING COMPLETED")
    print("==================")
    print("\nModel comparison:")
    print(comparison_df.to_string(index=False))

    print("\nSaved files:")
    print(f"Comparison: {comparison_path}")
    print(f"Forecasts:  {forecast_path}")
    print(f"Metadata:   {metadata_path}")

    print("\nBest models:")
    print(json.dumps(best_models, indent=4))


if __name__ == "__main__":
    main()
