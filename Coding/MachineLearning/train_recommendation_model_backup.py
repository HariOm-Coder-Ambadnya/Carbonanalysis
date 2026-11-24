
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT
    / "Coding"
    / "MachineLearning"
    / "results"
    / "ml_training_dataset.csv"
)

MODEL_DIR = (
    ROOT
    / "Coding"
    / "MachineLearning"
    / "models"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TRAIN_SCENARIOS = 45


def evaluate_model(name, model, X_test, y_test, all_labels):

    predictions = model.predict(X_test)

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    print("Accuracy:", round(
        accuracy_score(y_test, predictions), 4
    ))

    print("Balanced accuracy:", round(
        balanced_accuracy_score(y_test, predictions), 4
    ))

    print("Macro F1:", round(
        f1_score(
            y_test,
            predictions,
            labels=all_labels,
            average="macro",
            zero_division=0,
        ), 4
    ))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=all_labels,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=all_labels,
        )
    )

    return predictions

def train_target(df, target, feature_columns):

    train = df[df["scenario_order"] <= TRAIN_SCENARIOS].copy()
    test = df[df["scenario_order"] > TRAIN_SCENARIOS].copy()

    X_train = train[feature_columns]
    X_test = test[feature_columns]

    y_train = train[target].astype(str)
    y_test = test[target].astype(str)

    all_labels = sorted(
    set(y_train.unique()) | set(y_test.unique())
    )

    if y_train.nunique() < 2:
        raise ValueError(
            f"Training data contains fewer than two classes for {target}."
        )

    print("\nTarget:", target)
    print("Training scenarios:", len(train))
    print("Testing scenarios:", len(test))

    print("\nTraining class distribution:")
    print(y_train.value_counts().to_string())

    print("\nTesting class distribution:")
    print(y_test.value_counts().to_string())

    # Baseline: always predict the most common training class.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    evaluate_model(
    f"{target} - MAJORITY BASELINE",
    baseline,
    X_test,
    y_test,
    all_labels,
)

    # Random Forest recommendation model.
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    model.fit(X_train, y_train)

    predictions = evaluate_model(
    f"{target} - RANDOM FOREST",
    model,
    X_test,
    y_test,
    all_labels,
)

    model_path = MODEL_DIR / f"{target}_model.joblib"

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "target": target,
            "train_scenarios": TRAIN_SCENARIOS,
        },
        model_path,
    )

    print("\nSaved model:", model_path)

    prediction_results = test[
        ["scenario_id", "scenario_order", target]
    ].copy()

    prediction_results["predicted"] = predictions
    prediction_results["correct"] = (
        prediction_results[target].astype(str)
        == prediction_results["predicted"].astype(str)
    )

    return prediction_results


def main():

    df = pd.read_csv(DATA_FILE)

    # Only input features are used for prediction.
    # Never include labels, energy results, carbon results,
    # scenario identifiers, or scenario order as model inputs.
    excluded = {
        "scenario_id",
        "scenario_order",
        "scenario_type",
        "strategy",
        "deadline_minutes",
        "estimated_energy_kwh",
        "estimated_carbon_gco2e",
        "combined_score",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    X = df[feature_columns]

    # Check that all input features are numeric.
    non_numeric = X.select_dtypes(
        exclude=["number", "bool"]
    ).columns.tolist()

    if non_numeric:
        raise ValueError(
            f"Non-numeric model features found: {non_numeric}"
        )

    print("=" * 65)
    print("CARBON AND COST-AWARE LOAD BALANCING")
    print("ML RECOMMENDATION MODEL TRAINING")
    print("=" * 65)

    print("Total scenarios:", len(df))
    print("Feature count:", len(feature_columns))
    print("Training scenarios:", TRAIN_SCENARIOS)
    print("Testing scenarios:", len(df) - TRAIN_SCENARIOS)

    strategy_results = train_target(
        df,
        "strategy",
        feature_columns,
    )

    deadline_results = train_target(
        df,
        "deadline_minutes",
        feature_columns,
    )

    strategy_results.to_csv(
        MODEL_DIR / "strategy_test_predictions.csv",
        index=False,
    )

    deadline_results.to_csv(
        MODEL_DIR / "deadline_test_predictions.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("MODEL TRAINING COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
