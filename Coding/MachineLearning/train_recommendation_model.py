
from pathlib import Path

import joblib
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
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
    ROOT / "Coding" / "MachineLearning"
    / "results" / "ml_training_dataset.csv"
)

MODEL_DIR = (
    ROOT / "Coding" / "MachineLearning" / "models"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Preserve the original chronological final test split.
TRAIN_SCENARIOS = 45

# Internal validation: scenarios 001-036 train,
# scenarios 037-045 validate.
VALIDATION_TRAIN_SCENARIOS = 36


def create_models():

    return {
        "RANDOM_FOREST": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", RandomForestClassifier(
                    n_estimators=300,
                    max_depth=4,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )),
            ]),
            False,
        ),

        "BALANCED_RANDOM_FOREST": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", RandomForestClassifier(
                    n_estimators=400,
                    max_depth=5,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )),
            ]),
            True,
        ),

        "EXTRA_TREES_BALANCED": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", ExtraTreesClassifier(
                    n_estimators=400,
                    max_depth=5,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )),
            ]),
            True,
        ),
    }


def fit_candidate(model, X, y, use_balanced_weights):

    if use_balanced_weights:
        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y,
        )

        model.fit(
            X,
            y,
            classifier__sample_weight=sample_weights,
        )
    else:
        model.fit(X, y)

    return model


def calculate_metrics(y_true, predictions, labels):
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "macro_f1": f1_score(
            y_true,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
    }


def evaluate_model(name, model, X_test, y_test, all_labels):

    predictions = model.predict(X_test)

    metrics = calculate_metrics(
        y_test, predictions, all_labels
    )

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=all_labels,
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=all_labels,
    ))

    return predictions, metrics


def train_target(df, target, feature_columns):

    train = df[
        df["scenario_order"] <= TRAIN_SCENARIOS
    ].copy()

    test = df[
        df["scenario_order"] > TRAIN_SCENARIOS
    ].copy()

    validation_train = train[
        train["scenario_order"] <= VALIDATION_TRAIN_SCENARIOS
    ].copy()

    validation = train[
        train["scenario_order"] > VALIDATION_TRAIN_SCENARIOS
    ].copy()

    X_validation_train = validation_train[feature_columns]
    X_validation = validation[feature_columns]

    y_validation_train = validation_train[target].astype(str)
    y_validation = validation[target].astype(str)

    X_train = train[feature_columns]
    X_test = test[feature_columns]

    y_train = train[target].astype(str)
    y_test = test[target].astype(str)

    if y_validation_train.nunique() < 2:
        raise ValueError(
            f"Internal training data has fewer than two "
            f"classes for {target}."
        )

    if y_train.nunique() < 2:
        raise ValueError(
            f"Training data has fewer than two classes for {target}."
        )

    all_labels = sorted(y_train.unique())

    print("\n" + "#" * 65)
    print("TARGET:", target)
    print("#" * 65)

    print("Training scenarios:", len(train))
    print("Internal training scenarios:", len(validation_train))
    print("Internal validation scenarios:", len(validation))
    print("Final untouched test scenarios:", len(test))

    print("\nTraining class distribution:")
    print(y_train.value_counts().to_string())

    print("\nFinal test class distribution:")
    print(y_test.value_counts().to_string())

    # Compare candidates using only scenarios 001-045.
    validation_results = []

    for name, (model, use_balanced_weights) in create_models().items():

        fit_candidate(
            model,
            X_validation_train,
            y_validation_train,
            use_balanced_weights,
        )
        predictions = model.predict(X_validation)

        metrics = calculate_metrics(
            y_validation,
            predictions,
            all_labels,
        )

        print(f"\n{name} validation metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        validation_results.append({
            "model": name,
            **metrics,
        })

    results_df = pd.DataFrame(validation_results)

    # Select using macro F1 first, then balanced accuracy,
    # then ordinary accuracy.
    results_df = results_df.sort_values(
        by=[
            "macro_f1",
            "balanced_accuracy",
            "accuracy",
        ],
        ascending=False,
    )

    results_path = (
        MODEL_DIR / f"{target}_validation_comparison.csv"
    )

    results_df.to_csv(results_path, index=False)

    selected_name = results_df.iloc[0]["model"]

    print("\nSelected model:", selected_name)
    print("Validation comparison saved:", results_path)

    # Retrain the selected algorithm on all 45 training scenarios.
        # Retrain the selected algorithm on all 45 training scenarios.
    final_model, use_balanced_weights = create_models()[selected_name]

    fit_candidate(
        final_model,
        X_train,
        y_train,
        use_balanced_weights,
    )

    # Evaluate the untouched final test set only after selection.
    predictions, test_metrics = evaluate_model(
        f"{target} - FINAL MODEL ({selected_name})",
        final_model,
        X_test,
        y_test,
        all_labels,
    )

    model_path = MODEL_DIR / f"{target}_model.joblib"

    joblib.dump(
        {
            "model": final_model,
            "feature_columns": feature_columns,
            "target": target,
            "train_scenarios": TRAIN_SCENARIOS,
            "selected_algorithm": selected_name,
            "validation_metrics": results_df.to_dict("records"),
            "test_metrics": test_metrics,
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

    df = df.sort_values(
        "scenario_order"
    ).reset_index(drop=True)

    if df["scenario_order"].duplicated().any():
        raise ValueError("Duplicate scenario_order values found.")

    if len(df) != 56:
        raise ValueError(
            f"Expected 56 scenarios, found {len(df)}."
        )

    if df["scenario_order"].tolist() != list(range(1, 57)):
        raise ValueError(
            "Expected chronological scenario_order 1 through 56."
        )

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

    non_numeric = X.select_dtypes(
        exclude=["number", "bool"]
    ).columns.tolist()

    if non_numeric:
        raise ValueError(
            f"Non-numeric model features found: {non_numeric}"
        )

    if X.isna().all().any():
        raise ValueError("A feature contains only missing values.")

    print("=" * 65)
    print("CARBON AND COST-AWARE LOAD BALANCING")
    print("IMPROVED ML RECOMMENDATION MODEL")
    print("=" * 65)

    print("Total scenarios:", len(df))
    print("Feature count:", len(feature_columns))
    print("Training scenarios: 001-045")
    print("Final untouched test scenarios: 046-056")

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
