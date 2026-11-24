
import joblib
from pathlib import Path

model_dir = Path("Coding/MachineLearning/models")

for filename in [
    "strategy_model.joblib",
    "deadline_minutes_model.joblib"
]:
    data = joblib.load(model_dir / filename)

    print("\n" + "=" * 70)
    print(filename)
    print("=" * 70)

    print("Target:", data["target"])
    print("Algorithm:", data["selected_algorithm"])
    print("Training scenarios:", data["train_scenarios"])
    print("Number of features:", len(data["feature_columns"]))

    print("\nExact feature columns:")
    for index, column in enumerate(data["feature_columns"], start=1):
        print(f"{index:02}. {column}")

    print("\nPipeline:")
    print(data["model"])

    print("\nModel classes:")
    print(data["model"].classes_)
