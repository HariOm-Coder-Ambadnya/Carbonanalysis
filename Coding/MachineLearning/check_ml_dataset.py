
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

FILE = (
    ROOT
    / "Coding"
    / "MachineLearning"
    / "results"
    / "ml_training_dataset.csv"
)

df = pd.read_csv(FILE)

print("=" * 65)
print("ML DATASET QUALITY CHECK")
print("=" * 65)

print("\nShape:")
print(df.shape)

print("\nMissing values:")
missing = df.isnull().sum()
print(missing[missing > 0].to_string() if missing.any() else "None")

print("\nConstant feature columns:")
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
    col for col in df.columns if col not in excluded
]

constant_columns = [
    col for col in feature_columns
    if df[col].nunique(dropna=False) <= 1
]

print(constant_columns if constant_columns else "None")

print("\nWorkload type ratios:")
for col in [
    "immediate_job_ratio",
    "delay_tolerant_job_ratio",
]:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].describe().to_string())

print("\nFeature statistics:")
print(df[feature_columns].describe().round(3).to_string())

print("\nTarget distribution:")
print(df["strategy"].value_counts().to_string())

print("\nDeadline distribution:")
print(df["deadline_minutes"].value_counts().sort_index().to_string())

print("\nScenario order:")
print(
    df[["scenario_id", "scenario_order"]]
    .head()
    .to_string(index=False)
)

print("\nQuality check completed.")
