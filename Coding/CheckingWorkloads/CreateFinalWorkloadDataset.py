
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

WORKLOAD_FILE = (
    BASE_DIR
    / "Coding"
    / "Renewddatasets"
    / "workload_1min_complete.csv"
)

CONCURRENCY_FILE = (
    BASE_DIR
    / "Coding"
    / "ConcurrencyAnalysis"
    / "concurrency_1min.csv"
)

OUTPUT_DIR = BASE_DIR / "Coding" / "FinalWorkloadDataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "final_workload_dataset.csv"

# --------------------------------------------------
# 2. LOAD DATASETS
# --------------------------------------------------

print("Loading workload and concurrency datasets...")

workload = pd.read_csv(WORKLOAD_FILE)
concurrency = pd.read_csv(CONCURRENCY_FILE)

print(f"Workload shape: {workload.shape}")
print(f"Concurrency shape: {concurrency.shape}")

# --------------------------------------------------
# 3. VALIDATE MERGE KEYS
# --------------------------------------------------

if workload["minute"].duplicated().any():
    raise ValueError("Duplicate minute values in workload dataset.")

if concurrency["minute"].duplicated().any():
    raise ValueError("Duplicate minute values in concurrency dataset.")

if not workload["minute"].equals(concurrency["minute"]):
    raise ValueError(
        "Minute indexes do not match. Check the input datasets."
    )

# --------------------------------------------------
# 4. MERGE DATASETS
# --------------------------------------------------

final_df = pd.merge(
    workload,
    concurrency[
        [
            "minute",
            "average_concurrency",
            "maximum_concurrency"
        ]
    ],
    on="minute",
    how="inner",
    validate="one_to_one"
)

# --------------------------------------------------
# 5. VALIDATE FINAL DATASET
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL DATASET VALIDATION")
print("=" * 60)

print(f"Total rows: {len(final_df):,}")
print(f"Total columns: {len(final_df.columns)}")

print(
    f"Missing average concurrency values: "
    f"{final_df['average_concurrency'].isna().sum()}"
)

print(
    f"Missing maximum concurrency values: "
    f"{final_df['maximum_concurrency'].isna().sum()}"
)

if len(final_df) != 20160:
    raise ValueError("Expected exactly 20,160 minute intervals.")

if final_df["average_concurrency"].isna().any():
    raise ValueError("Missing average concurrency values.")

if final_df["maximum_concurrency"].isna().any():
    raise ValueError("Missing maximum concurrency values.")

if (final_df["average_concurrency"] < 0).any():
    raise ValueError("Negative average concurrency detected.")

if (final_df["maximum_concurrency"] < 0).any():
    raise ValueError("Negative maximum concurrency detected.")

# --------------------------------------------------
# 6. SAVE FINAL DATASET
# --------------------------------------------------

final_df.to_csv(OUTPUT_FILE, index=False)

print("\nFINAL DATASET COLUMNS")
print("-" * 60)

for column in final_df.columns:
    print(column)

print("\n" + "=" * 60)
print("FINAL WORKLOAD DATASET CREATED")
print("=" * 60)

print(f"Saved to: {OUTPUT_FILE}")


# --------------------------------------------------
# 7. INSPECT PEAK CONCURRENCY
# --------------------------------------------------

print("\nTOP 10 CONCURRENCY INTERVALS")
print("-" * 60)

# Generate hour from minute
final_df["hour"] = final_df["minute"] / 60

# Add workload category from the previously generated
# workload pattern analysis dataset
pattern_file = (
    BASE_DIR
    / "Coding"
    / "WorkloadAnalysis"
    / "final_workload_with_patterns.csv"
)

if pattern_file.exists():

    patterns = pd.read_csv(pattern_file)

    if patterns["minute"].duplicated().any():
        raise ValueError("Duplicate minute values in pattern dataset.")

    category_data = patterns[["minute", "workload_category"]]

    final_df = final_df.merge(
        category_data,
        on="minute",
        how="left",
        validate="one_to_one"
    )

else:
    print("Pattern dataset not found. Workload category omitted.")

columns_to_display = [
    "minute",
    "hour",
    "invocation_count",
    "total_execution_seconds",
    "average_concurrency",
    "maximum_concurrency"
]

if "workload_category" in final_df.columns:
    columns_to_display.append("workload_category")

print(
    final_df.nlargest(
        10,
        "maximum_concurrency"
    )[columns_to_display].to_string(index=False)
)

# Save again with additional derived columns
final_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nUpdated dataset saved to: {OUTPUT_FILE}")
