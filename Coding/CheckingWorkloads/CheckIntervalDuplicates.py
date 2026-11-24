import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "AzureFunctionsInvocationTraceForTwoWeeksJan2021"
    / "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
)

OUTPUT_DIR = BASE_DIR / "Coding" / "WorkloadAnalysis"

TARGET_MINUTE = 18610

print("Loading original Azure Functions trace...")

df = pd.read_csv(INPUT_FILE)

df["start_timestamp"] = df["end_timestamp"] - df["duration"]
df["minute"] = (df["start_timestamp"] // 60).astype("int64")

interval = df[df["minute"] == TARGET_MINUTE].copy()

# Count exact duplicate rows across the original four source columns
source_columns = ["app", "func", "end_timestamp", "duration"]

duplicate_mask = interval.duplicated(
    subset=source_columns,
    keep="first"
)

duplicate_count = int(duplicate_mask.sum())

# Remove exact duplicates for comparison only
deduplicated = interval.drop_duplicates(
    subset=source_columns
)

print("\n" + "=" * 60)
print("EXACT DUPLICATE INVESTIGATION")
print("=" * 60)

print(f"\nOriginal records: {len(interval)}")
print(f"Exact duplicate records: {duplicate_count}")
print(f"Records after deduplication: {len(deduplicated)}")

print("\nEXECUTION TIME COMPARISON")
print("-" * 60)

original_total = interval["duration"].sum()
deduplicated_total = deduplicated["duration"].sum()

print(f"Original total execution seconds: {original_total:,.3f}")
print(f"Deduplicated total execution seconds: {deduplicated_total:,.3f}")
print(f"Difference: {original_total - deduplicated_total:,.3f}")

if original_total > 0:
    print(
        "Percentage of execution time removed: "
        f"{((original_total - deduplicated_total) / original_total) * 100:.4f}%"
    )

# Most frequent app/function combinations
print("\nTOP 10 APP/FUNCTION COMBINATIONS")
print("-" * 60)

combinations = (
    interval.groupby(["app", "func"])
    .agg(
        invocation_count=("duration", "size"),
        total_execution_seconds=("duration", "sum"),
        average_duration=("duration", "mean"),
        maximum_duration=("duration", "max")
    )
    .sort_values("invocation_count", ascending=False)
    .head(10)
    .reset_index()
)

print(combinations.to_string(index=False))

# Save comparison data
combinations.to_csv(
    OUTPUT_DIR / "interval_18610_app_function_summary.csv",
    index=False
)

print("\nInvestigation completed.")