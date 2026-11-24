import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "AzureFunctionsInvocationTraceForTwoWeeksJan2021"
    / "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
)

OUTPUT_DIR = BASE_DIR / "Coding" / "WorkloadAnalysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_MINUTE = 18610

print("Loading original Azure Functions trace...")

df = pd.read_csv(INPUT_FILE)

# Derive invocation start time
df["start_timestamp"] = df["end_timestamp"] - df["duration"]

# Assign each invocation to its starting minute
df["minute"] = (df["start_timestamp"] // 60).astype("int64")

# Select records belonging to the target minute
interval = df[df["minute"] == TARGET_MINUTE].copy()

print("\n" + "=" * 60)
print(f"INVESTIGATION OF MINUTE {TARGET_MINUTE}")
print("=" * 60)

print(f"\nTotal invocations: {len(interval)}")
print(f"Total execution seconds: {interval['duration'].sum():,.3f}")
print(f"Average duration: {interval['duration'].mean():,.3f}")
print(f"Median duration: {interval['duration'].median():,.3f}")
print(f"Maximum duration: {interval['duration'].max():,.3f}")
print(f"P95 duration: {interval['duration'].quantile(0.95):,.3f}")

# Show the 20 longest invocations
longest = interval.nlargest(20, "duration")

print("\nTOP 20 LONGEST INVOCATIONS")
print("-" * 60)

print(
    longest[
        ["app", "func", "start_timestamp", "end_timestamp", "duration"]
    ].to_string(index=False)
)

# Duration distribution
print("\nDURATION DISTRIBUTION")
print("-" * 60)

print(
    interval["duration"]
    .describe(percentiles=[0.50, 0.75, 0.90, 0.95, 0.99])
    .to_string()
)

# Save the selected original records for further inspection
output_file = OUTPUT_DIR / "anomalous_interval_18610.csv"
interval.to_csv(output_file, index=False)

print(f"\nOriginal records saved to: {output_file}")