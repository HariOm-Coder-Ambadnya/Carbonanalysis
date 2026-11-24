
import pandas as pd

# 1. Load original dataset
file_path = r"C:\Users\rishi\Documents\Cloud_G45_Project\AzureFunctionsInvocationTraceForTwoWeeksJan2021\AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"

df = pd.read_csv(file_path)

print("Original records:", len(df))

# 2. Ensure numeric columns
df["end_timestamp"] = pd.to_numeric(df["end_timestamp"])
df["duration"] = pd.to_numeric(df["duration"])

# 3. Calculate invocation start time
df["start_timestamp"] = (
    df["end_timestamp"] - df["duration"]
)

# 4. Assign each invocation to a one-minute interval
df["minute"] = (
    df["start_timestamp"] // 60
).astype("int64")


# 5. Aggregate workload information
def create_workload_dataset(data, output_file):

    workload = data.groupby("minute").agg(
        invocation_count=("func", "count"),

        total_execution_seconds=("duration", "sum"),

        average_duration=("duration", "mean"),

        median_duration=("duration", "median"),

        p95_duration=(
            "duration",
            lambda x: x.quantile(0.95)
        ),

        zero_duration_count=(
            "duration",
            lambda x: (x == 0).sum()
        ),

        unique_applications=("app", "nunique"),

        unique_functions=("func", "nunique")
    ).reset_index()

    workload.to_csv(output_file, index=False)

    print("\nSaved:", output_file)
    print("Total one-minute intervals:", len(workload))
    print("\nWorkload preview:")
    print(workload.head(10))

    print("\nWorkload statistics:")
    print(workload.describe())

    return workload


# 6. Baseline: Keep all original invocation records
baseline = create_workload_dataset(
    df,
    r"C:\Users\rishi\Documents\Cloud_G45_Project\Coding\Renewddatasets\workload_1min_baseline.csv"
)


# 7. Comparison: Remove exact duplicate rows
deduplicated_df = df.drop_duplicates(
    subset=[
        "app",
        "func",
        "end_timestamp",
        "duration"
    ]
)

print(
    "\nRecords after removing exact duplicates:",
    len(deduplicated_df)
)

deduplicated = create_workload_dataset(
    deduplicated_df,
    r"C:\Users\rishi\Documents\Cloud_G45_Project\Coding\Renewddatasets\workload_1min_deduplicated.csv"
)

print("\nWorkload aggregation completed.")
