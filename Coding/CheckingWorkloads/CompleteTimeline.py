
import pandas as pd

file_path = r"C:\Users\rishi\Documents\Cloud_G45_Project\Coding\Renewddatasets\workload_1min_baseline.csv"

df = pd.read_csv(file_path)

# Create a complete timeline of 20,160 minutes
full_timeline = pd.DataFrame({
    "minute": range(20160)
})

# Add the existing workload data
complete_df = full_timeline.merge(
    df,
    on="minute",
    how="left"
)

# Columns that should be zero when no invocation occurred
zero_columns = [
    "invocation_count",
    "total_execution_seconds",
    "zero_duration_count",
    "unique_applications",
    "unique_functions"
]

# Replace missing values with zero
complete_df[zero_columns] = (
    complete_df[zero_columns].fillna(0)
)

# Convert count and index columns to integers
integer_columns = [
    "minute",
    "invocation_count",
    "zero_duration_count",
    "unique_applications",
    "unique_functions"
]

complete_df[integer_columns] = (
    complete_df[integer_columns].astype("int64")
)

# Keep average, median and P95 as NaN for empty intervals

output_file = r"C:\Users\rishi\Documents\Cloud_G45_Project\Coding\Renewddatasets\workload_1min_complete.csv"

complete_df.to_csv(output_file, index=False)

print("========== COMPLETE TIMELINE ==========")
print("Total intervals:", len(complete_df))

print(
    "Intervals with invocations:",
    (complete_df["invocation_count"] > 0).sum()
)

print(
    "Intervals without invocations:",
    (complete_df["invocation_count"] == 0).sum()
)

print("\nColumn data types:")
print(complete_df.dtypes)

print("\nMissing values:")
print(complete_df.isnull().sum())

print("\nFirst 10 rows:")
print(complete_df.head(10))

print("\nSaved:", output_file)
