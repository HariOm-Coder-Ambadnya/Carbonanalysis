
import pandas as pd
import numpy as np
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]

file_path = (
    project_dir
    / "Coding"
    / "Renewddatasets"
    / "workload_1min_complete.csv"
)

output_dir = project_dir / "Coding" / "Renewddatasets"
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(file_path)

print("=" * 60)
print("CARBON AND COST-AWARE SERVERLESS LOAD BALANCING")
print("WORKLOAD STATISTICAL ANALYSIS")
print("=" * 60)


# --------------------------------------------------
# 2. DATA VALIDATION
# --------------------------------------------------

required_columns = [
    "minute",
    "invocation_count",
    "total_execution_seconds",
    "average_duration",
    "p95_duration"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

if df.empty:
    raise ValueError("Dataset is empty.")

if df["invocation_count"].isna().any():
    raise ValueError("Invocation count contains missing values.")

if (df["invocation_count"] < 0).any():
    raise ValueError("Negative invocation counts detected.")

if df["minute"].duplicated().any():
    raise ValueError("Duplicate minute values detected.")


# --------------------------------------------------
# 3. INVOCATION COUNT STATISTICS
# --------------------------------------------------

invocations = df["invocation_count"]

total_minutes = len(df)
active_minutes = int((invocations > 0).sum())
idle_minutes = int((invocations == 0).sum())

mean_invocations = invocations.mean()
std_invocations = invocations.std()

coefficient_variation = (
    std_invocations / mean_invocations
    if mean_invocations != 0
    else np.nan
)

invocation_stats = {
    "Total intervals": total_minutes,
    "Active intervals": active_minutes,
    "Idle intervals": idle_minutes,
    "Idle percentage": (idle_minutes / total_minutes) * 100,
    "Total invocations": int(invocations.sum()),
    "Minimum invocations per minute": int(invocations.min()),
    "Maximum invocations per minute": int(invocations.max()),
    "Mean invocations per minute": mean_invocations,
    "Median invocations per minute": invocations.median(),
    "95th percentile invocations": invocations.quantile(0.95),
    "Standard deviation": std_invocations,
    "Coefficient of variation": coefficient_variation
}


# --------------------------------------------------
# 4. EXECUTION DURATION STATISTICS
# --------------------------------------------------

# Ignore empty minutes because they have no execution duration.

active_df = df[df["invocation_count"] > 0].copy()

average_durations = active_df["average_duration"].dropna()
p95_durations = active_df["p95_duration"].dropna()

# Weighted average across all invocations.
# This is more accurate than averaging minute-level averages.

total_execution_time = active_df["total_execution_seconds"].sum()
total_invocations = active_df["invocation_count"].sum()

overall_weighted_average = (
    total_execution_time / total_invocations
    if total_invocations > 0
    else np.nan
)

duration_stats = {
    "Total execution time (seconds)": total_execution_time,
    "Overall weighted average duration (seconds)": overall_weighted_average,
    "Mean of minute-level average durations (seconds)": average_durations.mean(),
    "Median of minute-level average durations (seconds)": average_durations.median(),
    "Maximum minute-level average duration (seconds)": average_durations.max(),
    "Mean of minute-level P95 durations (seconds)": p95_durations.mean(),
    "Maximum minute-level P95 duration (seconds)": p95_durations.max(),
    "95th percentile of minute-level P95 durations (seconds)": p95_durations.quantile(0.95)
}


# --------------------------------------------------
# 5. PEAK WORKLOAD ANALYSIS
# --------------------------------------------------

peak_index = df["invocation_count"].idxmax()
peak_row = df.loc[peak_index]

peak_minute = int(peak_row["minute"])
peak_hour = peak_minute / 60

peak_stats = {
    "Peak workload minute": peak_minute,
    "Peak workload hour": peak_hour,
    "Peak invocation count": int(peak_row["invocation_count"]),
    "Execution time during peak minute (seconds)": peak_row["total_execution_seconds"]
}


# --------------------------------------------------
# 6. DISPLAY RESULTS
# --------------------------------------------------

def display_statistics(title, statistics):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for key, value in statistics.items():

        if isinstance(value, (float, np.floating)):
            print(f"{key}: {value:,.4f}")
        else:
            print(f"{key}: {value}")


display_statistics("INVOCATION COUNT STATISTICS", invocation_stats)

display_statistics("EXECUTION DURATION STATISTICS", duration_stats)

display_statistics("PEAK WORKLOAD STATISTICS", peak_stats)


# --------------------------------------------------
# 7. SAVE RESULTS
# --------------------------------------------------

all_statistics = {
    **invocation_stats,
    **duration_stats,
    **peak_stats
}

results_df = pd.DataFrame(
    list(all_statistics.items()),
    columns=["Metric", "Value"]
)

output_file = output_dir / "workload_statistical_analysis.csv"

results_df.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)

print(f"Results saved to:\n{output_file}")
