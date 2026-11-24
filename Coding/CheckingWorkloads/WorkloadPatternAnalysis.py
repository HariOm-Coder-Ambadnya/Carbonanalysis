
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==================================================
# 1. PATH CONFIGURATION
# ==================================================

project_dir = Path(__file__).resolve().parents[2]

dataset_path = (
    project_dir
    / "Coding"
    / "Renewddatasets"
    / "workload_1min_complete.csv"
)

output_dir = project_dir / "Coding" / "WorkloadAnalysis"

graph_dir = output_dir / "Graphs"

output_dir.mkdir(parents=True, exist_ok=True)
graph_dir.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD DATASET
# ==================================================

df = pd.read_csv(dataset_path)

required_columns = [
    "minute",
    "invocation_count",
    "total_execution_seconds",
    "average_duration",
    "p95_duration"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

if df.empty:
    raise ValueError("Dataset is empty.")

if df["minute"].duplicated().any():
    raise ValueError("Duplicate minute values detected.")

df = df.sort_values("minute").reset_index(drop=True)

df["hour"] = df["minute"] / 60


# ==================================================
# 3. ROLLING AVERAGES
# ==================================================

# Rolling averages help us identify sustained trends
# rather than focusing only on individual spikes.

df["invocations_15min_avg"] = (
    df["invocation_count"].rolling(
        window=15,
        min_periods=1
    ).mean()
)

df["invocations_60min_avg"] = (
    df["invocation_count"].rolling(
        window=60,
        min_periods=1
    ).mean()
)

df["execution_time_15min_avg"] = (
    df["total_execution_seconds"].rolling(
        window=15,
        min_periods=1
    ).mean()
)

df["execution_time_60min_avg"] = (
    df["total_execution_seconds"].rolling(
        window=60,
        min_periods=1
    ).mean()
)

# Rolling duration averages are calculated over active minutes.
df["average_duration_60min_avg"] = (
    df["average_duration"].rolling(
        window=60,
        min_periods=15
    ).mean()
)


# ==================================================
# 4. CORRELATION ANALYSIS
# ==================================================

correlation_columns = [
    "invocation_count",
    "total_execution_seconds",
    "average_duration",
    "p95_duration"
]

correlation_data = df[correlation_columns]

pearson_correlation = correlation_data.corr(
    method="pearson"
)

spearman_correlation = correlation_data.corr(
    method="spearman"
)

pearson_correlation.to_csv(
    output_dir / "pearson_correlation.csv"
)

spearman_correlation.to_csv(
    output_dir / "spearman_correlation.csv"
)


# ==================================================
# 5. WORKLOAD CLASSIFICATION
# ==================================================

# Use active minutes to calculate demand thresholds.
# Idle minutes receive their own category.

active_counts = df.loc[
    df["invocation_count"] > 0,
    "invocation_count"
]

p50 = active_counts.quantile(0.50)
p90 = active_counts.quantile(0.90)
p99 = active_counts.quantile(0.99)


def classify_workload(count):

    if count == 0:
        return "Idle"

    elif count <= p50:
        return "Low"

    elif count <= p90:
        return "Normal"

    elif count <= p99:
        return "High"

    else:
        return "Extreme"


df["workload_category"] = (
    df["invocation_count"].apply(classify_workload)
)


# ==================================================
# 6. WORKLOAD CATEGORY SUMMARY
# ==================================================

category_summary = (
    df.groupby("workload_category")
    .agg(
        interval_count=("minute", "count"),
        average_invocations=("invocation_count", "mean"),
        maximum_invocations=("invocation_count", "max"),
        average_execution_time=(
            "total_execution_seconds", "mean"
        )
    )
    .reset_index()
)

category_summary["percentage"] = (
    category_summary["interval_count"] / len(df) * 100
)

category_summary.to_csv(
    output_dir / "workload_category_summary.csv",
    index=False
)


# ==================================================
# 7. PEAK WORKLOAD INTERVALS
# ==================================================

top_20_peaks = (
    df.nlargest(20, "invocation_count")[
        [
            "minute",
            "hour",
            "invocation_count",
            "total_execution_seconds",
            "average_duration",
            "p95_duration",
            "workload_category"
        ]
    ]
    .reset_index(drop=True)
)

top_20_peaks.to_csv(
    output_dir / "top_20_peak_intervals.csv",
    index=False
)


# ==================================================
# 8. GRAPH 1 - INVOCATION ROLLING AVERAGES
# ==================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["hour"],
    df["invocation_count"],
    alpha=0.25,
    label="Original invocations"
)

plt.plot(
    df["hour"],
    df["invocations_15min_avg"],
    label="15-minute rolling average"
)

plt.plot(
    df["hour"],
    df["invocations_60min_avg"],
    linewidth=2,
    label="60-minute rolling average"
)

plt.title("Workload Trends: Invocation Count")
plt.xlabel("Time (hours)")
plt.ylabel("Invocations per minute")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    graph_dir / "invocation_rolling_average.png",
    dpi=300
)

plt.close()


# ==================================================
# 9. GRAPH 2 - EXECUTION TIME ROLLING AVERAGES
# ==================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["hour"],
    df["total_execution_seconds"],
    alpha=0.25,
    label="Original execution time"
)

plt.plot(
    df["hour"],
    df["execution_time_15min_avg"],
    label="15-minute rolling average"
)

plt.plot(
    df["hour"],
    df["execution_time_60min_avg"],
    linewidth=2,
    label="60-minute rolling average"
)

plt.title("Execution Workload Trends")
plt.xlabel("Time (hours)")
plt.ylabel("Total execution seconds per minute")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    graph_dir / "execution_time_rolling_average.png",
    dpi=300
)

plt.close()


# ==================================================
# 10. GRAPH 3 - INVOCATIONS VS EXECUTION TIME
# ==================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["invocation_count"],
    df["total_execution_seconds"],
    alpha=0.35
)

plt.title("Invocation Count vs Total Execution Time")
plt.xlabel("Invocations per minute")
plt.ylabel("Total execution seconds per minute")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    graph_dir / "invocations_vs_execution_time.png",
    dpi=300
)

plt.close()


# ==================================================
# 11. GRAPH 4 - WORKLOAD CATEGORY DISTRIBUTION
# ==================================================

category_order = ["Idle", "Low", "Normal", "High", "Extreme"]

category_counts = (
    df["workload_category"]
    .value_counts()
    .reindex(category_order, fill_value=0)
)

plt.figure(figsize=(10, 6))

plt.bar(
    category_counts.index,
    category_counts.values
)

plt.title("Workload Category Distribution")
plt.xlabel("Workload Category")
plt.ylabel("Number of Minutes")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    graph_dir / "workload_category_distribution.png",
    dpi=300
)

plt.close()


# ==================================================
# 12. DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 60)
print("WORKLOAD PATTERN ANALYSIS")
print("=" * 60)

print("\nWORKLOAD THRESHOLDS")
print("-" * 60)

print(f"P50 threshold: {p50:.2f}")
print(f"P90 threshold: {p90:.2f}")
print(f"P99 threshold: {p99:.2f}")

print("\nWORKLOAD CATEGORY SUMMARY")
print("-" * 60)

print(category_summary.to_string(index=False))

print("\nPEARSON CORRELATION")
print("-" * 60)

print(pearson_correlation.round(4))

print("\nSPEARMAN CORRELATION")
print("-" * 60)

print(spearman_correlation.round(4))

print("\nTOP 20 PEAK WORKLOAD INTERVALS")
print("-" * 60)

print(top_20_peaks.head(10).to_string(index=False))


# ==================================================
# 13. SAVE FINAL DATASET
# ==================================================

df.to_csv(
    output_dir / "workload_pattern_analysis.csv",
    index=False
)

print("\n" + "=" * 60)
print("WORKLOAD PATTERN ANALYSIS COMPLETED")
print("=" * 60)

print(f"Results directory: {output_dir}")
print(f"Graphs directory: {graph_dir}")
