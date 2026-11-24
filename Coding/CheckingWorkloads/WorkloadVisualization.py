import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. LOAD COMPLETE WORKLOAD DATASET
# ==========================================

file_path = r"C:\Users\rishi\Documents\Cloud_G45_Project\Coding\Renewddatasets\workload_1min_complete.csv"

df = pd.read_csv(file_path)

print("========== WORKLOAD VISUALIZATION ==========")
print("Total records:", len(df))


# ==========================================
# 2. PREPARE TIME COLUMN
# ==========================================

# Convert minute index into elapsed hours
df["hours"] = df["minute"] / 60


# ==========================================
# 3. CREATE OUTPUT DIRECTORY
# ==========================================

output_dir = Path(
    r"C:\Users\rishi\Documents\Cloud_G45_Project\Coding\WorkloadGraphs"
)

output_dir.mkdir(parents=True, exist_ok=True)


# ==========================================
# 4. PLOT INVOCATION COUNT
# ==========================================

plt.figure(figsize=(14, 5))

plt.plot(
    df["hours"],
    df["invocation_count"],
    linewidth=0.8
)

plt.title(
    "Azure Functions Invocation Count Over Time",
    fontsize=15
)

plt.xlabel("Time (hours)")
plt.ylabel("Number of Invocations per Minute")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    output_dir / "invocation_count.png",
    dpi=300
)

plt.show()


# ==========================================
# 5. PLOT AVERAGE EXECUTION DURATION
# ==========================================

plt.figure(figsize=(14, 5))

plt.plot(
    df["hours"],
    df["average_duration"],
    linewidth=0.8
)

plt.title(
    "Average Function Execution Duration Over Time",
    fontsize=15
)

plt.xlabel("Time (hours)")
plt.ylabel("Average Duration (seconds)")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    output_dir / "average_duration.png",
    dpi=300
)

plt.show()


# ==========================================
# 6. PLOT P95 EXECUTION DURATION
# ==========================================

plt.figure(figsize=(14, 5))

plt.plot(
    df["hours"],
    df["p95_duration"],
    linewidth=0.8
)

plt.title(
    "P95 Function Execution Duration Over Time",
    fontsize=15
)

plt.xlabel("Time (hours)")
plt.ylabel("P95 Duration (seconds)")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    output_dir / "p95_duration.png",
    dpi=300
)

plt.show()


# ==========================================
# 7. DISPLAY SUMMARY
# ==========================================

print("\n========== VISUALIZATION SUMMARY ==========")

print("Total intervals:", len(df))

print(
    "Maximum invocations in one minute:",
    df["invocation_count"].max()
)

print(
    "Minimum invocations in one minute:",
    df["invocation_count"].min()
)

print(
    "Average invocations per minute:",
    round(df["invocation_count"].mean(), 2)
)

print(
    "Peak workload minute:",
    df.loc[
        df["invocation_count"].idxmax(),
        "minute"
    ]
)

print("\nGraphs saved in:")
print(output_dir)

print("\nWorkload visualization completed successfully.")
