
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_DIR
    / "Coding"
    / "FinalWorkloadDataset"
    / "final_workload_dataset.csv"
)

OUTPUT_DIR = PROJECT_DIR / "Coding" / "CarbonAnalysis"
OUTPUT_FILE = OUTPUT_DIR / "workload_timeline.csv"

# Documented dataset start date.
# UTC midnight is a provisional alignment assumption.
TRACE_START = pd.Timestamp("2021-01-31 00:00:00", tz="UTC")

# Load the final workload dataset
df = pd.read_csv(INPUT_FILE)

# Validate expected number of intervals
assert len(df) == 20160, "Unexpected number of workload intervals"
assert df["minute"].is_unique, "Duplicate minute indexes found"

# Reconstruct the timeline.
# Each row represents one minute from the trace start.
df["relative_time_seconds"] = df["minute"] * 60

df["timestamp_utc"] = (
    TRACE_START
    + pd.to_timedelta(df["relative_time_seconds"], unit="s")
)

# Derive the corresponding India Standard Time.
df["timestamp_ist"] = df["timestamp_utc"].dt.tz_convert(
    "Asia/Kolkata"
)

# Save a separate timeline dataset
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("=" * 55)
print("WORKLOAD TIMELINE CREATED")
print("=" * 55)

print("Rows:", len(df))
print("First timestamp UTC:", df["timestamp_utc"].min())
print("Last timestamp UTC:", df["timestamp_utc"].max())
print("First timestamp IST:", df["timestamp_ist"].min())
print("Last timestamp IST:", df["timestamp_ist"].max())

print("\nSaved to:")
print(OUTPUT_FILE)
