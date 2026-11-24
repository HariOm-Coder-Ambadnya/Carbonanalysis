import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_DIR
    / "Coding"
    / "CarbonAnalysis"
    / "snapshots_2026-02-10_IN-WE-2021-hourly.csv"
)

OUTPUT_DIR = PROJECT_DIR / "Coding" / "CarbonAnalysis"

OUTPUT_FILE = (
    OUTPUT_DIR / "carbon_intensity_2021_window.csv"
)

# --------------------------------------------------
# 2. LOAD ORIGINAL DATASET
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Original dataset rows:", len(df))

# --------------------------------------------------
# 3. CONVERT TIMESTAMP TO UTC
# --------------------------------------------------

df["timestamp_utc"] = pd.to_datetime(
    df["Datetime (UTC)"],
    utc=True
)

# --------------------------------------------------
# 4. DEFINE EXPERIMENT PERIOD
# --------------------------------------------------

START_DATE = pd.Timestamp("2021-01-31 00:00:00", tz="UTC")

END_DATE = pd.Timestamp("2021-02-14 00:00:00", tz="UTC")

# --------------------------------------------------
# 5. FILTER REQUIRED OBSERVATIONS
# --------------------------------------------------

filtered = df[
    (df["timestamp_utc"] >= START_DATE)
    & (df["timestamp_utc"] < END_DATE)
].copy()

# --------------------------------------------------
# 6. SELECT REQUIRED COLUMNS
# --------------------------------------------------

filtered = filtered[
    [
        "timestamp_utc",
        "Zone name",
        "Zone id",
        "Carbon intensity gCO₂eq/kWh (direct)",
        "Carbon intensity gCO₂eq/kWh (Life cycle)",
        "Data source",
        "Data estimated",
        "Data estimation method"
    ]
]

filtered = filtered.rename(
    columns={
        "Zone name": "zone_name",
        "Zone id": "zone_id",
        "Carbon intensity gCO₂eq/kWh (direct)": "carbon_intensity_direct",
        "Carbon intensity gCO₂eq/kWh (Life cycle)": "carbon_intensity_lifecycle",
        "Data source": "data_source",
        "Data estimated": "data_estimated",
        "Data estimation method": "data_estimation_method"
    }
)

# --------------------------------------------------
# 7. VALIDATION
# --------------------------------------------------

print("\n" + "=" * 55)
print("CARBON DATASET VALIDATION")
print("=" * 55)

print("Filtered rows:", len(filtered))
print("Expected rows: 336")

print("First timestamp:", filtered["timestamp_utc"].min())
print("Last timestamp:", filtered["timestamp_utc"].max())

print("Unique zones:", filtered["zone_id"].unique())

print(
    "Missing values:",
    filtered.isnull().sum().sum()
)

print(
    "Duplicate timestamps:",
    filtered["timestamp_utc"].duplicated().sum()
)

print(
    "Estimated observations:",
    filtered["data_estimated"].sum()
)

# Do not silently continue if the expected observations are missing.
assert len(filtered) == 336, (
    f"Expected 336 hourly observations, got {len(filtered)}"
)

assert filtered["timestamp_utc"].is_unique

assert filtered["zone_id"].nunique() == 1

assert filtered["zone_id"].iloc[0] == "IN-WE"

assert not filtered.isnull().any().any()

# --------------------------------------------------
# 8. SAVE PROCESSED DATASET
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

filtered.to_csv(OUTPUT_FILE, index=False)

print("\nCarbon intensity summary (lifecycle):")

print(
    filtered["carbon_intensity_lifecycle"].describe()
)

print("\nCarbon intensity summary (direct):")

print(
    filtered["carbon_intensity_direct"].describe()
)

print("\n" + "=" * 55)
print("CARBON DATASET CREATED SUCCESSFULLY")
print("=" * 55)

print("Saved to:", OUTPUT_FILE)
