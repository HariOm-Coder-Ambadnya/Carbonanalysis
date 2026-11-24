
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]

WORKLOAD_FILE = (
    PROJECT_DIR
    / "Coding"
    / "CarbonAnalysis"
    / "workload_timeline.csv"
)

CARBON_FILE = (
    PROJECT_DIR
    / "Coding"
    / "CarbonAnalysis"
    / "carbon_intensity_2021_window.csv"
)

OUTPUT_DIR = PROJECT_DIR / "Coding" / "CarbonAnalysis"

OUTPUT_FILE = (
    OUTPUT_DIR / "workload_carbon_dataset.csv"
)

# --------------------------------------------------
# 2. LOAD DATASETS
# --------------------------------------------------

print("Loading workload and carbon datasets...")

workload = pd.read_csv(WORKLOAD_FILE)

carbon = pd.read_csv(CARBON_FILE)

print("Workload shape:", workload.shape)
print("Carbon shape:", carbon.shape)

# --------------------------------------------------
# 3. CONVERT TIMESTAMPS TO UTC
# --------------------------------------------------

workload["timestamp_utc"] = pd.to_datetime(
    workload["timestamp_utc"],
    utc=True
)

carbon["timestamp_utc"] = pd.to_datetime(
    carbon["timestamp_utc"],
    utc=True
)

# --------------------------------------------------
# 4. CREATE HOURLY JOIN KEY
# --------------------------------------------------

# Every workload minute is assigned to its
# corresponding UTC hour.

workload["hour_timestamp_utc"] = (
    workload["timestamp_utc"].dt.floor("h")
)

# Carbon observations are already hourly.
carbon["hour_timestamp_utc"] = (
    carbon["timestamp_utc"]
)

# --------------------------------------------------
# 5. VALIDATE CARBON HOURLY KEYS
# --------------------------------------------------

assert carbon["hour_timestamp_utc"].is_unique, (
    "Duplicate hourly carbon timestamps found."
)

assert len(carbon) == 336, (
    "Expected exactly 336 hourly carbon observations."
)

# --------------------------------------------------
# 6. MERGE DATASETS
# --------------------------------------------------

merged = workload.merge(
    carbon.drop(columns=["timestamp_utc"]),
    on="hour_timestamp_utc",
    how="left",
    validate="many_to_one",
    indicator=True
)

# --------------------------------------------------
# 7. VALIDATE MERGE
# --------------------------------------------------

print("\n" + "=" * 60)
print("WORKLOAD + CARBON DATASET VALIDATION")
print("=" * 60)

print("Merged rows:", len(merged))
print("Expected rows:", 20160)

print("\nMerge status:")
print(merged["_merge"].value_counts())

unmatched = (merged["_merge"] != "both").sum()

print("\nUnmatched workload minutes:", unmatched)

print(
    "Missing lifecycle carbon values:",
    merged["carbon_intensity_lifecycle"].isna().sum()
)

print(
    "Missing direct carbon values:",
    merged["carbon_intensity_direct"].isna().sum()
)

print(
    "Unique carbon hours:",
    merged["hour_timestamp_utc"].nunique()
)

# --------------------------------------------------
# 8. STRICT VALIDATION
# --------------------------------------------------

assert len(merged) == 20160

assert unmatched == 0

assert merged["carbon_intensity_lifecycle"].notna().all()

assert merged["carbon_intensity_direct"].notna().all()

assert merged["hour_timestamp_utc"].nunique() == 336

assert merged["minute"].is_unique

# --------------------------------------------------
# 9. CLEAN UP MERGE INDICATOR
# --------------------------------------------------

merged.drop(columns=["_merge"], inplace=True)

# --------------------------------------------------
# 10. SAVE FINAL DATASET
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

merged.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print("FINAL WORKLOAD-CARBON DATASET CREATED")
print("=" * 60)

print("Rows:", len(merged))
print("Columns:", len(merged.columns))

print("\nFinal columns:")
for column in merged.columns:
    print("-", column)

print("\nFirst five records:")
print(merged.head().to_string(index=False))

print("\nSaved to:")
print(OUTPUT_FILE)
