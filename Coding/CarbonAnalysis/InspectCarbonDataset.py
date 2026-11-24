
import pandas as pd
from pathlib import Path

# Locate the downloaded dataset
PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_DIR
    / "Coding"
    / "CarbonAnalysis"
    / "snapshots_2026-02-10_IN-WE-2021-hourly.csv"
)

# Load CSV
df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("ELECTRICITY MAPS DATASET INSPECTION")
print("=" * 60)

# Basic information
print("\n1. DATASET SHAPE")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n2. COLUMN NAMES")
print(df.columns.tolist())

print("\n3. DATA TYPES")
print(df.dtypes)

print("\n4. FIRST 5 ROWS")
print(df.head().to_string())

print("\n5. LAST 5 ROWS")
print(df.tail().to_string())

print("\n6. MISSING VALUES")
print(df.isnull().sum())

print("\n7. DUPLICATE ROWS")
print(df.duplicated().sum())

print("\n8. DATASET SIZE")
print(f"{INPUT_FILE.stat().st_size / (1024 * 1024):.2f} MB")

print("\nInspection completed.")
