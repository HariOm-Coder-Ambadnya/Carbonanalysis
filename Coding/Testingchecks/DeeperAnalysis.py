
import pandas as pd

file_path = r"C:\Users\rishi\Documents\Cloud_G45_Project\AzureFunctionsInvocationTraceForTwoWeeksJan2021\AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"

df = pd.read_csv(file_path)

# Convert columns to numeric
df["end_timestamp"] = pd.to_numeric(df["end_timestamp"])
df["duration"] = pd.to_numeric(df["duration"])

# Basic statistics
print("\n========== DATASET INFORMATION ==========")
print("Total records:", len(df))
print("Unique applications:", df["app"].nunique())
print("Unique functions:", df.groupby("app")["func"].nunique().sum())

# Duration statistics
print("\n========== EXECUTION DURATION ==========")
print(df["duration"].describe())

# Timestamp range
print("\n========== TIMESTAMP RANGE ==========")
print("Start:", df["end_timestamp"].min())
print("End:", df["end_timestamp"].max())
print("Time coverage (hours):",
      (df["end_timestamp"].max() -
       df["end_timestamp"].min()) / 3600)

# Invalid values
print("\n========== DATA QUALITY ==========")
print("Duplicate rows:", df.duplicated().sum())
print("Zero/negative durations:",
      (df["duration"] <= 0).sum())
print("Negative timestamps:",
      (df["end_timestamp"] < 0).sum())

# Calculate invocation start time
df["start_timestamp"] = (
    df["end_timestamp"] - df["duration"]
)

print("\n========== SAMPLE WITH START TIME ==========")
print(df.head())

print("\nAnalysis completed!")
