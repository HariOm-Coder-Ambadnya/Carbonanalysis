
import pandas as pd

file_path = r"C:\Users\rishi\Documents\Cloud_G45_Project\AzureFunctionsInvocationTraceForTwoWeeksJan2021\AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"

df = pd.read_csv(file_path)

# Separate duration categories
zero_duration = df[df["duration"] == 0]
negative_duration = df[df["duration"] < 0]
positive_duration = df[df["duration"] > 0]

print("========== DURATION INVESTIGATION ==========")

print("Zero duration:", len(zero_duration))
print("Negative duration:", len(negative_duration))
print("Positive duration:", len(positive_duration))

print("\n========== ZERO DURATION EXAMPLES ==========")
print(zero_duration.head(10))

print("\n========== NEGATIVE DURATION EXAMPLES ==========")
print(negative_duration.head(10))

print("\n========== DUPLICATE INVESTIGATION ==========")

duplicates = df[df.duplicated(keep=False)]

print("Rows involved in duplicate groups:", len(duplicates))
print(duplicates.head(10))

print("\n========== POSITIVE DURATION STATISTICS ==========")
print(positive_duration["duration"].describe())

# Check calculated invocation start timestamps
df["start_timestamp"] = (
    df["end_timestamp"] - df["duration"]
)

print("\n========== START TIMESTAMP INVESTIGATION ==========")
print("Negative start timestamps:",
      (df["start_timestamp"] < 0).sum())

print("\nInvestigation completed.")
