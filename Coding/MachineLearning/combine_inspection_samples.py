import pandas as pd
from pathlib import Path

input_dir = Path("Coding/MachineLearning/inspection_samples")
output_file = Path("Coding/MachineLearning/combined_inspection_samples.csv")

csv_files = sorted(input_dir.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No sample CSV files found.")

frames = []

for file in csv_files:
    print(f"Reading: {file.name}")

    df = pd.read_csv(file)
    df["source_file"] = file.name

    frames.append(df)

combined = pd.concat(frames, ignore_index=True, sort=False)

combined.to_csv(output_file, index=False)

print("\nCombination completed!")
print("Files combined:", len(csv_files))
print("Total rows:", len(combined))
print("Total columns:", len(combined.columns))
print("Output:", output_file)
