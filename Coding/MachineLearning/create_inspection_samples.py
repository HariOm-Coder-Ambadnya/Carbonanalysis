import csv
from pathlib import Path

results_dir = Path("Coding/LoadBalancing/results")
output_dir = Path("Coding/MachineLearning/inspection_samples")
output_dir.mkdir(parents=True, exist_ok=True)

for csv_file in results_dir.rglob("*.csv"):
    if csv_file.stat().st_size < 50 * 1024 * 1024:
        continue

    relative = csv_file.relative_to(results_dir)
    sample_file = output_dir / (str(relative).replace("\\", "__").replace("/", "__"))

    with csv_file.open("r", newline="", encoding="utf-8-sig", errors="replace") as source:
        reader = csv.reader(source)

        try:
            header = next(reader)
        except StopIteration:
            continue

        with sample_file.open("w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target)
            writer.writerow(header)

            for _, row in zip(range(100), reader):
                writer.writerow(row)

    print(f"Sample created: {sample_file}")

print("\nInspection samples generated.")
