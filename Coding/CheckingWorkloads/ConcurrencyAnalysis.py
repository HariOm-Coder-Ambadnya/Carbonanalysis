
import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "AzureFunctionsInvocationTraceForTwoWeeksJan2021"
    / "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
)

OUTPUT_DIR = BASE_DIR / "Coding" / "ConcurrencyAnalysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MINUTES = 20160
SECONDS_PER_MINUTE = 60
TOTAL_SECONDS = MINUTES * SECONDS_PER_MINUTE

print("=" * 60)
print("SERVERLESS CONCURRENCY ANALYSIS")
print("=" * 60)

# --------------------------------------------------
# 2. LOAD ORIGINAL INVOCATION DATA
# --------------------------------------------------

print("\nLoading original Azure Functions trace...")

df = pd.read_csv(INPUT_FILE)

print(f"Original records: {len(df):,}")

# Derive invocation start timestamps
df["start_timestamp"] = df["end_timestamp"] - df["duration"]

# Clip invocation intervals to the observation window
starts = df["start_timestamp"].to_numpy(dtype=np.float64)
ends = df["end_timestamp"].to_numpy(dtype=np.float64)

starts = np.maximum(starts, 0)
ends = np.minimum(ends, TOTAL_SECONDS)

# Ignore intervals with no positive execution duration
valid = ends > starts

starts = starts[valid]
ends = ends[valid]

print(f"Valid positive-duration invocations: {len(starts):,}")

# --------------------------------------------------
# 3. CREATE START AND END EVENTS
# --------------------------------------------------

print("\nCreating invocation events...")

# Invocation start = +1
# Invocation end   = -1

event_times = np.concatenate([
    starts,
    ends,
    np.array([0.0, float(TOTAL_SECONDS)])
])

event_deltas = np.concatenate([
    np.ones(len(starts), dtype=np.int64),
    -np.ones(len(ends), dtype=np.int64),
    np.array([0, 0], dtype=np.int64)
])

# Sort events by timestamp
order = np.argsort(event_times, kind="stable")

event_times = event_times[order]
event_deltas = event_deltas[order]

# Combine events occurring at exactly the same timestamp
unique_times, first_indices = np.unique(
    event_times,
    return_index=True
)

combined_deltas = np.add.reduceat(
    event_deltas,
    first_indices
)

event_times = unique_times
event_deltas = combined_deltas

# Running concurrency immediately after each event timestamp
concurrency = np.cumsum(event_deltas)

print(f"Unique event timestamps: {len(event_times):,}")

if concurrency.min() < 0:
    raise ValueError("Negative concurrency detected. Check event processing.")

if concurrency[-1] != 0:
    raise ValueError("Final concurrency is not zero. Check event boundaries.")

# --------------------------------------------------
# 4. CALCULATE CUMULATIVE CONCURRENCY-TIME
# --------------------------------------------------

# Between two consecutive event timestamps, concurrency
# remains constant.

time_differences = np.diff(event_times)

area_segments = (
    time_differences * concurrency[:-1]
)

# Prefix integral:
# prefix[i] = total concurrency-seconds before event i

prefix_integral = np.zeros(
    len(event_times),
    dtype=np.float64
)

prefix_integral[1:] = np.cumsum(area_segments)


def cumulative_concurrency_time(timestamp):
    """
    Calculate the integral of concurrency from time zero
    up to the supplied timestamp.
    """

    timestamp = np.asarray(timestamp, dtype=np.float64)

    index = (
        np.searchsorted(event_times, timestamp, side="right") - 1
    )

    index = np.maximum(index, 0)

    return (
        prefix_integral[index]
        + (timestamp - event_times[index]) * concurrency[index]
    )


# --------------------------------------------------
# 5. CALCULATE PER-MINUTE CONCURRENCY
# --------------------------------------------------

print("\nCalculating minute-level concurrency...")

minute_starts = (
    np.arange(MINUTES, dtype=np.float64)
    * SECONDS_PER_MINUTE
)

minute_ends = minute_starts + SECONDS_PER_MINUTE

# Time-weighted average concurrency
integral_starts = cumulative_concurrency_time(minute_starts)
integral_ends = cumulative_concurrency_time(minute_ends)

average_concurrency = (
    integral_ends - integral_starts
) / SECONDS_PER_MINUTE

# Maximum concurrency during each minute
maximum_concurrency = np.zeros(MINUTES, dtype=np.int64)

for minute in range(MINUTES):

    start = minute_starts[minute]
    end = minute_ends[minute]

    # Include concurrency already active at the minute boundary
    left = (
        np.searchsorted(event_times, start, side="right") - 1
    )

    # Events strictly before the next minute boundary
    right = np.searchsorted(
        event_times,
        end,
        side="left"
    )

    maximum_concurrency[minute] = int(
        concurrency[left:right].max()
    )

# --------------------------------------------------
# 6. SAVE MINUTE-LEVEL RESULTS
# --------------------------------------------------

result = pd.DataFrame({
    "minute": np.arange(MINUTES),
    "hour": np.arange(MINUTES) / 60,
    "average_concurrency": average_concurrency,
    "maximum_concurrency": maximum_concurrency
})

output_file = OUTPUT_DIR / "concurrency_1min.csv"

result.to_csv(output_file, index=False)

# --------------------------------------------------
# 7. OVERALL STATISTICS
# --------------------------------------------------

print("\n" + "=" * 60)
print("OVERALL CONCURRENCY STATISTICS")
print("=" * 60)

print(f"Total minutes: {MINUTES:,}")
print(f"Maximum concurrency: {concurrency.max():,}")
print(
    f"Overall time-weighted average concurrency: "
    f"{prefix_integral[-1] / TOTAL_SECONDS:.4f}"
)

print(
    f"Mean of minute-level maximum concurrency: "
    f"{result['maximum_concurrency'].mean():.4f}"
)

print(
    f"P95 of minute-level maximum concurrency: "
    f"{result['maximum_concurrency'].quantile(0.95):.4f}"
)

print(
    f"P99 of minute-level maximum concurrency: "
    f"{result['maximum_concurrency'].quantile(0.99):.4f}"
)

# --------------------------------------------------
# 8. INVESTIGATE MINUTE 18610
# --------------------------------------------------

target_minute = 18610

target = result.loc[
    result["minute"] == target_minute
].iloc[0]

print("\n" + "=" * 60)
print("CONCURRENCY AT MINUTE 18610")
print("=" * 60)

print(f"Average concurrency: {target['average_concurrency']:.4f}")
print(f"Maximum concurrency: {target['maximum_concurrency']}")

# --------------------------------------------------
# 9. TOP 10 PEAK CONCURRENCY MINUTES
# --------------------------------------------------

print("\nTOP 10 PEAK CONCURRENCY MINUTES")
print("-" * 60)

print(
    result.nlargest(10, "maximum_concurrency")
    .to_string(index=False)
)

# --------------------------------------------------
# 10. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 60)
print("CONCURRENCY ANALYSIS COMPLETED")
print("=" * 60)

print(f"Results saved to: {output_file}")