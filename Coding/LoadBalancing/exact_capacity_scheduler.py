
from pathlib import Path
import math
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "Coding" / "LoadBalancing"
    / "results" / "classified_workload.csv"
)

CARBON_FILE = (
    ROOT / "Coding" / "CarbonAnalysis"
    / "workload_carbon_dataset.csv"
)

OUTPUT_DIR = (
    ROOT / "Coding" / "LoadBalancing"
    / "results" / "exact_capacity"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEADLINES = [15, 30, 60]

MAX_CONCURRENCY = 3027
SLOT_SECONDS = 1


# ============================================================
# RANGE MAX / RANGE ADD SEGMENT TREE
# ============================================================


class CapacityTree:

    def __init__(self, size):

        self.n = max(1, int(size))

        self.base = 1
        while self.base < self.n:
            self.base *= 2

        self.maximum = np.zeros(
            2 * self.base, dtype=np.int32
        )

        self.lazy = np.zeros(
            2 * self.base, dtype=np.int32
        )

    @property
    def peak(self):
        """Return the maximum concurrency currently stored in the tree."""
        return int(self.maximum[1])

    def push(self, node):

        value = self.lazy[node]

        if value != 0 and node < self.base:

            left = node * 2
            right = left + 1

            self.maximum[left] += value
            self.maximum[right] += value

            self.lazy[left] += value
            self.lazy[right] += value

            self.lazy[node] = 0

    def add(self, ql, qr, value):

        ql = max(0, int(ql))
        qr = min(self.n, int(qr))

        if ql >= qr:
            return

        self._add(
            1, 0, self.base,
            ql, qr, int(value)
        )

    def _add(self, node, left, right, ql, qr, value):

        if ql >= right or qr <= left:
            return

        if ql <= left and right <= qr:

            self.maximum[node] += value
            self.lazy[node] += value
            return

        self.push(node)

        mid = (left + right) // 2

        self._add(
            node * 2, left, mid,
            ql, qr, value
        )

        self._add(
            node * 2 + 1, mid, right,
            ql, qr, value
        )

        self.maximum[node] = max(
            self.maximum[node * 2],
            self.maximum[node * 2 + 1]
        )

    def query_max(self, ql, qr):

        ql = max(0, int(ql))
        qr = min(self.n, int(qr))

        if ql >= qr:
            return 0

        return self._query(
            1, 0, self.base,
            ql, qr
        )

    def _query(self, node, left, right, ql, qr):

        if ql >= right or qr <= left:
            return 0

        if ql <= left and right <= qr:
            return int(self.maximum[node])

        self.push(node)

        mid = (left + right) // 2

        left_max = self._query(
            node * 2, left, mid,
            ql, qr
        )

        right_max = self._query(
            node * 2 + 1, mid, right,
            ql, qr
        )

        return max(left_max, right_max)


# ============================================================
# TIME SLOT CONVERSION
# ============================================================

def occupied_slots(start, duration):

    if duration <= 0:
        return None

    first = math.floor(start / SLOT_SECONDS)

    end = start + duration

    last_exclusive = math.ceil(
        end / SLOT_SECONDS
    )

    return first, last_exclusive


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    jobs = pd.read_csv(INPUT_FILE)
    carbon = pd.read_csv(CARBON_FILE)

    required = [
        "start_timestamp",
        "duration",
        "workload_type"
    ]

    for col in required:
        if col not in jobs.columns:
            raise ValueError(
                f"Missing column: {col}"
            )

    if "minute" not in carbon.columns:
        raise ValueError("Missing carbon minute column.")

    if "carbon_intensity_lifecycle" not in carbon.columns:
        raise ValueError(
            "Missing carbon intensity column."
        )

    jobs["start_timestamp"] = pd.to_numeric(
        jobs["start_timestamp"], errors="coerce"
    )

    jobs["duration"] = pd.to_numeric(
        jobs["duration"], errors="coerce"
    )

    if jobs[
        ["start_timestamp", "duration"]
    ].isna().any().any():

        raise ValueError("Invalid job timestamps or durations.")

    if (
        (jobs["start_timestamp"] < 0).any()
        or (jobs["duration"] < 0).any()
    ):
        raise ValueError("Negative timestamps or durations found.")

    jobs["workload_type"] = (
        jobs["workload_type"]
        .astype(str)
        .str.upper()
    )

    valid_types = {
        "IMMEDIATE",
        "DELAY_TOLERANT"
    }

    if not set(jobs["workload_type"].unique()).issubset(valid_types):
        raise ValueError("Unexpected workload types.")

    carbon = carbon.sort_values(
        "minute"
    ).reset_index(drop=True)

    minutes = carbon["minute"].to_numpy(dtype=int)

    if not np.array_equal(
        minutes,
        np.arange(len(minutes))
    ):
        raise ValueError(
            "Carbon timeline must be continuous from minute zero."
        )

    carbon_values = carbon[
        "carbon_intensity_lifecycle"
    ].to_numpy(dtype=float)

    if not np.isfinite(carbon_values).all():
        raise ValueError("Invalid carbon intensity values.")

    if (jobs["start_timestamp"] >= len(carbon_values) * 60).any():
        raise ValueError(
            "Some job start timestamps are outside the carbon timeline."
        )

    return jobs, carbon_values


# ============================================================
# INITIALISE CAPACITY TREE
# ============================================================

def create_capacity_tree(jobs):

    latest_end = (
        jobs["start_timestamp"] + jobs["duration"]
    ).max()

    tree_size = max(
        1,
        math.ceil(latest_end / SLOT_SECONDS) + 1
    )

    tree = CapacityTree(tree_size)

    immediate = jobs[
        jobs["workload_type"] == "IMMEDIATE"
    ]

    print(
        f"Loading {len(immediate):,} immediate invocations..."
    )

    for start, duration in zip(
        immediate["start_timestamp"].to_numpy(),
        immediate["duration"].to_numpy()
    ):

        slots = occupied_slots(
            float(start), float(duration)
        )

        if slots is None:
            continue

        left, right = slots

        tree.add(left, right, 1)

    if tree.peak > MAX_CONCURRENCY:

        raise RuntimeError(
            "Immediate workload alone exceeds the experimental "
            f"capacity ceiling: {tree.peak} > {MAX_CONCURRENCY}. "
            "The one-second conservative model is infeasible "
            "with this ceiling."
        )

    print(
        f"Immediate workload peak: {tree.peak:,}"
    )

    return tree


# ============================================================
# SCHEDULE DELAY-TOLERANT JOBS
# ============================================================

def run_scheduler(jobs, carbon_values, deadline):

    tree = create_capacity_tree(jobs)

    result = jobs.copy()

    original = result[
        "start_timestamp"
    ].to_numpy(dtype=float)

    durations = result[
        "duration"
    ].to_numpy(dtype=float)

    categories = result[
        "workload_type"
    ].to_numpy()

    scheduled = original.copy()

    original_minutes = np.floor(
        original / 60
    ).astype(int)

    scheduled_minutes = original_minutes.copy()

    original_carbon = carbon_values[
        original_minutes
    ]

    scheduled_carbon = original_carbon.copy()

    waiting = np.zeros(len(result), dtype=float)

    tolerant_indices = np.flatnonzero(
        categories == "DELAY_TOLERANT"
    )

    # Process by original arrival time.
    tolerant_indices = tolerant_indices[
        np.argsort(
            original[tolerant_indices],
            kind="stable"
        )
    ]

    delayed_count = 0

    print(
        f"Scheduling {len(tolerant_indices):,} "
        "delay-tolerant invocations..."
    )

    for position, idx in enumerate(tolerant_indices):

        start = original[idx]
        duration = durations[idx]

        original_minute = original_minutes[idx]

        original_offset = start % 60

        last_candidate = min(
            original_minute + deadline,
            len(carbon_values) - 1
        )

        candidates = np.arange(
            original_minute,
            last_candidate + 1
        )

        # Candidate minutes ordered by carbon intensity.
        order = np.lexsort(
            (
                candidates,
                carbon_values[candidates]
            )
        )

        candidates = candidates[order]

        selected_minute = None
        selected_start = None

        for candidate in candidates:

            candidate = int(candidate)

            # Do not choose a higher-carbon period.
            if (
                carbon_values[candidate]
                > original_carbon[idx]
            ):
                continue

            candidate_start = (
                candidate * 60 + original_offset
            )

            wait = candidate_start - start

            if wait < 0 or wait > deadline * 60:
                continue

            slots = occupied_slots(
                candidate_start, duration
            )

            if slots is None:

                selected_minute = candidate
                selected_start = candidate_start
                break

            left, right = slots

            peak = tree.query_max(left, right)

            if peak < MAX_CONCURRENCY:

                selected_minute = candidate
                selected_start = candidate_start

                tree.add(left, right, 1)

                break

        if selected_minute is None:

            raise RuntimeError(
                f"No capacity-feasible slot found for job {idx}. "
                "Try increasing the deadline or reviewing "
                "the experimental capacity ceiling."
            )

        scheduled[idx] = selected_start
        scheduled_minutes[idx] = selected_minute

        waiting[idx] = selected_start - start

        scheduled_carbon[idx] = (
            carbon_values[selected_minute]
        )

        if waiting[idx] > 0:
            delayed_count += 1

        if (position + 1) % 100000 == 0:

            print(
                f"Processed {position + 1:,} / "
                f"{len(tolerant_indices):,} jobs"
            )

    result["original_start_timestamp"] = original
    result["scheduled_start_timestamp"] = scheduled
    result["scheduled_end_timestamp"] = scheduled + durations

    result["original_minute"] = original_minutes
    result["scheduled_minute"] = scheduled_minutes

    result["waiting_time_seconds"] = waiting

    result["original_carbon_intensity"] = original_carbon
    result["scheduled_carbon_intensity"] = scheduled_carbon

    result["scheduling_strategy"] = (
        f"EXACT_CAPACITY_{deadline}MIN"
    )

    summary = {
        "deadline_minutes": deadline,
        "total_invocations": len(result),
        "immediate_invocations": int(
            np.sum(categories == "IMMEDIATE")
        ),
        "delay_tolerant_invocations": len(tolerant_indices),
        "delayed_invocations": delayed_count,
        "delayed_percentage": (
            delayed_count / max(len(tolerant_indices), 1) * 100
        ),
        "average_waiting_seconds": (
            waiting[tolerant_indices].mean()
        ),
        "maximum_waiting_seconds": waiting.max(),
        "average_original_carbon_intensity": (
            original_carbon.mean()
        ),
        "average_scheduled_carbon_intensity": (
            scheduled_carbon.mean()
        ),
        "maximum_one_second_slot_concurrency": tree.peak,
        "capacity_limit": MAX_CONCURRENCY,
        "time_slot_seconds": SLOT_SECONDS
    }

    return result, summary


# ============================================================
# VALIDATE
# ============================================================

def validate(result, deadline):

    immediate = result[
        result["workload_type"] == "IMMEDIATE"
    ]

    assert (
        result["scheduled_start_timestamp"]
        >= result["original_start_timestamp"]
    ).all()

    assert (
        result["waiting_time_seconds"]
        <= deadline * 60 + 1e-8
    ).all()

    assert np.allclose(
        result["scheduled_end_timestamp"],
        result["scheduled_start_timestamp"]
        + result["duration"]
    )

    assert np.allclose(
        immediate["scheduled_start_timestamp"],
        immediate["original_start_timestamp"]
    )

    assert (
        result["scheduled_carbon_intensity"]
        <= result["original_carbon_intensity"] + 1e-9
    ).all()

    print("PASS: Deadline, carbon and immediate-job checks.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nINTERVAL-BASED CAPACITY-CONSTRAINED SCHEDULER")
    print("=" * 60)

    jobs, carbon_values = load_data()

    print(f"Total invocations: {len(jobs):,}")
    print(f"Capacity ceiling: {MAX_CONCURRENCY:,}")
    print(f"Time resolution: {SLOT_SECONDS} second")

    summaries = []

    for deadline in DEADLINES:

        print(f"\nRUNNING {deadline}-MINUTE SCENARIO")
        print("-" * 50)

        result, summary = run_scheduler(
            jobs,
            carbon_values,
            deadline
        )

        validate(result, deadline)

        output_file = (
            OUTPUT_DIR
            / f"exact_capacity_schedule_{deadline}min.csv"
        )

        result.to_csv(
            output_file,
            index=False
        )

        summaries.append(summary)

        print(
            f"Delayed jobs: {summary['delayed_invocations']:,}"
        )

        print(
            f"Average waiting: "
            f"{summary['average_waiting_seconds']:.2f} seconds"
        )

        print(
            f"Average carbon intensity: "
            f"{summary['average_original_carbon_intensity']:.2f}"
            " -> "
            f"{summary['average_scheduled_carbon_intensity']:.2f}"
        )

        print(
            f"Maximum one-second slot concurrency: "
            f"{summary['maximum_one_second_slot_concurrency']:,}"
        )

        print(f"Saved: {output_file}")

        del result

    summary_df = pd.DataFrame(summaries)

    summary_file = (
        OUTPUT_DIR / "exact_capacity_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    print("\nALL SCENARIOS COMPLETED")
    print("=" * 60)

    print(summary_df.to_string(index=False))
    print(f"\nSaved: {summary_file}")


if __name__ == "__main__":
    main()
