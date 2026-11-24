
import numpy as np
import pandas as pd

from config import (
    RAW_WORKLOAD_FILE,
    OUTPUT_DIR,
    IMMEDIATE_WORKLOAD_PERCENTAGE,
    DELAY_TOLERANT_WORKLOAD_PERCENTAGE
)


# --------------------------------------------------
# 1. LOAD ORIGINAL WORKLOAD
# --------------------------------------------------

def load_workload():

    print("\nLoading original Azure Functions workload...")

    # Read the original invocation trace.
    # Do not use the aggregated workload-carbon dataset here.
    df = pd.read_csv(RAW_WORKLOAD_FILE)

    required_columns = [
        "app",
        "func",
        "end_timestamp",
        "duration"
    ]

    # Check required columns
    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    # Check missing values
    if df[required_columns].isnull().any().any():

        raise ValueError(
            "Input contains missing values in required columns."
        )

    # Validate numeric columns
    df["end_timestamp"] = pd.to_numeric(
        df["end_timestamp"],
        errors="raise"
    )

    df["duration"] = pd.to_numeric(
        df["duration"],
        errors="raise"
    )

    # Check negative duration
    if (df["duration"] < 0).any():

        raise ValueError(
            "Negative invocation durations found."
        )

    # Check finite numeric values
    if not np.isfinite(
        df[["end_timestamp", "duration"]].to_numpy()
    ).all():

        raise ValueError(
            "Non-finite timestamp or duration values found."
        )

    print(f"Original invocations: {len(df):,}")

    return df


# --------------------------------------------------
# 2. CLASSIFY WORKLOAD DETERMINISTICALLY
# --------------------------------------------------

def classify_workload(df):

    print("\nClassifying workload...")

    immediate_ratio = IMMEDIATE_WORKLOAD_PERCENTAGE

    tolerant_ratio = DELAY_TOLERANT_WORKLOAD_PERCENTAGE

    # Validate percentages
    if not np.isclose(
        immediate_ratio + tolerant_ratio,
        1.0
    ):

        raise ValueError(
            "Immediate and delay-tolerant percentages must sum to 1."
        )

    if not 0 <= immediate_ratio <= 1:

        raise ValueError(
            "Immediate workload percentage must be between 0 and 1."
        )

    total_invocations = len(df)

    if total_invocations == 0:

        raise ValueError(
            "The workload dataset is empty."
        )

    # Calculate exact number of immediate invocations
    immediate_count = round(
        total_invocations * immediate_ratio
    )

    # Fixed seed guarantees reproducibility
    rng = np.random.default_rng(seed=42)

    # Generate a random ordering of row indices
    shuffled_indices = rng.permutation(
        total_invocations
    )

    # Initially classify every invocation as delay-tolerant
    classifications = np.full(
        total_invocations,
        "DELAY_TOLERANT",
        dtype=object
    )

    # Select immediate workload indices
    immediate_indices = shuffled_indices[
        :immediate_count
    ]

    classifications[immediate_indices] = "IMMEDIATE"

    # Preserve original records and add new columns
    df = df.copy()

    # Calculate invocation start time
    df["start_timestamp"] = (
        df["end_timestamp"] - df["duration"]
    )

    # Assign simulated workload category
    df["workload_type"] = classifications

    print("Workload classification completed.")

    return df


# --------------------------------------------------
# 3. GENERATE CLASSIFICATION SUMMARY
# --------------------------------------------------

def generate_summary(df):

    summary = (
        df["workload_type"]
        .value_counts()
        .rename_axis("workload_type")
        .reset_index(name="invocation_count")
    )

    summary["percentage"] = (
        summary["invocation_count"]
        / len(df)
        * 100
    )

    print("\nWORKLOAD CLASSIFICATION SUMMARY")
    print("================================")

    print(
        summary.to_string(
            index=False,
            formatters={
                "percentage": "{:.2f}%".format
            }
        )
    )

    return summary


# --------------------------------------------------
# 4. VALIDATE CLASSIFICATION
# --------------------------------------------------

def validate_classification(df):

    print("\nVALIDATING CLASSIFICATION")
    print("=========================")

    total = len(df)

    immediate = (
        df["workload_type"] == "IMMEDIATE"
    ).sum()

    tolerant = (
        df["workload_type"] == "DELAY_TOLERANT"
    ).sum()

    # Check total records
    assert total == immediate + tolerant, (
        "Classification record count mismatch."
    )

    # Check expected immediate count
    expected_immediate = round(
        total * IMMEDIATE_WORKLOAD_PERCENTAGE
    )

    assert immediate == expected_immediate, (
        "Immediate workload count does not match configuration."
    )

    # Check allowed categories
    valid_categories = {
        "IMMEDIATE",
        "DELAY_TOLERANT"
    }

    assert set(
        df["workload_type"].unique()
    ).issubset(valid_categories), (
        "Unexpected workload category found."
    )

    # Check start timestamp
    assert "start_timestamp" in df.columns

    assert np.allclose(
        df["start_timestamp"],
        df["end_timestamp"] - df["duration"]
    ), "Start timestamp calculation failed."

    print("PASS: Total invocation count preserved.")
    print("PASS: Immediate workload percentage correct.")
    print("PASS: Delay-tolerant workload percentage correct.")
    print("PASS: Only valid workload categories present.")
    print("PASS: Start timestamps calculated correctly.")

    print("\nAll classification checks passed.")


# --------------------------------------------------
# 5. SAVE RESULTS
# --------------------------------------------------

def save_results(df, summary):

    output_file = (
        OUTPUT_DIR / "classified_workload.csv"
    )

    summary_file = (
        OUTPUT_DIR / "workload_classification_summary.csv"
    )

    print("\nSaving classification results...")

    df.to_csv(
        output_file,
        index=False
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print(f"Classified dataset: {output_file}")

    print(f"Summary dataset: {summary_file}")


# --------------------------------------------------
# 6. MAIN EXECUTION
# --------------------------------------------------

def main():

    # Load original trace
    df = load_workload()

    # Assign workload categories
    df = classify_workload(df)

    # Validate classification
    validate_classification(df)

    # Generate summary
    summary = generate_summary(df)

    # Save output files
    save_results(df, summary)

    print("\nFINAL CLASSIFICATION RESULTS")
    print("============================")

    print(
        f"Total invocations: {len(df):,}"
    )

    print(
        "Immediate invocations:",
        (df["workload_type"] == "IMMEDIATE").sum()
    )

    print(
        "Delay-tolerant invocations:",
        (df["workload_type"] == "DELAY_TOLERANT").sum()
    )

    print("\nWORKLOAD CLASSIFICATION SUCCESSFUL!")


if __name__ == "__main__":
    main()
