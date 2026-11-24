
from pathlib import Path
import sys
import math
import importlib
import traceback

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ==================================================
# 1. PROJECT PATHS AND MODEL LOADING
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "Coding" / "MachineLearning" / "models"

SCHEDULER_DIR = PROJECT_ROOT / "Coding" / "LoadBalancing"


def load_model(filename):
    path = MODEL_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    return joblib.load(path)


strategy_artifact = load_model("strategy_model.joblib")
deadline_artifact = load_model("deadline_minutes_model.joblib")

strategy_model = strategy_artifact["model"]
deadline_model = deadline_artifact["model"]

FEATURE_COLUMNS = strategy_artifact["feature_columns"]

if FEATURE_COLUMNS != deadline_artifact["feature_columns"]:
    raise RuntimeError("Model feature columns do not match.")

if len(FEATURE_COLUMNS) != 32:
    raise RuntimeError("Expected exactly 32 model features.")


# ==================================================
# 2. FASTAPI
# ==================================================

app = FastAPI(
    title="CarbonControl ML API",
    description=(
        "Carbon-aware serverless workload analysis, "
        "ML recommendations and multi-objective scheduling."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# 3. FEATURE EXTRACTION
# Replicates build_ml_dataset.py
# ==================================================

def safe_mean(series):
    return float(series.mean()) if len(series) else 0.0


def safe_std(series):
    return float(series.std()) if len(series) > 1 else 0.0


def safe_quantile(series, quantile):
    return float(series.quantile(quantile)) if len(series) else 0.0


def extract_features(workload_records, carbon_records):

    required_workload = {
        "app",
        "func",
        "start_timestamp",
        "end_timestamp",
        "duration",
        "workload_type",
    }

    required_carbon = {
        "minute",
        "carbon_intensity_lifecycle",
    }

    if not workload_records:
        raise ValueError("At least one workload record is required.")

    if not carbon_records:
        raise ValueError("At least one carbon record is required.")

    workload = pd.DataFrame(workload_records)
    carbon = pd.DataFrame(carbon_records)

    if not required_workload.issubset(workload.columns):
        missing = required_workload - set(workload.columns)
        raise ValueError(
            f"Missing workload fields: {sorted(missing)}"
        )

    if not required_carbon.issubset(carbon.columns):
        missing = required_carbon - set(carbon.columns)
        raise ValueError(
            f"Missing carbon fields: {sorted(missing)}"
        )

    for column in [
        "duration",
        "start_timestamp",
        "end_timestamp",
    ]:
        workload[column] = pd.to_numeric(
            workload[column],
            errors="coerce",
        )

    carbon["carbon_intensity_lifecycle"] = pd.to_numeric(
        carbon["carbon_intensity_lifecycle"],
        errors="coerce",
    )

    workload = workload.dropna(
        subset=[
            "duration",
            "start_timestamp",
            "end_timestamp",
        ]
    ).copy()

    carbon_values = carbon[
        "carbon_intensity_lifecycle"
    ].dropna()

    if workload.empty:
        raise ValueError(
            "No valid workload records remain after numeric validation."
        )

    if carbon_values.empty:
        raise ValueError(
            "No valid carbon intensity readings were supplied."
        )

    total_jobs = len(workload)

    workload_types = (
        workload["workload_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    type_counts = workload_types.value_counts()

    immediate_jobs = sum(
        count
        for name, count in type_counts.items()
        if "immediate" in name
    )

    delayed_jobs = sum(
        count
        for name, count in type_counts.items()
        if "delay" in name or "tolerant" in name
    )

    durations = workload["duration"]
    arrivals = workload["start_timestamp"]

    if (durations < 0).any():
        raise ValueError("Workload durations cannot be negative.")

    if (arrivals < 0).any():
        raise ValueError("Arrival timestamps cannot be negative.")

    if (workload["end_timestamp"] < arrivals).any():
        raise ValueError("End timestamps cannot precede start timestamps.")

    total_execution_seconds = float(durations.sum())

    arrival_minute = arrivals / 60.0

    arrival_bins = pd.cut(
        arrival_minute,
        bins=[-np.inf, 60, 120, 180, 240, 300, np.inf],
        labels=False,
        include_lowest=True,
    )

    arrival_distribution = (
        arrival_bins.value_counts(normalize=True)
        .reindex(range(6), fill_value=0)
    )

    unique_function_pairs = (
        workload[["app", "func"]]
        .drop_duplicates()
        .shape[0]
    )

    features = {
        "total_jobs": total_jobs,
        "unique_applications": workload["app"].nunique(),
        "unique_functions": workload["func"].nunique(),
        "unique_function_pairs": unique_function_pairs,

        "immediate_job_ratio": immediate_jobs / total_jobs,
        "delay_tolerant_job_ratio": delayed_jobs / total_jobs,

        "duration_mean": safe_mean(durations),
        "duration_median": float(durations.median()),
        "duration_std": safe_std(durations),
        "duration_max": float(durations.max()),

        "duration_p75": safe_quantile(durations, 0.75),
        "duration_p90": safe_quantile(durations, 0.90),
        "duration_p95": safe_quantile(durations, 0.95),
        "duration_p99": safe_quantile(durations, 0.99),

        "long_job_ratio_10s": float((durations >= 10).mean()),
        "long_job_ratio_30s": float((durations >= 30).mean()),
        "long_job_ratio_60s": float((durations >= 60).mean()),

        "total_execution_seconds": total_execution_seconds,
        "execution_time_per_job": total_execution_seconds / total_jobs,

        "arrival_mean_seconds": safe_mean(arrivals),
        "arrival_std_seconds": safe_std(arrivals),

        "carbon_mean": safe_mean(carbon_values),
        "carbon_median": float(carbon_values.median()),
        "carbon_std": safe_std(carbon_values),
        "carbon_min": float(carbon_values.min()),
        "carbon_max": float(carbon_values.max()),
    }

    for hour in range(6):
        features[f"arrival_ratio_hour_{hour + 1}"] = float(
            arrival_distribution.loc[hour]
        )

    feature_values = [
        features[column]
        for column in FEATURE_COLUMNS
    ]

    return pd.DataFrame(
        [feature_values],
        columns=FEATURE_COLUMNS,
    )


# ==================================================
# 4. VALIDATE AND NORMALIZE INPUT DATA
# ==================================================

def prepare_workload(workload_records):

    required = {
        "app",
        "func",
        "start_timestamp",
        "end_timestamp",
        "duration",
        "workload_type",
    }

    workload = pd.DataFrame(workload_records)

    missing = required - set(workload.columns)

    if missing:
        raise ValueError(
            f"Missing workload fields: {sorted(missing)}"
        )

    for column in [
        "start_timestamp",
        "end_timestamp",
        "duration",
    ]:
        workload[column] = pd.to_numeric(
            workload[column],
            errors="coerce",
        )

    if workload[
        ["start_timestamp", "end_timestamp", "duration"]
    ].isna().any().any():
        raise ValueError("Workload contains invalid numeric values.")

    if not np.isfinite(
        workload[
            ["start_timestamp", "end_timestamp", "duration"]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Workload contains infinite values.")

    if (workload["start_timestamp"] < 0).any():
        raise ValueError("Start timestamps cannot be negative.")

    if (workload["duration"] <= 0).any():
        raise ValueError(
            "Scheduler requires positive execution durations."
        )

    if (
        workload["end_timestamp"] < workload["start_timestamp"]
    ).any():
        raise ValueError("End timestamp precedes start timestamp.")

    calculated_duration = (
        workload["end_timestamp"] - workload["start_timestamp"]
    )

    if not np.allclose(
        calculated_duration,
        workload["duration"],
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValueError(
            "Each duration must equal end_timestamp - start_timestamp."
        )

    workload["workload_type"] = (
        workload["workload_type"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    allowed_types = {"IMMEDIATE", "DELAY_TOLERANT"}

    invalid_types = (
        set(workload["workload_type"].unique()) - allowed_types
    )

    if invalid_types:
        raise ValueError(
            "Scheduler accepts only IMMEDIATE and DELAY_TOLERANT. "
            f"Invalid types: {sorted(invalid_types)}"
        )

    return workload.reset_index(drop=True)


# ==================================================
# 5. CARBON DATA ALIGNMENT
# ==================================================

def prepare_carbon_data(carbon_records):

    carbon = pd.DataFrame(carbon_records)

    required = {
        "minute",
        "carbon_intensity_lifecycle",
    }

    missing = required - set(carbon.columns)

    if missing:
        raise ValueError(
            f"Missing carbon fields: {sorted(missing)}"
        )

    carbon["minute"] = pd.to_numeric(
        carbon["minute"],
        errors="coerce",
    )

    carbon["carbon_intensity_lifecycle"] = pd.to_numeric(
        carbon["carbon_intensity_lifecycle"],
        errors="coerce",
    )

    if carbon[
        ["minute", "carbon_intensity_lifecycle"]
    ].isna().any().any():
        raise ValueError("Carbon data contains missing or invalid values.")

    if not np.isfinite(
        carbon[
            ["minute", "carbon_intensity_lifecycle"]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Carbon data contains infinite values.")

    if (carbon["carbon_intensity_lifecycle"] < 0).any():
        raise ValueError("Carbon intensity cannot be negative.")

    if (carbon["minute"] < 0).any():
        raise ValueError("Carbon minute values cannot be negative.")

    if not np.allclose(
        carbon["minute"],
        np.round(carbon["minute"]),
    ):
        raise ValueError("Carbon minute values must be integers.")

    carbon["minute"] = carbon["minute"].astype(int)

    if carbon["minute"].duplicated().any():
        raise ValueError("Duplicate carbon minute values detected.")

    carbon = carbon.sort_values("minute").reset_index(drop=True)

    # Support the original 336-hour experiment dataset.
    # Its minute column represents hour indices 0 through 335.
    if (
        len(carbon) == 336
        and carbon["minute"].tolist() == list(range(336))
    ):
        hourly_values = carbon[
            "carbon_intensity_lifecycle"
        ].to_numpy()

        expanded_values = np.repeat(hourly_values, 60)

        carbon = pd.DataFrame({
            "minute": np.arange(len(expanded_values)),
            "carbon_intensity_lifecycle": expanded_values,
        })

    expected_minutes = np.arange(
        carbon["minute"].min(),
        carbon["minute"].max() + 1,
    )

    if not np.array_equal(
        carbon["minute"].to_numpy(),
        expected_minutes,
    ):
        raise ValueError(
            "Carbon minutes must form a continuous sequence without gaps."
        )

    return carbon


# ==================================================
# 6. EXACT MINUTE-LEVEL CONCURRENCY AND EXECUTION
# ==================================================

def build_minute_metrics(workload, carbon):

    starts = workload["start_timestamp"].to_numpy(dtype=float)
    ends = workload["end_timestamp"].to_numpy(dtype=float)

    maximum_end = float(ends.max())

    number_of_minutes = int(math.ceil(maximum_end / 60.0))

    number_of_minutes = max(number_of_minutes, 1)

    if carbon["minute"].min() > 0:
        raise ValueError(
            "Carbon data must start at minute zero for this workload."
        )

    if carbon["minute"].max() < number_of_minutes - 1:
        raise ValueError(
            "Carbon data does not cover the complete workload timeline. "
            f"Required through minute {number_of_minutes - 1}; "
            f"available through minute {carbon['minute'].max()}."
        )

    # --------------------------------------------------
    # Create exact start and end events.
    # --------------------------------------------------

    event_times = np.concatenate([
        starts,
        ends,
        np.array([0.0]),
    ])

    event_deltas = np.concatenate([
        np.ones(len(starts), dtype=np.int64),
        -np.ones(len(ends), dtype=np.int64),
        np.array([0], dtype=np.int64),
    ])

    order = np.argsort(event_times, kind="stable")

    event_times = event_times[order]
    event_deltas = event_deltas[order]

    unique_times, first_indices = np.unique(
        event_times,
        return_index=True,
    )

    combined_deltas = np.add.reduceat(
        event_deltas,
        first_indices,
    )

    event_times = unique_times
    event_deltas = combined_deltas

    concurrency = np.cumsum(event_deltas)

    if concurrency.min() < 0:
        raise ValueError("Negative concurrency detected.")

    prefix_integral = np.zeros(
        len(event_times),
        dtype=np.float64,
    )

    if len(event_times) > 1:
        prefix_integral[1:] = np.cumsum(
            np.diff(event_times) * concurrency[:-1]
        )

    def cumulative_concurrency_time(timestamp):

        index = (
            np.searchsorted(
                event_times,
                timestamp,
                side="right",
            ) - 1
        )

        index = np.maximum(index, 0)

        return (
            prefix_integral[index]
            + (
                timestamp - event_times[index]
            ) * concurrency[index]
        )

    # --------------------------------------------------
    # Minute-level averages and maximum concurrency.
    # --------------------------------------------------

    minute_starts = (
        np.arange(number_of_minutes, dtype=float) * 60.0
    )

    minute_ends = minute_starts + 60.0

    integral_starts = cumulative_concurrency_time(minute_starts)
    integral_ends = cumulative_concurrency_time(minute_ends)

    average_concurrency = (
        integral_ends - integral_starts
    ) / 60.0

    maximum_concurrency = np.zeros(
        number_of_minutes,
        dtype=np.int64,
    )

    for minute in range(number_of_minutes):

        start = minute_starts[minute]
        end = minute_ends[minute]

        left = (
            np.searchsorted(
                event_times,
                start,
                side="right",
            ) - 1
        )

        right = np.searchsorted(
            event_times,
            end,
            side="left",
        )

        maximum_concurrency[minute] = int(
            concurrency[left:right].max()
        )

    # --------------------------------------------------
    # Calculate total execution seconds per minute.
    # Each invocation contributes only its overlap.
    # --------------------------------------------------

    total_execution_seconds = np.zeros(
        number_of_minutes,
        dtype=np.float64,
    )

    for start, end in zip(starts, ends):

        first_minute = int(math.floor(start / 60.0))
        last_minute = int(math.floor((end - 1e-12) / 60.0))

        for minute in range(first_minute, last_minute + 1):

            minute_start = minute * 60.0
            minute_end = minute_start + 60.0

            overlap = max(
                0.0,
                min(end, minute_end) - max(start, minute_start),
            )

            total_execution_seconds[minute] += overlap

    # --------------------------------------------------
    # Attach minute-level carbon readings.
    # --------------------------------------------------

    carbon_lookup = carbon.set_index(
        "minute"
    )["carbon_intensity_lifecycle"]

    minute_numbers = np.arange(number_of_minutes)

    carbon_values = carbon_lookup.reindex(
        minute_numbers
    )

    if carbon_values.isna().any():
        raise ValueError(
            "Some workload minutes have no carbon intensity value."
        )

    minute_metrics = pd.DataFrame({
        "minute": minute_numbers,
        "total_execution_seconds": total_execution_seconds,
        "average_concurrency": average_concurrency,
        "maximum_concurrency": maximum_concurrency,
        "carbon_intensity_lifecycle": carbon_values.to_numpy(),
    })

    return minute_metrics


# ==================================================
# 7. IMPORT ORIGINAL SCHEDULER
# ==================================================

def get_scheduler():

    if not SCHEDULER_DIR.exists():
        raise FileNotFoundError(
            f"Scheduler directory not found: {SCHEDULER_DIR}"
        )

    scheduler_path = str(SCHEDULER_DIR)

    if scheduler_path not in sys.path:
        sys.path.insert(0, scheduler_path)

    try:
        scheduler_module = importlib.import_module(
            "multi_objective_scheduler"
        )

        return scheduler_module

    except Exception as error:
        raise RuntimeError(
            "Could not import multi_objective_scheduler.py. "
            "Check its config.py imports and project file paths. "
            f"Original error: {error}"
        ) from error


# ==================================================
# 8. ORIGINAL VS SCHEDULED COMPARISONS
# ==================================================

def safe_percentage_saving(original, scheduled):

    if original is None or original == 0:
        return None

    return round(
        ((original - scheduled) / original) * 100.0,
        4,
    )


# def calculate_comparisons(original, scheduled):

#     original = original.copy()
#     scheduled = scheduled.copy()

#     original_energy = float(
#         (
#             original["original_energy_intensity"]
#             * original["duration"]
#         ).sum()
#     )

#     scheduled_energy = float(
#         (
#             scheduled["scheduled_energy_intensity"]
#             * scheduled["duration"]
#         ).sum()
#     )

#     original_carbon = float(
#         (
#             original["original_carbon_intensity"]
#             * original["duration"]
#         ).sum()
#     )

#     scheduled_carbon = float(
#         (
#             scheduled["scheduled_carbon_intensity"]
#             * scheduled["duration"]
#         ).sum()
#     )

#     original_cost = float(
#         scheduled["estimated_cost_usd"].sum()
#     )

#     scheduled_cost = original_cost

#     original_objective = float(
#         scheduled["original_objective_score"].mean()
#     )

#     scheduled_objective = float(
#         scheduled["scheduled_objective_score"].mean()
#     )

#     total_waiting = float(
#         scheduled["waiting_time_seconds"].sum()
#     )

#     delayed_jobs = int(
#         scheduled["was_delayed"].sum()
#     )

#     total_jobs = len(scheduled)

#     return {
#         "original": {
#             "total_jobs": total_jobs,
#             "estimated_energy_kwh": original_energy,
#             "carbon_weighted_execution": original_carbon,
#             "estimated_cost_usd": original_cost,
#             "average_objective_score": original_objective,
#         },
#         "scheduled": {
#             "total_jobs": total_jobs,
#             "estimated_energy_kwh": scheduled_energy,
#             "carbon_weighted_execution": scheduled_carbon,
#             "estimated_cost_usd": scheduled_cost,
#             "average_objective_score": scheduled_objective,
#             "delayed_jobs": delayed_jobs,
#             "immediate_jobs": total_jobs - delayed_jobs,
#             "total_waiting_seconds": total_waiting,
#             "average_waiting_seconds": (
#                 total_waiting / total_jobs if total_jobs else 0.0
#             ),
#         },
#         "savings": {
#             "energy_saving_percent": safe_percentage_saving(
#                 original_energy,
#                 scheduled_energy,
#             ),
#             "carbon_weighted_execution_saving_percent": (
#                 safe_percentage_saving(
#                     original_carbon,
#                     scheduled_carbon,
#                 )
#             ),
#             "cost_saving_percent": safe_percentage_saving(
#                 original_cost,
#                 scheduled_cost,
#             ),
#             "objective_score_improvement_percent": (
#                 safe_percentage_saving(
#                     original_objective,
#                     scheduled_objective,
#                 )
#             ),
#         },
#     }



def calculate_comparisons(scheduled):

    scheduled = scheduled.copy()

    required_columns = {
        "duration",
        "original_energy_intensity",
        "scheduled_energy_intensity",
        "original_carbon_intensity",
        "scheduled_carbon_intensity",
        "estimated_cost_usd",
        "original_objective_score",
        "scheduled_objective_score",
        "waiting_time_seconds",
        "was_delayed",
        "workload_type",
    }

    missing = required_columns - set(scheduled.columns)

    if missing:
        raise ValueError(
            f"Missing comparison columns: {sorted(missing)}"
        )

    # ----------------------------------------------
    # ENERGY COMPARISON
    # ----------------------------------------------

    original_energy = float(
        (
            scheduled["original_energy_intensity"]
            * scheduled["duration"]
        ).sum()
    )

    scheduled_energy = float(
        (
            scheduled["scheduled_energy_intensity"]
            * scheduled["duration"]
        ).sum()
    )

    # ----------------------------------------------
    # CARBON COMPARISON
    # ----------------------------------------------

    original_carbon = float(
        (
            scheduled["original_carbon_intensity"]
            * scheduled["duration"]
        ).sum()
    )

    scheduled_carbon = float(
        (
            scheduled["scheduled_carbon_intensity"]
            * scheduled["duration"]
        ).sum()
    )

    # ----------------------------------------------
    # COST COMPARISON
    # ----------------------------------------------

    # Current scheduler provides one estimated cost
    # per invocation, not separate original/scheduled costs.

    original_cost = float(
        scheduled["estimated_cost_usd"].sum()
    )

    scheduled_cost = original_cost

    # ----------------------------------------------
    # OBJECTIVE SCORE
    # ----------------------------------------------

    original_objective = float(
        scheduled["original_objective_score"].mean()
    )

    scheduled_objective = float(
        scheduled["scheduled_objective_score"].mean()
    )

    # ----------------------------------------------
    # WAITING TIME AND JOB STATISTICS
    # ----------------------------------------------

    total_jobs = len(scheduled)

    delayed_jobs = int(
        scheduled["was_delayed"].sum()
    )

    immediate_jobs = int(
        (
            scheduled["workload_type"] == "IMMEDIATE"
        ).sum()
    )

    delay_tolerant_jobs = total_jobs - immediate_jobs

    total_waiting = float(
        scheduled["waiting_time_seconds"].sum()
    )

    # ----------------------------------------------
    # FINAL COMPARISON RESPONSE
    # ----------------------------------------------

    return {

        "original": {
            "total_jobs": total_jobs,
            "estimated_energy_kwh": original_energy,
            "carbon_weighted_execution": original_carbon,
            "estimated_cost_usd": original_cost,
            "average_objective_score": original_objective,
        },

        "scheduled": {
            "total_jobs": total_jobs,
            "estimated_energy_kwh": scheduled_energy,
            "carbon_weighted_execution": scheduled_carbon,
            "estimated_cost_usd": scheduled_cost,
            "average_objective_score": scheduled_objective,

            "delayed_jobs": delayed_jobs,
            "immediate_jobs": immediate_jobs,
            "delay_tolerant_jobs": delay_tolerant_jobs,

            "total_waiting_seconds": total_waiting,

            "average_waiting_seconds": (
                total_waiting / total_jobs
                if total_jobs else 0.0
            ),
        },

        "savings": {

            "energy_saving_percent": safe_percentage_saving(
                original_energy,
                scheduled_energy,
            ),

            "carbon_weighted_execution_saving_percent": (
                safe_percentage_saving(
                    original_carbon,
                    scheduled_carbon,
                )
            ),

            "cost_saving_percent": safe_percentage_saving(
                original_cost,
                scheduled_cost,
            ),

            "objective_score_improvement_percent": (
                safe_percentage_saving(
                    original_objective,
                    scheduled_objective,
                )
            ),
        },
    }


# ==================================================
# 9. RUN MULTI-OBJECTIVE SCHEDULER
# ==================================================

def run_scheduler(workload, carbon, deadline_minutes):

    scheduler = get_scheduler()

    minute_metrics = build_minute_metrics(
        workload,
        carbon,
    )

    minute_metrics = scheduler.calculate_minute_energy_metrics(
        minute_metrics
    )

    required_metrics = {
        "carbon_intensity_lifecycle",
        "energy_per_execution_second",
        "total_execution_seconds",
        "average_concurrency",
        "maximum_concurrency",
    }

    missing = required_metrics - set(minute_metrics.columns)

    if missing:
        raise RuntimeError(
            f"Scheduler metrics missing columns: {sorted(missing)}"
        )

    scheduled = scheduler.run_multi_objective_scheduler(
        workload.copy(),
        minute_metrics.copy(),
        int(deadline_minutes),
    )

    print("\n========== SCHEDULER OUTPUT COLUMNS ==========")
    print(scheduled.columns.tolist())
    print("\n========== SCHEDULER OUTPUT SAMPLE ==========")
    print(scheduled.head().to_string())

    if scheduled is None or scheduled.empty:
        raise RuntimeError("Scheduler returned no scheduling results.")

    scheduler.validate_schedule(
        workload.copy(),
        scheduled.copy(),
        int(deadline_minutes),
    )

    # Original schedule values are available in the scheduler output.
    comparison_source = scheduled.copy()

    comparison_source = comparison_source.rename(
        columns={
            "original_carbon_intensity": "original_carbon_intensity",
            "scheduled_carbon_intensity": "scheduled_carbon_intensity",
            "original_energy_intensity": "original_energy_intensity",
            "scheduled_energy_intensity": "scheduled_energy_intensity",
            "original_objective_score": "original_objective_score",
            "scheduled_objective_score": "scheduled_objective_score",
        }
    )

    comparisons = calculate_comparisons(
        comparison_source,
    )

    # The scheduler's own summary provides workload-type statistics.
    summary = scheduler.generate_summary(scheduled.copy())

    # Return a JSON-safe representation of each invocation.
    schedule_records = (
        scheduled.replace({np.nan: None})
        .to_dict(orient="records")
    )

    return {
        "comparisons": comparisons,
        "scheduler_summary": (
            summary.replace({np.nan: None}).to_dict(orient="records")
            if isinstance(summary, pd.DataFrame)
            else summary
        ),
        "schedule": schedule_records,
        "minute_metrics_count": len(minute_metrics),
        "scheduler_strategy": "MULTI_OBJECTIVE",
    }


# ==================================================
# 10. HEALTH AND MODEL INFORMATION
# ==================================================

@app.get("/")
def home():
    return {
        "application": "CarbonControl ML API",
        "status": "running",
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "strategy_model_loaded": True,
        "deadline_model_loaded": True,
        "feature_count": len(FEATURE_COLUMNS),
        "models_retrained": False,
        "scheduler_directory_exists": SCHEDULER_DIR.exists(),
    }


@app.get("/api/model-info")
def model_info():
    return {
        "strategy_model": {
            "algorithm": strategy_artifact["selected_algorithm"],
            "target": strategy_artifact["target"],
            "classes": [
                str(value) for value in strategy_model.classes_
            ],
        },
        "deadline_model": {
            "algorithm": deadline_artifact["selected_algorithm"],
            "target": deadline_artifact["target"],
            "classes": [
                str(value) for value in deadline_model.classes_
            ],
        },
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "models_retrained": False,
    }


# ==================================================
# 11. PREDICTION + REAL SCHEDULING ENDPOINT
# ==================================================

@app.post("/api/predict")
def predict(payload: dict):

    try:

        workload_records = payload.get("workload")
        carbon_records = payload.get("carbon_data")

        if not isinstance(workload_records, list):
            raise HTTPException(
                status_code=422,
                detail="'workload' must be a list of workload records.",
            )

        if not isinstance(carbon_records, list):
            raise HTTPException(
                status_code=422,
                detail="'carbon_data' must be a list of carbon readings.",
            )

        # ----------------------------------------------
        # Prepare inputs
        # ----------------------------------------------

        workload = prepare_workload(workload_records)

        carbon = prepare_carbon_data(carbon_records)

        # ----------------------------------------------
        # Extract original 32 model features
        # ----------------------------------------------

        input_data = extract_features(
            workload.to_dict(orient="records"),
            carbon.to_dict(orient="records"),
        )

        # ----------------------------------------------
        # ML predictions
        # ----------------------------------------------

        strategy_prediction = str(
            strategy_model.predict(input_data)[0]
        )

        deadline_prediction = int(
            deadline_model.predict(input_data)[0]
        )

        if deadline_prediction <= 0:
            raise ValueError(
                f"Invalid predicted deadline: {deadline_prediction}"
            )

        # ----------------------------------------------
        # Execute actual scheduler
        # ----------------------------------------------

        if strategy_prediction != "MULTI_OBJECTIVE":
            raise ValueError(
                "The saved model predicted "
                f"'{strategy_prediction}', but this API currently "
                "integrates only the MULTI_OBJECTIVE scheduler. "
                "The prediction is returned as an error rather than "
                "silently running a different strategy."
            )

        scheduling_result = run_scheduler(
            workload,
            carbon,
            deadline_prediction,
        )

        return {
            "strategy": strategy_prediction,
            "deadline_minutes": deadline_prediction,
            "model_type": "Random Forest",
            "feature_count": len(FEATURE_COLUMNS),
            "models_retrained": False,
            "scheduler_executed": True,
            "comparisons": scheduling_result["comparisons"],
            "scheduler_summary": scheduling_result["scheduler_summary"],
            "schedule": scheduling_result["schedule"],
            "minute_metrics_count": (
                scheduling_result["minute_metrics_count"]
            ),
            "scheduler_strategy": (
                scheduling_result["scheduler_strategy"]
            ),
            "derived_features": (
                input_data.iloc[0].to_dict()
            ),
            "message": (
                "ML recommendation generated and the original "
                "Multi-Objective Scheduler executed successfully."
            ),
        }

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

    except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Prediction or scheduling failed: {str(error)}",
        )
