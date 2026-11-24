"""Generate the six report figures for Carbon and Cost-Aware Serverless Load Balancing.

Run from any directory:
    python generate_six_graphs.py

The script reads the project's scheduling CSVs and model/evaluation CSVs. It does
not invent missing energy or carbon results. If an evaluation summary is missing,
it reports exactly which file/metric is needed instead of plotting fabricated data.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. PROJECT PATHS (edit PROJECT_ROOT only if your folder is elsewhere)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Users\rishi\Documents\Cloud_G45_Project")
RESULTS = PROJECT_ROOT / "Coding" / "LoadBalancing" / "results"
ENERGY_RESULTS = PROJECT_ROOT / "Coding" / "EnergyModel" / "results"
CARBON_DATA = PROJECT_ROOT / "Coding" / "CarbonAnalysis" / "carbon_intensity_2021_window.csv"
WORKLOAD_CARBON = PROJECT_ROOT / "Coding" / "CarbonAnalysis" / "workload_carbon_dataset.csv"
OUTPUT_DIR = RESULTS / "report_graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Include the strategy files actually created in the project.
SCHEDULE_FILES = {
    "Baseline": RESULTS / "baseline_schedule.csv",
    "Carbon-aware (15 min)": RESULTS / "carbon_aware_schedule_15min.csv",
    "Carbon-aware (30 min)": RESULTS / "carbon_aware_schedule_30min.csv",
    "Carbon-aware (60 min)": RESULTS / "carbon_aware_schedule_60min.csv",
    "Multi-objective (15 min)": RESULTS / "multi_objective_schedule_15min.csv",
    "Multi-objective (30 min)": RESULTS / "multi_objective_schedule_30min.csv",
    "Multi-objective (60 min)": RESULTS / "multi_objective_schedule_60min.csv",
    "Forecast-assisted (15 min)": RESULTS / "forecast_assisted" / "forecast_assisted_schedule_15min.csv",
    "Forecast-assisted (30 min)": RESULTS / "forecast_assisted" / "forecast_assisted_schedule_30min.csv",
    "Forecast-assisted (60 min)": RESULTS / "forecast_assisted" / "forecast_assisted_schedule_60min.csv",
    "Capacity-constrained (15 min)": RESULTS / "capacity_constrained" / "capacity_constrained_schedule_15min.csv",
    "Capacity-constrained (30 min)": RESULTS / "capacity_constrained" / "capacity_constrained_schedule_30min.csv",
    "Capacity-constrained (60 min)": RESULTS / "capacity_constrained" / "capacity_constrained_schedule_60min.csv",
    "Exact-capacity (15 min)": RESULTS / "exact_capacity" / "exact_capacity_schedule_15min.csv",
    "Exact-capacity (30 min)": RESULTS / "exact_capacity" / "exact_capacity_schedule_30min.csv",
    "Exact-capacity (60 min)": RESULTS / "exact_capacity" / "exact_capacity_schedule_60min.csv",
}

# Evaluation summaries may have been produced by the project's evaluators.
# Add a path here if your evaluator writes its summary to another location.
EVALUATION_FILES = [
    RESULTS / "consolidated_evaluation" / "consolidated_strategy_comparison.csv",
    RESULTS / "forecast_assisted" / "evaluation" / "strategy_evaluation.csv",
    RESULTS / "exact_capacity" / "evaluation" / "exact_capacity_evaluation.csv",
    ENERGY_RESULTS / "time_aware_energy_summary.csv",
]

# Cost assumptions documented in the report. Gross Consumption-plan estimate;
# 1 GB memory is assumed. This is an estimate, not an Azure invoice or quote.
EXECUTION_RATE_PER_MILLION_USD = 0.20
GB_SECOND_RATE_USD = 0.000016
ASSUMED_MEMORY_GB = 1.0

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.titlesize": 14,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def first_col(df: pd.DataFrame, aliases: list[str], required: bool = True) -> str | None:
    lookup = {norm(c): c for c in df.columns}
    for alias in aliases:
        if norm(alias) in lookup:
            return lookup[norm(alias)]
    if required:
        raise KeyError(
            f"Could not find any of columns {aliases}. Available columns: {list(df.columns)}"
        )
    return None


def numeric_series(df: pd.DataFrame, aliases: list[str], label: str) -> pd.Series:
    col = first_col(df, aliases)
    values = pd.to_numeric(df[col], errors="coerce")
    if values.isna().all():
        raise ValueError(f"Column '{col}' for {label} contains no numeric values")
    return values


def load_schedules() -> dict[str, pd.DataFrame]:
    schedules = {}
    missing = []
    for label, path in SCHEDULE_FILES.items():
        if not path.exists():
            missing.append(str(path))
            continue
        df = pd.read_csv(path)
        if df.empty:
            warnings.warn(f"Skipping empty schedule: {path}")
            continue
        schedules[label] = df
    if "Baseline" not in schedules:
        raise FileNotFoundError(
            f"Baseline schedule missing: {SCHEDULE_FILES['Baseline']}"
        )
    if missing:
        print(f"\nNote: {len(missing)} optional schedule file(s) were not found; skipped.")
        for item in missing:
            print("  -", item)
    return schedules


def schedule_metrics(label: str, df: pd.DataFrame) -> dict:
    wait_col = first_col(df, ["waiting_time_seconds", "waiting_time", "wait_seconds"], required=False)
    if wait_col:
        waits = pd.to_numeric(df[wait_col], errors="coerce").fillna(0).clip(lower=0)
    else:
        original_col = first_col(df, ["start_timestamp", "original_start_timestamp", "end_timestamp"], required=False)
        scheduled_col = first_col(df, ["scheduled_start_timestamp", "start_timestamp"], required=False)
        if not original_col or not scheduled_col or original_col == scheduled_col:
            raise KeyError(f"{label}: need waiting_time_seconds or original and scheduled start columns")
        waits = (pd.to_numeric(df[scheduled_col], errors="coerce") -
                 pd.to_numeric(df[original_col], errors="coerce")).fillna(0).clip(lower=0)

    start_col = first_col(df, ["scheduled_start_timestamp", "start_timestamp"])
    end_col = first_col(df, ["scheduled_end_timestamp"] , required=False)
    if end_col is None:
        duration_col = first_col(df, ["duration", "duration_seconds"])
        starts = pd.to_numeric(df[start_col], errors="coerce")
        ends = starts + pd.to_numeric(df[duration_col], errors="coerce")
    else:
        starts = pd.to_numeric(df[start_col], errors="coerce")
        ends = pd.to_numeric(df[end_col], errors="coerce")
    valid = starts.notna() & ends.notna() & (ends > starts)
    events = []
    for s, e in zip(starts[valid], ends[valid]):
        events.append((float(s), 1))
        events.append((float(e), -1))
    # End events sort before start events at equal timestamps: [start, end).
    events.sort(key=lambda x: (x[0], x[1]))
    active = maximum = 0
    for _, delta in events:
        active += delta
        maximum = max(maximum, active)
    return {
        "strategy": label,
        "invocations": int(len(df)),
        "mean_wait_seconds": float(waits.mean()),
        "total_wait_seconds": float(waits.sum()),
        "max_concurrency": int(maximum),
    }


def find_metric_table() -> pd.DataFrame:
    """Find a strategy-level table containing energy, carbon and/or cost metrics."""
    for path in EVALUATION_FILES:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"Could not read {path}: {exc}")
            continue
        if not df.empty:
            print(f"Using evaluation data from: {path}")
            return df
    raise FileNotFoundError(
        "No non-empty evaluation summary found. Expected one of:\n" +
        "\n".join(str(p) for p in EVALUATION_FILES) +
        "\nThe energy/carbon/cost graphs require strategy-level evaluation results; "
        "the energy summary alone may contain capacity scenarios rather than scheduling strategies."
    )


def make_metric_lookup(df: pd.DataFrame) -> tuple[str, dict[str, str]]:
    strategy_col = first_col(df, ["strategy", "strategy_name", "scheduling_strategy", "method"])
    aliases = {
        "energy": ["energy_kwh", "total_energy_kwh", "estimated_energy_kwh", "energy_consumption_kwh"],
        "carbon": ["carbon_kgco2eq", "carbon_kg", "total_carbon_kgco2eq", "carbon_emissions_kg", "lifecycle_carbon_kg"],
        "cost": ["cost_usd", "estimated_cost_usd", "total_cost_usd", "cost"],
    }
    found = {}
    for key, names in aliases.items():
        col = first_col(df, names, required=False)
        if col:
            found[key] = col
    return strategy_col, found



def match_eval_rows(labels, eval_df, strategy_col):
    """
    Match each plot label to exactly one evaluation row using
    strategy family and deadline_minutes.
    """
    out = eval_df.copy()

    def normalize(value):
        return re.sub(r"[^a-z0-9]", "", str(value).lower())

    strategy_aliases = {
    "baseline": "baseline",
    "carbonaware": "carbonaware",
    "multiobjective": "multiobjective",
    "forecastassisted": "forecastassisted",

    # Graph label -> consolidated CSV strategy
    "capacityconstrained": "approxcapacity",
    "approxcapacity": "approxcapacity",

    # Graph label -> consolidated CSV strategy
    "exactcapacity": "onesecondslotcapacity",
    "onesecondslotcapacity": "onesecondslotcapacity",
}

    if "deadline_minutes" not in out.columns:
        raise ValueError(
            "Evaluation CSV is missing deadline_minutes."
        )

    out["_strategy_key"] = (
        out[strategy_col].astype(str).map(normalize)
    )

    matched_rows = []

    for label in labels:
        normalized_label = normalize(label)

        # Identify the strategy family.
        strategy_key = None

        for alias in strategy_aliases:
            if alias in normalized_label:
                strategy_key = strategy_aliases[alias]
                break

        if strategy_key is None:
            raise ValueError(
                f"Could not identify strategy from label: {label}"
            )

        # Baseline is repeated for all deadlines but has identical
        # metrics. Use the 15-minute reference row for its plot label.
        deadline_match = re.search(
            r"(15|30|60)", str(label)
        )

        deadline = (
            int(deadline_match.group(1))
            if deadline_match
            else 15
        )

        matches = out[
            (out["_strategy_key"] == strategy_key)
            & (out["deadline_minutes"] == deadline)
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one evaluation row for "
                f"{label}; found {len(matches)}."
            )

        row = matches.iloc[0].copy()
        row["_plot_label"] = label
        matched_rows.append(row)

    if not matched_rows:
        return pd.DataFrame()

    return pd.DataFrame(matched_rows).reset_index(drop=True)


def save_bar(data: pd.DataFrame, x: str, y: str, title: str, ylabel: str, filename: str,
             rotate: int = 35, zero: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    vals = pd.to_numeric(data[y], errors="coerce")
    ax.bar(data[x].astype(str), vals)
    ax.set_title(title, pad=12, weight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Scheduling strategy")
    if zero:
        ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)
    print("Created:", OUTPUT_DIR / filename)


def main() -> None:
    schedules = load_schedules()
    schedule_rows = [schedule_metrics(name, frame) for name, frame in schedules.items()]
    schedule_df = pd.DataFrame(schedule_rows)
    schedule_df.to_csv(OUTPUT_DIR / "schedule_derived_metrics.csv", index=False)

    # Figure 5: concurrency (recomputed from actual scheduled intervals).
    save_bar(schedule_df, "strategy", "max_concurrency",
             "Maximum Scheduled Concurrency by Strategy", "Maximum concurrent invocations",
             "concurrency_comparison.png")

    # Figure 4: mean waiting time from each schedule's own recorded wait column.
    save_bar(schedule_df, "strategy", "mean_wait_seconds",
             "Average Invocation Waiting Time by Strategy", "Mean waiting time (seconds)",
             "waiting_time_comparison.png")

    # The remaining report graphs need the evaluator's strategy-level metrics.
    eval_df = find_metric_table()
    strategy_col, metric_cols = make_metric_lookup(eval_df)
    matched = match_eval_rows(list(schedules.keys()), eval_df, strategy_col)
    if matched.empty:
        raise ValueError(
            f"Could not match evaluation strategy names in column '{strategy_col}' "
            f"to schedule labels. Values found: {eval_df[strategy_col].dropna().unique().tolist()}"
        )
    matched.to_csv(OUTPUT_DIR / "matched_evaluation_rows.csv", index=False)

    if "energy" not in metric_cols or "carbon" not in metric_cols:
        raise KeyError(
            "Evaluation table must contain both energy (kWh) and lifecycle carbon (kgCO2eq) "
            f"columns. Detected columns: {list(eval_df.columns)}. Found metric columns: {metric_cols}. "
            "Do not substitute the capacity-only time_aware_energy_summary.csv for strategy-level results."
        )
    matched["energy_plot"] = pd.to_numeric(matched[metric_cols["energy"]], errors="coerce")
    matched["carbon_plot"] = pd.to_numeric(matched[metric_cols["carbon"]], errors="coerce")
    if matched[["energy_plot", "carbon_plot"]].isna().all().any():
        raise ValueError("Energy or carbon values are missing/non-numeric in matched evaluation rows")

    save_bar(matched, "_plot_label", "energy_plot", "Estimated Energy Consumption by Strategy",
             "Energy consumption (kWh)", "energy_comparison.png")
    save_bar(matched, "_plot_label", "carbon_plot", "Lifecycle Carbon Emissions by Strategy",
             "Lifecycle carbon emissions (kgCO2eq)", "carbon_comparison.png")

    baseline_rows = matched[matched["_plot_label"].str.lower().eq("baseline")]
    if baseline_rows.empty:
        raise ValueError("Baseline evaluation row not found; cannot calculate relative carbon reduction")
    baseline_carbon = float(baseline_rows.iloc[0]["carbon_plot"])
    if baseline_carbon <= 0:
        raise ValueError("Baseline carbon must be positive to calculate relative reduction")
    matched["carbon_reduction_percent"] = (
        (baseline_carbon - matched["carbon_plot"]) / baseline_carbon * 100.0
    )
    save_bar(matched, "_plot_label", "carbon_reduction_percent",
             "Relative Lifecycle Carbon Reduction vs Baseline", "Carbon reduction (%)",
             "carbon_reduction_comparison.png", zero=False)

    # Cost is calculated from actual invocation count and duration with the
    # report's stated 1-GB assumption and published rate assumptions.
    # If the evaluation table has a cost column, use that; otherwise compute
    # gross modeled cost from each strategy's schedule (same workload = same cost).
    if "cost" in metric_cols:
        matched["cost_plot"] = pd.to_numeric(matched[metric_cols["cost"]], errors="coerce")
    else:
        cost_rows = []
        for label, frame in schedules.items():
            duration_col = first_col(frame, ["duration", "duration_seconds"])
            durations = pd.to_numeric(frame[duration_col], errors="coerce").fillna(0).clip(lower=0)
            executions = len(frame)
            gb_seconds = durations.sum() * ASSUMED_MEMORY_GB
            total = executions / 1_000_000 * EXECUTION_RATE_PER_MILLION_USD + gb_seconds * GB_SECOND_RATE_USD
            cost_rows.append({"strategy": label, "cost_plot": total})
        cost_df = pd.DataFrame(cost_rows)
        save_bar(cost_df, "strategy", "cost_plot",
                 "Estimated Gross Azure Functions Consumption-plan Cost", "Estimated cost (USD)",
                 "cost_comparison.png")
    if "cost" in metric_cols:
        save_bar(matched, "_plot_label", "cost_plot",
                 "Estimated Gross Azure Functions Consumption-plan Cost", "Estimated cost (USD)",
                 "cost_comparison.png")

    print("\nAll six report figures are saved in:", OUTPUT_DIR)
    print("Also saved schedule_derived_metrics.csv and matched_evaluation_rows.csv for audit.")
    print("Reminder: cost is a modeled estimate under the report's 1-GB and pricing assumptions;")
    print("it is not a measured Azure bill. Verify the evaluation table's metric units before submission.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nERROR:", exc, file=sys.stderr)
        sys.exit(1)
