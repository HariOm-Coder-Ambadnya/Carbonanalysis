
from pathlib import Path

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Aggregated dataset containing workload and carbon information
INPUT_FILE = (
    PROJECT_ROOT
    / "Coding"
    / "CarbonAnalysis"
    / "workload_carbon_dataset.csv"
)

# Original Azure Functions invocation trace
RAW_WORKLOAD_FILE = (
    PROJECT_ROOT
    / "AzureFunctionsInvocationTraceForTwoWeeksJan2021"
    / "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
)

# Load balancing output directory
OUTPUT_DIR = (
    PROJECT_ROOT
    / "Coding"
    / "LoadBalancing"
    / "results"
)


# --------------------------------------------------
# SIMULATION CONFIGURATION
# --------------------------------------------------

INTERVAL_SECONDS = 60

# Initial simulated workload classification
IMMEDIATE_WORKLOAD_PERCENTAGE = 0.70
DELAY_TOLERANT_WORKLOAD_PERCENTAGE = 0.30

# Maximum waiting-time scenarios (minutes)
DEADLINE_SCENARIOS_MINUTES = [15, 30, 60]

# Scheduling interval
SCHEDULING_INTERVAL_MINUTES = 1


# --------------------------------------------------
# CARBON CONFIGURATION
# --------------------------------------------------

CARBON_COLUMN = "carbon_intensity_lifecycle"

# Lower carbon intensity is preferred
CARBON_OPTIMIZATION_ENABLED = True


# --------------------------------------------------
# ENERGY CONFIGURATION
# --------------------------------------------------

ENERGY_SCENARIO = "capacity_20"

CONCURRENCY_CAPACITY = 20

IDLE_POWER_W = 44.0
PEAK_POWER_W = 125.0


# --------------------------------------------------
# COST CONFIGURATION
# --------------------------------------------------

MEMORY_ALLOCATION_MB = 1024

EXECUTION_PRICE_PER_MILLION = 0.20
PRICE_PER_GB_SECOND = 0.000016


# --------------------------------------------------
# OBJECTIVE FUNCTION WEIGHTS
# --------------------------------------------------

ENERGY_WEIGHT = 0.30
CARBON_WEIGHT = 0.50
COST_WEIGHT = 0.20


# --------------------------------------------------
# OUTPUT SETTINGS
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# VALIDATE REQUIRED INPUT FILES
# --------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Aggregated workload-carbon dataset not found: {INPUT_FILE}"
    )

if not RAW_WORKLOAD_FILE.exists():
    raise FileNotFoundError(
        f"Original invocation trace not found: {RAW_WORKLOAD_FILE}"
    )


# --------------------------------------------------
# CONFIGURATION INFORMATION
# --------------------------------------------------

print("LOAD BALANCING CONFIGURATION LOADED")

print(f"Project root: {PROJECT_ROOT}")
print(f"Original workload: {RAW_WORKLOAD_FILE}")
print(f"Aggregated dataset: {INPUT_FILE}")
print(f"Output directory: {OUTPUT_DIR}")

print(
    f"Immediate workload: "
    f"{IMMEDIATE_WORKLOAD_PERCENTAGE * 100:.0f}%"
)

print(
    f"Delay-tolerant workload: "
    f"{DELAY_TOLERANT_WORKLOAD_PERCENTAGE * 100:.0f}%"
)

print(f"Deadline scenarios: {DEADLINE_SCENARIOS_MINUTES}")
