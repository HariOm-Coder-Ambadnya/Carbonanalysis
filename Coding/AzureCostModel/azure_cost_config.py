
from pathlib import Path

# Project root: Cloud_G45_Project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Original Azure Functions invocation trace
INPUT_FILE = (
    PROJECT_ROOT
    / "AzureFunctionsInvocationTraceForTwoWeeksJan2021"
    / "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
)

OUTPUT_DIR = PROJECT_ROOT / "Coding" / "AzureCostModel" / "results"

# Reconstructed trace timeline.
# This is an assumption used in the existing project timeline.
TRACE_START_UTC = "2021-01-31 00:00:00"

# Azure Functions Consumption pricing reference assumptions.
# Verify region-specific rates before presenting them as a price quote.
COST_CURRENCY = "USD"

# Reference rates used by the model.
# Keep these configurable because Azure pricing can change.
EXECUTION_PRICE_PER_MILLION = 0.20
PRICE_PER_GB_SECOND = 0.000016

# Documented Consumption-plan monthly free grants.
FREE_EXECUTIONS_PER_MONTH = 1_000_000
FREE_GB_SECONDS_PER_MONTH = 400_000

# Assumed memory allocations, in MB.
# These are sensitivity scenarios, NOT measured memory usage.
MEMORY_SCENARIOS_MB = [512, 1024, 1536]

# Azure Consumption billing assumptions.
MEMORY_ROUNDING_MB = 128
MAX_BILLABLE_MEMORY_MB = 1536
MIN_EXECUTION_DURATION_SECONDS = 0.100
DURATION_ROUNDING_SECONDS = 0.001

# If True, calculate illustrative cost after monthly free grants.
# Assumes this trace is the only workload consuming the subscription grants.
CALCULATE_FREE_GRANT_SCENARIO = True
