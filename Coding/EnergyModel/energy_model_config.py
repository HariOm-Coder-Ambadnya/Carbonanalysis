
"""
Energy Model Configuration

IMPORTANT:
All power and concurrency values below are experimental assumptions.
They are NOT official Azure Functions hardware specifications.

Replace or validate them against suitable research literature before
using the results as final research findings.
"""

INPUT_FILE = (
    "Coding/CarbonAnalysis/workload_carbon_dataset.csv"
)

OUTPUT_DIR = "Coding/EnergyModel/results"

# Each simulated instance is assumed to handle this many
# concurrent invocations.
SCENARIOS = {
    "capacity_10": {
        "concurrency_capacity": 10,
        "idle_power_w": 34.0,
        "peak_power_w": 125.0
    },

    "capacity_20": {
        "concurrency_capacity": 20,
        "idle_power_w": 44.0,
        "peak_power_w": 125.0
    },

    "capacity_50": {
        "concurrency_capacity": 50,
        "idle_power_w": 44.0,
        "peak_power_w": 140.0
    }
}

# Each record represents one minute.
INTERVAL_SECONDS = 60

# Carbon intensity unit: grams of CO2 equivalent per kWh.
CARBON_UNIT = "gCO2eq/kWh"

PRIMARY_CARBON_COLUMN = "carbon_intensity_lifecycle"
SECONDARY_CARBON_COLUMN = "carbon_intensity_direct"
