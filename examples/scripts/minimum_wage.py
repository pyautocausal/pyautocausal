import pandas as pd
from pathlib import Path
from pyautocausal.pipelines.example_graph import (
    create_panel_graph, export_outputs
)
from pyautocausal.pipelines.example_graph.utils import print_execution_summary, setup_output_directories

# Load the minimum wage data
raw_data = pd.read_csv("pyautocausal/examples/data/mpdta.csv")

# Preprocess data by changing column names to match pipeline expectations
# The mpdta dataset has:
# - year: time variable
# - countyreal: unit identifier (county codes)  
# - lemp: log employment (outcome variable)
# - treat: treatment indicator (0/1) - but this is "absorbing" (always 1 once treated)
# - lpop: log population (covariate)
# - first.treat: first treatment year - this has the ACTUAL treatment timing
data = raw_data.rename(columns={
    "countyreal": "id_unit",  # county identifier
    "year": "t",              # time variable
    "lemp": "y"               # log employment as outcome
})

# Reconstruct the treatment variable properly using first.treat
# The original 'treat' column is absorbing (1 for all periods of units that ever get treated)
# We need to make it 0 before treatment and 1 from treatment onwards
print("Reconstructing treatment variable using first.treat column...")

# Create proper treatment indicator based on first.treat timing
def reconstruct_treatment(row):
    """Reconstruct treatment: 0 before first.treat, 1 from first.treat onwards"""
    if pd.isna(row['first.treat']) or row['first.treat'] == 0:
        # Never treated units
        return 0
    elif row['t'] >= row['first.treat']:
        # Treated in current period (treatment started)
        return 1
    else:
        # Not yet treated (before treatment start)
        return 0

data['treat'] = data.apply(reconstruct_treatment, axis=1)

# Keep additional covariates
data = data[["id_unit", "t", "treat", "y", "lpop"]]  # Include lpop as covariate

print("Data Overview:")
print(f"Data shape: {data.shape}")
print(f"Unique counties: {data['id_unit'].nunique()}")
print(f"Time periods: {data['t'].min()}-{data['t'].max()}")
print(f"Treatment distribution:\n{data['treat'].value_counts()}")


# Check if we now have staggered treatment
from pyautocausal.pipelines.library.conditions import has_staggered_treatment
print(f"Staggered treatment detected: {has_staggered_treatment(data)}")

# Show treatment start timing distribution
treatment_starts = data[data['treat'] == 1].groupby('id_unit')['t'].min()
print(f"Treatment start times: {sorted(treatment_starts.value_counts().sort_index().items())}")

print(f"Sample data:\n{data.head(10)}")

# Define output path
output_path = Path("pyautocausal/examples/outputs/minimum_wage")
output_path.mkdir(parents=True, exist_ok=True)
setup_output_directories(output_path)

# Initialize graph
graph = create_panel_graph(output_path)

# Save data for reference - same location as notebook
notebooks_path = output_path / "notebooks"
notebooks_path.mkdir(parents=True, exist_ok=True)
data_csv_path = notebooks_path / "minimum_wage.csv"
data.to_csv(data_csv_path, index=False)

print(f"\nProcessed data saved to {data_csv_path}")

# Run the causal pipeline
graph.fit(df=data)

# Results summary and export
print("\n======= Execution Summary =======")
print_execution_summary(graph)
print("-" * 50)

export_outputs(graph, output_path, "minimum_wage.csv")
print("\n======= Minimum Wage Analysis Finished =======")
