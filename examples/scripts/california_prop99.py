# Matplotlib imports (removed non-interactive backend setting)
import matplotlib

import pandas as pd
from pathlib import Path
from pyautocausal.pipelines.example_graph import (
    create_panel_graph, export_outputs
)
from pyautocausal.pipelines.example_graph.utils import print_execution_summary, setup_output_directories


# The data already has the correct column names: id_unit, t, treat, y, plus control variables

DATA_PATH = "pyautocausal/examples/data/california_prop99.csv"

data = pd.read_csv(DATA_PATH)

data = data.rename(columns={ "year": "t", "treated": "treat", "cigsale": "y"})

data.t = data.t.astype(int)
data["id_unit"] = pd.factorize(data.state)[0]

data = data.fillna(-999)

# Define output path
output_path = Path("pyautocausal/examples/outputs/california_prop99")
output_path.mkdir(parents=True, exist_ok=True)
setup_output_directories(output_path)

# Initialize graph
graph = create_panel_graph(output_path)

# Save data for reference - same location as notebook
notebooks_path = output_path / "notebooks"
notebooks_path.mkdir(parents=True, exist_ok=True)
data_csv_path = notebooks_path / "california_prop99.csv"
data.to_csv(data_csv_path, index=False)

print(f"Processed data saved to {data_csv_path}")

graph.fit(df=data)

# Results summary and export
print("\n======= Execution Summary =======")
print_execution_summary(graph)
print("-" * 50)

export_outputs(graph, output_path, "california_prop99.csv")
print("\n======= Example Graph Run Finished =======")

