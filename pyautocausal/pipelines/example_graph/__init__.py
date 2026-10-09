"""PyAutoCausal Example Graph

A comprehensive causal inference pipeline that automatically selects appropriate 
methods based on data characteristics.

The pipeline supports:
- Cross-sectional analysis (single period)
- Synthetic DiD (single treated unit, multiple periods)  
- Standard DiD (multiple periods, insufficient for staggered)
- Event study (multiple periods, non-staggered treatment)
- Staggered DiD with Callaway & Sant'Anna methods

## Quick Start

```python
from pyautocausal.pipelines.example_graph import causal_pipeline
from pathlib import Path
import pandas as pd

# Create the pipeline
output_path = Path('output')
graph = causal_pipeline(output_path)

# Run with your data
result = graph.fit(df=your_dataframe)
```

## Architecture

The pipeline is organized into separate modules:

- `core.py`: Core decision structure and routing logic
- `branches.py`: Specific causal analysis method implementations  
- `utils.py`: Helper functions for setup and output
- `main.py`: Main execution and demo functionality
"""

from pathlib import Path
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.orchestration.run_context import RunContext
# All branch imports are now done locally within each function as needed
from .core import (
    _create_shared_head,
    _validate_missing_policy,
    create_panel_decision_structure,
    configure_panel_decision_paths,
    create_cross_sectional_decision_structure,
)


def create_panel_graph(output_dir: Path, *, missing_strategy: str = "drop_rows",
                       max_missing_fraction: float = 0.1) -> ExecutableGraph:
    """Create the panel graph with an explicit missing-data policy.

    ``drop_rows`` permits missingness up to ``max_missing_fraction`` per column
    (default 10%); ``reject`` refuses all missingness. Cleaning records drops in
    DataFrame.attrs['cleaning_metadata'] and revalidates the remaining design.
    """
    _validate_missing_policy(missing_strategy, max_missing_fraction)
    graph = ExecutableGraph()
    graph.configure_runtime(output_path=output_dir, run_context=RunContext(metadata={
        "missing_strategy": missing_strategy, "max_missing_fraction": max_missing_fraction,
    }))
    
    # 1. Shared Head
    _create_shared_head(graph)
    
    # 2. Panel Decision Structure
    create_panel_decision_structure(graph)
    
    # 3. Add Analysis Branches (using complete functions with saving functionality)
    from .branches import (
        create_did_branch,
        create_event_study_branch, 
        create_synthetic_did_branch,
        create_staggered_did_branch,
        create_hainmueller_synth_branch
    )
    
    create_did_branch(graph)
    create_event_study_branch(graph)
    create_synthetic_did_branch(graph)
    create_hainmueller_synth_branch(graph)
    create_staggered_did_branch(graph)
    
    # 4. Configure Paths
    configure_panel_decision_paths(graph)
    
    return graph


def create_cross_sectional_graph(output_dir: Path, *, missing_strategy: str = "drop_rows",
                                 max_missing_fraction: float = 0.1) -> ExecutableGraph:
    """Create the cross-sectional graph with an explicit missing-data policy.

    ``drop_rows`` permits missingness up to ``max_missing_fraction`` per column
    (default 10%); ``reject`` refuses all missingness. Both treatment groups must
    remain after cleaning. Unit IDs are optional and may be strings.
    """
    _validate_missing_policy(missing_strategy, max_missing_fraction)
    graph = ExecutableGraph()
    graph.configure_runtime(output_path=output_dir, run_context=RunContext(metadata={
        "missing_strategy": missing_strategy, "max_missing_fraction": max_missing_fraction,
    }))
    # 1. Shared Head
    _create_shared_head(graph)
    
    # 2. Cross-Sectional Decision Structure
    create_cross_sectional_decision_structure(graph)
    
    # 3. Add Analysis Branches (using complete function with saving functionality)
    from .branches import create_cross_sectional_branch
    create_cross_sectional_branch(graph)
    
    
    return graph


def simple_graph(*, missing_strategy: str = "drop_rows", max_missing_fraction: float = 0.1) -> ExecutableGraph:
    """Create a simple graph for testing purposes.
    
    This creates a basic pipeline with cross-sectional branch only
    for testing serialization and basic functionality.

    This function is a bit duplicative (i.e. it's a very thin wrapper) but we maintain it for backwards compatibility.
    
    Returns:
        Configured ExecutableGraph ready for execution
    """
    
    _validate_missing_policy(missing_strategy, max_missing_fraction)
    graph = ExecutableGraph()
    graph.configure_runtime(run_context=RunContext(metadata={
        "missing_strategy": missing_strategy, "max_missing_fraction": max_missing_fraction,
    }))

    # Create the shared head nodes
    _create_shared_head(graph)
    
    # Add simple cross-sectional branches for testing
    create_cross_sectional_decision_structure(graph)
    from .branches import create_cross_sectional_branch
    create_cross_sectional_branch(graph)
    
    return graph


# Import main execution function and utilities for convenience
from .utils import export_outputs

# Backward compatibility alias for the test
_export_outputs = export_outputs

__all__ = [
    'create_panel_graph',
    'create_cross_sectional_graph',
    'simple_graph', 
    'export_outputs',
    '_export_outputs'
] 