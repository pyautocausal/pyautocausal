# PyAutoCausal

**Automated causal inference pipelines for data scientists**

## Why Causal Inference Matters in Tech

As data scientists, we're often asked to go beyond correlation and answer causal questions: 
- "Did our new recommendation algorithm actually increase user engagement, or was it just seasonal trends?"
- "What's the true impact of our premium subscription tier on customer retention?"
- "How much did our marketing campaign increase conversions versus organic growth?"
- "Did our product redesign cause the drop in user activity, or was it market conditions?"

These questions can't be answered with standard predictive models or A/B tests alone. Real-world constraints often prevent randomized experiments:
- **Ethical concerns**: Can't randomly deny users important features
- **Business constraints**: Can't risk revenue on large-scale experiments  
- **Natural experiments**: Sometimes changes happen organically (competitor exits, policy changes)
- **Historical analysis**: Need to evaluate past decisions without experimental data

## The Challenge of Observational Data

When working with observational data (logs, user behavior, historical metrics), we face fundamental challenges:

1. **Confounding**: Users who adopt premium features might be inherently more engaged
2. **Selection bias**: Treatment assignment isn't random
3. **Time-varying effects**: Impact changes over time
4. **Heterogeneous effects**: Different user segments respond differently

Traditional ML models are built for prediction, not causal inference. They'll happily exploit confounders and selection bias to maximize accuracy, giving you precisely wrong answers to causal questions.

## PyAutoCausal: Causal Inference Made Practical

PyAutoCausal automates the complex decision tree of modern causal inference methods. Instead of manually implementing and choosing between dozens of estimators, PyAutoCausal:

1. **Analyzes your data structure** to understand treatment timing, units, and available controls
2. **Selects appropriate methods** based on your data characteristics
3. **Validates assumptions** and warns about potential violations
4. **Executes analysis** with proper statistical inference
5. **Exports results** in formats ready for stakeholder communication

## Quick Example: Measuring Feature Impact

```python
from pyautocausal.pipelines.example_graph import causal_pipeline
import pandas as pd

# Your product data with treatment (feature rollout) and outcome (engagement)
data = pd.DataFrame({
    'id_unit': [...],        # User identifier
    't': [...],              # Time periods
    'treat': [...],          # 1 if user has feature, 0 otherwise
    'y': [...],              # Your KPI (DAU, sessions, revenue, etc.)
    'x1': [...],             # User characteristics
    'x2': [...]              # Additional controls
})

# PyAutoCausal automatically:
# - Detects this is panel data with staggered treatment
# - Chooses modern DiD methods (e.g., Callaway-Sant'Anna)
# - Handles heterogeneous treatment effects
# - Produces event study plots

pipeline = causal_pipeline(output_path="./feature_impact_analysis")
pipeline.fit(df=data)

# Results include:
# - Average treatment effect with confidence intervals
# - Dynamic effects over time since treatment
# - Heterogeneity analysis across user segments
# - Diagnostic plots and assumption checks
```

## Real Tech Applications

### Product & Feature Analysis
- **Feature rollout impact**: Measure true lift from new features beyond selection effects
- **UI/UX changes**: Isolate design impact from user self-selection
- **Pricing changes**: Estimate elasticity when users choose their plans
- **Platform migrations**: Quantify the causal effect of moving users to new systems

### Marketing & Growth
- **Campaign effectiveness**: Separate campaign impact from organic trends
- **Channel attribution**: Understand true incremental value of marketing channels
- **Retention interventions**: Measure causal impact of win-back campaigns
- **Geographic expansions**: Estimate market entry effects using synthetic controls

### Business Operations
- **Policy changes**: Evaluate impact of new policies on user behavior
- **Competitive effects**: Measure how competitor actions affect your metrics
- **Seasonal adjustments**: Separate true treatment effects from seasonality
- **Long-term impacts**: Track how effects evolve over months/years

## Why Automation Matters

Modern causal inference has seen an explosion of methods in recent years. Choosing correctly requires deep knowledge of:
- Parallel trends assumptions
- Staggered treatment timing
- Heterogeneous treatment effects
- Two-way fixed effects bias
- Synthetic control construction

PyAutoCausal encodes this expertise, automatically routing your analysis through the appropriate methods while maintaining transparency about assumptions and limitations.

## Installation

```bash
pip install pyautocausal
```

Or for development:

```bash
git clone https://github.com/yourusername/pyautocausal.git
cd pyautocausal
poetry install
```

## Core Concepts

### Graph-Based Pipeline Architecture

PyAutoCausal organizes causal analysis as directed graphs of computational nodes:

```python
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.persistence.output_config import OutputConfig, OutputType

# Build custom pipelines using the graph API
graph = (ExecutableGraph()
    .configure_runtime(output_path="./outputs")
    .create_input_node("data", input_dtype=pd.DataFrame)
    .create_decision_node("has_multiple_periods", 
                         condition=lambda df: df['t'].nunique() > 1,
                         predecessors=["data"])
    .create_node("cross_sectional_analysis", 
                cross_sectional_estimator,
                predecessors=["has_multiple_periods"])
    .create_node("panel_analysis",
                panel_estimator, 
                predecessors=["has_multiple_periods"])
    .when_false("has_multiple_periods", "cross_sectional_analysis")
    .when_true("has_multiple_periods", "panel_analysis")
)

graph.fit(data=your_dataframe)
```

### Automated Method Selection

The framework automatically routes your data through appropriate causal inference methods:

- **Cross-sectional** (single time period) → OLS with robust inference
- **Panel with single treated unit** → Synthetic control methods
- **Panel with multiple treatment timing** → Modern DiD estimators
- **Staggered treatment adoption** → Callaway-Sant'Anna, BACON decomposition
- **Large datasets** → Double/debiased machine learning approaches

### Built-in Validation

Every analysis includes:
- **Data quality checks**: Missing values, duplicates, proper formatting
- **Assumption testing**: Parallel trends, common support, balance
- **Robustness checks**: Alternative specifications and estimators
- **Diagnostic plots**: Visual assumption validation

## Project Structure

```
pyautocausal/
├── orchestration/          # Core graph execution framework
│   ├── graph.py            # ExecutableGraph class and execution logic
│   ├── nodes.py            # Node types (standard, decision, input)
│   └── ...
├── pipelines/              # Pre-built causal inference workflows
│   ├── library/            # Reusable causal analysis components
│   │   ├── specifications.py  # Treatment/outcome specifications
│   │   ├── estimators.py      # Statistical estimators
│   │   ├── conditions.py      # Data characteristic detectors
│   │   ├── plots.py           # Visualization functions
│   │   └── ...
│   └── example_graph.py    # Main causal inference pipeline
├── causal_methods/         # Core statistical methods
│   └── double_ml.py        # DoubleML implementation
├── persistence/            # Output handling and export
│   ├── notebook_export.py  # Jupyter notebook generation
│   ├── output_config.py    # Output format configuration
│   └── ...
└── utils/                  # Utility functions
```

## Next Steps

- **📖 [Getting Started Guide](docs/getting-started.md)** - Step-by-step tutorial
- **📊 [Causal Methods Reference](docs/causal-methods.md)** - All available estimators
- **🔧 [Pipeline Development](docs/pipeline-guide.md)** - Building custom workflows
- **📋 [Data Requirements](docs/data-requirements.md)** - Input formats and validation
- **💡 [Examples](docs/examples/)** - Real-world case studies

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## License

[MIT License](LICENSE)

## Citation

If you use PyAutoCausal in your research, please cite:

```bibtex
@software{pyautocausal,
  title={PyAutoCausal: Automated Causal Inference Pipelines},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/pyautocausal}
}
```
