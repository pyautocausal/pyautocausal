# Getting Started with PyAutoCausal

This guide walks you through your first causal inference analysis using PyAutoCausal, from data preparation to interpreting results.

## Prerequisites

- **Python 3.8+** with pandas, numpy
- **Basic causal inference knowledge**: Understanding of treatment/control, confounders, and parallel trends
- **Panel data**: At minimum, you need units observed over time with treatment variation

## Installation

```bash
pip install pyautocausal
```

## Your First Analysis: Feature Rollout Impact

Let's analyze whether a new app feature actually increased user engagement.

### Step 1: Prepare Your Data

PyAutoCausal expects your data in a specific format. Here's what you need:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Example: Feature rollout data
# Real data would come from your analytics warehouse
np.random.seed(42)

# Create sample data: users, time periods, feature rollout
users = range(1000)
dates = pd.date_range('2024-01-01', periods=12, freq='W')

data = []
for user in users:
    for i, date in enumerate(dates):
        # Feature rolled out to different users at different times
        if user < 300:
            treat_start = 4  # Early adopters
        elif user < 600:
            treat_start = 6  # Mid rollout
        elif user < 800:
            treat_start = 8  # Late rollout
        else:
            treat_start = 15  # Never treated (control)
        
        treated = 1 if i >= treat_start else 0
        
        # Engagement with treatment effect + individual + time trends
        baseline_engagement = 10 + (user % 50) / 10  # User heterogeneity
        time_trend = i * 0.1  # Overall growth
        treatment_effect = 2.5 * treated if treated else 0
        noise = np.random.normal(0, 1)
        
        engagement = baseline_engagement + time_trend + treatment_effect + noise
        
        data.append({
            'id_unit': user,      # Required: Unit identifier
            't': i,               # Required: Time period
            'treat': treated,     # Required: Treatment indicator
            'y': engagement,      # Required: Outcome variable
            'x1': user % 10,      # Optional: User segment
            'x2': np.random.normal(0, 1)  # Optional: Time-varying control
        })

df = pd.DataFrame(data)
print(f"Dataset shape: {df.shape}")
print(f"Treatment timing:\n{df.groupby('id_unit')['treat'].apply(lambda x: x.idxmax() if x.sum() > 0 else 'Never').value_counts()}")
```

### Step 2: Run the Analysis

```python
from pyautocausal.pipelines.example_graph import causal_pipeline

# Create the causal inference pipeline
pipeline = causal_pipeline(output_path="./feature_analysis_results")

# PyAutoCausal automatically detects:
# - This is panel data (multiple periods)
# - Treatment is staggered (different adoption times)
# - Multiple treated units exist
# → Routes to Callaway-Sant'Anna estimator

pipeline.fit(df=df)
```

### Step 3: Examine the Results

PyAutoCausal saves comprehensive results to your output directory:

```
feature_analysis_results/
├── plots/
│   ├── callaway_santanna_never_treated_plot.png    # Event study plot
│   └── staggered_event_study_plot.png              # Alternative visualization
├── text/
│   ├── callaway_santanna_never_treated_results.txt # Statistical results
│   └── save_stag_output.txt                        # Panel regression
└── notebooks/
    └── causal_pipeline_execution.ipynb             # Interactive analysis
```

**Key outputs to examine:**

1. **Statistical Results** (`callaway_santanna_never_treated_results.txt`):
```
Callaway-Sant'Anna Results:
Overall ATT: 2.48 (SE: 0.12, p < 0.001)
[Detailed cohort and time-specific effects...]
```

2. **Event Study Plot** (`callaway_santanna_never_treated_plot.png`):
Shows dynamic treatment effects over time since adoption

3. **Interactive Notebook** (`causal_pipeline_execution.ipynb`):
Jupyter notebook with all analysis steps for exploration and customization

### Step 4: Interpret Your Results

```python
# Load and examine detailed results
import matplotlib.pyplot as plt

# The notebook contains all intermediate results
# Key findings to look for:

# 1. Overall Average Treatment Effect (ATT)
# "The feature increased engagement by 2.48 units on average"

# 2. Dynamic Effects
# "Effects appear immediately and remain stable over time"

# 3. Heterogeneity Across Cohorts
# "Early vs late adopters show similar treatment effects"

# 4. Assumption Validation
# "Parallel trends assumption appears reasonable"
```

## Understanding What PyAutoCausal Did

The pipeline automatically:

1. **Detected data structure**:
   - Panel data: ✓ (multiple time periods)
   - Multiple treated units: ✓
   - Staggered treatment: ✓
   - Never-treated control group: ✓

2. **Selected appropriate method**:
   - Primary: Callaway-Sant'Anna with never-treated comparison
   - Alternative: Standard panel regression for robustness

3. **Validated assumptions**:
   - Checked for proper data formatting
   - Ensured treatment variation
   - Verified control group availability

4. **Generated outputs**:
   - Point estimates with confidence intervals
   - Dynamic treatment effects over time
   - Heterogeneity analysis across adoption cohorts
   - Diagnostic plots and assumption checks

## Common Data Scenarios

### Scenario 1: Single Time Period (Cross-sectional)
```python
# If your data has only one time period
cross_sectional_data = df[df['t'] == 0].copy()
pipeline.fit(df=cross_sectional_data)
# → Automatically uses OLS with robust standard errors
```

### Scenario 2: Single Treated Unit
```python
# Policy change affecting one state/region
single_unit_data = df[df['id_unit'].isin([0, 1, 2])].copy()  # One treated + controls
pipeline.fit(df=single_unit_data)
# → Automatically uses Synthetic Control methods
```

### Scenario 3: Large Dataset
```python
# When N×T is very large (>10k observations)
large_data = pd.concat([df] * 10)  # Simulate large dataset
pipeline.fit(df=large_data)
# → May use DoubleML for better performance
```

## Next Steps

- **📊 [Causal Methods Reference](causal-methods.md)** - Learn about all available estimators
- **📋 [Data Requirements](data-requirements.md)** - Detailed data formatting guide
- **🔧 [Pipeline Development](pipeline-guide.md)** - Build custom analysis workflows
- **💡 [Examples](examples/)** - More real-world case studies

## Troubleshooting

### Common Issues

**"No treatment variation detected"**
- Ensure your `treat` column has both 0s and 1s
- Check that treatment changes over time or across units

**"Insufficient control periods"**
- Make sure you have pre-treatment observations for treated units
- Consider using a different comparison group

**"Parallel trends violation"**
- Examine pre-treatment trends in your data
- Consider adding more control variables
- Try alternative identification strategies

### Getting Help

- Check the generated Jupyter notebook for detailed diagnostics
- Review assumption validation outputs in the text files
- Examine diagnostic plots for visual validation

The automated pipeline handles most cases, but domain knowledge remains crucial for interpreting results and validating assumptions. 