# Data Requirements

PyAutoCausal expects your data in a specific format to automatically detect the appropriate causal inference method. This guide explains the required and optional columns, data types, and formatting requirements.

## Core Data Structure

PyAutoCausal analyzes **panel data** - units observed over multiple time periods. Your dataset should be in **long format** where each row represents one unit-time observation.

### Required Columns

| Column | Name | Type | Description | Example Values |
|--------|------|------|-------------|----------------|
| **Unit ID** | `id_unit` | str/int | Unique identifier for each unit | `user_123`, `state_CA`, `1001` |
| **Time** | `t` | int | Time period indicator (0, 1, 2, ...) | `0, 1, 2, 3, 4` |
| **Treatment** | `treat` | int | Binary treatment indicator | `0` (control), `1` (treated) |
| **Outcome** | `y` | float | Dependent variable to measure | `revenue`, `engagement`, `clicks` |

### Optional Columns

| Column Pattern | Type | Description | Example |
|----------------|------|-------------|---------|
| `x1`, `x2`, `x_*` | float/int | Covariates/control variables | User age, income, segment |
| `x_cont_*` | float | Continuous covariates | Income, temperature, price |
| `x_cat_*` | str/int | Categorical covariates | Region, device type, plan |

## Example Dataset Structure

```python
import pandas as pd

# Correct format for PyAutoCausal
df = pd.DataFrame({
    'id_unit': [1, 1, 1, 2, 2, 2, 3, 3, 3],    # Unit identifier
    't': [0, 1, 2, 0, 1, 2, 0, 1, 2],          # Time periods
    'treat': [0, 0, 1, 0, 1, 1, 0, 0, 0],      # Treatment status
    'y': [10, 12, 18, 8, 15, 16, 9, 11, 12],   # Outcome variable
    'x1': [25, 25, 25, 30, 30, 30, 35, 35, 35], # Covariate (age)
    'x2': [1, 1, 1, 0, 0, 0, 1, 1, 1]          # Covariate (segment)
})

print(df)
```

```
   id_unit  t  treat   y  x1  x2
0        1  0      0  10  25   1
1        1  1      0  12  25   1
2        1  2      1  18  25   1    # Unit 1 becomes treated at t=2
3        2  0      0   8  30   0
4        2  1      1  15  30   0    # Unit 2 becomes treated at t=1
5        2  2      1  16  30   0
6        3  0      0   9  35   1
7        3  1      0  11  35   1    # Unit 3 never treated
8        3  2      0  12  35   1
```

## Data Formatting Requirements

### 1. Treatment Variable (`treat`)

**Must be binary (0/1):**
```python
# ✅ Correct
df['treat'] = [0, 0, 1, 1, 0]

# ❌ Incorrect - multiple treatment levels
df['treat'] = [0, 1, 2, 3, 0]  # Use separate indicators for multiple treatments

# ❌ Incorrect - string values  
df['treat'] = ['control', 'treated']  # Convert to 0/1
```

**Handle multiple treatments:**
```python
# For multiple treatments, create separate binary indicators
df['treat_feature_a'] = [0, 1, 1, 0, 0]
df['treat_feature_b'] = [0, 0, 1, 1, 0]

# Run separate analyses for each treatment
pipeline.fit(df=df.rename(columns={'treat_feature_a': 'treat'}))
```

### 2. Time Variable (`t`)

**Must be consecutive integers starting from 0:**
```python
# ✅ Correct
df['t'] = [0, 1, 2, 3, 4]

# ❌ Incorrect - gaps in time
df['t'] = [0, 1, 3, 4, 5]  # Missing t=2

# ❌ Incorrect - dates
df['t'] = ['2024-01-01', '2024-01-02']  # Convert to integers

# Convert dates to time periods
dates = pd.to_datetime(['2024-01-01', '2024-01-08', '2024-01-15'])
df['t'] = (dates - dates.min()).dt.days // 7  # Weekly periods: [0, 1, 2]
```

### 3. Unit ID (`id_unit`)

**Must uniquely identify units:**
```python
# ✅ Correct - unique identifiers
df['id_unit'] = ['user_123', 'user_456', 'user_789']

# ✅ Correct - integer IDs
df['id_unit'] = [1001, 1002, 1003]

# ❌ Incorrect - non-unique IDs
df['id_unit'] = ['A', 'A', 'B']  # 'A' appears multiple times for different observations
```

### 4. Missing Values

**Handle explicitly:**
```python
# ✅ Mark missing values as NaN
df['y'] = [10, np.nan, 12, 15]

# ❌ Don't use placeholder values
df['y'] = [10, -999, 12, 15]  # PyAutoCausal will treat -999 as real value
```

## Common Data Scenarios

### Scenario 1: Staggered Treatment Adoption

Different units adopt treatment at different times:

```python
# E-commerce feature rollout
data = pd.DataFrame({
    'id_unit': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
    't': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
    'treat': [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],  # Staggered adoption
    'y': [100, 105, 125, 130, 95, 115, 120, 125, 90, 92, 94, 110]
})
# → PyAutoCausal will use Callaway-Sant'Anna or similar modern DiD methods
```

### Scenario 2: Geographic Policy Analysis

State-level policy implementation:

```python
# Minimum wage policy analysis
states_data = pd.DataFrame({
    'id_unit': ['CA', 'CA', 'TX', 'TX', 'NY', 'NY'],
    't': [0, 1, 0, 1, 0, 1],
    'treat': [0, 1, 0, 0, 0, 1],  # CA and NY implement policy
    'y': [45000, 48000, 42000, 42500, 50000, 52000],  # Average wages
    'x1': [35.2, 35.3, 27.1, 27.2, 19.8, 19.9]  # Population (millions)
})
```

### Scenario 3: Marketing Campaign Analysis

```python
# Customer segment campaign targeting
campaign_data = pd.DataFrame({
    'id_unit': range(1000),  # Customer IDs
    't': [i // 1000 for i in range(4000)],  # 4 time periods
    'treat': [...],  # Campaign exposure
    'y': [...],  # Purchase amount
    'x1': [...],  # Customer lifetime value
    'x2': [...],  # Previous purchase frequency
    'x_cat_1': [...]  # Customer segment (Premium, Standard, Basic)
})
```

## Data Validation

PyAutoCausal automatically validates your data and provides helpful error messages:

```python
from pyautocausal.pipelines.example_graph import causal_pipeline

pipeline = causal_pipeline(output_path="./results")

# Common validation errors and fixes:
try:
    pipeline.fit(df=your_data)
except ValueError as e:
    print(f"Data validation error: {e}")
    # Follow the suggested fixes in the error message
```

### Common Validation Issues

**Missing required columns:**
```
Error: Required column 'id_unit' not found
→ Add unit identifier column as 'id_unit'
```

**Treatment variation issues:**
```
Error: No treatment variation detected
→ Ensure 'treat' column has both 0s and 1s
```

**Time period issues:**
```
Error: Time periods are not consecutive
→ Convert to consecutive integers: t = 0, 1, 2, ...
```

## Data Size Considerations

### Minimum Requirements
- **Cross-sectional**: ≥50 observations
- **Panel data**: ≥20 units × 3 time periods
- **Staggered treatment**: ≥10 units per cohort

### Performance Guidelines
- **Small data** (<1,000 obs): All methods available
- **Medium data** (1,000-10,000 obs): Most methods, may use robust algorithms
- **Large data** (>10,000 obs): May automatically select scalable methods (e.g., DoubleML)

## Best Practices

### 1. Data Quality
```python
# Check for common issues
print("Data shape:", df.shape)
print("Missing values:", df.isnull().sum())
print("Treatment variation:", df['treat'].value_counts())
print("Time periods:", sorted(df['t'].unique()))
print("Units per period:", df.groupby('t')['id_unit'].nunique())
```

### 2. Treatment Timing
```python
# Visualize treatment adoption
import matplotlib.pyplot as plt

treatment_timing = df[df['treat'] == 1].groupby('id_unit')['t'].min()
plt.hist(treatment_timing, bins=20)
plt.title('Treatment Adoption Timing')
plt.xlabel('Time Period')
plt.ylabel('Number of Units')
plt.show()
```

### 3. Outcome Variable
```python
# Check outcome distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
df.boxplot(column='y', by='treat')
plt.title('Outcome by Treatment')

plt.subplot(1, 3, 2)
df.boxplot(column='y', by='t')
plt.title('Outcome by Time')

plt.subplot(1, 3, 3)
df.groupby(['t', 'treat'])['y'].mean().unstack().plot()
plt.title('Average Outcome Over Time')
plt.show()
```

With properly formatted data, PyAutoCausal can automatically select and execute the most appropriate causal inference method for your analysis. 