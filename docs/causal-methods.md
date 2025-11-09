# Causal Methods Reference

PyAutoCausal implements a comprehensive suite of modern causal inference methods. This reference explains when each method is automatically selected, what assumptions they make, and how to interpret their results.

## Method Selection Logic

PyAutoCausal analyzes your data characteristics and automatically routes to the most appropriate estimator:

```
Data Analysis Flow:
└── Multiple time periods?
    ├── No → Cross-sectional Analysis (OLS)
    └── Yes → Panel Data Analysis
        └── Single treated unit?
            ├── Yes → Synthetic Control Methods
            └── No → Multiple treated units
                └── Staggered treatment timing?
                    ├── No → Standard DiD or Event Study
                    └── Yes → Modern DiD Methods
                        └── Never-treated units available?
                            ├── Yes → Callaway-Sant'Anna (never-treated)
                            └── No → Callaway-Sant'Anna (not-yet-treated)
```

## Available Methods

### 1. Cross-Sectional Analysis

**When used**: Single time period data (t=0 only)

**Method**: OLS with robust standard errors

**Assumptions**:
- No unmeasured confounders (conditional independence)
- Correct functional form specification
- No selection bias after controlling for observables

**Example**:
```python
# Single period data automatically triggers OLS
single_period_data = df[df['t'] == 0]
pipeline.fit(df=single_period_data)
```

**Outputs**:
- Treatment effect estimate with robust standard errors
- Coefficient table with p-values and confidence intervals
- Model diagnostics and R-squared

**Interpretation**:
```
Treatment Effect: 2.34 (SE: 0.45, p=0.000)
→ "Controlling for observables, treatment increases outcome by 2.34 units"
```

---

### 2. Difference-in-Differences (DiD)

**When used**: Panel data with treatment variation over time

**Method**: Two-way fixed effects regression

**Assumptions**:
- **Parallel trends**: Treatment and control groups would follow parallel paths absent treatment
- **No anticipation**: Units don't change behavior before treatment
- **Stable composition**: Same units observed throughout
- **Common shocks**: Time effects are common across groups

**Implementation**:
```
y_it = α_i + γ_t + β×Treat_it + X_it'δ + ε_it
```

Where:
- `α_i`: Unit fixed effects
- `γ_t`: Time fixed effects  
- `β`: Treatment effect (ATT)

**Example**:
```python
# 2 periods, clean treatment/control split
classic_did_data = df[(df['t'].isin([0, 1]))]
pipeline.fit(df=classic_did_data)
```

**Outputs**:
- Average Treatment Effect on the Treated (ATT)
- Event study plots (if multiple post-periods)
- Parallel trends diagnostic plots

---

### 3. Synthetic Control

**When used**: Single treated unit with multiple control units

**Method**: Synthetic Difference-in-Differences (Arkhangelsky et al.)

**Assumptions**:
- **No interference**: Treatment of one unit doesn't affect others
- **Parallel trends**: Synthetic control approximates counterfactual
- **Common factors**: Shared time-varying confounders across units

**Process**:
1. Constructs synthetic control as weighted average of untreated units
2. Ensures pre-treatment fit between treated and synthetic control
3. Estimates post-treatment differences

**Example**:
```python
# Policy change affecting single state/region
single_treated = df[df['id_unit'].isin(['CA'] + control_states)]
pipeline.fit(df=single_treated)
```

**Outputs**:
- Treatment effect trajectory over time
- Synthetic control weights
- Placebo tests using control units
- Pre-treatment fit diagnostics

---

### 4. Event Study Analysis

**When used**: Multiple periods with non-staggered treatment timing

**Method**: Dynamic difference-in-differences with event time

**Assumptions**:
- Standard DiD assumptions
- **No anticipation**: Effects only occur post-treatment
- **Treatment effect evolution**: Effects may change over time

**Implementation**:
```
y_it = α_i + γ_t + Σ β_k×D_it^k + X_it'δ + ε_it
```

Where `D_it^k` indicates k periods since treatment.

**Example**:
```python
# Multiple periods, treatment occurs at same time for all treated units
event_study_data = df  # Non-staggered treatment
pipeline.fit(df=event_study_data)
```

**Outputs**:
- Dynamic treatment effects (β₋₂, β₋₁, β₀, β₁, β₂, ...)
- Event study plot with confidence intervals
- Pre-treatment trend test (β₋₁ = β₋₂ = 0)

---

### 5. Callaway-Sant'Anna Method

**When used**: Staggered treatment adoption with heterogeneous effects

**Method**: Group-time average treatment effects with aggregation

**Assumptions**:
- **Conditional parallel trends**: Within comparison groups
- **No anticipation**: Treatment effects don't precede treatment
- **Irreversible treatment**: Once treated, always treated

**Approach**:
1. Estimates group-time specific effects ATT(g,t)
2. Aggregates to overall, dynamic, or group-specific effects
3. Uses never-treated or not-yet-treated comparison groups

**Two variants**:

#### Never-treated comparison:
```python
# Uses units never receiving treatment as controls
# Better if never-treated units are representative
```

#### Not-yet-treated comparison:
```python
# Uses units not yet treated at time t as controls  
# Better when all units eventually receive treatment
```

**Example**:
```python
# Staggered rollout with some never-treated units
staggered_data = df  # Different treatment start times
pipeline.fit(df=staggered_data)
```

**Outputs**:
- Overall ATT across all treated groups and periods
- Dynamic effects: ATT by time since treatment
- Group-specific effects: ATT by treatment cohort
- Event study plots with uniform confidence bands

---

### 6. BACON Decomposition

**When used**: Automatic diagnostics for two-way fixed effects with staggered treatment

**Method**: Goodman-Bacon decomposition of TWFE estimator

**Purpose**:
- Diagnoses potential bias in standard TWFE estimators
- Shows weight of each 2×2 DiD comparison
- Identifies problematic comparisons (already-treated as controls)

**Example**:
```python
# Automatically runs with staggered treatment data
# Provides diagnostic information about TWFE bias
```

**Outputs**:
- Decomposition of TWFE coefficient
- Weights on each comparison group
- Bias diagnostic plots

---

### 7. Double/Debiased Machine Learning

**When used**: Large datasets or high-dimensional controls

**Method**: Cross-fitted machine learning with Neyman orthogonal moments

**Assumptions**:
- **Conditional ignorability**: No unmeasured confounders given X
- **Overlap**: Positive probability of treatment for all covariate values
- **Regularity conditions**: For ML estimation consistency

**Process**:
1. Uses ML to predict treatment and outcome
2. Cross-fitting prevents overfitting bias
3. Estimates causal parameter using orthogonal moments

**Example**:
```python
# Large dataset triggers DoubleML
large_dataset = pd.concat([df] * 100)  # >10k observations
pipeline.fit(df=large_dataset)
```

**Outputs**:
- Causal parameter estimate with confidence intervals
- Variable importance from ML models
- Cross-fitting diagnostics

---

## Method Comparison

| Method | Data Structure | Key Advantage | Main Limitation |
|--------|---------------|---------------|-----------------|
| **OLS** | Cross-sectional | Simple, interpretable | Strong ignorability assumption |
| **Standard DiD** | 2×2 panel | Differences out fixed confounders | Parallel trends assumption |
| **Synthetic Control** | Single treated unit | Transparent counterfactual | Single unit generalization |
| **Event Study** | Multiple periods | Shows dynamic effects | Non-staggered treatment only |
| **Callaway-Sant'Anna** | Staggered treatment | Robust to heterogeneous effects | Complex interpretation |
| **DoubleML** | High-dimensional | Flexible confounder control | Black box ML components |

## Assumption Testing

PyAutoCausal automatically generates diagnostic tests:

### Parallel Trends Tests
```python
# Pre-treatment trend analysis
# Visual inspection of pre-treatment differences
# Statistical tests of differential trends
```

### Balance Tests
```python
# Covariate balance across treatment groups
# Overlap/common support diagnostics
# Propensity score distributions
```

### Robustness Checks
```python
# Alternative specifications
# Different comparison groups
# Sensitivity to sample restrictions
```

## Interpreting Results

### Treatment Effect Magnitudes

**Statistical Significance**:
- `p < 0.01`: Strong evidence of causal effect
- `p < 0.05`: Moderate evidence  
- `p > 0.05`: Insufficient evidence

**Economic Significance**:
- Compare effect size to outcome standard deviation
- Consider practical importance in business context
- Account for confidence interval width

### Dynamic Effects

**Event Study Interpretation**:
```
β₋₂ = 0.1 (p=0.8)  → No pre-trend (good)
β₋₁ = 0.2 (p=0.6)  → No anticipation (good)
β₀ = 1.5 (p=0.01)  → Immediate effect
β₁ = 2.1 (p=0.001) → Growing effect
β₂ = 2.0 (p=0.002) → Persistent effect
```

### Heterogeneity Analysis

**Group-Specific Effects**:
- Early vs. late adopters
- Different treatment intensities
- Subgroup analysis by covariates

## Best Practices

### 1. Always Examine Diagnostics
```python
# Check the generated diagnostic plots
# Review assumption test results  
# Examine pre-treatment trends
```

### 2. Consider Multiple Specifications
```python
# Run robustness checks with different methods
# Vary sample restrictions
# Test alternative comparison groups
```

### 3. Domain Knowledge Integration
```python
# Validate results against theoretical expectations
# Consider institutional details
# Assess external validity
```

### 4. Communicate Uncertainty
```python
# Report confidence intervals, not just point estimates
# Discuss assumption limitations
# Acknowledge potential confounders
```

The key advantage of PyAutoCausal is that it automatically selects the most appropriate method while maintaining transparency about assumptions and providing comprehensive diagnostics. However, domain expertise remains crucial for validating assumptions and interpreting results in context. 