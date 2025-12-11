# Design document of stock Causal Inference Graph

## Overview

This document outlines the design of a causal inference pipeline that automates data validation, method selection, and analysis based on data characteristics. The pipeline implements standard causal inference workflows while enforcing best practices.

## Purpose
1. Perform systematic exploratory data analysis (EDA) prior to causal modeling
2. Validate assumptions required for causal inference methods
3. Guide method selection based on data characteristics

## Data Schema Requirements

### Input Requirements

#### Supported File Formats
- CSV
- Parquet 
- Pandas DataFrame

#### Required Columns
| Column Type | Naming Convention | Description |
|------------|------------------|-------------|
| Outcome Variables | `y`, `y_1`, `y_2`, etc. | Target variables to measure causal effects |
| Unit Identifier | `id_unit` | Unique identifier for each observational unit |
| Time Variable | `period` | Time period indicator |
| Treatment Variables | `treat`, `treat_1`, `treat_2`, etc. | Treatment indicators with formats: |
|                    |                                      | • Binary: 0/1 coding |
|                    |                                      | • Continuous: Multiple values |
|                    |                                      | • Other formats will raise error |

#### Optional Columns  
| Column Type | Naming Convention | Description |
|------------|------------------|-------------|
| Covariates/Controls | `x`, `x_1`, `x_2`, etc. | Control variables |
| Continuous Covariates | `x_cont`, `x_cont_1`, etc. | Numeric control variables |
| Categorical Covariates | `x_cat`, `x_cat_1`, etc. | Categorical control variables |
| Treatment Timing | `treat_time` | When treatment occurred |
| Additional Unit IDs | `id_*` (e.g. `id_county`) | Secondary unit identifiers |

### Data Quality Requirements

| Requirement | Description |
|------------|-------------|
| Missing Values | Must be explicitly marked as NaN/null values |
| Duplicate Records | Must be either removed or explicitly handled with documented rationale |
| Variable Types | Must match their semantic meaning: |
|               | • Dates as datetime objects |
|               | • Categories as categorical/factor types |
|               | • Numeric values as appropriate numeric types |
| Data Consistency | Values must be consistent within columns (e.g., consistent units) |
| Data Range | Values must fall within valid/expected ranges for each variable |

## Primary Decision Points
1. Data type (cross-sectional vs. panel/longitudinal)
2. Treatment structure (binary, multi-valued, continuous)
3. Time structure (single period vs. multiple periods)
4. Missing data patterns (MCAR, MAR, MNAR)
5. Covariate balance between treatment groups
6. Sample size adequacy for statistical power

# Graph Flow

The graph proceeds in five main steps: (1) Preprocessing: validating and describing the data, (2) Specification Selection: determining the appropriate causal inference framework based on data characteristics, (3) Estimator Selection: choosing optimal estimation methods within the selected specification, (4) Model Estimation: fitting the selected models to the data, and (5) Post-estimation Analysis: generating model diagnostics, robustness checks, and interpretable outputs.


```mermaid
graph TD
    A[Input Data] --> B[Preprocessing]
    B --> C[Specification Selection]
    C --> D[Estimator Selection]
    D --> E[Post-estimation Output]

    %% All nodes in this graph are action nodes
    classDef actionNode fill:#d0e0ff,stroke:#3080cf,stroke-width:2px,color:black;
    class A,B,C,D,E actionNode;

```



## Preprocessing

This initial graph shows the basic data flow from input through validation, preprocessing, and descriptive analysis before entering the main analysis pipeline. It ensures data quality and prepares the dataset for causal analysis.

```mermaid
graph TD
    A[Input Data] --> B[Schema Validation]
    
    B --> C[Preprocessing]
    
    C --> D[Descriptives]
    D --> D1[Summary Statistics]
    D --> D2[Balancing Tables]
    
    C --> E[Analysis]
    
    %% All nodes in this graph are action nodes
    classDef actionNode fill:#d0e0ff,stroke:#3080cf,stroke-width:2px,color:black;
    class A,B,C,D,D1,D2,E actionNode;
```

## Analysis Graph Flow

This decision tree guides the selection of appropriate causal inference methodologies based on data structure, treatment patterns, and time periods. It helps determine whether standard approaches, difference-in-differences, or synthetic control methods are most appropriate for the dataset.

Possible specifications are:
- Standard Specification
- Difference-in-Differences (DiD) Specification
- Staggered DiD
- Santana DiD Specification

```mermaid
graph TD
    E[Analysis Input] --> F[Analysis]
    F --> F1[Multiple Periods]
    
    F1 -->|"False"| F4[Stand. Specif.]
     
    F1 -->|"True"| F7[Multiple Treated Units]
    F7 -->|"False"| F8[Stand. Specif]
    F7 -->|"More than 1"| F10[Multiple Pre-periods]
    
    F10 -->|"False"| F13[Multiple Post Periods]
    F10 -->|"True"| F12[Test Parallel Trends]

    F12 --> F13

    F13 -->|"False"| F15[DiD Specif]
    F13 -->|"True"| F14[Staggered Treatment]

    F14 -->|"False"| F15[DiD Specif]
    F14 -->|"True"| F16[Staggered DiD]

    F16 --> F17[DiD Specif]
    F16 --> F18[Santana DiD Specif]
    
    %% Node styling
    classDef decisionNode fill:#ffe0b0,stroke:#e09040,stroke-width:2px,color:black;
    classDef actionNode fill:#d0e0ff,stroke:#3080cf,stroke-width:2px,color:black;
    
    class F1,F3,F7,F10,F12,F14 decisionNode;
    class E,F,F4,F8,F13,F15,F16,F17,F18 actionNode;

  ```

## Post-Specification Analysis Flow

After selecting a specification, the model refinements determines whether standard OLS models, weighted approaches, or high-dimensional methods like Double Selection Lasso are most appropriate.

```mermaid
graph TD
    S[Selected Specification] --> C1[High Dimensionality]
    S[Selected Specification] --> S1[Single treated units]

    C1 --> |"True"| C2[DS Lasso]
    
    C1 --> |"False"| B1[Balanced]
    
    B1 -->|"True"| M1[Standard OLS]
    B1 -->|"False"| W1[Apply Weighting]
    S1 -->|"True"| W1

    W1 --> M1[Standard OLS]
    W1 --> P1[Weighted OLS]
    
    %% Node styling
    classDef decisionNode fill:#ffe0b0,stroke:#e09040,stroke-width:2px,color:black;
    classDef actionNode fill:#d0e0ff,stroke:#3080cf,stroke-width:2px,color:black;
    
    class S1,C1,B1 decisionNode;
    class S,C2,M1,W1,P1 actionNode;

```

## Post-Estimation Analysis Flow

Results and visualizations are saved while the model is passed to any downstream components in the broader analysis pipeline.

```mermaid
graph TD

    S[Estimated Model]  --> B1[Save Output]
    S --> B2[Save Plots]

    S--> A1[Rest of Graph]

    A1[Rest of Graph] --> A2[...]
    
    %% All nodes in this graph are action nodes
    classDef actionNode fill:#d0e0ff,stroke:#3080cf,stroke-width:2px,color:black;
    class S,B1,B2,A1,A2 actionNode;

```

## Node Color Legend

- <span style="background-color:#ffe0b0; color:black; padding:2px 6px; border:2px solid #e09040;">Decision Nodes</span>: Points where the pipeline branches based on data characteristics
- <span style="background-color:#d0e0ff; color:black; padding:2px 6px; border:2px solid #3080cf;">Action Nodes</span>: Processing steps, transformations, and outputs

## Node Overview


### Decision Nodes

Decision nodes test a condition and allow conditional execution of downstream nodes when true or false

### Action Nodes

Action nodes represent concrete processing steps and transformations in the pipeline:

#### Model Nodes

Model nodes implement specific estimation approaches wrapping packages like statsmodels or scikit and return baseestimator object

#### Model Output Nodes

Model output nodes take as input some Baseestimator and return string that can be save to txt file


#### Model Plot Nodes

Model plot nodes take as input some Baseestimator and save image or pdf file. 




