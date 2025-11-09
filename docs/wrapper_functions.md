# Working with Wrapper Functions and Notebook Export

PyAutoCausal's node-based architecture requires that function parameters match the names of predecessor nodes. While this provides clarity in your pipeline definition, it can lead to wrapper functions that obscure the actual implementation when exported to notebooks.

## The Challenge

Consider this common scenario:

```python
from external_library import complex_function

# The complex_function expects 'df' as parameter, but our node is named 'data'
def wrapper_function(data):
    return complex_function(df=data)

# In your graph
graph.create_node("process_data", wrapper_function, predecessors=["data"])
```

When this graph is exported to a notebook, users only see the wrapper function, not the actual implementation in `complex_function`.

## The Solution: `expose_in_notebook` Decorator

PyAutoCausal provides the `expose_in_notebook` decorator to solve this problem:

```python
from pyautocausal.persistence.notebook_decorators import expose_in_notebook
from external_library import complex_function

@expose_in_notebook(
    target_function=complex_function,
    arg_mapping={'data': 'df'}
)
def wrapper_function(data):
    return complex_function(df=data)
```

When the graph is exported to a notebook, both the wrapper and the target function will be included, with clear documentation on how they're related.

## Example

### Original Code

```python
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# Complex statistical function with specific parameter names
def run_ols_regression(df, formula=None):
    """Run OLS regression."""
    y = df['outcome']
    X = df[['treatment', 'covariate1', 'covariate2']]
    X = add_constant(X)
    model = OLS(y, X).fit()
    return model.summary()

# Wrapper function to adapt node names to parameter names
@expose_in_notebook(
    target_function=run_ols_regression,
    arg_mapping={'data': 'df'}
)
def ols_node_action(data):
    return run_ols_regression(df=data)

# In your graph
graph.create_node("ols", ols_node_action, predecessors=["data"])
```

### Generated Notebook

In the exported notebook, users will see:

```python
# This node uses a wrapper function that calls a target function with adapted arguments
# Argument mapping: 'data' → 'df'

def run_ols_regression(df, formula=None):
    """Run OLS regression."""
    y = df['outcome']
    X = df[['treatment', 'covariate1', 'covariate2']]
    X = add_constant(X)
    model = OLS(y, X).fit()
    return model.summary()

@expose_in_notebook(
    target_function=run_ols_regression,
    arg_mapping={'data': 'df'}
)
def ols_node_action(data):
    return run_ols_regression(df=data)

# In a different cell:
ols_output = ols_node_action(data_output)
# Alternatively, call the target function directly:
# ols_output = run_ols_regression(df=data_output)
```

This gives notebook users the full context and options for modifying the code.

## Benefits

- **Maintains clean graph structure**: Keep using node names that match parameter names
- **Exposes actual implementation**: Show users the real code, not just wrappers
- **Documents parameter mapping**: Clearly explains how parameters are adapted
- **Provides alternatives**: Users can choose to call either function
- **Preserves imports**: Automatically includes necessary import statements

## Advanced Usage

You can handle complex parameter mappings:

```python
@expose_in_notebook(
    target_function=complex_function,
    arg_mapping={
        'data': 'df',
        'settings': 'options',
        'alpha': 'significance_level'
    }
)
def wrapper(data, settings, alpha=0.05):
    return complex_function(
        df=data,
        options=settings,
        significance_level=alpha
    )
``` 