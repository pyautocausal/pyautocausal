"""Example usage of data validation and cleaning together in PyAutoCausal.

This module demonstrates how validation results flow into automated cleaning.
"""

import pandas as pd
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.persistence.output_config import OutputConfig, OutputType
from pyautocausal.data_validation import (
    DataValidator,
    DataValidatorConfig,
    NonEmptyDataCheck,
    RequiredColumnsCheck,
    ColumnTypesCheck,
    MissingDataCheck,
)
from pyautocausal.data_validation.checks.basic_checks import (
    ColumnTypesConfig,
)
from pyautocausal.data_cleaning import (
    DataCleaningPlanner,
    DataCleaner,
)


def create_validation_cleaning_graph() -> ExecutableGraph:
    """Create a graph that validates and then cleans data."""
    
    # Configure validation
    validator_config = DataValidatorConfig(
        check_configs={
            "column_types": ColumnTypesConfig(
                categorical_threshold=10,  # Columns with <= 10 unique values are categorical
                infer_categorical=True
            ),
        }
    )
    
    validator = DataValidator(
        checks=[
            NonEmptyDataCheck,
            RequiredColumnsCheck,
            ColumnTypesCheck,
            MissingDataCheck,
        ],
        config=validator_config
    )
    
    # Create the graph
    graph = ExecutableGraph()
    graph.configure_runtime(output_path="./cleaning_outputs")
    
    # Input node
    graph.create_input_node("df", input_dtype=pd.DataFrame)
    
    # Validation node
    graph.create_node(
        "validate",
        action_function=validator,
        predecessors=["df"],
        output_config=OutputConfig(
            output_filename="validation_results",
            output_type=OutputType.JSON
        ),
        save_node=True
    )
    
    # Planning node - creates cleaning plan from validation results
    graph.create_node(
        "plan_cleaning",
        action_function=DataCleaningPlanner(),
        predecessors=["validate"]
    )
    
    # Cleaning node - executes the plan
    # Note: takes both raw df and plan as inputs
    graph.create_node(
        "clean_data",
        action_function=lambda df, plan_cleaning: DataCleaner(plan_cleaning).clean(df),
        predecessors=["df", "plan_cleaning"],
        output_config=OutputConfig(
            output_filename="cleaning_metadata",
            output_type=OutputType.JSON
        ),
        save_node=True
    )
    
    # Continue with analysis using cleaned data
    graph.create_node(
        "analyze",
        action_function=lambda clean_data: {
            "message": "Analysis would happen here",
            "cleaned_shape": clean_data[0].shape,  # clean_data is tuple (df, metadata)
            "cleaning_summary": clean_data[1].to_dict()
        },
        predecessors=["clean_data"],
        output_config=OutputConfig(
            output_filename="analysis_results",
            output_type=OutputType.JSON
        ),
        save_node=True
    )
    
    return graph


def demonstrate_cleaning_flow():
    """Demonstrate the validation -> planning -> cleaning flow."""
    
    # Create sample data with issues that can be cleaned
    sample_data = pd.DataFrame({
        "user_id": ["A", "B", "C", "D", "E"] * 20,  # 100 rows, 5 unique values
        "status": ["active", "inactive", "pending", None, "active"] * 20,  # Has missing
        "score": [10, 20, 30, 40, 50] * 20,
        "category": ["X", "Y", "X", "Y", "Z"] * 20,  # 3 unique values
    })
    
    print("Original data shape:", sample_data.shape)
    print("\nColumn info:")
    print(sample_data.info())
    print("\nUnique values per column:")
    for col in sample_data.columns:
        print(f"  {col}: {sample_data[col].nunique()} unique values")
    
    # Create and run the graph
    graph = create_validation_cleaning_graph()
    graph.fit(df=sample_data)
    
    # Get the cleaned data from the graph
    clean_node = graph.get("clean_data")
    if clean_node.is_completed():
        cleaned_df, metadata = clean_node.get_result_value()
        
        print("\n" + "="*50)
        print("CLEANING RESULTS")
        print("="*50)
        print(f"\nCleaned data shape: {cleaned_df.shape}")
        print(f"Total operations: {metadata.total_operations}")
        print(f"Rows dropped: {metadata.total_rows_dropped}")
        print(f"Columns modified: {metadata.total_columns_added + metadata.total_columns_dropped}")
        
        print("\nTransformations applied:")
        for i, transform in enumerate(metadata.transformations, 1):
            print(f"{i}. {transform.operation_name}")
            if transform.columns_modified:
                print(f"   Modified: {', '.join(transform.columns_modified)}")
            if transform.details:
                print(f"   Details: {transform.details}")
        
        print("\nCleaned data types:")
        print(cleaned_df.dtypes)


def demonstrate_cleaning_without_graph():
    """Show how to use validation and cleaning outside of a graph."""
    
    # Create sample data
    df = pd.DataFrame({
        "color": ["red", "blue", "green", "red", "blue"] * 10,
        "size": ["S", "M", "L", "XL", None] * 10,
        "price": [10.5, 20.0, None, 15.5, 25.0] * 10,
    })
    
    # Step 1: Validate
    validator = DataValidator(
        checks=[ColumnTypesCheck],
        config=DataValidatorConfig(
            check_configs={
                "column_types": ColumnTypesConfig(
                    categorical_threshold=5,
                    infer_categorical=True
                )
            }
        )
    )
    
    validation_results = validator.validate(df)
    print("Validation complete. Inferred categorical columns:")
    for result in validation_results.individual_results:
        if result.check_name == "column_types":
            print(f"  {result.metadata.get('inferred_categorical', [])}")
    
    # Step 2: Plan cleaning
    planner = DataCleaningPlanner(validation_results)
    plan = planner.create_plan()
    print("\n" + plan.describe())
    
    # Step 3: Execute cleaning
    cleaned_df, metadata = plan(df)
    
    print(f"\nCleaning complete!")
    print(f"Original dtypes:\n{df.dtypes}")
    print(f"\nCleaned dtypes:\n{cleaned_df.dtypes}")


if __name__ == "__main__":
    print("Example 1: Full graph with validation and cleaning")
    print("-" * 70)
    demonstrate_cleaning_flow()
    
    print("\n\nExample 2: Standalone validation and cleaning")
    print("-" * 70)
    demonstrate_cleaning_without_graph() 