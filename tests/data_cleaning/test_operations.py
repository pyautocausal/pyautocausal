import pandas as pd
import pytest
from pyautocausal.data_cleaning.operations.categorical_operations import ConvertToCategoricalOperation
from pyautocausal.data_cleaning.operations.duplicate_operations import DropDuplicateRowsOperation
from pyautocausal.data_cleaning.operations.missing_data_operations import DropMissingRowsOperation
from pyautocausal.data_cleaning.operations.schema_operations import UpdateColumnTypesOperation
from pyautocausal.data_cleaning.hints import InferCategoricalHint, DropMissingRowsHint, DropDuplicateRowsHint, UpdateColumnTypesHint


@pytest.fixture
def sample_df():
    """A sample DataFrame for testing cleaning operations."""
    return pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": ["foo", "bar", "foo", "bar", "baz"],
            "C": [1.1, 2.2, 3.3, 4.4, 5.5],
            "D": [1, 2, 1, 2, 3],
        }
    )


def test_update_column_types(sample_df):
    op = UpdateColumnTypesOperation()
    hint = UpdateColumnTypesHint(type_mapping={"A": "str", "D": "category"})
    cleaned_df, _ = op.apply(sample_df, hint)
    assert cleaned_df["A"].dtype == "object" # Pandas uses 'object' for strings
    assert cleaned_df["D"].dtype == "category"
    assert cleaned_df["B"].dtype == "object"  # Unchanged


def test_convert_to_categorical(sample_df):
    op = ConvertToCategoricalOperation()
    hint = InferCategoricalHint(target_columns=["B", "D"], threshold=10, unique_counts={})
    cleaned_df, _ = op.apply(sample_df, hint)
    assert cleaned_df["B"].dtype == "category"
    assert cleaned_df["D"].dtype == "category"
    assert cleaned_df["A"].dtype == "int64"  # Unchanged


def test_drop_missing_rows():
    df = pd.DataFrame({"A": [1, 2, None, 4], "B": [1, None, 3, 4]})
    op = DropMissingRowsOperation()
    hint = DropMissingRowsHint(target_columns=["A"])
    cleaned_df, _ = op.apply(df, hint)
    assert len(cleaned_df) == 3
    assert cleaned_df["A"].isnull().sum() == 0

    hint = DropMissingRowsHint(target_columns=["A", "B"])
    cleaned_df, _ = op.apply(df, hint)
    assert len(cleaned_df) == 2
    assert cleaned_df.isnull().sum().sum() == 0


def test_drop_duplicate_rows():
    df = pd.DataFrame({"A": [1, 2, 1, 4], "B": ["a", "b", "a", "d"]})
    op = DropDuplicateRowsOperation()
    hint = DropDuplicateRowsHint()
    cleaned_df, _ = op.apply(df, hint)
    assert len(cleaned_df) == 3
    assert cleaned_df.iloc[2].to_dict() == {"A": 4, "B": "d"} 