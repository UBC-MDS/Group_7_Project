import pytest
import pandas as pd
import sys
import os

# Get the absolute path of the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# Add the 'src' directory to the Python path
sys.path.insert(0, project_root)

# tests for function count_missing_values
from src.viz.count_missing_values import count_missing_values

def test_with_no_missing_values():
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    assert count_missing_values(data).equals(pd.Series({'A': 0, 'B': 0}))

def test_with_all_missing_values():
    data = pd.DataFrame({'A': [None, None, None], 'B': [None, None, None]})
    assert count_missing_values(data).equals(pd.Series({'A': 3, 'B': 3}))

def test_with_some_missing_values():
    data = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, None]})
    assert count_missing_values(data).equals(pd.Series({'A': 1, 'B': 1}))


# tests for function plot_value_counts
from src.viz.plot_value_counts import plot_value_counts

def test_valid_input():
    df = pd.DataFrame({'col1': ['a', 'b', 'a']})
    # No assertion; just checking for absence of errors
    plot_value_counts(df, 'col1')

def test_non_existent_column():
    df = pd.DataFrame({'col1': ['a', 'b', 'a']})
    with pytest.raises(ValueError):
        plot_value_counts(df, 'non_existent')

def test_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        plot_value_counts(df, 'col1')