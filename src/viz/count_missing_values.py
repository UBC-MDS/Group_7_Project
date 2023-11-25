import pandas as pd

def count_missing_values(dataframe):
    """
    Counts the number of missing values in each column of a DataFrame.

    Parameters:
    ----------
    data_frame : pandas.DataFrame
        The input DataFrame to analyze for missing values.

    Returns:
    -------
    pandas.Series
        A Series where the index is the column names of the DataFrame, and the values
        are the counts of missing values in each column.
        
    Examples:
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6]})
    >>> missing_values = count_missing_values(data)
    >>> print(missing_values)
    A    1
    B    1
    dtype: int64
    """
    
    return dataframe.isna().sum()