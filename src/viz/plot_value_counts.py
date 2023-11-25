import pandas as pd
import matplotlib.pyplot as plt

def plot_value_counts(data_frame, column) -> None:
    """
    Plot the distribution of values in a specified column of a DataFrame as a bar chart.

    Parameters:
    ----------
    data_frame : pandas.DataFrame
        The input DataFrame containing the data to analyze.
    column : str
        The name of the column to plot value counts for.

    Examples:
    --------
    >>> import pandas as pd
    >>> plot_value_counts(data, 'subscribed')
    """
    if column not in data_frame.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    counts = data_frame[column].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind='bar', ax=ax)
    ax.set_title('Distribution of ' + column)
    ax.set_xlabel('Target')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()