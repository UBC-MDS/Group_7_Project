import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.viz.count_missing_values import count_missing_values
from src.viz.plot_value_counts import plot_value_counts


@click.command()
@click.option('--train_path', type=str, help="Train dataset path")
@click.option('--table_path', type=str, help="Path to save tables")
@click.option('--plot_path', type=str, help="Path to save plots")
def main(train_path, table_path, plot_path):
    """EDA of the dataset"""
    bank_marketing_train = pd.read_csv(train_path)

    print(bank_marketing_train.info())
    bank_marketing_summary = bank_marketing_train.describe(include = 'all')
    bank_marketing_summary.to_csv(os.path.join(table_path, "eda", "data_summary.csv"))
    missing_values = count_missing_values(bank_marketing_train)
    missing_values.to_csv(os.path.join(table_path, "eda", "missing_values.csv"))
    print("Length of training data: ", len(bank_marketing_train))

    # distribution of numerical columns
    numeric_cols = bank_marketing_train.select_dtypes(include=['number']).columns.to_list()
    plot = alt.Chart(bank_marketing_train).mark_bar(opacity=0.5).encode(
        x=alt.X(alt.repeat()).bin(maxbins=30),
        y=alt.Y("count()").stack(False).title("Count"),
        color="subscribed"
    ).repeat(
        numeric_cols,
        columns=2
    )
    plot.save(os.path.join(plot_path, "numeric_cols.png"))

    # target class imbalance
    fig, ax, bar = plot_value_counts(bank_marketing_train, 'subscribed')
    fig.savefig(os.path.join(plot_path, "class_imbalance.png"), format='png')

    # unique values in categorical columns
    cat_cols = ["job", "marital", "education"]
    for column in cat_cols:
        unique_values = list(bank_marketing_train[column].unique())
        print(f"{column}: {unique_values}")


if __name__ == '__main__':
    main()