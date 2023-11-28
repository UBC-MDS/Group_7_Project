import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.viz.count_missing_values import count_missing_values
from src.viz.plot_value_counts import plot_value_counts

@click.command()
@click.option('--data_id', type=int, help="Dataset ID")
@click.option('--ratio', type=float, help="Train, test split")

def main(data_id, ratio):
    """EDA of the dataset"""
    bank_marketing_train = pd.read_csv("data\raw\bank_marketing_train.csv")
    bank_marketing_test = pd.read_csv("data\raw\bank_marketing_test.csv")
    print(bank_marketing_train.info())
    bank_marketing_summary = bank_marketing_train.describe(include = 'all')
    print(bank_marketing_summary)
    missing_values = count_missing_values(bank_marketing_train)
    print(missing_values)
    print("Length of training data: ", len(bank_marketing_train))
    print("Length of testing data: ", len(bank_marketing_test))

    # distribution of numerical columns
    numeric_cols = bank_marketing_train.select_dtypes(include=['number']).columns.to_list()
    for i in numeric_cols:
        feature = i
        plt.figure(figsize=(4, 3))
        plot = bank_marketing_train.groupby("subscribed")[feature].plot.hist(bins=20, alpha = 0.5, legend = True, density = True, title = "Histogram of " + feature)
        plt.xlabel(feature)
        plt.show()

    # target class imbalance
    plot_value_counts(bank_marketing_train, 'subscribed')

    # unique values in categorical columns
    cat_cols = ["job", "marital", "education"]
    for column in cat_cols:
        unique_values = list(bank_marketing_train[column].unique())
        print(f"{column}: {unique_values}")


if __name__ == '__main__':
    main()