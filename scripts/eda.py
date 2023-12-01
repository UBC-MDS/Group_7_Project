import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.viz.count_missing_values import count_missing_values
from src.viz.plot_value_counts import plot_value_counts
import os

@click.command()
@click.option('--train_path', type=int, help="Train dataset path")
@click.option('--plot_path', type=str, help="Path to save plots")

def main(train_path, plot_path):
    """EDA of the dataset"""
    bank_marketing_train = pd.read_csv(train_path)

    print(bank_marketing_train.info())
    bank_marketing_summary = bank_marketing_train.describe(include = 'all')
    print(bank_marketing_summary)
    missing_values = count_missing_values(bank_marketing_train)
    print(missing_values)
    print("Length of training data: ", len(bank_marketing_train))

    # distribution of numerical columns
    numeric_cols = bank_marketing_train.select_dtypes(include=['number']).columns.to_list()
    for i in numeric_cols:
        feature = i
        plt.figure(figsize=(4, 3))
        plot = bank_marketing_train.groupby("subscribed")[feature].plot.hist(bins=20, alpha = 0.5, legend = True, density = True, title = "Histogram of " + feature)
        plt.xlabel(feature)
        #plt.show()
        plot.save(os.path.join(plot_path, "feature_"+feature+".png"))

    # target class imbalance
    plot_class_imbalance = plot_value_counts(bank_marketing_train, 'subscribed')
    plot.save(os.path.join(plot_class_imbalance, "class_imbalance.png"))


    # unique values in categorical columns
    cat_cols = ["job", "marital", "education"]
    for column in cat_cols:
        unique_values = list(bank_marketing_train[column].unique())
        print(f"{column}: {unique_values}")


if __name__ == '__main__':
    main()