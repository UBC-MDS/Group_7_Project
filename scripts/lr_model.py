import os
import sys
import click
import pickle
import numpy as np
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.lr_model import LRModel


@click.command()
@click.option('--preprocessor_path', type=str, help="PATH of the preprocessor object")
@click.option('--train_data_path', type=str, help="PATH of train dataset to read")
@click.option('--model_save_path', type=str, help="Path of the model file to save")
@click.option('--table_dir', type=str, help="Path of the table file to save")
@click.option('--plot_dir', type=str, help="Path of the plot file to save")
def main(preprocessor_path, train_data_path, model_save_path, table_dir, plot_dir):
    with open(preprocessor_path, 'rb') as file:
        preprocessor = pickle.load(file)

    train_df = pd.read_csv(train_data_path)
    # Hardcoding uninsightful columns to drop (based on TA advice)
    train_df = train_df.drop(["poutcome", "contact"], axis = 1)
    # Drop NAs
    train_df = train_df.dropna()
    X_train = train_df.drop(columns=["subscribed"])
    y_train = train_df["subscribed"]

    model = LRModel(10.0 ** np.arange(-3, 5, 1))
    model.set_preprocessor(preprocessor)
    model.search_cv(X_train, y_train)
    model.fit(X_train, y_train)
    with open(model_save_path, 'wb') as file:
        pickle.dump(model.model, file)
    accuracies_grid = model.get_accuracy_grid()
    accuracies_grid.to_csv(os.path.join(table_dir, "lr.csv"))
    plot = model.draw_search_plot()
    plot.save(os.path.join(plot_dir, "lr.png"))


if __name__ == "__main__":
    main()
