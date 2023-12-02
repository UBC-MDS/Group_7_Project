import os
import sys
import click
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.model import *


@click.command()
@click.option('--model_path', type=str, help="PATH of the model object")
@click.option('--test_data_path', type=str, help="PATH of the test data")
@click.option('--target_dir', type=str, help="Path of the table file to save")
def main(model_path, test_data_path, target_dir):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    test_df = pd.read_csv(test_data_path)
    # test_df = test_df.drop(["poutcome", "contact"], axis = 1)
    # Drop NAs
    test_df = test_df.dropna()
    X_test = test_df.drop(columns=["subscribed"])
    y_test = test_df["subscribed"]
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Compute F-beta score (beta = 5)
    preds_df = X_test.assign(
        predicted=model.predict(X_test)
    )
    f_beta_5_score = fbeta_score(
        y_test,
        preds_df['predicted'],
        beta=5,
        pos_label='yes'
    )

    result = pd.DataFrame({'accuracy': [accuracy], 'F-beta score (beta = 5)': [f_beta_5_score]})
    result.to_csv(os.path.join(target_dir, "best_model_score.csv"))

    confusion_matrix = pd.crosstab(
        y_test,
        preds_df['predicted'],
    )
    confusion_matrix.to_csv(os.path.join(target_dir, "best_model_confusion_matrix.csv"))


if __name__ == "__main__":
    main()
