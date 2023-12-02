import os
import sys
import click
import pickle
import numpy as np
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.rf_model import RFModel


@click.command()
@click.option('--result_folder_path', type=str, help="PATH of results folder")
def main(result_folder_path):
    knn_cv = pd.read_csv(os.path.join(result_folder_path, "cv", "knn_cv.csv"))
    lr_cv = pd.read_csv(os.path.join(result_folder_path, "cv", "lr_cv.csv"))
    svc_cv = pd.read_csv(os.path.join(result_folder_path, "cv", "svc_cv.csv"))
    rf_cv = pd.read_csv(os.path.join(result_folder_path, "cv", "rf_cv.csv"))
    model_comparison = { 
        "model_name": [
            "K-Nearest Neighbors",
            "SVC RBF",
            "Logistic Regression",
            "Random Forest",
        ],
        "mean_train_score": [],
        "mean_test_score": []
    }
    model_comparison["mean_train_score"].append(knn_cv.iloc[2, 1])
    model_comparison["mean_test_score"].append(knn_cv.iloc[1, 1])
    model_comparison["mean_train_score"].append(svc_cv.iloc[2, 1])
    model_comparison["mean_test_score"].append(svc_cv.iloc[1, 1])
    model_comparison["mean_train_score"].append(lr_cv.iloc[2, 1])
    model_comparison["mean_test_score"].append(lr_cv.iloc[1, 1])
    model_comparison["mean_train_score"].append(rf_cv.iloc[3, 1])
    model_comparison["mean_test_score"].append(rf_cv.iloc[2, 1])
    pd.DataFrame(model_comparison).to_csv(os.path.join(result_folder_path, "model_comparison.csv"))


if __name__ == "__main__":
    main()
