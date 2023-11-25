# File: model.py

import pytest
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import sys
import os

# Get the absolute path of the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# Add the 'src' directory to the Python path
sys.path.insert(0, project_root)
from src.models.knn_model import KNNModel


@pytest.fixture
def example_data():
    # Create a toy classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_clusters_per_class=2,
        random_state=123,
    )
    y = np.where(y == 1, 'yes', 'no')
    X_train = pd.DataFrame(X[:800, :], columns=["a", "b", "c", "d", "e"])
    y_train = pd.Series(y[:800], name="target")
    X_test = pd.DataFrame(X[800:, :], columns=["a", "b", "c", "d", "e"])
    y_test = pd.Series(y[800:], name="target")
    return X_train, y_train, X_test, y_test


def test_model_initialization():
    # Test the initialization of the Model class
    model_instance = KNNModel([1, 3])
    assert model_instance.cv == 5
    assert model_instance.preprocessor is None
    assert model_instance.model is None
    assert model_instance.param_grid == {"kneighborsclassifier__n_neighbors": [1, 3]}
    assert model_instance.cv_results == {}
    assert model_instance.param_name == {"param_kneighborsclassifier__n_neighbors": "k"}


def test_set_preprocessor(example_data):
    # Test setting the preprocessor
    model_instance = KNNModel([1, 3])
    preprocessor = StandardScaler()
    model_instance.set_preprocessor(preprocessor)
    assert model_instance.preprocessor == preprocessor
    assert model_instance.pipeline is not None


def test_search_cv(example_data):
    # Test hyperparameter tuning using cross-validation
    model_instance = KNNModel([1, 3])
    X_train, y_train, _, _ = example_data
    best_score, best_params = model_instance.search_cv(X_train, y_train)
    assert isinstance(best_score, float)
    assert isinstance(best_params, dict)
    assert "kneighborsclassifier__n_neighbors" in best_params
    assert isinstance(model_instance.model, Pipeline)
    assert isinstance(model_instance.model.named_steps['kneighborsclassifier'], KNeighborsClassifier)


def test_fit_and_predict(example_data):
    # Test fitting and predicting
    model_instance = KNNModel([1, 3])
    X_train, y_train, X_test, y_test = example_data
    model_instance.search_cv(X_train, y_train)
    model_instance.fit(X_train, y_train)
    predictions = model_instance.predict(X_test)
    assert isinstance(predictions, np.ndarray)
    assert predictions.size == y_test.size
    assert np.mean(predictions == y_test) > 0.5


def test_score(example_data):
    # Test scoring
    model_instance = KNNModel([1, 3])
    X_train, y_train, X_test, y_test = example_data
    model_instance.search_cv(X_train, y_train)
    model_instance.fit(X_train, y_train)
    score = model_instance.score(X_test, y_test)
    assert isinstance(score, float)


def test_get_cv_results(example_data):
    # Test getting cross-validation results
    model_instance = KNNModel([1, 3])
    X_train, y_train, X_test, y_test = example_data
    best_score, best_params = model_instance.search_cv(X_train, y_train)
    cv_results = model_instance.get_cv_results()
    assert isinstance(cv_results, dict)
    assert "mean_train_score" in cv_results
    assert "mean_test_score" in cv_results
    assert isinstance(cv_results["mean_train_score"], np.ndarray)
    assert isinstance(cv_results["mean_test_score"], np.ndarray)
    assert len(cv_results["mean_test_score"]) == 2
    assert len(cv_results["mean_train_score"]) == 2


def test_get_accuracy_grid(example_data):
    # Test getting accuracy grid
    model_instance = KNNModel([1, 3])
    X_train, y_train, X_test, y_test = example_data
    model_instance.search_cv(X_train, y_train)
    accuracy_grid = model_instance.get_accuracy_grid()
    assert isinstance(accuracy_grid, pd.DataFrame)
    assert accuracy_grid.shape == (2, 5)
    assert accuracy_grid.iloc[0, 0] == 3


def test_get_best_model_score(example_data):
    # Test getting the best model's score and hyperparameters
    model_instance = KNNModel([1, 3])
    X_train, y_train, X_test, y_test = example_data
    model_instance.search_cv(X_train, y_train)
    best_model_score = model_instance.get_best_model_score()
    assert isinstance(best_model_score, pd.Series)
    assert best_model_score.shape == (3,)
    assert best_model_score.iloc[2] == 1
