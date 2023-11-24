from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import fbeta_score, make_scorer
import pandas as pd
import numpy as np
import altair as alt

class Model():
    def __init__(self, cv=5, preprocessor=None):
        self.pipeline = None
        self.cv = cv
        self.preprocessor = preprocessor
        self.model = None
        self.param_grid = {}
        self.cv_results = {}
        self.param_name = []

    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor
        self.create_pipeline()

    def search_cv(self, X_train, y_train):
        search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=make_scorer(fbeta_score, pos_label='yes', beta=5),
            n_jobs=-1,
            return_train_score=True
        )
        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        self.cv_results = search.cv_results_
        return search.best_score_, search.best_params_
        
    def fit(self, X_train, y_train, **args):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)

    def get_cv_results(self):
        return self.cv_results

    def get_accuracy_grid(self):
        accuracies_grid = pd.DataFrame(self.get_cv_results())
        accuracies_grid = (
            accuracies_grid[
                list(self.param_name.keys()) + ["mean_test_score", "std_test_score"]
            ].assign(
                sem_test_score=accuracies_grid["std_test_score"] / self.cv**(1/2),
                sem_test_score_lower=lambda df: df["mean_test_score"] - (df["sem_test_score"]/2),
                sem_test_score_upper=lambda df: df["mean_test_score"] + (df["sem_test_score"]/2)
            ).rename(columns=self.param_name).drop(columns=["std_test_score"])
        )
        return accuracies_grid.sort_values("mean_test_score", ascending=False)

    def get_best_model_score(self):
        return pd.DataFrame(self.get_cv_results())[
            list(self.param_name.keys()) +
            [
                "mean_test_score", "mean_train_score", "rank_test_score"
            ]].set_index("rank_test_score").sort_index().iloc[1, :]

    def draw_search_plot(self):
        accuracies_grid = self.get_accuracy_grid()
        param = list(self.param_name.values())[0]
        line_n_point = alt.Chart(accuracies_grid, width=600).mark_line(color="black").encode(
            x=alt.X(param, scale=alt.Scale(type='log'), title=param),
            y=alt.Y("mean_test_score")
                .scale(zero=False) 
                .title("F-beta score (beta = 5)")
        )
        return line_n_point + line_n_point.mark_circle(color='black')

