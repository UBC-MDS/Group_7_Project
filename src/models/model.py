from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import fbeta_score, make_scorer

class Model():
    def __init__(self, cv=5, preprocessor=None):
        self.pipeline = None
        self.cv = cv
        self.preprocessor = preprocessor
        self.model = None
        self.param_grid = {}
        self.cv_results = {}

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


