from src.models.model import *
from sklearn.neighbors import KNeighborsClassifier

class KNNModel(Model):
    def __init__(self, k_range):
        super().__init__()
        self.create_pipeline()
        self.param_grid = { "kneighborsclassifier__n_neighbors": k_range }
        self.param_name = { "param_kneighborsclassifier__n_neighbors": "k" }

    def create_pipeline(self):
        if self.preprocessor:
            self.pipeline = make_pipeline(self.preprocessor, KNeighborsClassifier(n_jobs=-1))
        else:
            self.pipeline = make_pipeline(KNeighborsClassifier(n_jobs=-1))

    