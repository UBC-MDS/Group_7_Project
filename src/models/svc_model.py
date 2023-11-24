from src.models.model import *
from sklearn.svm import SVC

class SVCModel(Model):
    def __init__(self, C_range):
        super().__init__()
        self.create_pipeline()
        self.param_grid = { 'svc__C': C_range }
        self.param_name = { "param_svc__C": "C" }

    def create_pipeline(self):
        if self.preprocessor:
            self.pipeline = make_pipeline(self.preprocessor, SVC(class_weight={'no': 1, 'yes': 10}))
        else:
            self.pipeline = make_pipeline(SVC(class_weight={'no': 1, 'yes': 10}))
