from src.models.model import *
from sklearn.linear_model import LogisticRegression

class LRModel(Model):
    def __init__(self, C_range):
        super().__init__()
        self.create_pipeline()
        self.param_grid = { 'logisticregression__C': C_range }
        self.param_name = { 'param_logisticregression__C': 'C'}

    def create_pipeline(self):
        if self.preprocessor:
            self.pipeline = make_pipeline(self.preprocessor, LogisticRegression(class_weight={'no': 1, 'yes': 10}))
        else:
            self.pipeline = make_pipeline(LogisticRegression(class_weight={'no': 1, 'yes': 10}))
    