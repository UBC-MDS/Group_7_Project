from src.models.model import *
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class LRModel(Model):
    def __init__(self, C_range):
        super()
        if self.preprocessor:
            self.pipeline = make_pipeline(self.preprocessor, LogisticRegression(class_weight={'no': 1, 'yes': 10}))
        else:
            self.pipeline = make_pipeline(LogisticRegression(class_weight={'no': 1, 'yes': 10}))
        self.param_grid = { 'logisticregressional__C': C_range }

    
    