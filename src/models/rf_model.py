from src.models.model import *
from sklearn.ensemble import RandomForestClassifier

class RFModel(Model):
    def __init__(self, n_estimators, max_depth):
        super().__init__()
        self.create_pipeline()
        self.param_grid = { 
            'randomforestclassifier__max_depth': max_depth,
            'randomforestclassifier__n_estimators': n_estimators,
        }
        self.param_name = { 
            'param_randomforestclassifier__max_depth': 'max_depth',
            'param_randomforestclassifier__n_estimators': 'n_estimators',
        }
        
    def create_pipeline(self):
        if self.preprocessor:
            self.pipeline = make_pipeline(self.preprocessor, RandomForestClassifier(n_jobs=-1, class_weight={'no': 1, 'yes': 10}))
        else:
            self.pipeline = make_pipeline(RandomForestClassifier(n_jobs=-1, class_weight={'no': 1, 'yes': 10}))

    