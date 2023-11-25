from src.models.model import *
from sklearn.ensemble import RandomForestClassifier

class RFModel(Model):
    """
    Random Forest Model with hyperparameter tuning.

    Parameters:
    - n_estimators: int
        Number of trees in the forest.
    - max_depth: int or None
        Maximum depth of the trees in the forest.

    Inherits from the Model class.

    Attributes:
    - Inherits attributes from the Model class.
    
    Methods:
    - __init__(n_estimators, max_depth):
        Constructor method. Initializes the RFModel with a specified number of trees and maximum tree depth.
    
    - create_pipeline():
        Creates the machine learning pipeline for Random Forest with optional preprocessor.
    
    Example:
    ```python
    # Create an instance of the RFModel class with specific n_estimators and max_depth values
    rf_model_instance = RFModel(n_estimators=100, max_depth=10)

    # Set the preprocessor and create the pipeline
    rf_model_instance.set_preprocessor(my_preprocessor)

    # Perform hyperparameter tuning
    best_score, best_params = rf_model_instance.search_cv(X_train, y_train)
    ```

    Note: This class assumes the use of Altair for plotting. Make sure to have the necessary dependencies installed.
    """
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

    