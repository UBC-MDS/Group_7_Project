from src.models.model import *
from sklearn.svm import SVC

class SVCModel(Model):
    """
    Support Vector Classification Model with hyperparameter tuning.

    Parameters:
    - C_range: list or array-like
        Range of regularization parameter values for hyperparameter tuning.

    Inherits from the Model class.

    Attributes:
    - Inherits attributes from the Model class.
    
    Methods:
    - __init__(C_range):
        Constructor method. Initializes the SVCModel with a specified range of regularization parameter values.
    
    - create_pipeline():
        Creates the machine learning pipeline for Support Vector Classification with class weights.
    
    Example:
    ```python
    # Create an instance of the SVCModel class with a specific range of C values
    svc_model_instance = SVCModel(C_range=[0.1, 1, 10])

    # Set the preprocessor and create the pipeline
    svc_model_instance.set_preprocessor(my_preprocessor)

    # Perform hyperparameter tuning
    best_score, best_params = svc_model_instance.search_cv(X_train, y_train)
    ```

    Note: This class assumes the use of Altair for plotting. Make sure to have the necessary dependencies installed.
    """
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
