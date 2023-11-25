from src.models.model import *
from sklearn.neighbors import KNeighborsClassifier

class KNNModel(Model):
    """
    k-Nearest Neighbors Model with hyperparameter tuning.

    Parameters:
    - k_range: list or array-like
        Range of neighbors values for hyperparameter tuning.

    Inherits from the Model class.

    Attributes:
    - Inherits attributes from the Model class.
    
    Methods:
    - __init__(k_range):
        Constructor method. Initializes the KNNModel with a specified range of neighbors values.
    
    - create_pipeline():
        Creates the machine learning pipeline for k-Nearest Neighbors with optional preprocessor.
    
    Example:
    ```python
    # Create an instance of the KNNModel class with a specific range of k values
    knn_model_instance = KNNModel(k_range=[3, 5, 7])

    # Set the preprocessor and create the pipeline
    knn_model_instance.set_preprocessor(my_preprocessor)

    # Perform hyperparameter tuning
    best_score, best_params = knn_model_instance.search_cv(X_train, y_train)
    ```

    Note: This class assumes the use of Altair for plotting. Make sure to have the necessary dependencies installed.
    """
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

    