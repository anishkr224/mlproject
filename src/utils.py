import os
import sys
import dill     # For saving/loading Python objects (like pickle, but more flexible)

import numpy as np 
import pandas as pd

from sklearn.metrics import r2_score
from src.exception import CustomException

# Function: save_object
def save_object(file_path, obj):
    """
    Saves any Python object (model, transformer, etc.) to the given file path using dill.
    """
    try:
        # Extract directory path from the given file path
        dir_path = os.path.dirname(file_path)

        # Create directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in binary write mode and save the object
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

# Function: evaluate_models
def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Trains multiple models, evaluates them on train and test datasets using R2 score,
    and returns a dictionary of test scores for each model.
    
    Parameters:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        models: Dictionary of model_name: model_instance
    Returns:
        report: Dictionary of {model_name: test_r2_score}
    """
    try:
        report = {}  # To store the test scores

        # Iterate over all models in the dictionary
        for i in range(len(list(models))):
            model = list(models.values())[i]          # Get model instance
            model_name = list(models.keys())[i]       # Get model name

            model.fit(X_train, y_train)               # Train the model

            y_train_pred = model.predict(X_train)    # Predict on training data
            y_test_pred = model.predict(X_test)      # Predict on test data

            # Compute R2 scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store test score in the report dictionary
            report[model_name] = test_model_score

        return report  # Return dictionary of model scores

    except Exception as e:
        raise CustomException(e, sys)
