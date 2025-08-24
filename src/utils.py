import os
import sys
import dill     # For saving/loading Python objects (like pickle, but more flexible)

import numpy as np 
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
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
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Trains multiple models using GridSearchCV, evaluates them on train and test datasets using R2 score,
    and returns:
      - report: {model_name: test_r2_score}
      - trained_models: {model_name: trained_model_instance}
    """
    try:
        report = {}          # To store test scores
        trained_models = {}  # To store trained models

        # Iterate over all models in the dictionary
        for model_name, model in models.items():
            para = param.get(model_name, {})  # Get hyperparameters for this model

            # Perform GridSearchCV for hyperparameter tuning
            gs = GridSearchCV(model, para, cv=3, scoring="r2", n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)

            # Best estimator from grid search (already fitted on training data)
            best_model = gs.best_estimator_

            # model.fit(X_train, y_train)                 # Train the model

            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Compute R2 scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Save results
            report[model_name] = test_model_score
            trained_models[model_name] = best_model

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)
    
# Function: load_object
def load_object(file_path):
    """
    Loads a Python object from the given file path using dill.
    """
    try:
        # Open the file in read-binary ("rb") mode 
        with open(file_path, 'rb') as file_obj:
            # Use dill to deserialize (load) the object from the file
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
