import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

# Utility functions
from src.utils import save_object, evaluate_models

# Configuration Class
@dataclass
class ModelTrainerConfig:
    # Path to save the trained model
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

# Main Model Trainer Class
class ModelTrainer:
    def __init__(self):
        # Initialize configuration
        self.model_trainer_config = ModelTrainerConfig()

    # Train and Evaluate Models
    def initiate_model_trainer(self, train_array, test_array):
        """
        Trains multiple regression models on training data, evaluates them
        on test data, selects the best model based on R2 score, and saves it.
        """
        try:
            logging.info("Split train and test input data")

            # Separate features and target from train and test arrays
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # all columns except last as features
                train_array[:, -1],   # last column as target
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define multiple regression models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Evaluate all models using utility function
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            # Find the best model based on R2 score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # Raise exception if no good model found
            if best_model_score < 0.6:
                raise CustomException("No best model found!")

            logging.info(f"Best found model on both training and testing dataset: {best_model_name}")

            # Save the best model to disk
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict on test data using best model
            predicted = best_model.predict(X_test)

            # Calculate R2 score on test data
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)