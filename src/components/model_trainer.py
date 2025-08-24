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
            # Defines hyperparameter grids for each model
            params={
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "K-Neighbors Regressor":{
                    'n_neighbors':[3,5,7,9],
                    'weights':['uniform','distance'],
                    'algorithm':['ball_tree','kd_tree','brute']
                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            # Evaluate all models using utility function
            model_report, trained_models = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            # Find the best model based on R2 score
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = trained_models[best_model_name]

            # Raise exception if no good model found
            if best_model_score < 0.6:
                raise CustomException("No best model found!", sys)

            logging.info(f"Model performance report: {model_report}")
            logging.info(f"Best model: {best_model_name} with R2 score: {best_model_score}")


            # Save the best model to disk
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict on test data using best model
            predicted = best_model.predict(X_test)

            # Calculate R2 score on test data
            r2_square = r2_score(y_test, predicted)
            return best_model_name, r2_square

        except Exception as e:
            raise CustomException(e, sys)