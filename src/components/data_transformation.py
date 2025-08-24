# Importing system-specific parameters and functions
import sys
# Used for creating data classes without having to write boilerplate code
from dataclasses import dataclass

# Numerical and data manipulation libraries
import numpy as np 
import pandas as pd

# Scikit-learn utilities for preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Custom exception and logging utilities
from src.exception import CustomException
from src.logger import logging
import os

# Utility function to save objects
from src.utils import save_object

# Configuration class to define file path for saving the preprocessor object
@dataclass
class DataTransformationConfig:
    # File path where the preprocessor object will be saved
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

# Main class responsible for performing data transformation
class DataTransformation:
    def __init__(self):
        # Initialize configuration
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for creating and returning
        a data preprocessing pipeline object.
        '''
        try:
            # Define numerical and categorical column names
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]

            # Define pipeline for numerical columns
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Fill missing values with median
                    ("scaler", StandardScaler())                    # Standardize features
                ]
            )

            # Define pipeline for categorical columns
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing values with mode
                    ("one_hot_encoder", OneHotEncoder()),                  # Convert categories to one-hot
                    ("scaler", StandardScaler(with_mean=False))            # Scale encoded values
                ]
            )

            # Logging column information
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine both pipelines into one column transformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            # Return the final preprocessing object
            return preprocessor
        
        except Exception as e:
            # Raise a custom exception in case of any error
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function reads train and test datasets, applies preprocessing,
        and returns the transformed arrays and path to the preprocessor object.
        '''
        try:
            # Load training and testing data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            # Get the preprocessor object
            preprocessing_obj = self.get_data_transformer_object()

            # Define target and numerical columns
            target_column_name = "math score"
            numerical_columns = ["writing score", "reading score"]

            # Separate input and target features
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Apply transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed features and targets
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            # Save the preprocessor object for later use
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return transformed train/test data and preprocessor file path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            # Handle errors using custom exception
            raise CustomException(e, sys)
