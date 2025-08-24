import sys
import os
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

# Prediction Pipeline Class
class PredictPipeline:
    def __init__(self):
        pass

    # Method to make predictions on input features
    def predict(self, features):
        try:
            # Load the preprocessor and model objects
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join("artifacts","model.pkl")

            preprocessor = load_object(file_path = preprocessor_path)
            model = load_object(file_path = model_path)

            # Transform the input features using the preprocessor
            data_scaled = preprocessor.transform(features)

            # Make predictions using the trained model
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)

# Data class to capture user input features
class CustomData:
        # Initialize with user input features
        def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):
             
            self.gender = gender
            self.race_ethnicity = race_ethnicity
            self.parental_level_of_education = parental_level_of_education
            self.lunch = lunch
            self.test_preparation_course = test_preparation_course
            self.reading_score = reading_score
            self.writing_score = writing_score

        # Method to convert input data to a pandas DataFrame
        def get_data_as_data_frame(self):
            try:
                custom_data_input_dict = {
                    "gender": [self.gender],
                    "race/ethnicity": [self.race_ethnicity],
                    "parental level of education": [self.parental_level_of_education],
                    "lunch": [self.lunch],
                    "test preparation course": [self.test_preparation_course],
                    "reading score": [self.reading_score],
                    "writing score": [self.writing_score],
                    }
                return pd.DataFrame(custom_data_input_dict)
            
            except Exception as e:
                 raise CustomException(e, sys)

