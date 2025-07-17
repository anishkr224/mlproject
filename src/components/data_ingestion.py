import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass  # Used to define configuration class easily

# Configuration class using @dataclass to automatically generate init and other methods
@dataclass
class DataIngestionConfig:
    # Paths where data will be saved
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

# Main class for data ingestion
class DataIngestion:
    def __init__(self):
        # Initializes configuration
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Reading dataset from CSV file
            df = pd.read_csv('notebook\data\StudentsPerformance.csv')
            logging.info('Read the dataset as dataframe')

            # Create directory if it doesn't exist for storing artifacts
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data to file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")

            # Split dataset into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test data to respective files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Return paths for downstream usage
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)

# Entry point for script execution
if __name__ == "__main__":
    # Create an object of DataIngestion
    obj = DataIngestion()
    
    # Start data ingestion process
    obj.initiate_data_ingestion()