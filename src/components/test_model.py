from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

# Step 1: Data Ingestion
# Create an object of DataIngestion class
obj = DataIngestion()

# Call the ingestion method to fetch raw dataset, split it into train & test sets, and return their file paths
train_data, test_data = obj.initiate_data_ingestion()

# Step 2: Data Transformation
# Create an object of DataTransformation class
data_transformation = DataTransformation()

# Perform preprocessing:
# - Impute missing values
# - Scale numerical features
# - Encode categorical features
# Returns transformed train and test arrays, and the path to preprocessor object
train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

# Step 3: Model Training
# Create an object of ModelTrainer class
modeltrainer = ModelTrainer()

# Train multiple models using transformed train and test arrays and save the best one based on R2 score
# Print the R2 score of the best model found
print(modeltrainer.initiate_model_trainer(train_arr, test_arr))

# Summary:
# DataIngestion → Fetches raw data, splits it into train/test
# DataTransformation → Prepares the data for ML pipeline (scaling, encoding)
# ModelTrainer → Trains and evaluates the machine learning model
