# Import the required classes from components
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation, DataTransformationConfig

# Step 1: Create an object of DataIngestion class
obj = DataIngestion()

# Step 2: Call the ingestion method to get train and test dataset paths
train_data, test_data = obj.initiate_data_ingestion()

# Step 3: Create an object of DataTransformation class
data_transformation = DataTransformation()

# Step 4: Perform transformation (scaling, encoding, etc.) on train and test datasets
data_transformation.initiate_data_transformation(train_data, test_data)

# DataIngestion → Fetches & splits data.
# DataTransformation → Prepares data for ML pipeline.