import os
import sys
import dill # For saving/loading Python objects (similar to pickle)

import numpy as np 
import pandas as pd

from src.exception import CustomException


# Function to save any Python object to a given file path
def save_object(file_path, obj):
    try:
        # Extract directory path from the given file_path
        dir_path = os.path.dirname(file_path)

        # Create the directory if it doesn't exist
        # (exist_ok=True → no error if directory already exists)
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in 'write binary' mode
        with open(file_path, 'wb') as file_obj:
            # Dump (save) the Python object into this file using dill
            dill.dump(obj, file_obj)

    except Exception as e:
        # If any error occurs, raise it as a CustomException with sys info
        raise CustomException(e, sys)
