# Import the sys module to interact with the Python interpreter : to access exception information like traceback
import sys

# Import your custom logging setup
from src.logger import logging

# This function creates a detailed error message with:
# - File name where the error occurred
# - Line number of the error
# - Actual error message
def error_message_detail(error, error_detail: sys):
    # Extracting traceback object from the system : sys.exc_info() gives (type, value, traceback) of the last exception.
    _, _, exc_tb = error_detail.exc_info()
    
    # Getting the name of the file where the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Create a detailed error message string with filename, line number, and error
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))

    return error_message


# Creating a custom exception class by extending the built-in Exception class that overrides the default Python error message to include file and line details.
class CustomException(Exception):
    # Constructor for the custom exception
    def __init__(self, error_message, error_detail: sys):
        # Calling the parent class constructor with the base error message
        super().__init__(error_message)

        # Generate and store the detailed error message using the helper function
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    # Overriding the __str__() method to return the detailed error message when printed
    # __str__ method : Ensures the custom exception prints the full message when raised.
    def __str__(self):
        return self.error_message