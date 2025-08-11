import sys
import traceback
from typing import Tuple, Any

def error_message_detail(error: Exception, error_detail: Tuple[Any, Any, Any]) -> str:
    """
    Generates a detailed error message including filename and line number.

    Parameters:
        error (Exception): The caught exception object.
        error_detail (Tuple[Any, Any, Any]): The result of sys.exc_info().

    Returns:
        str: A detailed error message string.
    """
    _, _, exc_tb = error_detail  # exc_tb is the traceback object
    filename = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error in script: {filename} at line {line_number}: {str(error)}"

# Example usage:
# try:
#     1 / 0
# except Exception as e:
#     error_detail = sys.exc_info()
#     print(error_message_detail(e, error_detail))

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message