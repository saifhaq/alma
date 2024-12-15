import sys
import traceback
from functools import wraps
from typing import Callable


def benchmark_error_handler(decorated_func: Callable) -> Callable:
    """
    Decorator to catch exceptions and return them as a dictionary.

    Inputs:
    - decorated_func (callable): the benchmarking function to wrap/decorate, should return a
        dictionary.

    Outputs:
    - wrapper (callable): the wrapped function.
    """

    @wraps(decorated_func)
    def wrapper(*args, **kwargs) -> dict:
        """
        Wrapper to catch exceptions and return them as a dictionary.
        This wraps `benchmark` and captures its exceptions, returning them as a dictionary.
        This allows us to stitch together a coherent traceback through the multi-processing.

        Inputs:
        - args (Any): the positional arguments for the function
        - kwargs (Any): the keyword arguments for the function

        Outputs:
        - result (dict): the result of the function (should be dict), or an error dictionary
        """
        try:
            # Error-free path
            result: dict = decorated_func(*args, **kwargs)
            return result
        except Exception as e:
            # NOTE: here we do some processing to capture the "forward" traceback and return it.
            # This is useful for debugging, as we can construct a coherent traceback through the
            # multi-processing and error handling.

            # Get the wrapped function's code object info
            filename = decorated_func.__code__.co_filename
            line_number = (
                decorated_func.__code__.co_firstlineno + 1
            )  # This gets the line where the function is defined

            # Format the function call line in traceback style
            function_trace = f'\n  File "{filename}", line {line_number}, in {decorated_func.__name__}\n    {decorated_func.__name__}('

            # Get the full traceback object from here forwards
            exc_type, exc_value, exc_traceback = sys.exc_info()

            # Format the traceback starting from the original call
            # This includes all frames from script entry to the error
            forward_traceback = "".join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)
            )
            forward_traceback = forward_traceback.split(
                "Traceback (most recent call last):"
            )[1]

            # Combine the forward traceback with the function call
            forward_traceback = function_trace + forward_traceback

            return {
                "status": "error",
                "error": str(e),
                "traceback": forward_traceback,
            }

    return wrapper
