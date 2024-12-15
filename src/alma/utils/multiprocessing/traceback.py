import inspect
import os


def get_next_line_info() -> str:
    """
    Get the next line of code, formatted like a traceback.

    This is useufl for debugging, as we can construct a coherant traceback throuhg the
    multi-processing and error handling.
    """
    # Get the frame of the caller
    frame = inspect.currentframe().f_back

    # Get the filename, line number, and line content
    filename = frame.f_code.co_filename
    line_number = frame.f_lineno
    # Get the actual line of code
    with open(filename) as f:
        lines = f.readlines()
        line_content = lines[line_number].strip()

    # Format it like a traceback
    return f'  File "{os.path.abspath(filename)}", line {line_number+1}, in {frame.f_code.co_name}\n    {line_content}'
