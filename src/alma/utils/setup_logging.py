import contextlib
import io
import logging
import sys
from typing import Literal, Union


def setup_logging(
    log_file: Union[str, None] = None,
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
) -> None:
    """
    Sets up logging to print to console and optionally to a file.

    Inputs:
    - log_file (str): Path to a log file. If None, logs will not be saved to a file.
    - level (Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]): The logging level to use.

    Outputs:
    - None
    """
    log_format = "%(levelname)s: [%(filename)s:%(lineno)d] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    log_level = getattr(logging, level)

    # Clear existing handlers
    logging.getLogger().handlers.clear()

    # Create handlers first
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        force=True,
        handlers=handlers,
    )


@contextlib.contextmanager
def suppress_output(active: bool = False):
    """
    Context manager to suppress logging of internal programs / imports.

    Inputs:
    - active (bool): if active, it suppresses the internal outputs/logs of the wrapped
    program.

    Outputs:
    None
    """
    if active:
        # Suppress stdout/stderr
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = io.StringIO()

        # Suppress logging
        logging.disable(logging.CRITICAL)  # Disable all logging calls below CRITICAL

        try:
            yield
        finally:
            # Restore stdout/stderr
            sys.stdout, sys.stderr = stdout, stderr

            # Restore logging
            logging.disable(logging.NOTSET)  # Re-enable logging
    else:
        yield  # We still need to yield even when not active
