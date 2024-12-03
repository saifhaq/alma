import logging
from typing import Literal, Union


def setup_logging(log_file: Union[str, None] = None, level: Literal[str] = "INFO") -> None:
    """
    Sets up logging to print to console and optionally to a file.

    Inputs:
    - log_file (str): Path to a log file. If None, logs will not be saved to a file.
    - level (Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]): The logging level to use.

    Outputs:
    - None
    """
    # log_format = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] %(message)s"
    log_format = "%(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    match level:
        case "DEBUG":
            log_level = logging.DEBUG
        case "INFO":
            log_level = logging.INFO
        case "WARNING":
            log_level = logging.WARNING
        case "ERROR":
            log_level = logging.ERROR
        case "CRITICAL":
            log_level = logging.CRITICAL
        case _:
            raise ValueError(f"Invalid log level: {level}")

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler(),
        ],
    )
