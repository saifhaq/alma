import logging
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def display_all_results(
    all_results: Dict[str, Dict[str, Any]],
    display_function: Callable = print,
    include_traceback_for_errors: bool = True,
) -> None:
    """
    Display all the benchmarking results.

    Inputs:
    - all_results (Dict[str, Dict[str, Any]]): The results of the benchmarking for each conversion
        method. The key is the conversion method, and the value is a dictionary containing the
        results of the benchmarking. These can be floats giving the timings, or strings if there
        is an error containing the traceback.
    - display_function (Callable): The function to use to display the results. Default is print.
    - include_traceback_for_errors (bool): Whether to include the traceback for errors. Default is True.

    Outputs:
    None
    """
    display_function("\n\nAll results:")
    for conversion_method, result in all_results.items():
        display_function(f"{conversion_method} results:")
        if result["status"] == "success":
            log_results(result, display_function=display_function)
        elif result["status"] == "error":
            log_failure(
                result,
                display_function=display_function,
                include_traceback=include_traceback_for_errors,
            )
        display_function("\n")


def log_results(
    results: Dict[str, Any], display_function: Callable = logger.info
) -> None:
    """
    Logs the benchmarking results.

    Inputs:
    - results (Dict[str, float]): The results of the benchmarking. This contains the following:
        - device (torch.device): the device the benchmarking was done on.
        - total_elapsed_time (float): The total elapsed time for the benchmark.
        - total_inf_time (float): The total time taken for inference.
        - total_samples (int): The total number of samples benchmarked.
        - batch_size (int): The batch size used during inference.
        - throughput (float): The throughput of the model.
    - display_function (Callable): The function to use to display the results. Default is logger.info.
        Can be `print`, etc.

    Outputs:
    None
    """
    display_function(f"Device: {results['device']}")
    display_function(f"Total elapsed time: {results['total_elapsed_time']:.4f} seconds")
    display_function(
        f"Total inference time (model only): {results['total_inf_time']:.4f} seconds"
    )
    display_function(
        f"Total samples: {results['total_samples']} - Batch size: {results['batch_size']}"
    )
    display_function(f"Throughput: {results['throughput']:.2f} samples/second")


def log_failure(
    error_result: Dict[str, str],
    display_function: Callable = logger.info,
    include_traceback: bool = True,
) -> None:
    """
    Log the error message, when logging the results of a conversion benchmark.

    Inputs:
    - error_result (Dict[str, str]): the error message and traceback.
    - display_function (Callable): The function to use to display the results. Default is logger.info.
        Can be `print`, etc.
    - include_traceback (bool): Whether to include the traceback. Default is True.

    Outputs:
    None
    """
    display_function(f"Benchmarking failed, error: {error_result['error']}")
    if include_traceback:
        display_function(error_result["traceback"])
