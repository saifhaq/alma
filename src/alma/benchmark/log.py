import logging
from typing import Any, Callable, Dict

from .metrics import (
    BenchmarkError,
    BenchmarkMetrics,
    TextGenerationPipelineMetrics,
    TorchModuleMetrics,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def display_all_results(
    all_results: Dict[str, Dict[str, Any]],
    display_function: Callable = print,
    include_errors: bool = True,
    include_traceback_for_errors: bool = False,
) -> None:
    """
    Display all the benchmarking results.

    Inputs:
    - all_results (Dict[str, Dict[str, Any]]): The results of the benchmarking for each conversion
        method. The key is the conversion method, and the value is a dictionary containing the
        results of the benchmarking. These can be floats giving the timings, or strings if there
        is an error containing the traceback.
    - display_function (Callable): The function to use to display the results. Default is print.
    - include_errors (bool): Whether to include the error message for errors. Default is True.
    - include_traceback_for_errors (bool): Whether to include the traceback for errors. Default is True.

    Outputs:
    None
    """
    display_function("\n\nAll results:")
    for conversion_method, result in all_results.items():
        display_function(f"{conversion_method} results:")
        if isinstance(result, BenchmarkMetrics):
            log_results(result, display_function=display_function)
        elif isinstance(result, BenchmarkError):
            log_failure(
                result,
                display_function=display_function,
                include_errors=include_errors,
                include_traceback=include_traceback_for_errors,
            )
        display_function("\n")


def log_results(
    results: BenchmarkMetrics, display_function: Callable = logger.info
) -> None:
    """
    Logs the benchmarking results.

    Inputs:
    - results (BenchmarkMetrics): The results of the benchmarking. This contains the following:
        - device (torch.device): the device the benchmarking was done on.
        - total_elapsed_time (float): The total elapsed time for the benchmark.
        - total_inf_time (float): The total time taken for inference.
        - total_samples (int): The total number of samples benchmarked.
        - batch_size (int): The batch size used during inference.
        - throughput (float): The throughput of the model.
        It may also contain benchmark-specific metrics, e.g. total_output_tokens for
        `TextGenerationPipelineMetrics`.
    - display_function (Callable): The function to use to display the results. Default is logger.info.
        Can be `print`, etc.

    Outputs:
    None
    """
    display_function(f"Device: {results.device}")
    display_function(f"Data dtype: {results.data_dtype}")
    display_function(f"Total elapsed time: {results.total_elapsed_time:.3f} seconds")
    if isinstance(results, TextGenerationPipelineMetrics):
        text_gen_pipeline_logging(results, display_function)
    if isinstance(results, TorchModuleMetrics):
        torch_module_logging(results, display_function)


def torch_module_logging(
    results: TorchModuleMetrics, display_function: Callable = logger.info
) -> None:
    """
    Some torch.nn.module specific result logging.
    Inputs:
    - results (TorchModuleMetrics): The results of the torch.nn.Module specific benchmarking.
    - display_function (Callable): The function to use to display the results. Default is logger.info.
        Can be `print`, etc.
    """
    display_function(
        f"Total samples: {results.total_samples} - Batch size: {results.batch_size}"
    )
    display_function(f"Throughput: {results.throughput:.2f} samples/second")


def text_gen_pipeline_logging(
    results: TextGenerationPipelineMetrics, display_function: Callable = logger.info
) -> None:
    """
    Some Transformer TextGenerationPipeline specific result logging.

    Inputs:
    - results (TextGenerationPipelineMetrics): The results of the TextGenerationPipeline specific benchmarking.
    - display_function (Callable): The function to use to display the results. Default is logger.info.
        Can be `print`, etc.
    """
    display_function(
        f"Total prompts: {results.total_prompts} - Batch size: {results.batch_size}"
    )
    display_function(
        f"Total input tokens: {results.total_input_tokens} - Total output tokens: {results.total_output_tokens}"
    )
    display_function(
        f"Output token throughput: {results.output_throughput:.2f} tokens/second"
    )
    display_function(
        f"Input token throughput: {results.input_throughput:.2f} tokens/second"
    )
    display_function(f"Request rate: {results.request_rate:.2f} tokens/second")


def log_failure(
    error_result: BenchmarkError,
    display_function: Callable = logger.info,
    include_errors: bool = True,
    include_traceback: bool = True,
) -> None:
    """
    Log the error message, when logging the results of a conversion benchmark.

    Inputs:
    - error_result (BenchmarkError): the error message and traceback.
    - display_function (Callable): The function to use to display the results. Default is logger.info.
        Can be `print`, etc.
    - include_errors (bool): Whether to include the error message for errors. Default is True.
    - include_traceback (bool): Whether to include the traceback. Default is True.

    Outputs:
    None
    """
    display_function("Benchmarking failed")
    if include_errors:
        display_function(f"Error: {error_result.error}")
    if include_traceback:
        display_function(error_result.traceback)
