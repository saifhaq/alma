import logging
import traceback
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader

from .benchmark import benchmark
from .benchmark.benchmark_config import BenchmarkConfig
from .conversions.conversion_options import MODEL_CONVERSION_OPTIONS, ConversionOption
from .dataloader.create import create_single_tensor_dataloader
from .utils.checks import check_consistent_batch_size, check_inputs
from .utils.device import setup_device
from .utils.multiprocessing import benchmark_process_wrapper

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def benchmark_model(
    model: Union[torch.nn.Module, Callable],
    config: BenchmarkConfig,
    conversions: Optional[List[ConversionOption]] = None,
    data: Optional[torch.Tensor] = None,
    data_loader: Optional[DataLoader] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark the model on different conversion methods. If provided, the dataloader will be used.
    Else, a random dataloader will be created, in which case the `data` tensor must be provided
    to initialize the dataloader.

    The model is benchmarked on the given conversion methods, and the results are returned.
    The `config` BenchmarkConfig must contain the following:
    - n_samples (int): The number of samples to benchmark on
    - batch_size (int): The batch size to use for benchmarking
    - device (torch.device): The device to benchmark on.
    - multiprocessing (Optional[bool], default True): Whether or not to use multiprocessing to have isolated
        testing environments per conversion method. This helps keep the global torch state consistent
        when each method is benchmarked.
    - fail_on_error (Optional[bool], default False): whether or not to fail fast, or fail gracefully.
        If we fail gracefully, we continue benchmarking other methods if one fails, and store
        the error message and traceback in the returned struct.

    Args:
        model (Union[torch.nn.Module, Callable]): The model to benchmark. If a callable is provided,
            it should return the model instance. This helps when using multiprocessing, as the model
            can be instantiated inside isolated child processes.
        config (BenchmarkConfig): A validated Pydantic configuration for benchmarking.
        conversions (Optional[List[ConversionOption]]): List of `ConversionOption` objects to benchmark.
            If None, all available conversion methods will be used.
        data (Optional[torch.Tensor]): Input data for benchmarking, required if no `data_loader` is provided.
        data_loader (Optional[DataLoader]): DataLoader for data samples. If provided, it takes precedence.

    Returns:
    - all_results (Dict[str, Dict[str, Any]]): The results of the benchmarking for each conversion method.
        The key is the conversion method, and the value is a tuple containing the total elapsed
        time, the total time taken, the total number of samples, and the throughput of the model.
        If the conversion method failed and we fail gracefully, the value will be a dictionary
        containing the error and traceback.            If a method fails, its value contains the error message and traceback.
    """
    # Check the inputs
    check_inputs(model, config, conversions, data, data_loader)

    # If the conversions are not provided, we use all available conversions
    if conversions is None:
        conversions = list(MODEL_CONVERSION_OPTIONS.values())

    # The number of samples to benchmark on, batch size, and device to benchmark on
    n_samples: int = config.n_samples
    batch_size: int = config.batch_size
    device: torch.device = config.device
    # Whether or not to use multiprocessing for isolated testing environments (which protects against
    # conversion methods contaminating the global torch state), and whether to fail quickly or gracefully.
    # By default, we enable multiprocessing, and fail gracefully.
    multiprocessing: bool = config.multiprocessing
    fail_on_error: bool = config.fail_on_error
    # Creates a dataloader with random data, of the same size as the input data sample
    # If the data_loader has been provided by the user, we use that one
    if not isinstance(data_loader, DataLoader):
        data_loader = create_single_tensor_dataloader(
            tensor_size=data.size(),
            num_tensors=n_samples,
            random_type="normal",
            random_params={"mean": 0.0, "std": 2.0},
            batch_size=batch_size,
        )

    all_results: Dict[str, Dict[str, Any]] = {}

    for conversion_option in conversions:
        conversion_mode = conversion_option.mode
        device_override = conversion_option.device_override

        check_consistent_batch_size(conversion_mode, n_samples, batch_size)

        torch.cuda.empty_cache()
        logger.info(f"Benchmarking model using conversion: {conversion_mode}")

        # Potential device override, depending on if the conversion method is device-specific and
        # the provided override options
        device = setup_device(
            device,
            allow_cuda=config.allow_cuda,
            allow_mps=config.allow_mps,
            allow_device_override=config.allow_device_override,
            selected_conversion=conversion_option,
        )

        # Benchmark the model
        result, stacktrace = benchmark_process_wrapper(
            multiprocessing,
            benchmark,
            device,
            model,
            conversion_mode,
            data_loader,
            n_samples,
        )
        all_results[conversion_mode] = result

        # If the conversion failed, we raise an exception if we are failing fast
        if result["status"] == "error":
            logger.error(f"Benchmark failed for conversion: {conversion_mode}")
            # We combine the stacktrace from the child process with the stacktrace from the parent process
            result["traceback"] = stacktrace + result["traceback"]
            if fail_on_error:
                raise RuntimeError(result["traceback"])

    return all_results


# Example Usage
if __name__ == "__main__":
    from .benchmark.benchmark_config import BenchmarkConfig

    model = torch.nn.Linear(10, 1)
    config = BenchmarkConfig(
        n_samples=256,
        batch_size=32,
        allow_cuda=True,  # Set to False to test on CPU
        fail_on_error=True,
    )

    # Input data
    sample_data = torch.randn(32, 10)

    conversions = [
        MODEL_CONVERSION_OPTIONS[0],  # EAGER mode
        MODEL_CONVERSION_OPTIONS[2],  # ONNX_CPU mode
    ]

    # Run benchmark
    try:
        results = benchmark_model(
            model, config, conversions=conversions, data=sample_data
        )
        print("Benchmark Results:")
        for conversion, result in results.items():
            print(f"{conversion}: {result}")
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
