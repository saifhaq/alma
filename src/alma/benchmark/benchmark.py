import logging
from typing import Callable, Dict, Union

import torch
import torch._dynamo
from torch.utils.data import DataLoader

from ..conversions.select import select_forward_call_function
from ..utils.data import get_sample_data
from ..utils.multiprocessing import benchmark_error_handler
from ..utils.times import inference_time_benchmarking  # should we use this?
from .log import log_results
from .time import time_forward

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@benchmark_error_handler
def benchmark(
    device: torch.device,
    model: Union[torch.nn.Module, Callable],
    conversion: str,
    data_loader: DataLoader,
    n_samples: int,
) -> Dict[str, float]:
    """
    Benchmark the model using the given data loader. This function will benchmark the model using the
    given conversion method.

    NOTE: We wrap this function in an error handler to catch any exceptions that occur during the
    benchmarking process. This will allow us to pass the error and traceback back through any
    multiprocessing handler.

    Inputs:
    - device (torch.device): The device we are targetting.
    - model (Union[torch.nn.Module, Callable]): The model to benchmark, or callable which returns
    the model (used for memory efficiency when using multi-processing, as a means of creating
    isolated test environments).
    - conversion (str): The conversion method to use for benchmarking.
    - data_loader (DataLoader): The DataLoader to get samples of data from.
    - n_samples (int): The number of samples to benchmark on.

    Outputs:
    - total_elapsed_time (float): The total elapsed time for the benchmark.
    - total_time (float): The total time taken for inference.
    - total_samples (int): The total number of samples benchmarked.
    - throughput (float): The throughput of the model.
    """
    # If the model is a callable, call it to get the model
    if not isinstance(model, torch.nn.Module):
        logger.info(f"Initializing model inside {conversion} benchmarking")
        model = model()
        assert isinstance(
            model, torch.nn.Module
        ), "The provided callable should return a PyTorch model"

    # Send the model to device
    model = model.to(device)

    # Set to eval mode
    model.eval()

    # Get sample of data from dataloader
    data = get_sample_data(data_loader, device)

    # Get the forward call of the model, which we will benchmark. We also return the device we will
    # benchmark on, since some conversions are only supported for certain devices, e.g.
    # PyTorch native quantized conversions requires CPU
    forward_call = select_forward_call_function(model, conversion, data, device)

    # Clear all caches, etc.
    torch._dynamo.reset()

    # Benchmarking loop
    result = time_forward(
        forward_call,
        data_loader,
        n_samples,
        device,
        conversion,
        warmup_iterations=10,
    )

    result["status"] = "success"
    if logger.root.level <= logging.DEBUG:
        log_results(result)

    return result
