import logging
import time
from typing import Any, Dict, Union

import torch
import torch._dynamo
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..conversions.conversion_options import ConversionOption
from ..conversions.select import select_forward_call_function
from ..dataloader.create import create_single_tensor_dataloader
from ..utils.data import get_sample_data
from ..utils.multiprocessing import benchmark_error_handler, init_lazy_model
from .benchmark_config import BenchmarkConfig
from .log import log_results
from .warmup import warmup

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@benchmark_error_handler
def benchmark(
    device: torch.device,
    model: Any,
    config: BenchmarkConfig,
    conversion: ConversionOption,
    data: Optional[torch.Tensor] = None,
    data_loader: Optional[DataLoader] = None,
) -> Dict[str, Union[torch.device, float, int, str, torch.dtype]]:
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
    - config (BenchmarkConfig): The configuration for the benchmarking.
    - conversion (ConversionOption): The conversion method to benchmark.
    - data (Optional[torch.Tensor]): The input data to benchmark the model on.
    - data_loader (Optional[DataLoader]): The DataLoader to get samples of data from.

    Outputs:
    - total_elapsed_time (float): The total elapsed time for the benchmark.
    - total_time (float): The total time taken for inference.
    - total_samples (int): The total number of samples benchmarked.
    - throughput (float): The throughput of the model.
    """
    # If the model is a LazyLoad instance, load it to initialize the model
    model = init_lazy_model(model)

    # Get the number of samples to benchmark
    n_samples = config.n_samples
    batch_size = config.batch_size

    # Creates a dataloader with random data, of the same size as the input data sample
    # If the data_loader has been provided by the user, we use that one
    if not isinstance(data_loader, DataLoader):
        data_loader = create_single_tensor_dataloader(
            tensor_size=data.size(),
            num_tensors=n_samples,
            random_type="normal",
            random_params={"mean": 0.0, "std": 2.0},
            batch_size=batch_size,
            dtype=conversion.data_dtype,
        )
    else:
        # If a data loader is provided, we check that the data dtype matches the conversion dtype
        data = get_sample_data(data_loader, device)
        assert (
            data.dtype == conversion.data_dtype
        ), f"The data loader dtype ({data.dtype}) does not match the conversion dtype ({conversion.data_dtype})."

    import ipdb; ipdb.set_trace(); import pprint
    # Send the model to device
    model = model.to(device)

    # Set to eval mode
    model.eval()

    # Initialize benchmark variables
    total_samples = 0

    # Get sample of data from dataloader. This overwrites the data tensor provided by the user
    data = get_sample_data(data_loader, device)

    # Get the forward call of the model, which we will benchmark. We also return the device we will
    # benchmark on, since some conversions are only supported for certain devices, e.g.
    # PyTorch native quantized conversions requires CPU
    forward_call = select_forward_call_function(model, conversion.mode, data, device)
    import ipdb; ipdb.set_trace(); import pprint

    # Clear all caches, etc.
    torch._dynamo.reset()

    # Warmup
    warmup(forward_call, data_loader, device)

    # Setup CUDA events for more accurate GPU timing if available
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
    else:
        start_time = time.perf_counter()

    # Calculate number of full batches needed
    n_batches = (n_samples + config.batch_size - 1) // config.batch_size

    # Benchmarking loop - only process full batches
    with torch.no_grad():
        for i, (data, _) in enumerate(
            tqdm(
                data_loader,
                desc=f"Benchmarking {conversion.mode} on {device}",
                total=n_batches,
            )
        ):
            # Transfer data to device
            data = data.to(device, non_blocking=config.non_blocking)

            # Run forward pass without per-batch timing
            _ = forward_call(data)

            total_samples += data.size(0)

            if total_samples > n_samples:
                break

    # End timing and synchronize if needed
    if device.type == "cuda":
        end_event.record()
        end_event.synchronize()
        total_elapsed_time = (
            start_event.elapsed_time(end_event) / 1000.0
        )  # Convert ms to seconds
    else:
        end_time = time.perf_counter()
        total_elapsed_time = end_time - start_time

    throughput = total_samples / total_elapsed_time if total_elapsed_time > 0 else 0

    result: Dict[str, Any] = {
        "device": device,
        "total_elapsed_time": total_elapsed_time,
        "total_samples": total_samples,
        "batch_size": config.batch_size,
        "throughput": throughput,
        "status": "success",
        "data_dtype": conversion.data_dtype,
    }
    if logger.root.level <= logging.DEBUG:
        log_results(result)

    return result
