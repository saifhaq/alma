import logging
import time
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..conversions.select import select_forward_call_function
from ..utils.data import args_to_dict, get_sample_data
from ..utils.times import inference_time_benchmarking  # should we use this?
from .log import log_results
from .warmup import warmup


def benchmark(
    model: torch.nn.Module,
    conversion: str,
    device: torch.device,
    data_loader: DataLoader,
    n_samples: int,
    logger: logging.Logger,
) -> Dict[str, float]:
    """
    Benchmark the model using the given data loader. This function will benchmark the model using the
    given conversion method.

    Inputs:
    - model (torch.nn.Module): The model to benchmark.
    - conversion (str): The conversion method to use for benchmarking.
    - device (torch.device): The device we are targetting.
    - data_loader (DataLoader): The DataLoader to get samples of data from.
    - n_samples (int): The number of samples to benchmark on.
    - logger (logging.Logger): The logger to use for logging.

    Outputs:
    - total_elapsed_time (float): The total elapsed time for the benchmark.
    - total_time (float): The total time taken for inference.
    - total_samples (int): The total number of samples benchmarked.
    - throughput (float): The throughput of the model.
    """
    total_inf_time = 0.0
    total_samples = 0
    num_batches = 0

    # Get sample of data from dataloader
    data = get_sample_data(data_loader, device)

    # Get the forward call of the model, which we will benchmark
    forward_call = select_forward_call_function(model, conversion, data, logger)

    # Warmup
    warmup(forward_call, data_loader, device)

    # Benchmarking loop
    start_time = time.perf_counter()  # Start timing for the entire process
    with torch.no_grad():
        for data, _ in tqdm(data_loader, desc="Benchmarking"):
            if total_samples >= n_samples:
                break

            # data = data.to(device, non_blocking=True)
            data = data.to(device)
            batch_start_time = time.perf_counter()
            _ = forward_call(data)
            batch_end_time = time.perf_counter()

            batch_size = min(data.size(0), n_samples - total_samples)
            total_inf_time += batch_end_time - batch_start_time
            total_samples += batch_size
            num_batches += 1

            if total_samples >= n_samples:
                break

    end_time = time.perf_counter()  # End timing for the entire process

    total_elapsed_time = end_time - start_time
    throughput = total_samples / total_elapsed_time if total_elapsed_time > 0 else 0
    result: Dict[str, float] = {
        "total_elapsed_time": total_elapsed_time,
        "total_inf_time": total_inf_time,
        "total_samples": total_samples,
        "throughput": throughput,
    }
    if logger.root.level <= logging.DEBUG:
        log_results(result)

    return result
