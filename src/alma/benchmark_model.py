import argparse
import logging
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .conversions.select import select_forward_call_function
from .utils.data import get_sample_data
from .utils.times import inference_time_benchmarking  # should we use this?


def benchmark_model(
    model: torch.nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    n_samples: int,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    """
    Benchmark the model using the given data loader. From the args, we can select which model
    conversion method to use.

    Inputs:
    - model (torch.nn.Module): The model to benchmark.
    - device (torch.device): The device to run the model on.
    - data_loader (DataLoader): The DataLoader to get samples of data from.
    - n_samples (int): The number of samples to benchmark on.
    - args (argparse.Namespace): The command line arguments.
    - logger (logging.Logger): The logger to use for logging.

    Outputs:
    None
    """
    total_time = 0.0
    total_samples = 0
    num_batches = 0

    # Get sample of data, used in some of the compilation methods
    data = get_sample_data(data_loader, device)

    # Get the forward call of the model, which we will benchmark
    forward_call = select_forward_call_function(model, args, data, logger)

    # Warmup
    counter = 0
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            _ = forward_call(data)
            counter += 1
            if counter > 3:
                break

    start_time = time.perf_counter()  # Start timing for the entire process

    with torch.no_grad():
        for data, _ in tqdm(data_loader, desc="Benchmarking"):
            if total_samples >= n_samples:
                break

            # data = data.to(device, non_blocking=True)
            data = data.to(device)
            batch_start_time = time.time()
            _ = forward_call(data)
            batch_end_time = time.time()

            batch_size = min(data.size(0), n_samples - total_samples)
            total_time += batch_end_time - batch_start_time
            total_samples += batch_size
            num_batches += 1

            if total_samples >= n_samples:
                break

    end_time = time.perf_counter()  # End timing for the entire process

    total_elapsed_time = end_time - start_time
    throughput = total_samples / total_elapsed_time if total_elapsed_time > 0 else 0
    logger.info(f"Total elapsed time: {total_elapsed_time:.4f} seconds")
    logger.info(f"Total inference time (model only): {total_time:.4f} seconds")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Throughput: {throughput:.2f} samples/second")
