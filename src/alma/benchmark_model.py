import argparse
import logging
import time
from typing import Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .conversions.select import select_forward_call_function
from .dataloader.create import create_single_tensor_dataloader
from .utils.times import inference_time_benchmarking  # should we use this?


def benchmark_model(
    model: torch.nn.Module,
    device: torch.device,
    data: torch.Tensor,
    args: argparse.Namespace,
    logger: logging.Logger,
    data_loader: Union[DataLoader, None] = None,
) -> None:
    """
    Benchmark the model using the given data loader. From the args, we can select which model
    conversion method to use.

    Inputs:
    - model (torch.nn.Module): The model to benchmark.
    - data (torch.Tensor): The data to use for benchmarking.
    - device (torch.device): The device to run the model on.
    - args (argparse.Namespace): The command line arguments.
    - logger (logging.Logger): The logger to use for logging.
    - data_loader (DataLoader): The DataLoader to get samples of data from. If provided, this will
            be used. Else, a random dataloader will be created.

    Outputs:
    None
    """
    total_time = 0.0
    total_samples = 0
    num_batches = 0

    # The number of samples to benchmark on
    n_samples: int = args.n_samples

    # Creates a dataloader with random data, of the same size as the input data sample
    # If the data_loader has been provided by the user, we use that one
    if not data_loader:
        data_loader = create_single_tensor_dataloader(
            tensor_size=data.size(),
            num_tensors=int(
                n_samples * 1.5
            ),  # 1.5 is a magic number to ensure we have enough samples in the dataloader
            random_type="normal",
            random_params={"mean": 0.0, "std": 2.0},
            batch_size=args.batch_size,
        )

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
