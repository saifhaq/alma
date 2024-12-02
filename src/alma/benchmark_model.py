import argparse
import logging
from typing import Callable, List, Union

import torch
from torch.utils.data import DataLoader

from .benchmark import benchmark, log_results
from .conversions.select import MODEL_CONVERSION_OPTIONS
from .dataloader.create import create_single_tensor_dataloader
from .utils.setup_logging import setup_logging
from .utils.times import inference_time_benchmarking  # should we use this?


def benchmark_model(
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    conversions: Union[List[str], None] = None,
    data: Union[torch.Tensor, None] = None,
    data_loader: Union[DataLoader, None] = None,
) -> None:
    """
    Benchmark the model using the given data loader. From the args, we can select which model
    conversion method to use.

    Inputs:
    - model (torch.nn.Module): The model to benchmark.
    - device (torch.device): The device to run the model on.
    - args (argparse.Namespace): The command line arguments.
    - conversions (List[str]): The list of conversion methods to benchmark. If None, all of the
        available conversion methods will be benchmarked.
    - data (torch.Tensor): The data to use for benchmarking. If provided, and the data loader has
            not been provided, then the data shape will be used as the basis for the data loader.
    - data_loader (DataLoader): The DataLoader to get samples of data from. If provided, this will
            be used. Else, a random dataloader will be created.

    Outputs:
    None
    """
    setup_logging()

    # Either the `data` Tensor must be provided, or a data loader
    if data is None:
        assert (
            data_loader is not None
        ), "If data is not provided, the data_loader must be provided"
    if data_loader is None:
        assert (
            data is not None
        ), "If data_loader is not provided, a data tensor must be provided"

    # If the conversions are not provided, we use all available conversions
    if conversions is None:
        conversions = list(MODEL_CONVERSION_OPTIONS.values())

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

    all_results = {}
    for conversion_method in conversions:
        logging.info(f"Benchmarking model using conversion: {conversion_method}")
        times: tuple = benchmark(
            model, conversion_method, device, data_loader, n_samples, logging
        )
        all_results[conversion_method] = times

    logging.info("\n\nAll results:")
    for conversion_method, result in all_results.items():
        logging.info(f"{conversion_method} results:")
        total_elapsed_time, total_time, total_samples, throughput = result
        log_results(logging, total_elapsed_time, total_time, total_samples, throughput)
