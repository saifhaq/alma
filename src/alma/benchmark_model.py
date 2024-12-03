import argparse
from typing import Callable, List, Union, Any, Dict

import torch
from torch.utils.data import DataLoader

from .benchmark import benchmark, log_results
from .conversions.select import MODEL_CONVERSION_OPTIONS
from .dataloader.create import create_single_tensor_dataloader
from .utils.times import inference_time_benchmarking  # should we use this?

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def benchmark_model(
    model: torch.nn.Module,
    config: Dict[str, Any],
    conversions: Union[List[str], None] = None,
    data: Union[torch.Tensor, None] = None,
    data_loader: Union[DataLoader, None] = None,
    # logger: Union[logging.Logger, None] = None,
    # verbose: bool = False,
) -> None:
    """
    Benchmark the model on different conversion methods. If provided, the dataloader will be used.
    Else, a random dataloader will be created, in which case the `data` tensor must be provided
    to initialize the dataloader.
    The model is benchmarked on the given conversion methods, and the results are logged.
    If no logger is provided, a default logger is created. If the verbose flag is set, the logger
    level is set to DEBUG. Otherwise the INFO level is used.
    `config` contains the following:
    - n_samples: int     # The number of samples to benchmark on
    - batch_size: int    # The batch size to use for benchmarking

    Inputs:
    - model (torch.nn.Module): The model to benchmark.
    - config (Dict[str, Any]): The configuration for the benchmarking. This contains the number of
        samples to benchmark on, and the batch size to use for benchmarking.
    - conversions (List[str]): The list of conversion methods to benchmark. If None, all of the
        available conversion methods will be benchmarked.
    - data (torch.Tensor): The data to use for benchmarking. If provided, and the data loader has
            not been provided, then the data shape will be used as the basis for the data loader.
    - data_loader (DataLoader): The DataLoader to get samples of data from. If provided, this will
            be used. Else, a random dataloader will be created.
    - logger (logging.Logger): The logger to use for logging. If None, a default logger is created.
    - verbose (bool): Whether to set the logger level to DEBUG. If False, the INFO level is used.

    Outputs:
    - all_results (Dict[str, tuple]): The results of the benchmarking for each conversion method.
        The key is the conversion method, and the value is a tuple containing the total elapsed
        time, the total time taken, the total number of samples, and the
    """
    # # If no logger is provided, we set up the default logger
    # if logger is None:
    #     setup_logging()
    # # If the verbose flag is set, we set the logger level to DEBUG
    # if verbose:
    #     logging.getLogger().setLevel(logging.DEBUG)

    # We determine the device to run the model on
    # NOTE: this will only work for single-device set ups. Benchmarking on multiple devices is not
    # currently supported.
    device: torch.device = next(model.parameters()).device

    # Check the configuration
    assert "batch_size" in config, "The batch size must be provided in the config"
    assert "n_samples" in config, "The number of samples (n_samples) to benchmark on must be provided in the config"

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
    n_samples: int = config["n_samples"]

    # Creates a dataloader with random data, of the same size as the input data sample
    # If the data_loader has been provided by the user, we use that one
    if not data_loader:
        data_loader = create_single_tensor_dataloader(
            tensor_size=data.size(),
            num_tensors=n_samples,
            random_type="normal",
            random_params={"mean": 0.0, "std": 2.0},
            batch_size=config["batch_size"],
        )

    all_results: Dict[str, Dict[str, float]] = {}
    for conversion_method in conversions:
        logging.info(f"Benchmarking model using conversion: {conversion_method}")
        times: Dict[str, float] = benchmark(
            model, conversion_method, device, data_loader, n_samples, logging
        )
        all_results[conversion_method] = times

    logging.info("\n\nAll results:")
    for conversion_method, result in all_results.items():
        logging.info(f"{conversion_method} results:")
        log_results(result)
        print("\n")

    return all_results
