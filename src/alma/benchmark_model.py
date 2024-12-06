import logging
import traceback
from typing import Any, Dict, List, Union

import torch
from torch.utils.data import DataLoader

from .benchmark import benchmark
from .conversions.select import MODEL_CONVERSION_OPTIONS
from .dataloader.create import create_single_tensor_dataloader
from .utils.checks import check_consistent_batch_size, check_inputs
from .utils.times import inference_time_benchmarking  # should we use this?

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def benchmark_model(
    model: torch.nn.Module,
    config: Dict[str, Any],
    conversions: Union[List[str], None] = None,
    data: Union[torch.Tensor, None] = None,
    data_loader: Union[DataLoader, None] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark the model on different conversion methods. If provided, the dataloader will be used.
    Else, a random dataloader will be created, in which case the `data` tensor must be provided
    to initialize the dataloader.
    The model is benchmarked on the given conversion methods, and the results are returned.
    The `config` dict must contain the following:
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

    Outputs:
    - all_results (Dict[str, Dict[str, Any]]): The results of the benchmarking for each conversion method.
        The key is the conversion method, and the value is a tuple containing the total elapsed
        time, the total time taken, the total number of samples, and the throughput of the model.
        If the conversion method failed, the value will be a dictionary containing the error and
        traceback.
    """
    # Set to eval mode
    model.eval()

    # We determine the device to run the model on
    # NOTE: this will only work for single-device set ups. Benchmarking on multiple devices is not
    # currently supported.
    device: torch.device = next(model.parameters()).device

    # Check the inputs
    check_inputs(model, config, conversions, data, data_loader)

    # If the conversions are not provided, we use all available conversions
    if conversions is None:
        conversions = list(MODEL_CONVERSION_OPTIONS.values())

    # The number of samples to benchmark on, batch size, and whether or not to give verbose logging
    n_samples: int = config["n_samples"]
    batch_size: int = config["batch_size"]

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
        check_consistent_batch_size(conversion_method, n_samples, batch_size)

        logger.info(f"Benchmarking model using conversion: {conversion_method}")
        try:
            result: Dict[str, float] = benchmark(
                model, conversion_method, device, data_loader, n_samples
            )
            result["status"] = "success"
            all_results[conversion_method] = result
        except Exception as e:
            # If there is an error, we log the error. In the returned "results", we include the
            # full traceback
            error_msg = (
                f"Benchmarking conversion {conversion_method} failed. Error: {e}"
            )
            logger.error(error_msg)
            all_results[conversion_method] = {
                "status": "error",
                "error": e,
                "traceback": traceback.format_exc(),
            }

    return all_results
