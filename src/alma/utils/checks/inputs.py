import logging
from typing import Callable, List, Union

import torch
from torch.utils.data import DataLoader

from .config import check_config
from .data import check_data_or_dataloader

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def check_inputs(
    model: Union[torch.nn.Module, Callable],
    config: dict,
    conversions: Union[List[str], None],
    data: Union[torch.Tensor, None],
    data_loader: Union[DataLoader, None],
):
    """
    Check the inputs to the benchmark_model function.

    Inputs:
    - model (Union[torch.nn.Module, Callable]): The model to benchmark, or a callable function to get the model.
    - config (dict): The configuration for the benchmarking.
    - conversions (list): The list of conversion methods to benchmark.
    - data (torch.Tensor): The data to use for benchmarking (initialising the dataloader).
    - data_loader (DataLoader): The DataLoader to get samples of data from.

    Outputs:
    None
    """
    # Basic type checks
    check_input_type(model, config, conversions, data, data_loader)

    # Check the config
    check_config(config)

    # Check that either the data or data_loader is provided
    check_data_or_dataloader(data, data_loader)

    # Check model is valid (either torch module or callable)
    check_model(model, config)
    # If the model is a torch.nn.Mpodule and multiprocessing is enabled, we log a warning
    # that this is not memory efficient
    if isinstance(model, torch.nn.Module) and config.get("multiprocessing", True):
        warning_msg = """Multiprocessing is enabled, and the model is a torch.nn.Module. This is not memory efficient,
as the model will be pickled and sent to each child process, which will require the model to be stored in memory
twice. If the model is large, this may cause memory issues. Consider using a callable to return the model, which
will be created in each child process, rather than the parent process. See `examples/mnist/mem_efficient_benchmark_rand_tensor.py`
for an example."""
        logger.warning(warning_msg)


def check_input_type(model, config, conversions, data, data_loader) -> None:
    """
    Checks that the inputs are of the correct types
    """
    assert isinstance(
        model, (torch.nn.Module, Callable)
    ), "The model must be a torch.nn.Module or callable"
    assert isinstance(config, dict), "The config must be a dictionary"
    assert isinstance(conversions, (list, type(None))), "The conversions must be a list"
    assert isinstance(
        data, (torch.Tensor, type(None))
    ), "The data must be a torch.Tensor"
    assert isinstance(
        data_loader, (DataLoader, type(None))
    ), "The data_loader must be a DataLoader"
