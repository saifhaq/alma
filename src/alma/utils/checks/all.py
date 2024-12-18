from typing import Callable, List, Union

import torch
from torch.utils.data import DataLoader

from alma.benchmark.benchmark_config import BenchmarkConfig

from .config import check_config
from .data import check_data_or_dataloader
from .inputs import check_input_type
from .model import check_model


def check_inputs(
    model: Union[torch.nn.Module, Callable],
    config: BenchmarkConfig,
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

    # Check model is valid for multiprocessing (if callable, should be pickle-able)
    check_model(model, config)
