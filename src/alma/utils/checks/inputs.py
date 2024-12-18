from typing import Callable, List

import torch
from torch.utils.data import DataLoader

from alma.benchmark.benchmark_config import BenchmarkConfig


def check_input_type(model, config, conversions, data, data_loader) -> None:
    """
    Checks that the inputs are of the correct types
    """
    assert isinstance(
        model, (torch.nn.Module, Callable)
    ), "The model must be a torch.nn.Module or callable"
    assert isinstance(
        config, BenchmarkConfig
    ), "The config must be of type BenchmarkConfig"
    assert isinstance(conversions, (list, type(None))), "The conversions must be a list"
    assert isinstance(
        data, (torch.Tensor, type(None))
    ), "The data must be a torch.Tensor"
    assert isinstance(
        data_loader, (DataLoader, type(None))
    ), "The data_loader must be a DataLoader"
