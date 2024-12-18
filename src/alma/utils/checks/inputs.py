from typing import Callable, List

import torch
from torch.utils.data import DataLoader

from alma.benchmark.benchmark_config import BenchmarkConfig
from alma.conversions.conversion_options import ConversionOption


def check_input_type(model, config, conversions, data, data_loader) -> None:
    """
    Checks that the inputs are of the correct types.
    """
    assert isinstance(
        model, (torch.nn.Module, Callable)
    ), "The model must be a torch.nn.Module or callable"
    assert isinstance(
        config, BenchmarkConfig
    ), "The config must be of type BenchmarkConfig"

    assert isinstance(
        conversions, (list, type(None))
    ), f"Expected conversions to be a list or None, got {type(conversions)}."
    if conversions is not None:
        for conv in conversions:
            assert isinstance(conv, ConversionOption), (
                f"Expected each element of conversions to be of type ConversionOption, "
                f"but got {type(conv)}."
            )

    assert isinstance(
        data, (torch.Tensor, type(None))
    ), "The data must be a torch.Tensor"
    assert isinstance(
        data_loader, (DataLoader, type(None))
    ), "The data_loader must be a DataLoader"
