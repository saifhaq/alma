from typing import List, Union

import torch
from torch.utils.data import DataLoader


def check_inputs(
    model: torch.nn.Module,
    config: dict,
    conversions: Union[List[str], None],
    data: Union[torch.Tensor, None],
    data_loader: Union[DataLoader, None],
):
    """
    Check the inputs to the benchmark_model function.

    Inputs:
    - model (torch.nn.Module): The model to benchmark.
    - config (dict): The configuration for the benchmarking.
    - conversions (list): The list of conversion methods to benchmark.
    - data (torch.Tensor): The data to use for benchmarking (initialising the dataloader).
    - data_loader (DataLoader): The DataLoader to get samples of data from.

    Outputs:
    None
    """
    assert isinstance(model, torch.nn.Module), "The model must be a torch.nn.Module"
    assert isinstance(config, dict), "The config must be a dictionary"
    assert isinstance(conversions, (list, type(None))), "The conversions must be a list"
    assert isinstance(
        data, (torch.Tensor, type(None))
    ), "The data must be a torch.Tensor"
    assert isinstance(
        data_loader, (DataLoader, type(None))
    ), "The data_loader must be a DataLoader"

    # Check the configuration
    assert "batch_size" in config, "The batch size must be provided in the config"
    assert (
        "n_samples" in config
    ), "The number of samples (n_samples) to benchmark on must be provided in the config"

    # Either the `data` Tensor must be provided, or a data loader
    if data is None:
        assert (
            data_loader is not None
        ), "If data is not provided, the data_loader must be provided"
    if data_loader is None:
        assert (
            data is not None
        ), "If data_loader is not provided, a data tensor must be provided"
    else:
        assert (
            data is None
        ), "If a data loader is provided, the data tensor must be None"


def check_consistent_batch_size(
    conversion_method: str, n_samples: int, batch_size: int
) -> None:
    """
    Check that the batch size will always be consistent, which is a requirement for certain
    conversion methods.

    Inputs:
    - conversion_method (str): the conversion method being evaluated.
    - n_samples (int): the number of samples to use during benchmarking. This may have to be a perfect
        multiple of the batch size, depending on the conversion method.
    - batch_size (int): the batch size to use during benchmarking.
    """
    if "COMPILE" in conversion_method:
        if n_samples % batch_size != 0:
            error_msg = f"""n_samples must be a multiple of batch_size for compilation based methods, 
otherwise the last batch will fail due to having a different shape. n-samples was {n_samples}, and 
batch size was {batch_size} """
            raise ValueError(error_msg)
