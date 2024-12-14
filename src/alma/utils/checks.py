from typing import List, Union, Optional, Callable

import logging
import torch
from torch.utils.data import DataLoader
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class BenchmarkConfig(BaseModel):
    n_samples: int
    batch_size: int
    device: Union[torch.device, str]
    multiprocessing: Optional[bool]
    fail_fast: Optional[bool]


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
    # badoc type checks
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

    # Check the configuration, there are some required fields
    try:
        _ = BenchmarkConfig.model_validate(config)
    except ValueError as e:
        print(
            f"Validation error: {e}, Please ensure the `config` has all of the required fields."
        )
    # Check the device
    is_valid_torch_device(config["device"])

    # Either the `data` Tensor must be provided, or a data loader
    if data is None:
        error_msg = "If data is not provided, the data_loader must be provided"
        assert data_loader is not None, error_msg
    if data_loader is None:
        error_msg = "If data_loader is not provided, the data tensor must be provided"
        assert data is not None, error_msg
    else:
        error_msg = "If a data loader is provided, the data tensor must be None"
        assert data is None, error_msg

    # If the model is a torch.nn.Mpodule and multiprocessing is enabled, we log a warning
    # that this is not memory efficient
    if isinstance(model, torch.nn.Module) and config.get("multiprocessing", True):
        warning_msg = """Multiprocessing is enabled, and the model is a torch.nn.Module. This is not memory efficient,
as the model will be pickled and sent to each child process, which will require the model to be stored in memory
twice. If the model is large, this may cause memory issues. Consider using a callable to return the model, which
will be created in each child process, rather than the parent process. See `examples/mnist/mem_efficient_benchmark_rand_tensor.py`
for an example."""
        logger.warning(warning_msg)


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


def is_valid_torch_device(device_str: str) -> bool:
    """
    Check if a string is a valid torch device.
    """
    try:
        torch.device(device_str)
        return True
    except RuntimeError:
        return False
