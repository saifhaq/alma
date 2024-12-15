import logging
from typing import Any, Optional, Union

import torch
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class BenchmarkConfig(BaseModel):
    n_samples: int
    batch_size: int
    device: Any
    multiprocessing: Optional[bool]
    fail_on_error: Optional[bool]


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


def is_valid_torch_device(device: Union[str, torch.device]) -> torch.device:
    """
    Check if a string is a valid torch device.
    """
    if isinstance(device, str):
        try:
            device = torch.device(device)
            return device
        except RuntimeError:
            raise
    else:
        assert device, "device should be of type torch.device or str"
        return device


def check_config(config: dict) -> None:
    """
    Check the config is valid.

    Inputs:
    - comfig (dict): the config

    Outputs:
    None
    """
    # Check the device in the config
    assert "device" in config, "`device` must be provided in config"
    device = is_valid_torch_device(config["device"])
    config["device"] = device

    # Check the configuration, there are some required fields
    _ = BenchmarkConfig.model_validate(config)
