from typing import Optional

import torch
from pydantic import BaseModel, Field, model_validator

from alma.utils.device import setup_device


class BenchmarkConfig(BaseModel):
    """
    Configuration model for benchmarking a machine learning model.

    Attributes:
        n_samples (int): Number of samples to benchmark. Defaults to 128.
        batch_size (int): Batch size for benchmarking. Defaults to 128.
        multiprocessing (bool): Enables multiprocessing during benchmarking. Defaults to True.
        fail_on_error (bool): Fails benchmarking on any error. Defaults to True.
        allow_device_override (bool): Allows automatic device selection override. Defaults to True.
        allow_cuda (bool): Allows CUDA usage if available. Defaults to True.
        allow_mps (bool): Allows MPS usage if available. Defaults to True.
        device (torch.device): Target device for benchmarking. Auto-selected if not provided.
    """

    n_samples: int = Field(
        default=128, gt=0, description="Number of samples to benchmark."
    )
    batch_size: int = Field(
        default=128, gt=0, description="Batch size for benchmarking."
    )
    multiprocessing: bool = Field(
        default=True, description="Enable multiprocessing support."
    )
    fail_on_error: bool = Field(
        default=True, description="Fail immediately on any error."
    )
    allow_device_override: bool = Field(
        default=True, description="Allow device override selection."
    )
    allow_cuda: bool = Field(
        default=True, description="Allow CUDA acceleration if available."
    )
    allow_mps: bool = Field(
        default=True, description="Allow MPS acceleration if available."
    )
    device: Optional[torch.device] = Field(
        default=None, description="Device for benchmarking."
    )

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="before")
    def select_device(cls, values):
        """
        Automatically selects a device if none is provided, respecting allow_cuda and allow_mps settings.
        """
        if values.get("device") is None:
            allow_cuda = values.get("allow_cuda", True) and torch.cuda.is_available()
            allow_mps = (
                values.get("allow_mps", True) and torch.backends.mps.is_available()
            )
            allow_override = values.get("allow_device_override", True)

            values["device"] = setup_device(
                allow_cuda=allow_cuda,
                allow_mps=allow_mps,
                allow_device_override=allow_override,
            )
        return values
