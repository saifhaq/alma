import logging
from typing import Callable, Dict, Literal

import torch
from torch.export.exported_program import ExportedProgram

from .compile import (
    get_compiled_forward_call_eager_fallback,
    get_compiled_model_forward_call,
)
from .utils.checks.type import check_model_type
from .utils.export import get_exported_model

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_export_compiled_forward_call(
    model: torch.nn.Module,
    data: torch.Tensor,
    backend: Literal[str] = "inductor-default",
) -> Callable:
    """
    Get the forward call function for the exported model using torch.compile.

    Inputs:
    - model (torch.nn.Module): The model to get the forward call for.
    - data (torch.Tensor): A sample of data to pass through the model.
    - backend (Literal[str]): The backend to use for torch.compile. Currently supported options in
        PyTorch are given by torch._dynamo.list_backends():
        ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']
        However, we have also split 'inductor' into 'inductor-default', 'inductor-max-autotune', and
        'inductor-reduce-overhead'. See here for an explanation of each:
        https://pytorch.org/get-started/pytorch-2.0/#user-experience

    Outputs:
    - forward (Callable): The forward call function for the model.
    """

    # Export the model
    model = get_exported_model(model, data)

    check_model_type(model, ExportedProgram)

    forward = get_compiled_model_forward_call(model.module(), data, backend)

    return forward


def get_export_compiled_forward_call_eager_fallback(
    model: torch.nn.Module, data: torch.Tensor, backend: Literal[str] = "inductor"
) -> Callable:
    """
    Get the forward call function for the exported model using torch.compile. If torch.compile
    fails, we fall back on Eager mode.

    Inputs:
    - model (torch.nn.Module): The model to get the forward call for.
    - data (torch.Tensor): A sample of data to pass through the model.
    - backend (Literal[str]): The backend to use for torch.compile. Currently supported options in
        PyTorch are given by torch._dynamo.list_backends():
        ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']

    Outputs:
    - forward (Callable): The forward call function for the model.
    """

    # Export the model
    model = get_exported_model(model, data)

    check_model_type(model, ExportedProgram)

    forward = get_compiled_forward_call_eager_fallback(model.module(), data, backend)

    return forward
