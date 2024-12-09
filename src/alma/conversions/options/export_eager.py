from typing import Callable

import torch
from torch.export.exported_program import ExportedProgram

from .utils.checks.type import check_model_type
from .utils.export import get_exported_model


def get_export_eager_forward_call(
    model: torch.nn.Module, data: torch.Tensor
) -> Callable:
    """
    Get eager mode forward call of export (shouldn't be much faster than basic eager
    mode, the only difference is we perhaps remove some of the Python wrapper functions
    around the Aten ops)

    Inputs:
    - model (torch.nn.Module): The model to get the forward call for.
    - data (torch.Tensor): A sample of data to pass through the model.

    Outputs:
    - forward (Callable): The forward call function for the model.

    """
    # Get exported model
    model = get_exported_model(model, data)

    check_model_type(model, ExportedProgram)
    forward = model.module().forward

    return forward
