import logging
from typing import Callable

import torch
from torch.export.exported_program import ExportedProgram

from .utils import get_exported_model


def get_export_compiled_forward_call(
    model: torch.nn.Module, data: torch.Tensor, logging: logging.Logger
) -> Callable:
    """
    Get the forward call function for the exported model using torch.compile.

    Inputs:
    - model (torch.nn.Module): The model to get the forward call for.
    - data (torch.Tensor): A sample of data to pass through the model.
    - logging (logging.Logger): The logger to use for logging

    Outputs:
    - forward (Callable): The forward call function for the model.
    """

    # Export the model
    model = get_exported_model(model, data, logging)

    assert isinstance(
        model, ExportedProgram
    ), f"model must be of type ExportedProgram, got {type(model)}"

    # Set the comilation settings
    compile_settings = {
        # Good for small models
        # 'mode': "reduce-overhead",
        # Slow to compile, but should find the "best" option
        "mode": "max-autotune",
        # Compiles entire program into 1 graph, but only works with a restricted subset of Python
        # (e.g. no data dependent control flow)
        "fullgraph": True,
    }

    # Compile the model, and get the forward call
    forward = torch.compile(model.module(), **compile_settings)  # , backend="inductor")

    return forward
