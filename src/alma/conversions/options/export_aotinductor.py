import logging
import os
from typing import Callable, Literal

import torch._export
import torch._inductor
from torch.export.exported_program import ExportedProgram
from torch.fx.graph_module import GraphModule

from ...utils.setup_logging import suppress_output
from .export_quant import get_quant_exported_model
from .utils.checks.type import check_model_type
from .utils.export import get_exported_model

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_export_aot_inductor_forward_call(
    model: ExportedProgram | GraphModule, data: torch.Tensor, device: torch.device
) -> Callable:
    """
    Get the forward call function for the exported model using AOTInductor.

    Inputs:
    - model (Union[ExportedProram, GraphModule]): The model to get the forward call for.
    - data (torch.Tensor): A sample of data to pass through the model.
    - device (torch.device): The device we are loading the AOTInductor-lowered model to.

    Outputs:
    - forward (Callable): The forward call function for the model.
    """

    # Export the model
    model = get_exported_model(model, data)

    check_model_type(model, ExportedProgram)

    return get_AOTInductor_lowered_model_forward_call(model, data, device)


def get_AOTInductor_lowered_model_forward_call(
    model: ExportedProgram | GraphModule, data: torch.Tensor, device: torch.device
) -> Callable:
    """
    Get the forward call function for the model using AOTInductor.

    Inputs:
    - model (Union[ExportedProram, GraphModule]): The model to get the forward call for.
    - data (torch.Tensor): A sample of data to pass through the model.
    - device (torch.device): The device we are loading the AOTInductor-lowered model to.

    Outputs:
    - forward (Callable): The forward call function for the model.
    """

    check_model_type(model, (ExportedProgram, GraphModule))
    logger.info("Lowering the model with AOTInductor")

    # Compile the exported program to a `.so` using ``AOTInductor``
    # E.g. this can be run as a C++ file, or called from Python
    with suppress_output(logger.root.level >= logging.DEBUG):
        if isinstance(model, ExportedProgram):
            with torch.no_grad():
                so_path = torch._inductor.aot_compile(model.module(), (data,))
        else:
            with torch.no_grad():
                so_path = torch._inductor.aot_compile(model, (data,))

        # Load and run the .so file in Python.
        # To load and run it in a C++ environment, see:
        # https://pytorch.org/docs/main/torch.compiler_aot_inductor.html
        forward = torch._export.aot_load(so_path, device=device.type)

    return forward
