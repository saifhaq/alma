import logging
import os
from typing import Callable, Literal

import torch._export
import torch._inductor
from torch.export.exported_program import ExportedProgram
from torch.fx.graph_module import GraphModule

from .export_quant import get_quant_exported_model
from .utils.check_type import check_model_type
from .utils.export import get_exported_model

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_export_aot_inductor_forward_call(
    model: ExportedProgram | GraphModule, data: torch.Tensor
) -> Callable:
    """
    Get the forward call function for the exported model using AOTInductor.

    Inputs:
    - model (Union[ExportedProram, GraphModule]): The model to get the forward call for.
    - data (torch.Tensor): A sample of data to pass through the model.

    Outputs:
    - forward (Callable): The forward call function for the model.
    """

    # Export the model
    model = get_exported_model(model, data)

    # NOTE: not sure if this is correct. It may be correct for compile, or for compile+export.
    # If the latter, move this into the export sub directory.
    check_model_type(model, ExportedProgram)

    return get_AOTInductor_lowered_model_forward_call(model, data)


def get_AOTInductor_lowered_model_forward_call(
    model: ExportedProgram | GraphModule, data: torch.Tensor
) -> Callable:
    """
    Get the forward call function for the model using AOTInductor.

    Inputs:
    - model (Union[ExportedProram, GraphModule]): The model to get the forward call for.
    - data (torch.Tensor): A sample of data to pass through the model.

    Outputs:
    - forward (Callable): The forward call function for the model.
    """

    check_model_type(model, (ExportedProgram, GraphModule))
    logger.info("Lowering the model with AOTInductor")

    # Compile the exported program to a `.so` using ``AOTInductor``
    # E.g. this can be run as a C++ file, or called from Python
    if isinstance(model, ExportedProgram):
        with torch.no_grad():
            so_path = torch._inductor.aot_compile(model.module(), (data,))
    else:
        with torch.no_grad():
            so_path = torch._inductor.aot_compile(model, (data,))

    # Load and run the .so file in Python.
    # To load and run it in a C++ environment, see:
    # https://pytorch.org/docs/main/torch.compiler_aot_inductor.html
    forward = torch._export.aot_load(so_path, device="cuda")

    return forward


def get_quant_export_aot_inductor_forward_call(
    model,
    data: torch.Tensor,
    int_or_dequant_op: Literal["int", "dequant"],
) -> Callable:
    """
    Get the forward call function for the exported quantized model using AOTInductor.
    We first produce the quantized exported model, then call AOtInductor to lower it.

    Inputs:
    - model (torch.nn.Module): The model to export
    - data (torch.Tensor): Sample data to feed through the model for tracing.
    - int_or_dequant_op (Literal["int", "dequant"]): do we use integer arithmetic operations on
            quantized layers, or do we dequantize just prior to the op

    Outputs:
    - forward (Callable): The forward call function for the model.
    """

    model = get_quant_exported_model(model, data, int_or_dequant_op)
    forward = get_export_aot_inductor_forward_call(model, data, logger)
    return forward
