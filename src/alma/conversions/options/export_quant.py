import copy
import logging
from argparse import Namespace
from typing import Callable, Literal

import torch
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torch.export.exported_program import ExportedProgram

from .utils.check_type import check_model_type


def get_quant_exported_model(
    model,
    data: torch.Tensor,
    logging: logging.Logger,
    int_or_dequant_op: Literal["int", "dequant"],
) -> torch.fx.graph_module.GraphModule:
    """
    Export the model using torch.export.export_for_training and convert_pt2e to get a quantized
    exported model.

    Inputs:
    - model (torch.nn.Module): The model to export
    - data (torch.Tensor): Sample data to feed through the model for tracing.
    - logging (logging.Logger): The logger to use for logging
    - int_or_dequant_op (Literal["int", "dequant"]): do we use integer arithmetic operations on
            quantized layers, or do we dequantize just prior to the op

    Outputs:
    model (torch.export.Model): The exported model
    """

    logging.info(
        "Running torch.export.export_for_training on the model to get a quantized exported model"
    )

    # We do this check early to save time, used when converting the quantized model
    if int_or_dequant_op == "int":
        int_op = True
    elif int_or_dequant_op == "dequant":
        int_op = False
    else:
        raise ValueError(
            f"`int_or_dequant_op` shoudl equal `int` or `dequant, not {int_or_dequant_op}"
        )

    # Step 1. program capture
    # This is available for pytorch 2.5+, for more details on lower pytorch versions
    # please check `Export the model with torch.export` section
    # we get a model with aten ops
    m_export: torch.fx.graph_module.GraphModule = torch.export.export_for_training(
        model, (data,)
    ).module()

    # Step 2. quantization
    # TODO: mess around with affine quantization
    to_quant_model = copy.deepcopy(m_export)
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())

    # PTQ step
    m_fq: torch.fx.graph_module.GraphModule = prepare_pt2e(to_quant_model, quantizer)

    # TODO: calibration omitted for quantization

    # Lower the quantized model
    # use_reference_optimization=True means that one uses integer arithmetic, False means that one
    # does operations in floating point and dequantizes prior.

    m_q: torch.fx.graph_module.GraphModule = convert_pt2e(
        m_fq, use_reference_representation=int_op
    )
    m_q.graph.print_tabular()

    # we have a model with aten ops doing integer computations when possible
    check_model_type(m_q, torch.fx.graph_module.GraphModule)

    return m_q


def get_quant_exported_forward_call(
    model,
    data: torch.Tensor,
    logging: logging.Logger,
    int_or_dequant_op: Literal["int", "dequant"],
) -> Callable:
    """
    Get the torch export + quantized model forward call.

    Inputs:
    - model (torch.nn.Module): The model to export
    - data (torch.Tensor): Sample data to feed through the model for tracing.
    - logging (logging.Logger): The logger to use for logging
    - int_or_dequant_op (Literal["int", "dequant"]): do we use integer arithmetic operations on
            quantized layers, or do we dequantize just prior to the op

    Outputs:
    - forward (Callable): The forward call function for the model.
    """

    model = get_quant_exported_model(model, data, logging, int_or_dequant_op)

    check_model_type(model, expected_type=torch.fx.graph_module.GraphModule)
    forward = model.forward

    return forward
