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

from ...utils.setup_logging import suppress_output
from .utils.checks.type import check_model_type

# Create a module-level logger
logger = logging.getLogger(__name__)
# Don't add handlers - let the application configure logging
logger.addHandler(logging.NullHandler())


def get_quant_exported_model(
    model: torch.nn.Module,
    data: torch.Tensor,
    int_or_dequant_op: Literal["int", "dequant"],
    run_decompositions: bool,
) -> torch.fx.graph_module.GraphModule:
    """
    Export the model using torch.export.export_for_training and convert_pt2e to get a quantized
    exported model.

    Inputs:
    - model (torch.nn.Module): The model to export
    - data (torch.Tensor): Sample data to feed through the model for tracing.
    - int_or_dequant_op (Literal["int", "dequant"]): do we use integer arithmetic operations on
            quantized layers, or do we dequantize just prior to the op
    - run_decompositions (bool): do we, after all of our processing, re-export the model and run
            `run_decompositions`?

    Outputs:
    model (torch.export.Model): The exported model
    """

    logger.info(
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

    # Feed some data throuhg the model, if only to intialise the observers and supress the warnings
    with suppress_output(logger.root.level >= logging.DEBUG):
        with torch.no_grad():
            for module in m_fq.modules():
                if hasattr(m_fq, "observer_enabled") or hasattr(m_fq, "static_enabled"):
                    m_fq.enable_observer()
                    m_fq.enable_fake_quant()
            _ = m_fq(data)
            for module in m_fq.modules():
                if hasattr(m_fq, "observer_enabled") or hasattr(m_fq, "static_enabled"):
                    m_fq.disable_observer()

    # Lower the quantized model
    # use_reference_optimization=True means that one uses integer arithmetic, False means that one
    # does operations in floating point and dequantizes prior.
    m_q: torch.fx.graph_module.GraphModule = convert_pt2e(
        m_fq, use_reference_representation=int_op
    )

    if run_decompositions:
        # Run decompositions, which is the same as exporting it for inference (as of torch 2.5.0)
        # See here: https://github.com/pytorch/pytorch/blob/0ecba5756166f45f547ee1f8bce5c216154cdba3/torch/export/__init__.py#L260
        # Running decompositions requires an exported model, so we re-export it.
        m_export_q: ExportedProgram = torch.export.export_for_training(m_q, (data,))

        # The below should be available in torch 2.6.0
        # decomp_table = torch.export.exported_program.default_decompositions()
        m_q = m_export_q.run_decompositions().module()
        # # m_q = m_export_q.run_decompositions(decomp_table=decomp_table).module()

    with suppress_output(logger.root.level >= logging.DEBUG):
        with torch.no_grad():
            _ = m_q(data)

    logger.debug("Quantized model graph:")
    if logger.root.level <= logging.DEBUG:
        logger.debug(m_q.graph.print_tabular())

    # we have a model with aten ops doing integer computations when possible
    check_model_type(m_q, torch.fx.graph_module.GraphModule)

    return m_q


def get_quant_exported_forward_call(
    model,
    data: torch.Tensor,
    int_or_dequant_op: Literal["int", "dequant"],
    run_decompositions: bool,
) -> Callable:
    """
    Get the torch export + quantized model forward call.

    Inputs:
    - model (torch.nn.Module): The model to export
    - data (torch.Tensor): Sample data to feed through the model for tracing.
    - int_or_dequant_op (Literal["int", "dequant"]): do we use integer arithmetic operations on
            quantized layers, or do we dequantize just prior to the op
    - run_decompositions (bool): do we, after all of our processing, re-export the model and run
            `run_decompositions`?
    Outputs:
    - forward (Callable): The forward call function for the model.
    """

    model = get_quant_exported_model(model, data, int_or_dequant_op, run_decompositions)

    check_model_type(model, expected_type=torch.fx.graph_module.GraphModule)
    forward = model.forward

    return forward
