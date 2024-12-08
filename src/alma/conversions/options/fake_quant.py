import logging
from typing import Callable, List

import torch
import torch.fx as fx
import torch.quantization as tq
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx

from .utils.checks.type import check_model_type
from .utils.qconfigs import fake_quant_act, fixed_0255, learnable_act, learnable_weights

# One needs to set their quantization backend engine to what is appropriate for their system.
# torch.backends.quantized.engine = 'x86'
torch.backends.quantized.engine = "qnnpack"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_fake_quantized_model_forward_call(
    model: torch.nn.Module,
    data: torch.Tensor,
) -> Callable:
    """
    Quantize the model using FX Graph mode, and return the forward call.

    Inputs:
    - model (torch.nn.Module): The model to quantize.
    - data (torch.Tensor): Sample data to feed through the model for tracing and PTQ.

    Returns:
    - forward (Callable): the forward call of the FX Graph quantized model.
    """

    # FX graph mode fake-quantization
    fx_model: fx.GraphModule = fx_quantize(model, data)

    # Return forward call
    return fx_model.forward


def fx_quantize(
    model: torch.nn.Module,
    data: torch.Tensor,
) -> Callable:
    """
    Wrapper around the FX Graph mode quantization of the model, which will deal with error
    handling.

    Returns a fx-graph traced quantized model.

    Inputs:
    - model (torch.nn.Module): The model to quantize.
    - data (torch.Tensor): A sample of input data to use for tracing.

    Outputs:
    - fx_model (torch.fx.GraphModule): The quantized model in FX graph mode.
    """
    # FX graph mode fake-quantization
    try:
        fx_model: fx.GraphModule = _fx_quantize(model, data)
    except Exception as e:
        error_msg = f"""Fake quantization failed. Check that your model is traceable, and make sure 
that your quantization backend engine is set to match your hardware. E.g. 
torch.backends.quantized.engine = "qnnpack" | "x86". Error: {e}"""
        logger.error(error_msg)

    # Check model type
    check_model_type(fx_model, fx.GraphModule)

    # Make sure to send the model to the desired device
    fx_model = fx_model.to(data.device)
    return fx_model


def _fx_quantize(
    model: torch.nn.Module,
    data: torch.Tensor,
) -> torch.fx.GraphModule:
    """
    Quantizes the model with FX graph tracing. Does not do PTQ or QAT, and
    just provides default quantization parameters that are entirely not optimized.

    Returns a fx-graph traced quantized model.

    Inputs:
    - model (torch.nn.Module): The model to quantize.
    - data (torch.Tensor): A sample of input data to use for tracing.

    Outputs:
    - fx_model (torch.fx.GraphModule): The quantized model in FX graph mode.

    """

    logger.info("Quantizing model with FX graph mode")

    # These values define the quantization scales of the activations and weights. They are set
    # to arbitrary values for the sake of this benchmarking.
    default_act_scale = 2.0
    default_weight_scale = 2.0

    # Define global (default) qconfig
    qconfig_global = tq.QConfig(
        activation=learnable_act(range=default_act_scale),
        weight=tq.default_fused_per_channel_wt_fake_quant,
    )

    # Assign global qconfig
    qconfig_mapping = QConfigMapping().set_global(qconfig_global)

    # We loop through the modules so that we can access the `out_channels`/`out_features` attribute
    for name, module in model.named_modules():
        # Convolutional layers
        if hasattr(module, "out_channels"):
            qconfig = tq.QConfig(
                activation=learnable_act(range=default_act_scale),
                weight=learnable_weights(
                    range=default_weight_scale, channels=module.out_channels
                ),
            )
            qconfig_mapping.set_module_name(name, qconfig)
        # Linear layers
        elif hasattr(module, "out_features"):
            qconfig = tq.QConfig(
                activation=learnable_act(range=default_act_scale),
                weight=learnable_weights(
                    range=default_weight_scale, channels=module.out_features
                ),
            )
            qconfig_mapping.set_module_name(name, qconfig)

    # Do symbolic tracing and quantization
    model.eval()
    fx_model = prepare_qat_fx(model, qconfig_mapping, (data,))

    # Prints the graph as a table
    if logger.root.level <= logging.DEBUG:
        logger.debug("\nFake quantized model graph:\n")
        logger.debug(fx_model.graph.print_tabular())
    return fx_model
