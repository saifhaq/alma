import logging

import torch
import torch.quantization as tq
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx

from .qconfigs import fake_quant_act, fixed_0255, learnable_act, learnable_weights


def fx_quantize(
    model: torch.nn.Module,
    data: torch.Tensor,
    logger: logging.Logger,
) -> torch.fx.GraphModule:
    """
    Quantizes the model with FX graph tracing. Does not do PTQ or QAT, and
    just provides default quantization parameters.

    Returns a fx-graph traced quantized model.

    Inputs:
    - model (torch.nn.Module): The model to quantize.
    - data (torch.Tensor): A sample of input data to use for tracing.
    - logger (logging.Logger): The logger to use for logging.

    Outputss:
    - fx_model (torch.fx.GraphModule): The quantized model in FX graph mode.
    """
    logger.info("Quantizing model with FX graph mode")
    warning_message = "This is a default FX Graph mode quantization method. Please customize the qconfigs for your model."
    logger.warning(warning_message)

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
        logger.debug("\nGraph as a Table:\n")
        logger.debug(fx_model.graph.print_tabular())
    
    return fx_model
