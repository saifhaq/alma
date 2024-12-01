import logging

import torch
import torch.quantization as tq
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx

from .qconfigs import fake_quant_act, fixed_0255, learnable_act, learnable_weights


def fx_quantize(
    model: torch.nn.Module,
    data: torch.Tensor,
    act_qconfig: tq.QConfig,
    weight_qconfig: tq.QConfig,
    act_scale: int,
    logger: logging.Logger,
) -> torch.fx.GraphModule:
    """
    Quantizes the model with FX graph tracing. Does not do PTQ or QAT, and
    just provides default quantization parameters.

    Returns a fx-graph traced quantized model.

    Inputs:
    - model (torch.nn.Module): The model to quantize.
    - data (torch.Tensor): A sample of input data to use for tracing.
    - act_qconfig (tq.QConfig): The qconfig to use for activation quantization.
    - weight_qconfig (tq.QConfig): The qconfig to use for weight quantization.
    - act_scale (int): The initial scale of the activation quantization.
    - logger (logging.Logger): The logger to use for logging.

    Outputss:
    - fx_model (torch.fx.GraphModule): The quantized model in FX graph mode.
    """
    logger.info("Quantizing model with FX graph mode")

    # Define global (default) qconfig
    qconfig_global = tq.QConfig(
        activation=learnable_act(range=act_scale),
        weight=tq.default_fused_per_channel_wt_fake_quant,
    )

    # Assign global qconfig
    qconfig_mapping = QConfigMapping().set_global(qconfig_global)

    # We loop through the modules so that we can access the `out_channels`/`out_features` attribute
    for name, module in model.named_modules():
        # Convolutional layers
        if hasattr(module, "out_channels"):
            qconfig = tq.QConfig(
                activation=learnable_act(range=act_scale),
                weight=learnable_weights(channels=module.out_channels),
            )
            qconfig_mapping.set_module_name(name, qconfig)
        # Linear layers
        elif hasattr(module, "out_features"):
            qconfig = tq.QConfig(
                activation=learnable_act(range=act_scale),
                weight=learnable_weights(channels=module.out_features),
            )
            qconfig_mapping.set_module_name(name, qconfig)

    # Do symbolic tracing and quantization
    model.eval()
    fx_model = prepare_qat_fx(model, qconfig_mapping, (data,))

    # Prints the graph as a table
    logger.info("\nGraph as a Table:\n")
    fx_model.graph.print_tabular()
    return fx_model
