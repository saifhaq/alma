import logging
from typing import Callable

import torch
import torch.quantization as tq

from .qconfigs import fake_quant_act, fixed_0255, learnable_act, learnable_weights


def eager_quantize(
    model: torch.nn.Module, fuse_modules: Callable, logger: logging.Logger
) -> torch.nn.Module:
    """
    Quantizes the model with eager mode. Does not do PTQ or QAT, and
    just provides default quantization parameters.

    Returns a eager quantized model.

    By default, we use the following quantization parameters:
    - Activation: Learnable fake quantization with a starting range of 2.
    - Weight: Learnable fake quantization with per-channel quantization.
    - Input: Fixed fake quantization with a range of 0-255.

    Inputs:
    - model (torch.nn.Module): The model to quantize.
    - logger (logging.Logger): The logger to use for logging.

    Outputs:
    - quant_model (torch.nn.Module): The quantized model.
    """

    logger.info("Quantizing model with eager mode")
    warning_message = "This is a default eager quantization method. Please customize the qconfigs for your model."
    logger.warning(warning_message)

    default_act_scale = 2
    default_weight_scale = 2

    # Fuse the layers
    fused_model = fuse_modules(model, logger)

    # We loop through the modules so that we can access the `out_channels` attribute
    for name, module in fused_model.named_modules():
        # Convolutional layers
        if hasattr(module, "out_channels"):
            qconfig = tq.QConfig(
                activation=learnable_act(range=default_act_scale),
                weight=learnable_weights(
                    range=default_weight_scale, channels=module.out_channels
                ),
            )
        # Linear layers
        elif hasattr(module, "out_features"):
            qconfig = tq.QConfig(
                activation=learnable_act(range=default_act_scale),
                weight=learnable_weights(
                    range=default_weight_scale, channels=module.out_features
                ),
            )
        else:
            qconfig = tq.QConfig(
                activation=learnable_act(range=default_act_scale),
                weight=tq.default_fused_per_channel_wt_fake_quant,
            )
        module.qconfig = qconfig

    qconfig = tq.QConfig(
        activation=fixed_0255, weight=tq.default_fused_per_channel_wt_fake_quant
    )
    fused_model.quant_input.qconfig = qconfig

    # Do eager mode quantization
    fused_model.train()
    quant_model = torch.ao.quantization.prepare_qat(fused_model, inplace=False)

    return quant_model
