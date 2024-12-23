"""
Focused on https://github.com/pytorch/ao for on-GPU quantization in PyTorch.
"""

import logging
from typing import Callable

import torch

from .compile import get_compiled_model

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_torchao_quantize_model(
    model: torch.nn.Module,
    quantization_mode: Callable,
):
    """
    Quantize the model using torchao quantization. The model has to be cast to bf16
    first.

    Inputs:
    - model (torch.nn.Module): The model to quantize.
    - quantization_mode (Callable): The quantization mode to use.

    Outputs:
    - model (torch.nn.Module): The quantized model.
    """
    from torchao.quantization.quant_api import int4_weight_only, quantize_

    if quantization_mode == None:
        quantization_mode = int4_weight_only

    logger.info("Casting model to bf16")
    model = model.to(torch.bfloat16)
    logger.info("Running torchao quantize_ on the model")
    quantize_(model, quantization_mode())

    return model


def get_torchao_quantize_forward_call(
    model: torch.nn.Module,
    quantization_mode: Callable,
) -> Callable:
    """
    Get the forward call for the quantized model.

    Inputs:
    - model (torch.nn.Module): The model to quantize.
    - quantization_mode (Callable): The quantization mode to use.

    Outputs:
    - forward (Callable): The forward call for the quantized model.
    """
    from torchao.quantization.quant_api import int4_weight_only

    if quantization_mode == None:
        quantization_mode = int4_weight_only

    model = get_torchao_quantize_model(model, quantization_mode)

    def forward(data: torch.Tensor):
        """
        The forward call for the quantized model.

        Inputs:
        - data (torch.Tensor): data to feed through the model.

        Outputs:
        - output (torch.Tensor).
        """
        data = data.to(torch.bfloat16)
        output = model(data)
        return output

    return forward


def get_compiled_torchao_quantize_forward_call(
    model: torch.nn.Module,
    data: torch.Tensor,
    backend: str,
    quantization_mode: Callable,
) -> Callable:
    """
    Get the forward call for the quantized model. We do after compiling the model.

    Inputs:
    - model (torch.nn.Module): The model to quantize.
    - data (torch.Tensor): The input data to pass through the model.
    - backend (str): the backend for torch.compile to use.
    - quantization_mode (Callable): The quantization mode to use.

    Outputs:
    - forward (Callable): The forward call for the quantized model.
    """
    from torchao.quantization.quant_api import int4_weight_only

    if quantization_mode == None:
        quantization_mode = int4_weight_only

    model = get_compiled_model(model, data, backend=backend)
    forward = get_torchao_quantize_forward_call(model, quantization_mode)

    return forward
