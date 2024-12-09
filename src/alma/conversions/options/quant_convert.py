import logging
from typing import Callable, Tuple

import torch
import torch.fx as fx
from torch.ao.quantization.quantize_fx import convert_fx

from .fake_quant import fx_quantize

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_converted_quantized_model_forward_call(
    model: torch.nn.Module,
    data: torch.Tensor,
) -> Tuple[Callable, torch.device]:
    """
    Fake quantizes the model using FX Graph mode, and then converts it to "true" int8.
    It then returns the forward call.

    Inputs:
    - model (torch.nn.Module): The model to quantize.
    - data (torch.Tensor): Sample data to feed through the model for tracing and PTQ.

    Returns:
    - forward (Callable): the forward call of the converted FX Graph quantized model.
    """

    # FX graph mode fake-quantization
    fx_model: fx.GraphModule = fx_quantize(model, data)

    # # Send the model to CPU: PyTorch native conversion is currently CPU-only
    # fx_model = fx_model.to("cpu")
    # device = torch.device("cpu")

    # Convert the model
    model_quantized = convert_fx(fx_model)

    # Return forward call
    return model_quantized.forward, device
