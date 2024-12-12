import logging
from typing import Optional, Union

import torch
from optimum.quanto import freeze, quantize
from optimum.quanto.tensor.qtype import qtype, qtypes
from torch._dynamo.eval_frame import OptimizedModule

from .compile import get_compiled_model
from .utils.checks.type import check_model_type

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_optimum_quanto_forward_call(
    model: torch.nn.Module,
    data: torch.Tensor,
    weights: Optional[Union[qtype, str]] = None,
    activations: Optional[Union[qtype, str]] = None,
) -> callable:
    """
    Fake quantizes the model using FX Graph mode, and then converts it to "true" int8.
    It then returns the forward call.

    Inputs:
    - model (torch.nn.Module): The model to quantize.
    - data (torch.Tensor): Sample data to feed through the model for tracing and PTQ.
    - weights (Union[qtype, str]): The type of quantization to give the weights of the model.
    - activations (Union[qtype, str]): The type of quantization to give the activations of the model.

    Returns:
    - forward (callable): the forward call of the HuggingFace Optimum Quanto quantized model.
    """
    model = get_optimum_quanto_model(model, data, weights, activations)
    forward = model.forward

    # Test feeding data through the model
    _ = forward(data)

    return forward


def get_optimum_quanto_model(
    model: torch.nn.Module,
    data: torch.Tensor,
    weights: Optional[Union[qtype, str]] = None,
    activations: Optional[Union[qtype, str]] = None,
) -> callable:
    """
    Fake-quantizes the model using HuggingFace's optimum quanto, and then "freezes" the weights as actual
    quantized weights.

    Inputs:
    - model (torch.nn.Module): The model to quantize.
    - data (torch.Tensor): Sample data to feed through the model for tracing and PTQ.
    - weights (Union[qtype, str]): The type of quantization to give the weights of the model.
    - activations (Union[qtype, str]): The type of quantization to give the activations of the model.

    Returns:
    - model (torch.nn.Module): the HuggingFace Optimum Quanto quantized model. The quantized model
    continues to be an "Eager" model, and not a GraphModule.
    """
    if weights:
        assert isinstance(weights, qtype)
    if activations:
        assert isinstance(activations, qtype)

    # Fake quantized
    quantize(model, weights=weights, activations=activations)

    # Weights are replaced with quantized versions
    freeze(model)

    return model


def get_optimum_quant_compiled_model(
    model: torch.nn.Module,
    data: torch.Tensor,
    weights: Optional[Union[qtype, str]] = None,
    activations: Optional[Union[qtype, str]] = None,
    backend: str = "inductor",
) -> OptimizedModule:
    """
    Fake-quantizes the model using HuggingFace's optimum quanto, and then "freezes" the weights as actual
    quantized weights. It then runs torch.compile on the model, with the provided backend.

    Inputs:
    - model (torch.nn.Module): The model to quantize.
    - data (torch.Tensor): Sample data to feed through the model for tracing and PTQ.
    - weights (Union[qtype, str]): The type of quantization to give the weights of the model.
    - activations (Union[qtype, str]): The type of quantization to give the activations of the model.
    - backend (str): the backend for torch.compile. Default is torch inductor.

    Returns:
    - model (OptimizedModule): the HuggingFace Optimum Quanto quantized, and then torch.compiled, model.
    """
    model = get_optimum_quanto_model(model, data, weights, activations)

    compiled_model = get_compiled_model(model, data, backend)

    return compiled_model


def get_optimum_quant_model_compiled_forward_call(
    model: torch.nn.Module,
    data: torch.Tensor,
    weights: Optional[Union[qtype, str]] = None,
    activations: Optional[Union[qtype, str]] = None,
    backend: str = "inductor",
) -> callable:
    """
    Fake-quantizes the model using HuggingFace's optimum quanto, and then "freezes" the weights as actual
    quantized weights. It then runs torch.compile on the model, with the provided backend.

    Inputs:
    - model (torch.nn.Module): The model to quantize.
    - data (torch.Tensor): Sample data to feed through the model for tracing and PTQ.
    - weights (Union[qtype, str]): The type of quantization to give the weights of the model.
    - activations (Union[qtype, str]): The type of quantization to give the activations of the model.
    - backend (str): the backend for torch.compile. Default is torch inductor.

    Returns:
    - forward (callable): the forward call of the HuggingFace Optimum Quanto quantized and compiled model.
    """
    model = get_optimum_quant_compiled_model(model, data, weights, activations, backend)

    check_model_type(model, OptimizedModule)
    return model.forward
