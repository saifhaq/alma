from typing import Callable, List

import torch
from onnxruntime.quantization import QuantFormat, quantize_dynamic, quantize_static

from .onnx import _get_onnx_forward_call, save_onnx_model


def get_onnx_static_quant_forward_call(
    model: torch.nn.Module,
    data: torch.Tensor,
    onnx_model_path: str = "model/model.onnx",
    quant_onnx_model_path: str = "model/quant_model.onnx",
    quant_format: QuantFormat = QuantFormat.QOperator,
    onnx_providers: List[str] = ["CPUExecutionProvider"],
):
    """
    Get the forward call function for the model using ONNX and static quantization.

    Inputs:
    - model (Any): The model to get the forward call for.
    - data (torch.Tensor): A sample of data to pass through the model.
    - onnx_model_path (str): the path to save the ONNX model to.
    - quant_onnx_model_path (str): the path to save the quantized ONNX model to.
    - quant_format (QuantFormat): the format for wuantize doperators to be in, e.g. directly as quantized
        operators or dequantized on the fly and operations in fp.
    - onnx_provider (str): the ONNX execution provider to use.

    Outputs:
    - onnx_forward (Callable): The forward call function for the model.
    """

    assert quant_format in [QuantFormat.QOperator, QuantFormat.DQO]

    # We first save the unquantized ONNX model
    save_onnx_model(model, data, onnx_model_path)

    # Save the quantized model
    quantize_static(
        onnx_model_path,
        quant_onnx_model_path,
        per_channel=True,
        quant_format=quant_format,
    )

    # Get onnx forward call
    onnx_forward: Callable = _get_onnx_forward_call(
        quant_onnx_model_path, onnx_provider
    )

    return onnx_forward
