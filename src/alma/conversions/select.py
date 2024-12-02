import argparse
import logging
from pathlib import Path
from typing import Any, Callable

import torch

from .options.compile import get_compiled_model_forward_call
from .options.export_aotinductor import (
    get_export_aot_inductor_forward_call,
    get_quant_export_aot_inductor_forward_call,
)
from .options.export_compile import get_export_compiled_forward_call
from .options.export_eager import get_export_eager_forward_call
from .options.export_quant import (
    get_quant_exported_forward_call,
    get_quant_exported_model,
)
from .options.onnx import get_onnx_dynamo_forward_call, get_onnx_forward_call

# from .options.tensorrt import get_tensorrt_dynamo_forward_call # commented out because it messes up imports if not on CUDA

MODEL_CONVERSION_OPTIONS = {
    0: "EXPORT+COMPILE",
    1: "EXPORT+AOT_INDUCTOR",
    2: "EXPORT+EAGER",
    3: "EXPORT+TENSORRT",
    4: "ONNX+DYNAMO_EXPORT",
    5: "EXPORT+INT_QUANTIZED",
    6: "EXPORT+FLOAT_QUANTIZED",
    7: "EXPORT+INT-QUANTIZED+AOT_INDUCTOR",
    8: "EXPORT+FLOAT-QUANTIZED+AOT_INDUCTOR",
    9: "COMPILE",
    10: "EAGER",
    11: "TENSORRT",
    12: "ONNX_CPU",
    13: "ONNX_GPU",
    14: "CONVERT_QUANTIZED",
    15: "FAKE_QUANTIZED",
}


def select_forward_call_function(
    model: Any,
    conversion: str,
    data: torch.Tensor,
    logging: logging.Logger,
) -> Callable:
    """
    Get the forward call function for the model. The complexity is because there are multiple
    ways to export the model, and the forward call is different for each.

    Inputs:
    - model (Any): The model to get the forward call for.
    - conversion (str): The conversion method to use for the model.
    - data (torch.Tensor): A sample of data to pass through the model, which may be needed for
    some of the export methods.
    - logging (logging.Logger): The logger to use for logging.

    Outputs:
    - forward (Callable): The forward call function for the model.
    """

    match conversion:
        ###############
        # WITH EXPORT #
        ###############
        case "EXPORT+COMPILE":
            # This is torch compile, fed into torch export
            forward = get_export_compiled_forward_call(model, data, logging)

        case "EXPORT+AOT_INDUCTOR":
            forward = get_export_aot_inductor_forward_call(model, data, logging)

        case "EXPORT+EAGER":
            forward = get_export_eager_forward_call(model, data, logging)

        case "EXPORT+TENSORRT":
            # forward = get_tensorrt_dynamo_forward_call(model, data, logging)
            raise NotImplementedError(
                "Installing torch_tensorrt is taking forever, have to do"
            )

        case "ONNX+DYNAMO_EXPORT":
            forward = get_onnx_dynamo_forward_call(model, data, logging)

        case "EXPORT+INT_QUANTIZED":
            forward = get_quant_exported_forward_call(
                model, data, logging, int_or_dequant_op="int"
            )

        case "EXPORT+FLOAT_QUANTIZED":
            forward = get_quant_exported_forward_call(
                model, data, logging, int_or_dequant_op="dequant"
            )

        case "EXPORT+INT_QUANTIZED+AOT_INDUCTOR":
            forward = get_quant_export_aot_inductor_forward_call(
                model, data, logging, int_or_dequant_op="int"
            )

        case "EXPORT+FLOAT_QUANTIZED+AOT_INDUCTOR":
            forward = get_quant_export_aot_inductor_forward_call(
                model, data, logging, int_or_dequant_op="dequant"
            )

        ##################
        # WITHOUT EXPORT #
        ##################
        case "COMPILE":
            # Torch.compile without export. This is basically just a regular forward call,
            # maybe with some fusing
            forward = get_compiled_model_forward_call(model, data, logging)

        case "EAGER":
            # Regular eager model forward call
            forward = model.forward

        case "TENSORRT":
            # forward = get_tensorrt_dynamo_forward_call(model, data)
            raise NotImplementedError("Installing tensor RT is having some issues, fix")

        case "ONNX_CPU":
            onnx_model_path = Path("model/model.onnx")
            onnx_backend = "CPUExecutionProvider"
            forward = get_onnx_forward_call(
                model, data, logging, onnx_model_path, onnx_backend
            )

        case "ONNX_GPU":
            onnx_model_path = Path("model/model.onnx")
            onnx_backend = "CUDAExecutionProvider"
            forward = get_onnx_forward_call(
                model, data, logging, onnx_model_path, onnx_backend
            )

        case "CONVERT_QUANTIZED":
            pass

        case "FAKE_QUANTIZED":
            pass

        case _:
            assert (
                conversion in MODEL_CONVERSION_OPTIONS.values()
            ), f"The option {conversion} is not supported"
            raise NotImplementedError("Option not currently supported")

    return forward
