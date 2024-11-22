import argparse
import logging
from pathlib import Path
from typing import Any, Callable

import torch

from .compile import get_compiled_model_forward_call
from .export.aotinductor import (
    get_export_aot_inductor_forward_call,
    get_quant_export_aot_inductor_forward_call,
)
from .export.compile import get_export_compiled_forward_call
from .export.eager import get_export_eager_forward_call
from .export.quant import get_quant_exported_forward_call, get_quant_exported_model
from .onnx import get_onnx_forward_call

# from .tensorrt import get_tensorrt_dynamo_forward_call # commented out because it messes up imports if not on CUDA

MODEL_CONVERSION_OPTIONS = {
    0: "EXPORT+COMPILE",
    1: "EXPORT+AOT_INDUCTOR",
    2: "EXPORT+EAGER",
    3: "EXPORT+TENSORRT",
    4: "EXPORT+ONNX",
    5: "EXPORT+QUANTIZED",
    6: "EXPORT+INT-QUANTIZED+AOT_INDUCTOR",
    7: "EXPORT+FLOAT-QUANTIZED+AOT_INDUCTOR",
    8: "COMPILE",
    9: "EAGER",
    10: "TENSORRT",
    11: "ONNX-CPU",
    12: "CONVERT_QUANTIZED",
    13: "FAKE_QUANTIZED",
}


def select_forward_call_function(
    model: Any,
    args: argparse.Namespace,
    data: torch.Tensor,
    logging: logging.Logger,
) -> Callable:
    """
    Get the forward call function for the model. The complexity is because there are multiple
    ways to export the model, and the forward call is different for each.

    Inputs:
    - model (Any): The model to get the forward call for.
    - args (argparse.Namespace): The command line arguments.
    - data (torch.Tensor): A sample of data to pass through the model, which may be needed for
    some of the export methods.
    - logging (logging.Logger): The logger to use for logging.

    Outputs:
    - forward (Callable): The forward call function for the model.
    """

    conversion = MODEL_CONVERSION_OPTIONS[args.conversion]
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

        case "EXPORT+ONNX":
            raise NotImplementedError("Not Implemented")
            # forward = get_export_onnx_forward_call(model, data, logging)

        case "EXPORT+QUANTIZED":
            raise NotImplementedError("Not Implemented")
            # forward = get_quant_exported_forward_call(model, data, logging)

        case "EXPORT+INT-QUANTIZED+AOT_INDUCTOR":
            forward = get_quant_export_aot_inductor_forward_call(
                model, data, logging, int_or_dequant_op="int"
            )

        case "EXPORT+FLOAT-QUANTIZED+AOT_INDUCTOR":
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

        case "ONNX-CPU":
            onnx_model_path = Path("model/model.onnx")
            forward = get_onnx_forward_call(model, data, logging, onnx_model_path)

        case "CONVERT_QUANTIZED":
            pass

        case "FAKE_QUANTIZED":
            pass

        case _:
            assert conversion in MODEL_CONVERSION_OPTIONS.values()

    return forward
