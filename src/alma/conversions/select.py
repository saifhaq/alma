import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Tuple

import torch

from .options.compile import get_compiled_model_forward_call
from .options.export_aotinductor import (
    get_export_aot_inductor_forward_call,
    get_quant_export_aot_inductor_forward_call,
)
from .options.export_compile import get_export_compiled_forward_call
from .options.export_eager import get_export_eager_forward_call
from .options.export_quant import get_quant_exported_forward_call
from .options.fake_quant import get_fake_quantized_model_forward_call
from .options.onnx import get_onnx_dynamo_forward_call, get_onnx_forward_call
from .options.quant_convert import get_converted_quantized_model_forward_call

# from .options.tensorrt import get_tensorrt_dynamo_forward_call # commented out because it messes up imports if not on CUDA


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

MODEL_CONVERSION_OPTIONS = {
    0: "EXPORT+COMPILE_INDUCTOR",
    1: "EXPORT+COMPILE_CUDAGRAPH",
    2: "EXPORT+COMPILE_ONNXRT",
    3: "EXPORT+COMPILE_OPENXLA",
    4: "EXPORT+COMPILE_TVM",
    5: "EXPORT+AOT_INDUCTOR",
    6: "EXPORT+EAGER",
    7: "EXPORT+TENSORRT",
    8: "ONNX+DYNAMO_EXPORT",
    9: "EXPORT+INT_QUANTIZED",
    10: "EXPORT+FLOAT_QUANTIZED",
    11: "EXPORT+INT-QUANTIZED+AOT_INDUCTOR",
    12: "EXPORT+FLOAT-QUANTIZED+AOT_INDUCTOR",
    13: "COMPILE",
    14: "EAGER",
    15: "TENSORRT",
    16: "ONNX_CPU",
    17: "ONNX_GPU",
    18: "CONVERT_QUANTIZED",
    19: "FAKE_QUANTIZED",
}


def select_forward_call_function(
    model: Any,
    conversion: str,
    data: torch.Tensor,
) -> Tuple[Callable, torch.device]:
    """
    Get the forward call function for the model. The complexity is because there are multiple
    ways to export the model, and the forward call is different for each.

    Inputs:
    - model (Any): The model to get the forward call for.
    - conversion (str): The conversion method to use for the model.
    - data (torch.Tensor): A sample of data to pass through the model, which may be needed for
    some of the export methods.

    Outputs:
    - forward (Callable): The forward call function for the model.
    - device (torch.device): The device of the model
    """
    device = data.device
    match conversion:
        ###############
        # WITH EXPORT #
        ###############
        case "EXPORT+COMPILE_INDUCTOR":
            forward = get_export_compiled_forward_call(model, data, "inductor")

        case "EXPORT+COMPILE_CUDAGRAPH":
            forward = get_export_compiled_forward_call(model, data, "cudagraphs")

        case "EXPORT+COMPILE_ONNXRT":
            if not torch.onnx.is_onnxrt_backend_supported():
                # Make sure all dependencies are installed, see here for a discussion by the ONNX team:
                # https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/backends/onnxrt.py
                raise RuntimeError(
                    "Need to install all dependencies. See here for more details: https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/backends/onnxrt.py"
                )
            forward = get_export_compiled_forward_call(model, data, "onnxrt")

        case "EXPORT+COMPILE_OPENXLA":
            forward = get_export_compiled_forward_call(model, data, "openxla")

        case "EXPORT+COMPILE_TVM":
            forward = get_export_compiled_forward_call(model, data, "tvm")

        case "EXPORT+AOT_INDUCTOR":
            forward = get_export_aot_inductor_forward_call(model, data)

        case "EXPORT+EAGER":
            forward = get_export_eager_forward_call(model, data)

        case "EXPORT+TENSORRT":
            # forward = get_tensorrt_dynamo_forward_call(model, data)
            raise NotImplementedError(
                "Installing torch_tensorrt is taking forever, need to install."
            )

        case "ONNX+DYNAMO_EXPORT":
            forward = get_onnx_dynamo_forward_call(model, data)

        case "EXPORT+INT_QUANTIZED":
            forward = get_quant_exported_forward_call(
                model, data, int_or_dequant_op="int"
            )

        case "EXPORT+FLOAT_QUANTIZED":
            forward = get_quant_exported_forward_call(
                model, data, int_or_dequant_op="dequant"
            )

        case "EXPORT+INT_QUANTIZED+AOT_INDUCTOR":
            forward = get_quant_export_aot_inductor_forward_call(
                model, data, int_or_dequant_op="int"
            )

        case "EXPORT+FLOAT_QUANTIZED+AOT_INDUCTOR":
            forward = get_quant_export_aot_inductor_forward_call(
                model, data, int_or_dequant_op="dequant"
            )

        ##################
        # WITHOUT EXPORT #
        ##################
        case "COMPILE":
            # Torch.compile without export. This is basically just a regular forward call,
            # maybe with some fusing
            forward = get_compiled_model_forward_call(model, data)

        case "EAGER":
            # Regular eager model forward call
            forward = model.forward

        case "TENSORRT":
            # forward = get_tensorrt_dynamo_forward_call(model, data)
            raise NotImplementedError("Installing tensor RT is having some issues, fix")

        case "ONNX_CPU":
            # We create temporary file to save the onnx model
            with tempfile.TemporaryDirectory() as tmpdirname:
                onnx_model_path = Path(f"{tmpdirname}/model.onnx")
                onnx_backend = "CPUExecutionProvider"
                forward = get_onnx_forward_call(
                    model, data, onnx_model_path, onnx_backend
                )

        case "ONNX_GPU":
            # We create temporary file to save the onnx model
            with tempfile.TemporaryDirectory() as tmpdirname:
                onnx_model_path = Path(f"{tmpdirname}/model.onnx")
                onnx_backend = "CUDAExecutionProvider"
                forward = get_onnx_forward_call(
                    model, data, onnx_model_path, onnx_backend
                )

        case "CONVERT_QUANTIZED":
            # Also returns device, as PyTorch-natively converted models are only currently for CPU
            forward, device = get_converted_quantized_model_forward_call(model, data)
            pass

        case "FAKE_QUANTIZED":
            forward = get_fake_quantized_model_forward_call(model, data)

        case _:
            assert (
                conversion in MODEL_CONVERSION_OPTIONS.values()
            ), f"The option {conversion} is not supported"
            raise NotImplementedError("Option not currently supported")

    return forward, device
