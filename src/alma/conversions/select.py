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
from .options.tensorrt import get_tensorrt_dynamo_forward_call


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
    9: "EXPORT+AI8WI8_STATIC_QUANTIZED",
    10: "EXPORT+AI8WI8_FLOAT_QUANTIZED",
    11: "EXPORT+AI8WI8_STATIC_QUANTIZED+AOT_INDUCTOR",
    12: "EXPORT+AI8WI8_FLOAT_QUANTIZED+AOT_INDUCTOR",
    13: "EXPORT+AI8WI8_STATIC_QUANTIZED+RUN_DECOMPOSITION",
    14: "EXPORT+AI8WI8_FLOAT_QUANTIZED+RUN_DECOMPOSITION",
    15: "EXPORT+AI8WI8_STATIC_QUANTIZED+RUN_DECOMPOSITION+AOT_INDUCTOR",
    16: "EXPORT+AI8WI8_FLOAT_QUANTIZED+RUN_DECOMPOSITION+AOT_INDUCTOR",
    17: "COMPILE",
    18: "EAGER",
    19: "TENSORRT",
    20: "ONNX_CPU",
    21: "ONNX_GPU",
    22: "NATIVE_CONVERT_AI8WI8_STATIC_QUANTIZED",
    23: "NATIVE_FAKE_QUANTIZED_AI8WI8_STATIC",
}

def select_forward_call_function(
    model: Any,
    conversion: str,
    data: torch.Tensor,
    device: torch.device,
) -> Callable:
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
            # Check if 'openxla' backend is available
            if "openxla" not in torch._dynamo.list_backends():
                raise RuntimeError(
                    "OpenXLA backend is not available. Please ensure OpenXLA is installed and properly configured."
                )

            # Check if torch-xla is installed
            try:
                import torch_xla
            except ImportError:
                raise RuntimeError(
                    "The torch-xla package is not available. Please install torch-xla to use 'openxla' backend.\n"
                    "For installation instructions: https://github.com/pytorch/xla"
                )

            forward = get_export_compiled_forward_call(model, data, "openxla")

        case "EXPORT+COMPILE_TVM":
            # See here for some discussion of TVM backend: 
            # https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747/9

            # Check if 'tvm' backend is available
            if "tvm" not in torch._dynamo.list_backends():
                raise RuntimeError(
                    "TVM backend is not available. Please ensure TVM is installed and properly configured."
                )
            try:
                import torch_tvm
                torch_tvm.enable()
            except ImportError:
                raise RuntimeError(
                    "The torch-tvm package is not available. Please install torch-tvm to use 'tvm' backend.\n"
                )
            forward = get_export_compiled_forward_call(model, data, "tvm")

        case "EXPORT+AOT_INDUCTOR":
            forward = get_export_aot_inductor_forward_call(model, data, device)

        case "EXPORT+EAGER":
            forward = get_export_eager_forward_call(model, data)

        case "EXPORT+TENSORRT":
            try:
                import torch_tensorrt
            except ImportError:
                raise RuntimeError(
                    "Torch TensorRT backend is not available. Please ensure it is installed and properly configured."
                )     
            forward = get_tensorrt_dynamo_forward_call(model, data)
            raise NotImplementedError(
                "Installing torch_tensorrt is taking forever, need to install."
            )

        case "ONNX+DYNAMO_EXPORT":
            forward = get_onnx_dynamo_forward_call(model, data)

        case "EXPORT+AI8WI8_STATIC_QUANTIZED":
            forward = get_quant_exported_forward_call(
                model, data, int_or_dequant_op="int", run_decompositions=False
            )

        case "EXPORT+AI8WI8_FLOAT_QUANTIZED":
            forward = get_quant_exported_forward_call(
                model, data, int_or_dequant_op="dequant", run_decompositions=False
            )

        case "EXPORT+AI8WI8_STATIC_QUANTIZED+AOT_INDUCTOR":
            forward = get_quant_export_aot_inductor_forward_call(
                model, data, device, int_or_dequant_op="int", run_decompositions=False
            )

        case "EXPORT+AI8WI8_FLOAT_QUANTIZED+AOT_INDUCTOR":
            forward = get_quant_export_aot_inductor_forward_call(
                model, data, device, int_or_dequant_op="dequant", run_decompositions=False
            )

        case "EXPORT+AI8WI8_STATIC_QUANTIZED+RUN_DECOMPOSITION":
            # The difference with training (i.e. inference=False) is that at the end we re-run 
            # torch.export and then run `run_decompositions`, with the hope it might shorted the graph
            # a bit.
            forward = get_quant_exported_forward_call(
                model, data, int_or_dequant_op="int", run_decompositions=True
            )

        case "EXPORT+AI8WI8_FLOAT_QUANTIZED+RUN_DECOMPOSITION":
            forward = get_quant_exported_forward_call(
                model, data, int_or_dequant_op="dequant", run_decompositions=True
            )

        case "EXPORT+AI8WI8_STATIC_QUANTIZED+RUN_DECOMPOSITION+AOT_INDUCTOR":
            forward = get_quant_export_aot_inductor_forward_call(
                model, data, device, int_or_dequant_op="int", run_decompositions=True
            )

        case "EXPORT+AI8WI8_FLOAT_QUANTIZED+RUN_DECOMPOSITION+AOT_INDUCTOR":
            forward = get_quant_export_aot_inductor_forward_call(
                model, data, device, int_or_dequant_op="dequant", run_decompositions=True
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

        case "NATIVE_CONVERT_AI8WI8_STATIC_QUANTIZED":
            if device.type != "cpu":
                logger.warning("PyTorch native quantized model conversion is only supported for CPUs currently")
            forward = get_converted_quantized_model_forward_call(model, data)

        case "NATIVE_FAKE_QUANTIZED_AI8WI8_STATIC":
            forward = get_fake_quantized_model_forward_call(model, data)

        case _:
            error_msg = f"The option {conversion} is not supported"
            assert (
                conversion in MODEL_CONVERSION_OPTIONS.values()
            ), error_msg
            raise NotImplementedError(error_msg)

    return forward
