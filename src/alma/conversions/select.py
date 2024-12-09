import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Tuple

import torch

from .options.compile import (
    get_compiled_forward_call_eager_fallback,
    get_compiled_model_forward_call,
)
from .options.export_aotinductor import (
    get_export_aot_inductor_forward_call,
    get_quant_export_aot_inductor_forward_call,
)
from .options.export_compile import (
    get_export_compiled_forward_call,
    get_export_compiled_forward_call_eager_fallback,
)
from .options.export_eager import get_export_eager_forward_call
from .options.export_quant import get_quant_exported_forward_call
from .options.fake_quant import get_fake_quantized_model_forward_call
from .options.onnx import get_onnx_dynamo_forward_call, get_onnx_forward_call
from .options.quant_convert import get_converted_quantized_model_forward_call
from .options.tensorrt import get_tensorrt_dynamo_forward_call
from .options.utils.checks.imports import (
    check_onnxrt,
    check_openxla,
    check_tensort,
    check_tvm,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

MODEL_CONVERSION_OPTIONS = {
    0: "EXPORT+COMPILE_INDUCTOR_DEFAULT",
    1: "EXPORT+COMPILE_INDUCTOR_REDUCE_OVERHEAD",
    2: "EXPORT+COMPILE_INDUCTOR_MAX_AUTOTUNE",
    3: "EXPORT+COMPILE_CUDAGRAPH",
    4: "EXPORT+COMPILE_ONNXRT",
    5: "EXPORT+COMPILE_OPENXLA",
    6: "EXPORT+COMPILE_TVM",
    7: "EXPORT+COMPILE_INDUCTOR_DEFAULT_EAGER_FALLBACK",
    8: "EXPORT+AOT_INDUCTOR",
    9: "EXPORT+EAGER",
    10: "EXPORT+AI8WI8_STATIC_QUANTIZED",
    11: "EXPORT+AI8WI8_FLOAT_QUANTIZED",
    12: "EXPORT+AI8WI8_STATIC_QUANTIZED+AOT_INDUCTOR",
    13: "EXPORT+AI8WI8_FLOAT_QUANTIZED+AOT_INDUCTOR",
    14: "EXPORT+AI8WI8_STATIC_QUANTIZED+RUN_DECOMPOSITION",
    15: "EXPORT+AI8WI8_FLOAT_QUANTIZED+RUN_DECOMPOSITION",
    16: "EXPORT+AI8WI8_STATIC_QUANTIZED+RUN_DECOMPOSITION+AOT_INDUCTOR",
    17: "EXPORT+AI8WI8_FLOAT_QUANTIZED+RUN_DECOMPOSITION+AOT_INDUCTOR",
    18: "COMPILE_INDUCTOR_DEFAULT",
    19: "COMPILE_INDUCTOR_REDUCE_OVERHEAD",
    20: "COMPILE_INDUCTOR_MAX_AUTOTUNE",
    21: "COMPILE_CUDAGRAPH",
    22: "COMPILE_ONNXRT",
    23: "COMPILE_OPENXLA",
    24: "COMPILE_TVM",
    25: "COMPILE_INDUCTOR_DEFAULT_EAGER_FALLBACK",
    26: "EAGER",
    27: "TENSORRT",
    28: "ONNX_CPU",
    29: "ONNX_GPU",
    30: "ONNX+DYNAMO_EXPORT",
    31: "NATIVE_CONVERT_AI8WI8_STATIC_QUANTIZED",
    32: "NATIVE_FAKE_QUANTIZED_AI8WI8_STATIC",
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
    - device (torch.device): The device to run the benchmarking on.

    Outputs:
    - forward (Callable): The forward call function for the model.
    """
    device = data.device
    match conversion:
        ###############
        # WITH EXPORT #
        ###############
        case "EXPORT+COMPILE_INDUCTOR_DEFAULT":
            forward = get_export_compiled_forward_call(
                model, data, backend="inductor-default"
            )

        case "EXPORT+COMPILE_INDUCTOR_REDUCE_OVERHEAD":
            forward = get_export_compiled_forward_call(
                model, data, backend="inductor-reduce-overhead"
            )

        case "EXPORT+COMPILE_INDUCTOR_MAX_AUTOTUNE":
            forward = get_export_compiled_forward_call(
                model, data, backend="inductor-max-autotune"
            )

        case "EXPORT+COMPILE_CUDAGRAPH":
            forward = get_export_compiled_forward_call(
                model, data, backend="cudagraphs"
            )

        case "EXPORT+COMPILE_ONNXRT":
            # Very much an evolving API, not guaranteed to work.
            check_onnxrt()
            forward = get_export_compiled_forward_call(model, data, backend="onnxrt")

        case "EXPORT+COMPILE_OPENXLA":
            # Check if torch-xla is installed
            check_openxla()
            forward = get_export_compiled_forward_call(model, data, backend="openxla")

        case "EXPORT+COMPILE_TVM":
            check_tvm()
            forward = get_export_compiled_forward_call(model, data, backend="tvm")

        case "EXPORT+COMPILE_INDUCTOR_DEFAULT_EAGER_FALLBACK":
            forward = get_export_compiled_forward_call_eager_fallback(
                model,
                data,
                backend="inductor-default",
            )

        case "EXPORT+AOT_INDUCTOR":
            forward = get_export_aot_inductor_forward_call(model, data, device)

        case "EXPORT+EAGER":
            forward = get_export_eager_forward_call(model, data)

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
                model,
                data,
                device,
                int_or_dequant_op="dequant",
                run_decompositions=False,
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
                model,
                data,
                device,
                int_or_dequant_op="dequant",
                run_decompositions=True,
            )

        ##################
        # WITHOUT EXPORT #
        ##################
        case "COMPILE_INDUCTOR_DEFAULT":
            forward = get_compiled_model_forward_call(
                model, data, backend="inductor-default"
            )

        case "COMPILE_INDUCTOR_REDUCE_OVERHEAD":
            forward = get_compiled_model_forward_call(
                model, data, backend="inductor-reduce-overhead"
            )

        case "COMPILE_INDUCTOR_MAX_AUTOTUNE":
            forward = get_compiled_model_forward_call(
                model, data, backend="inductor-max-autotune"
            )

        case "COMPILE_CUDAGRAPH":
            forward = get_compiled_model_forward_call(model, data, backend="cudagraphs")

        case "COMPILE_ONNXRT":
            # Very much an evolving API, not guaranteed to work.
            check_onnxrt()
            forward = get_compiled_model_forward_call(model, data, backend="onnxrt")

        case "COMPILE_OPENXLA":
            check_openxla()
            forward = get_compiled_model_forward_call(model, data, backend="openxla")

        case "COMPILE_TVM":
            check_tvm()
            forward = get_compiled_model_forward_call(model, data, backend="tvm")

        case "COMPILE_INDUCTOR_DEFAULT_EAGER_FALLBACK":
            forward = get_compiled_forward_call_eager_fallback(
                model, data, backend="inductor-default"
            )

        case "EAGER":
            # Regular eager model forward call
            forward = model.forward

        case "TENSORRT":
            check_tensort()
            forward = get_tensorrt_dynamo_forward_call(model, data)

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
                logger.warning(
                    "PyTorch native quantized model conversion is only supported for CPUs currently"
                )
            forward = get_converted_quantized_model_forward_call(model, data)

        case "NATIVE_FAKE_QUANTIZED_AI8WI8_STATIC":
            forward = get_fake_quantized_model_forward_call(model, data)

        case _:
            error_msg = f"The option {conversion} is not supported"
            assert conversion in MODEL_CONVERSION_OPTIONS.values(), error_msg
            raise NotImplementedError(error_msg)

    return forward
