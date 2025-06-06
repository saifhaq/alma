import logging
import tempfile
from pathlib import Path
from typing import Any, Callable

import torch
from optimum.quanto import (
    qfloat8_e4m3fn,
    qfloat8_e4m3fnuz,
    qfloat8_e5m2,
    qint2,
    qint4,
    qint8,
)

from .conversion_options import MODEL_CONVERSION_OPTIONS
from .options.bf16 import get_bf16_eager_forward_call, get_bf16_model
from .options.compile import (
    get_compiled_forward_call_eager_fallback,
    get_compiled_model_forward_call,
)
from .options.export_aotinductor import get_export_aot_inductor_forward_call
from .options.export_compile import (
    get_export_compiled_forward_call,
    get_export_compiled_forward_call_eager_fallback,
)
from .options.export_eager import get_export_eager_forward_call
from .options.export_quant import get_quant_exported_forward_call
from .options.fake_quant import get_fake_quantized_model_forward_call
from .options.fp16 import get_fp16_eager_forward_call, get_fp16_model
from .options.jit_trace import get_jit_traced_model_forward_call
from .options.onnx import get_onnx_dynamo_forward_call, get_onnx_forward_call
from .options.optimum_quanto import (
    get_optimum_quant_model_compiled_forward_call,
    get_optimum_quanto_forward_call,
)
from .options.quant_edge_convert import get_converted_edge_quantized_model_forward_call
from .options.torchao_autoquant import get_torchao_autoquant_forward_call
from .options.torchao_quant import (
    get_compiled_torchao_quantize_forward_call,
    get_torchao_quantize_forward_call,
)
from .options.torchscript import get_torch_scripted_model_forward_call
from .options.utils.checks.imports import (
    check_onnxrt,
    check_openxla,
    check_tensort,
    check_tvm,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def select_forward_call_function(
    model: Any,
    conversion: str,
    data: torch.Tensor,
    device: torch.device,
) -> Callable:
    """
    Get the forward call function for the model, given a conversion type.

    Inputs:
    - model (Any): The model to get the forward call for.
    - conversion (str): The conversion method to use for the model.
    - data (torch.Tensor): A sample of data to pass through the model, which may be needed for
    some of the export methods.
    - device (torch.device): The device to run the benchmarking on. This is sometimes required
    for a conversion method.

    Outputs:
    - forward (Callable): The forward call function for the model.
    """
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

        case "EXPORT+COMPILE_CUDAGRAPHS":
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

        case "EXPORT+COMPILE_TENSORRT":
            check_tensort()
            forward = get_export_compiled_forward_call(model, data, backend="tensorrt")

        case "EXPORT+COMPILE_OPENVINO":
            forward = get_export_compiled_forward_call(model, data, backend="openvino")

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

        case "COMPILE_CUDAGRAPHS":
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

        case "COMPILE_TENSORRT":
            check_tensort()
            forward = get_compiled_model_forward_call(model, data, backend="tensorrt")

        case "COMPILE_OPENVINO":
            forward = get_export_compiled_forward_call(model, data, backend="openvino")

        case "COMPILE_INDUCTOR_EAGER_FALLBACK":
            forward = get_compiled_forward_call_eager_fallback(
                model, data, backend="inductor-default"
            )

        # TORCHAO
        case "COMPILE_INDUCTOR_MAX_AUTOTUNE+TORCHAO_AUTOQUANT_DEFAULT":
            forward = get_torchao_autoquant_forward_call(
                model, data, backend="inductor-max-autotune", use_autoquant_default=True
            )

        case "COMPILE_INDUCTOR_MAX_AUTOTUNE+TORCHAO_AUTOQUANT_NONDEFAULT":
            forward = get_torchao_autoquant_forward_call(
                model,
                data,
                backend="inductor-max-autotune",
                use_autoquant_default=False,
            )

        case "COMPILE_CUDAGRAPHS+TORCHAO_AUTOQUANT_DEFAULT":
            forward = get_torchao_autoquant_forward_call(
                model, data, backend="cudagraphs", use_autoquant_default=True
            )

        case "COMPILE_INDUCTOR_MAX_AUTOTUNE+TORCHAO_QUANT_I4_WEIGHT_ONLY":
            from torchao.quantization.quant_api import int4_weight_only

            forward = get_compiled_torchao_quantize_forward_call(
                model,
                data,
                backend="inductor-max-autotune",
                quantization_mode=int4_weight_only,
            )

        case "TORCHAO_QUANT_I4_WEIGHT_ONLY":
            from torchao.quantization.quant_api import int4_weight_only

            forward = get_torchao_quantize_forward_call(
                model, quantization_mode=int4_weight_only
            )

        case "EAGER":
            # Regular eager model forward call
            forward = model.forward

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
            forward = get_converted_edge_quantized_model_forward_call(model, data)

        case "NATIVE_FAKE_QUANTIZED_AI8WI8_STATIC":
            forward = get_fake_quantized_model_forward_call(model, data)

        case "JIT_TRACE":
            forward = get_jit_traced_model_forward_call(model, data)

        case "TORCH_SCRIPT":
            forward = get_torch_scripted_model_forward_call(model)

        ###############################
        # OPTIMUM QUANTO QUANTIZATION #
        ###############################
        case "OPTIMUM_QUANTO_AI8WI8":
            forward = get_optimum_quanto_forward_call(
                model, data, weights=qint8, activations=qint8
            )

        case "OPTIMUM_QUANTO_AI8WI4":
            forward = get_optimum_quanto_forward_call(
                model, data, weights=qint4, activations=qint8
            )

        case "OPTIMUM_QUANTO_AI8WI2":
            forward = get_optimum_quanto_forward_call(
                model, data, weights=qint2, activations=qint8
            )

        case "OPTIMUM_QUANTO_WI8":
            forward = get_optimum_quanto_forward_call(
                model, data, weights=qint8, activations=None
            )

        case "OPTIMUM_QUANTO_WI4":
            forward = get_optimum_quanto_forward_call(
                model, data, weights=qint4, activations=None
            )

        case "OPTIMUM_QUANTO_WI2":
            forward = get_optimum_quanto_forward_call(
                model, data, weights=qint2, activations=None
            )

        case "OPTIMUM_QUANTO_Wf8E4M3N":
            forward = get_optimum_quanto_forward_call(
                model, data, weights=qfloat8_e4m3fn, activations=None
            )

        case "OPTIMUM_QUANTO_Wf8E4M3NUZ":
            forward = get_optimum_quanto_forward_call(
                model, data, weights=qfloat8_e4m3fnuz, activations=None
            )

        case "OPTIMUM_QUANTO_Wf8E5M2":
            forward = get_optimum_quanto_forward_call(
                model, data, weights=qfloat8_e5m2, activations=None
            )

        case "OPTIMUM_QUANTO_Wf8E5M2+COMPILE_CUDAGRAPHS":
            forward = get_optimum_quant_model_compiled_forward_call(
                model,
                data,
                weights=qfloat8_e5m2,
                activations=None,
                backend="cudagraphs",
            )

        case "FP16+EAGER":
            forward = get_fp16_eager_forward_call(model)

        case "FP16+COMPILE_CUDAGRAPHS":
            model = get_fp16_model(model)
            forward = get_compiled_model_forward_call(model, data, backend="cudagraphs")

        case "FP16+COMPILE_INDUCTOR_DEFAULT":
            model = get_fp16_model(model)
            forward = get_compiled_model_forward_call(
                model, data, backend="inductor-default"
            )

        case "FP16+COMPILE_INDUCTOR_REDUCE_OVERHEAD":
            model = get_fp16_model(model)
            forward = get_compiled_model_forward_call(
                model, data, backend="inductor-reduce-overhead"
            )

        case "FP16+COMPILE_INDUCTOR_MAX_AUTOTUNE":
            model = get_fp16_model(model)
            forward = get_compiled_model_forward_call(
                model, data, backend="inductor-max-autotune"
            )

        case "FP16+COMPILE_INDUCTOR_EAGER_FALLBACK":
            model = get_fp16_model(model)
            forward = get_compiled_forward_call_eager_fallback(
                model, data, backend="inductor-default"
            )

        case "FP16+COMPILE_ONNXRT":
            model = get_fp16_model(model)
            check_onnxrt()
            forward = get_compiled_model_forward_call(model, data, backend="onnxrt")

        case "FP16+COMPILE_OPENXLA":
            model = get_fp16_model(model)
            check_openxla()
            forward = get_compiled_model_forward_call(model, data, backend="openxla")

        case "FP16+COMPILE_TVM":
            model = get_fp16_model(model)
            check_tvm()
            forward = get_compiled_model_forward_call(model, data, backend="tvm")

        case "FP16+COMPILE_TENSORRT":
            model = get_fp16_model(model)
            check_tensort()
            forward = get_compiled_model_forward_call(model, data, backend="tensorrt")

        case "FP16+COMPILE_OPENVINO":
            model = get_fp16_model(model)
            forward = get_compiled_model_forward_call(model, data, backend="openvino")

        case "FP16+EXPORT+COMPILE_CUDAGRAPHS":
            model = get_fp16_model(model)
            forward = get_export_compiled_forward_call(
                model, data, backend="cudagraphs"
            )

        case "FP16+EXPORT+COMPILE_INDUCTOR_DEFAULT":
            model = get_fp16_model(model)
            forward = get_export_compiled_forward_call(
                model, data, backend="inductor-default"
            )

        case "FP16+EXPORT+COMPILE_INDUCTOR_REDUCE_OVERHEAD":
            model = get_fp16_model(model)
            forward = get_export_compiled_forward_call(
                model, data, backend="inductor-reduce-overhead"
            )

        case "FP16+EXPORT+COMPILE_INDUCTOR_MAX_AUTOTUNE":
            model = get_fp16_model(model)
            forward = get_export_compiled_forward_call(
                model, data, backend="inductor-max-autotune"
            )

        case "FP16+EXPORT+COMPILE_INDUCTOR_DEFAULT_EAGER_FALLBACK":
            model = get_fp16_model(model)
            forward = get_export_compiled_forward_call_eager_fallback(
                model,
                data,
                backend="inductor-default",
            )

        case "FP16+EXPORT+COMPILE_ONNXRT":
            model = get_fp16_model(model)
            check_onnxrt()
            forward = get_export_compiled_forward_call(model, data, backend="onnxrt")

        case "FP16+EXPORT+COMPILE_OPENXLA":
            model = get_fp16_model(model)
            check_openxla()
            forward = get_export_compiled_forward_call(model, data, backend="openxla")

        case "FP16+EXPORT+COMPILE_TVM":
            model = get_fp16_model(model)
            check_tvm()
            forward = get_export_compiled_forward_call(model, data, backend="tvm")

        case "FP16+EXPORT+COMPILE_TENSORRT":
            model = get_fp16_model(model)
            check_tensort()
            forward = get_export_compiled_forward_call(model, data, backend="tensorrt")

        case "FP16+EXPORT+COMPILE_OPENVINO":
            model = get_fp16_model(model)
            forward = get_export_compiled_forward_call(model, data, backend="openvino")

        case "FP16+JIT_TRACE":
            model = get_fp16_model(model)
            forward = get_jit_traced_model_forward_call(model, data)

        case "FP16+TORCH_SCRIPT":
            model = get_fp16_model(model)
            forward = get_torch_scripted_model_forward_call(model)

        case "BF16+EAGER":
            forward = get_bf16_eager_forward_call(model)

        case "BF16+COMPILE_CUDAGRAPHS":
            model = get_bf16_model(model)
            forward = get_compiled_model_forward_call(model, data, backend="cudagraphs")

        case "BF16+COMPILE_INDUCTOR_DEFAULT":
            model = get_bf16_model(model)
            forward = get_compiled_model_forward_call(
                model, data, backend="inductor-default"
            )

        case "BF16+COMPILE_INDUCTOR_REDUCE_OVERHEAD":
            model = get_bf16_model(model)
            forward = get_compiled_model_forward_call(
                model, data, backend="inductor-reduce-overhead"
            )

        case "BF16+COMPILE_INDUCTOR_MAX_AUTOTUNE":
            model = get_bf16_model(model)
            forward = get_compiled_model_forward_call(
                model, data, backend="inductor-max-autotune"
            )

        case "BF16+COMPILE_INDUCTOR_EAGER_FALLBACK":
            model = get_bf16_model(model)
            forward = get_compiled_forward_call_eager_fallback(
                model, data, backend="inductor-default"
            )

        case "BF16+COMPILE_ONNXRT":
            model = get_bf16_model(model)
            check_onnxrt()
            forward = get_compiled_model_forward_call(model, data, backend="onnxrt")

        case "BF16+COMPILE_OPENXLA":
            model = get_bf16_model(model)
            check_openxla()
            forward = get_compiled_model_forward_call(model, data, backend="openxla")

        case "BF16+COMPILE_TVM":
            model = get_bf16_model(model)
            check_tvm()
            forward = get_compiled_model_forward_call(model, data, backend="tvm")

        case "BF16+COMPILE_TENSORRT":
            model = get_bf16_model(model)
            check_tensort()
            forward = get_compiled_model_forward_call(model, data, backend="tensorrt")

        case "BF16+COMPILE_OPENVINO":
            model = get_bf16_model(model)
            forward = get_compiled_model_forward_call(model, data, backend="openvino")

        case "BF16+EXPORT+COMPILE_CUDAGRAPHS":
            model = get_bf16_model(model)
            forward = get_export_compiled_forward_call(
                model, data, backend="cudagraphs"
            )

        case "BF16+EXPORT+COMPILE_INDUCTOR_DEFAULT":
            model = get_bf16_model(model)
            forward = get_export_compiled_forward_call(
                model, data, backend="inductor-default"
            )

        case "BF16+EXPORT+COMPILE_INDUCTOR_REDUCE_OVERHEAD":
            model = get_bf16_model(model)
            forward = get_export_compiled_forward_call(
                model, data, backend="inductor-reduce-overhead"
            )

        case "BF16+EXPORT+COMPILE_INDUCTOR_MAX_AUTOTUNE":
            model = get_bf16_model(model)
            forward = get_export_compiled_forward_call(
                model, data, backend="inductor-max-autotune"
            )

        case "BF16+EXPORT+COMPILE_INDUCTOR_DEFAULT_EAGER_FALLBACK":
            model = get_bf16_model(model)
            forward = get_export_compiled_forward_call_eager_fallback(
                model,
                data,
                backend="inductor-default",
            )

        case "BF16+EXPORT+COMPILE_ONNXRT":
            model = get_bf16_model(model)
            check_onnxrt()
            forward = get_export_compiled_forward_call(model, data, backend="onnxrt")

        case "BF16+EXPORT+COMPILE_OPENXLA":
            model = get_bf16_model(model)
            check_openxla()
            forward = get_export_compiled_forward_call(model, data, backend="openxla")

        case "BF16+EXPORT+COMPILE_TVM":
            model = get_bf16_model(model)
            check_tvm()
            forward = get_export_compiled_forward_call(model, data, backend="tvm")

        case "BF16+EXPORT+COMPILE_TENSORRT":
            model = get_bf16_model(model)
            check_tensort()
            forward = get_export_compiled_forward_call(model, data, backend="tensorrt")

        case "BF16+EXPORT+COMPILE_OPENVINO":
            model = get_bf16_model(model)
            forward = get_export_compiled_forward_call(model, data, backend="openvino")

        case "BF16+JIT_TRACE":
            model = get_bf16_model(model)
            forward = get_jit_traced_model_forward_call(model, data)

        case "BF16+TORCH_SCRIPT":
            model = get_bf16_model(model)
            forward = get_torch_scripted_model_forward_call(model)

        case "FP16+COMPILE_CUDAGRAPHS":
            model = get_fp16_model(model)
            forward = get_compiled_model_forward_call(model, data, backend="cudagraphs")

        case "FP16+COMPILE_INDUCTOR_DEFAULT":
            model = get_fp16_model(model)
            forward = get_compiled_model_forward_call(
                model, data, backend="inductor-default"
            )

        case "FP16+COMPILE_INDUCTOR_REDUCE_OVERHEAD":
            model = get_fp16_model(model)
            forward = get_compiled_model_forward_call(
                model, data, backend="inductor-reduce-overhead"
            )

        case "FP16+COMPILE_INDUCTOR_MAX_AUTOTUNE":
            model = get_fp16_model(model)
            forward = get_compiled_model_forward_call(
                model, data, backend="inductor-max-autotune"
            )

        case "FP16+COMPILE_INDUCTOR_EAGER_FALLBACK":
            model = get_fp16_model(model)
            forward = get_compiled_forward_call_eager_fallback(
                model, data, backend="inductor-default"
            )

        case "FP16+COMPILE_ONNXRT":
            model = get_fp16_model(model)
            check_onnxrt()
            forward = get_compiled_model_forward_call(model, data, backend="onnxrt")

        case "FP16+COMPILE_OPENXLA":
            model = get_fp16_model(model)
            check_openxla()
            forward = get_compiled_model_forward_call(model, data, backend="openxla")

        case "FP16+COMPILE_TVM":
            model = get_fp16_model(model)
            check_tvm()
            forward = get_compiled_model_forward_call(model, data, backend="tvm")

        case "FP16+COMPILE_TENSORRT":
            model = get_fp16_model(model)
            check_tensort()
            forward = get_compiled_model_forward_call(model, data, backend="tensorrt")

        case "FP16+COMPILE_OPENVINO":
            model = get_fp16_model(model)
            forward = get_compiled_model_forward_call(model, data, backend="openvino")

        case "FP16+EXPORT+COMPILE_CUDAGRAPHS":
            model = get_fp16_model(model)
            forward = get_export_compiled_forward_call(
                model, data, backend="cudagraphs"
            )

        case "FP16+EXPORT+COMPILE_INDUCTOR_DEFAULT":
            model = get_fp16_model(model)
            forward = get_export_compiled_forward_call(
                model, data, backend="inductor-default"
            )

        case "FP16+EXPORT+COMPILE_INDUCTOR_REDUCE_OVERHEAD":
            model = get_fp16_model(model)
            forward = get_export_compiled_forward_call(
                model, data, backend="inductor-reduce-overhead"
            )

        case "FP16+EXPORT+COMPILE_INDUCTOR_MAX_AUTOTUNE":
            model = get_fp16_model(model)
            forward = get_export_compiled_forward_call(
                model, data, backend="inductor-max-autotune"
            )

        case "FP16+EXPORT+COMPILE_INDUCTOR_DEFAULT_EAGER_FALLBACK":
            model = get_fp16_model(model)
            forward = get_export_compiled_forward_call_eager_fallback(
                model,
                data,
                backend="inductor-default",
            )

        case "FP16+EXPORT+COMPILE_ONNXRT":
            model = get_fp16_model(model)
            check_onnxrt()
            forward = get_export_compiled_forward_call(model, data, backend="onnxrt")

        case "FP16+EXPORT+COMPILE_OPENXLA":
            model = get_fp16_model(model)
            check_openxla()
            forward = get_export_compiled_forward_call(model, data, backend="openxla")

        case "FP16+EXPORT+COMPILE_TVM":
            model = get_fp16_model(model)
            check_tvm()
            forward = get_export_compiled_forward_call(model, data, backend="tvm")

        case "FP16+EXPORT+COMPILE_TENSORRT":
            model = get_fp16_model(model)
            check_tensort()
            forward = get_export_compiled_forward_call(model, data, backend="tensorrt")

        case "FP16+EXPORT+COMPILE_OPENVINO":
            model = get_fp16_model(model)
            forward = get_export_compiled_forward_call(model, data, backend="openvino")

        case "FP16+JIT_TRACE":
            model = get_fp16_model(model)
            forward = get_jit_traced_model_forward_call(model, data)

        case "FP16+TORCH_SCRIPT":
            model = get_fp16_model(model)
            forward = get_torch_scripted_model_forward_call(model)

        case _:
            error_msg = f"The option {conversion} is not supported"
            assert conversion in MODEL_CONVERSION_OPTIONS.values(), error_msg
            raise NotImplementedError(error_msg)

    return forward
