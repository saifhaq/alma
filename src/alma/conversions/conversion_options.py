from typing import List, Tuple, TypedDict, Union


class ConversionOption(TypedDict):
    mode: str
    device_override: Union[str, None]


def conversions_to_modes(conversions: List[Union[str, ConversionOption]]) -> List[str]:
    """
    Converts a list of conversions (which may be strings or ConversionOption dictionaries)
    into a list of mode strings.
    """
    modes = []
    for c in conversions:
        if isinstance(c, str):
            # Already a mode string
            modes.append(c)
        else:
            # c is a ConversionOption
            modes.append(c["mode"])
    return modes


MODEL_CONVERSION_OPTIONS: dict[int, ConversionOption] = {
    0: {"mode": "EAGER", "device_override": None},
    1: {"mode": "EXPORT+EAGER", "device_override": None},
    2: {"mode": "ONNX_CPU", "device_override": "CPU"},
    3: {"mode": "ONNX_GPU", "device_override": "CUDA"},
    4: {"mode": "ONNX+DYNAMO_EXPORT", "device_override": None},
    5: {"mode": "COMPILE_CUDAGRAPHS", "device_override": "CUDA"},
    6: {"mode": "COMPILE_INDUCTOR_DEFAULT", "device_override": None},
    7: {"mode": "COMPILE_INDUCTOR_REDUCE_OVERHEAD", "device_override": None},
    8: {"mode": "COMPILE_INDUCTOR_MAX_AUTOTUNE", "device_override": None},
    9: {"mode": "COMPILE_INDUCTOR_EAGER_FALLBACK", "device_override": None},
    10: {"mode": "COMPILE_ONNXRT", "device_override": "CUDA"},
    11: {"mode": "COMPILE_OPENXLA", "device_override": "XLA_GPU"},
    12: {"mode": "COMPILE_TVM", "device_override": None},
    13: {"mode": "EXPORT+AI8WI8_FLOAT_QUANTIZED", "device_override": None},
    14: {
        "mode": "EXPORT+AI8WI8_FLOAT_QUANTIZED+RUN_DECOMPOSITION",
        "device_override": None,
    },
    15: {"mode": "EXPORT+AI8WI8_STATIC_QUANTIZED", "device_override": None},
    16: {
        "mode": "EXPORT+AI8WI8_STATIC_QUANTIZED+RUN_DECOMPOSITION",
        "device_override": None,
    },
    17: {"mode": "EXPORT+AOT_INDUCTOR", "device_override": None},
    18: {"mode": "EXPORT+COMPILE_CUDAGRAPHS", "device_override": "CUDA"},
    19: {"mode": "EXPORT+COMPILE_INDUCTOR_DEFAULT", "device_override": None},
    20: {"mode": "EXPORT+COMPILE_INDUCTOR_REDUCE_OVERHEAD", "device_override": None},
    21: {"mode": "EXPORT+COMPILE_INDUCTOR_MAX_AUTOTUNE", "device_override": None},
    22: {
        "mode": "EXPORT+COMPILE_INDUCTOR_DEFAULT_EAGER_FALLBACK",
        "device_override": None,
    },
    23: {"mode": "EXPORT+COMPILE_ONNXRT", "device_override": "CUDA"},
    24: {"mode": "EXPORT+COMPILE_OPENXLA", "device_override": "XLA_GPU"},
    25: {"mode": "EXPORT+COMPILE_TVM", "device_override": None},
    26: {"mode": "NATIVE_CONVERT_AI8WI8_STATIC_QUANTIZED", "device_override": None},
    27: {"mode": "NATIVE_FAKE_QUANTIZED_AI8WI8_STATIC", "device_override": None},
    28: {"mode": "COMPILE_TENSORRT", "device_override": None},
    29: {"mode": "EXPORT+COMPILE_TENSORRT", "device_override": None},
    30: {"mode": "JIT_TRACE", "device_override": None},
    31: {"mode": "TORCH_SCRIPT", "device_override": None},
    32: {"mode": "OPTIMIM_QUANTO_AI8WI8", "device_override": None},
    33: {"mode": "OPTIMIM_QUANTO_AI8WI4", "device_override": None},
    34: {"mode": "OPTIMIM_QUANTO_AI8WI2", "device_override": None},
    35: {"mode": "OPTIMIM_QUANTO_WI8", "device_override": None},
    36: {"mode": "OPTIMIM_QUANTO_WI4", "device_override": None},
    37: {"mode": "OPTIMIM_QUANTO_WI2", "device_override": None},
    38: {"mode": "OPTIMIM_QUANTO_Wf8E4M3N", "device_override": None},
    39: {"mode": "OPTIMIM_QUANTO_Wf8E4M3NUZ", "device_override": None},
    40: {"mode": "OPTIMIM_QUANTO_Wf8E5M2", "device_override": None},
    41: {
        "mode": "OPTIMIM_QUANTO_Wf8E5M2+COMPILE_CUDAGRAPHS",
        "device_override": "cuda",
    },
}
