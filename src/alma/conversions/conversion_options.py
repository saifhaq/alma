from typing import List, Optional, Union

from pydantic import BaseModel, Field


class ConversionOption(BaseModel):
    """
    Represents a model conversion option.

    Attributes:
        mode (str): Conversion mode, e.g., "EAGER", "ONNX_CPU", etc.
        device_override (Optional[str]): Override device for conversion, e.g., "CPU", "CUDA", or None.
    """

    mode: str = Field(..., description="The mode or strategy for model conversion.")
    device_override: Optional[str] = Field(
        None,
        description="Optional override for the target device, e.g., 'CPU', 'CUDA', etc.",
    )


# Predefined conversion options for benchmarking
MODEL_CONVERSION_OPTIONS: dict[int, ConversionOption] = {
    0: ConversionOption(mode="EAGER"),
    1: ConversionOption(mode="EXPORT+EAGER"),
    2: ConversionOption(mode="ONNX_CPU", device_override="CPU"),
    3: ConversionOption(mode="ONNX_GPU", device_override="CUDA"),
    4: ConversionOption(mode="ONNX+DYNAMO_EXPORT"),
    5: ConversionOption(mode="COMPILE_CUDAGRAPHS", device_override="CUDA"),
    6: ConversionOption(mode="COMPILE_INDUCTOR_DEFAULT"),
    7: ConversionOption(mode="COMPILE_INDUCTOR_REDUCE_OVERHEAD"),
    8: ConversionOption(mode="COMPILE_INDUCTOR_MAX_AUTOTUNE"),
    9: ConversionOption(mode="COMPILE_INDUCTOR_EAGER_FALLBACK"),
    10: ConversionOption(mode="COMPILE_ONNXRT", device_override="CUDA"),
    11: ConversionOption(mode="COMPILE_OPENXLA", device_override="XLA_GPU"),
    12: ConversionOption(mode="COMPILE_TVM"),
    13: ConversionOption(mode="EXPORT+AI8WI8_FLOAT_QUANTIZED"),
    14: ConversionOption(mode="EXPORT+AI8WI8_FLOAT_QUANTIZED+RUN_DECOMPOSITION"),
    15: ConversionOption(mode="EXPORT+AI8WI8_STATIC_QUANTIZED"),
    16: ConversionOption(mode="EXPORT+AI8WI8_STATIC_QUANTIZED+RUN_DECOMPOSITION"),
    17: ConversionOption(mode="EXPORT+AOT_INDUCTOR"),
    18: ConversionOption(mode="EXPORT+COMPILE_CUDAGRAPHS", device_override="CUDA"),
    19: ConversionOption(mode="EXPORT+COMPILE_INDUCTOR_DEFAULT"),
    20: ConversionOption(mode="EXPORT+COMPILE_INDUCTOR_REDUCE_OVERHEAD"),
    21: ConversionOption(mode="EXPORT+COMPILE_INDUCTOR_MAX_AUTOTUNE"),
    22: ConversionOption(mode="EXPORT+COMPILE_INDUCTOR_DEFAULT_EAGER_FALLBACK"),
    23: ConversionOption(mode="EXPORT+COMPILE_ONNXRT", device_override="CUDA"),
    24: ConversionOption(mode="EXPORT+COMPILE_OPENXLA", device_override="XLA_GPU"),
    25: ConversionOption(mode="EXPORT+COMPILE_TVM"),
    26: ConversionOption(mode="NATIVE_CONVERT_AI8WI8_STATIC_QUANTIZED"),
    27: ConversionOption(mode="NATIVE_FAKE_QUANTIZED_AI8WI8_STATIC"),
    28: ConversionOption(mode="COMPILE_TENSORRT"),
    29: ConversionOption(mode="EXPORT+COMPILE_TENSORRT"),
    30: ConversionOption(mode="JIT_TRACE"),
    31: ConversionOption(mode="TORCH_SCRIPT"),
    32: ConversionOption(mode="OPTIMUM_QUANTO_AI8WI8"),
    33: ConversionOption(mode="OPTIMUM_QUANTO_AI8WI4"),
    34: ConversionOption(mode="OPTIMUM_QUANTO_AI8WI2"),
    35: ConversionOption(mode="OPTIMUM_QUANTO_WI8"),
    36: ConversionOption(mode="OPTIMUM_QUANTO_WI4"),
    37: ConversionOption(mode="OPTIMUM_QUANTO_WI2"),
    38: ConversionOption(mode="OPTIMUM_QUANTO_Wf8E4M3N"),
    39: ConversionOption(mode="OPTIMUM_QUANTO_Wf8E4M3NUZ"),
    40: ConversionOption(mode="OPTIMUM_QUANTO_Wf8E5M2"),
    41: ConversionOption(
        mode="OPTIMIM_QUANTO_Wf8E5M2+COMPILE_CUDAGRAPHS", device_override="CUDA"
    ),
    42: ConversionOption(mode="FP16+EAGER"),
    43: ConversionOption(mode="BF16+EAGER"),  
    44: ConversionOption(
        mode="COMPILE_INDUCTOR_MAX_AUTOTUNE+TORCHAO_AUTOQUANT_DEFAULT"
    ),
    45: ConversionOption(
        mode="COMPILE_INDUCTOR_MAX_AUTOTUNE+TORCHAO_AUTOQUANT_NONDEFAULT"
    ),
    46: ConversionOption(
        mode="COMPILE_CUDAGRAPHS+TORCHAO_AUTOQUANT_DEFAULT", device_override="CUDA"
    ),  # Works in certain circumstances, depends on weight tensor sizes
    47: ConversionOption(
        mode="COMPILE_INDUCTOR_MAX_AUTOTUNE+TORCHAO_QUANT_I4_WEIGHT_ONLY"
    ),  # Requires bf16 suuport
    48: ConversionOption(mode="TORCHAO_QUANT_I4_WEIGHT_ONLY"),  # Requires bf16 support
}


def conversions_to_modes(conversions: List[Union[str, ConversionOption]]) -> List[str]:
    """
    Converts a list of conversions (which may be strings or ConversionOption instances)
    into a list of mode strings.

    Args:
        conversions (List[Union[str, ConversionOption]]): List of conversion methods.

    Returns:
        List[str]: List of mode strings.
    """
    modes = []
    for c in conversions:
        if isinstance(c, str):
            # Already a mode string
            modes.append(c)
        elif isinstance(c, ConversionOption):
            # Extract mode from ConversionOption
            modes.append(c.mode)
        else:
            raise ValueError(
                f"Invalid conversion type: {type(c)}. Expected str or ConversionOption."
            )
    return modes


def mode_str_to_conversions(conversions: List[str]) -> List[ConversionOption]:
    """
    Converts a list of mode strings into a list of ConversionOption instances.

    Args:
        conversions (List[str]): List of mode strings.

    Returns:
        List[ConversionOption]: List of ConversionOption instances.
    """
    if not conversions:
        return []

    output = []
    for option in MODEL_CONVERSION_OPTIONS.values():
        name = option.mode
        if name in conversions:
            output.append(option)
    return output
