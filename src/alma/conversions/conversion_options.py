from typing import List, Optional, Union

import torch
from pydantic import BaseModel, Field, validator


class ConversionOption(BaseModel):
    """
    Represents a model conversion option.

    Attributes:
        mode (str): Conversion mode, e.g., "EAGER", "ONNX_CPU", etc.
        device_override (Optional[str]): Override device for conversion, e.g., "CPU", "CUDA", or None.
    """

    class Config:
        arbitrary_types_allowed = True  # Add this to allow torch.dtype

    mode: str = Field(..., description="The mode or strategy for model conversion.")
    device_override: Optional[str] = Field(
        None,
        description="Optional override for the target device, e.g., 'CPU', 'CUDA', etc.",
    )
    data_dtype: Optional[torch.dtype] = Field(  # Specify the type hint correctly
        default=torch.float32, description="The data type of the input data."
    )

    @validator("data_dtype", pre=True)
    def validate_dtype(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            # Convert string representation to torch.dtype
            return getattr(torch, v)
        if isinstance(v, torch.dtype):
            return v
        raise ValueError(f"Invalid dtype: {v}")


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
    30: ConversionOption(mode="COMPILE_OPENVINO"),
    31: ConversionOption(mode="JIT_TRACE"),
    32: ConversionOption(mode="TORCH_SCRIPT"),
    33: ConversionOption(mode="OPTIMUM_QUANTO_AI8WI8"),
    34: ConversionOption(mode="OPTIMUM_QUANTO_AI8WI4"),
    35: ConversionOption(mode="OPTIMUM_QUANTO_AI8WI2"),
    36: ConversionOption(mode="OPTIMUM_QUANTO_WI8"),
    37: ConversionOption(mode="OPTIMUM_QUANTO_WI4"),
    38: ConversionOption(mode="OPTIMUM_QUANTO_WI2"),
    39: ConversionOption(mode="OPTIMUM_QUANTO_Wf8E4M3N"),
    40: ConversionOption(mode="OPTIMUM_QUANTO_Wf8E4M3NUZ"),
    41: ConversionOption(mode="OPTIMUM_QUANTO_Wf8E5M2"),
    42: ConversionOption(
        mode="OPTIMIM_QUANTO_Wf8E5M2+COMPILE_CUDAGRAPHS", device_override="CUDA"
    ),
    43: ConversionOption(mode="FP16+EAGER", data_dtype=torch.float16),
    44: ConversionOption(mode="BF16+EAGER", data_dtype=torch.bfloat16),
    45: ConversionOption(
        mode="COMPILE_INDUCTOR_MAX_AUTOTUNE+TORCHAO_AUTOQUANT_DEFAULT"
    ),
    46: ConversionOption(
        mode="COMPILE_INDUCTOR_MAX_AUTOTUNE+TORCHAO_AUTOQUANT_NONDEFAULT"
    ),
    47: ConversionOption(
        mode="COMPILE_CUDAGRAPHS+TORCHAO_AUTOQUANT_DEFAULT", device_override="CUDA"
    ),  # Works in certain circumstances, depends on weight tensor sizes
    48: ConversionOption(
        mode="COMPILE_INDUCTOR_MAX_AUTOTUNE+TORCHAO_QUANT_I4_WEIGHT_ONLY"
    ),  # Requires bf16 suuport
    49: ConversionOption(mode="TORCHAO_QUANT_I4_WEIGHT_ONLY"),  # Requires bf16 support
    50: ConversionOption(
        mode="FP16+COMPILE_CUDAGRAPHS", device_override="CUDA", data_dtype=torch.float16
    ),
    51: ConversionOption(
        mode="FP16+COMPILE_INDUCTOR_DEFAULT", data_dtype=torch.float16
    ),
    52: ConversionOption(
        mode="FP16+COMPILE_INDUCTOR_REDUCE_OVERHEAD", data_dtype=torch.float16
    ),
    53: ConversionOption(
        mode="FP16+COMPILE_INDUCTOR_MAX_AUTOTUNE", data_dtype=torch.float16
    ),
    54: ConversionOption(
        mode="FP16+COMPILE_INDUCTOR_EAGER_FALLBACK", data_dtype=torch.float16
    ),
    55: ConversionOption(
        mode="FP16+COMPILE_ONNXRT", device_override="CUDA", data_dtype=torch.float16
    ),
    56: ConversionOption(
        mode="FP16+COMPILE_OPENXLA", device_override="XLA_GPU", data_dtype=torch.float16
    ),
    57: ConversionOption(mode="FP16+COMPILE_TVM", data_dtype=torch.float16),
    58: ConversionOption(mode="FP16+COMPILE_TENSORRT", data_dtype=torch.float16),
    59: ConversionOption(mode="FP16+COMPILE_OPENVINO", data_dtype=torch.float16),
    60: ConversionOption(
        mode="FP16+EXPORT+COMPILE_CUDAGRAPHS",
        device_override="CUDA",
        data_dtype=torch.float16,
    ),
    61: ConversionOption(
        mode="FP16+EXPORT+COMPILE_INDUCTOR_DEFAULT", data_dtype=torch.float16
    ),
    62: ConversionOption(
        mode="FP16+EXPORT+COMPILE_INDUCTOR_REDUCE_OVERHEAD", data_dtype=torch.float16
    ),
    63: ConversionOption(
        mode="FP16+EXPORT+COMPILE_INDUCTOR_MAX_AUTOTUNE", data_dtype=torch.float16
    ),
    64: ConversionOption(
        mode="FP16+EXPORT+COMPILE_INDUCTOR_DEFAULT_EAGER_FALLBACK",
        data_dtype=torch.float16,
    ),
    65: ConversionOption(
        mode="FP16+EXPORT+COMPILE_ONNXRT",
        device_override="CUDA",
        data_dtype=torch.float16,
    ),
    66: ConversionOption(
        mode="FP16+EXPORT+COMPILE_OPENXLA",
        device_override="XLA_GPU",
        data_dtype=torch.float16,
    ),
    67: ConversionOption(mode="FP16+EXPORT+COMPILE_TVM", data_dtype=torch.float16),
    68: ConversionOption(mode="FP16+EXPORT+COMPILE_TENSORRT", data_dtype=torch.float16),
    69: ConversionOption(mode="FP16+EXPORT+COMPILE_OPENVINO", data_dtype=torch.float16),
    70: ConversionOption(mode="FP16+JIT_TRACE", data_dtype=torch.float16),
    71: ConversionOption(mode="FP16+TORCH_SCRIPT", data_dtype=torch.float16),
    72: ConversionOption(
        mode="BF16+COMPILE_CUDAGRAPHS",
        device_override="CUDA",
        data_dtype=torch.bfloat16,
    ),
    73: ConversionOption(
        mode="BF16+COMPILE_INDUCTOR_DEFAULT", data_dtype=torch.bfloat16
    ),
    74: ConversionOption(
        mode="BF16+COMPILE_INDUCTOR_REDUCE_OVERHEAD", data_dtype=torch.bfloat16
    ),
    75: ConversionOption(
        mode="BF16+COMPILE_INDUCTOR_MAX_AUTOTUNE", data_dtype=torch.bfloat16
    ),
    76: ConversionOption(
        mode="BF16+COMPILE_INDUCTOR_EAGER_FALLBACK", data_dtype=torch.bfloat16
    ),
    77: ConversionOption(
        mode="BF16+COMPILE_ONNXRT", device_override="CUDA", data_dtype=torch.bfloat16
    ),
    78: ConversionOption(
        mode="BF16+COMPILE_OPENXLA",
        device_override="XLA_GPU",
        data_dtype=torch.bfloat16,
    ),
    79: ConversionOption(mode="BF16+COMPILE_TVM", data_dtype=torch.bfloat16),
    80: ConversionOption(mode="BF16+COMPILE_TENSORRT", data_dtype=torch.bfloat16),
    81: ConversionOption(mode="BF16+COMPILE_OPENVINO", data_dtype=torch.bfloat16),
    82: ConversionOption(
        mode="BF16+EXPORT+COMPILE_CUDAGRAPHS",
        device_override="CUDA",
        data_dtype=torch.bfloat16,
    ),
    83: ConversionOption(
        mode="BF16+EXPORT+COMPILE_INDUCTOR_DEFAULT", data_dtype=torch.bfloat16
    ),
    84: ConversionOption(
        mode="BF16+EXPORT+COMPILE_INDUCTOR_REDUCE_OVERHEAD", data_dtype=torch.bfloat16
    ),
    85: ConversionOption(
        mode="BF16+EXPORT+COMPILE_INDUCTOR_MAX_AUTOTUNE", data_dtype=torch.bfloat16
    ),
    86: ConversionOption(
        mode="BF16+EXPORT+COMPILE_INDUCTOR_DEFAULT_EAGER_FALLBACK",
        data_dtype=torch.bfloat16,
    ),
    87: ConversionOption(
        mode="BF16+EXPORT+COMPILE_ONNXRT",
        device_override="CUDA",
        data_dtype=torch.bfloat16,
    ),
    88: ConversionOption(
        mode="BF16+EXPORT+COMPILE_OPENXLA",
        device_override="XLA_GPU",
        data_dtype=torch.bfloat16,
    ),
    89: ConversionOption(mode="BF16+EXPORT+COMPILE_TVM", data_dtype=torch.bfloat16),
    90: ConversionOption(
        mode="BF16+EXPORT+COMPILE_TENSORRT", data_dtype=torch.bfloat16
    ),
    91: ConversionOption(
        mode="BF16+EXPORT+COMPILE_OPENVINO", data_dtype=torch.bfloat16
    ),
    92: ConversionOption(mode="BF16+JIT_TRACE", data_dtype=torch.bfloat16),
    93: ConversionOption(mode="BF16+TORCH_SCRIPT", data_dtype=torch.bfloat16),
    93: ConversionOption(mode="AOT_IREE"),
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
    # TODO: correct the order of the conversions, so that they are in the same order as the input
    for option in MODEL_CONVERSION_OPTIONS.values():
        name = option.mode
        if name in conversions:
            output.append(option)
    return output
