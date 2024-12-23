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
