import logging
import os
from typing import Optional

import torch

from alma.conversions.conversion_options import ConversionOption

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def handle_xla_device(mode: str, allow_cuda: bool = True) -> torch.device:
    """
    Configures XLA environment variables and selects the appropriate XLA device.

    Args:
        mode (str): The XLA mode to configure (e.g., "XLA_GPU", "XLA_CPU", "XLA_TPU").
        allow_cuda (bool): Whether CUDA usage is allowed if available.

    Returns:
        torch.device: The selected XLA device.
    """
    import torch_xla.core.xla_model as xm

    xla_device_map = {
        "XLA_GPU": ("CUDA", "1"),
        "XLA_CPU": ("CPU", None),
        "XLA_TPU": ("TPU", None),
    }

    # Dynamically determine mode based on CUDA availability and user preference
    if allow_cuda and torch.cuda.is_available():
        logger.info("CUDA is available and allowed; selecting XLA_GPU.")
        mode = "XLA_GPU"
    else:
        logger.info("CUDA is not available or not allowed; selecting XLA_CPU.")
        mode = "XLA_CPU"

    for key, (device, gpu_count) in xla_device_map.items():
        if key == mode:
            os.environ["PJRT_DEVICE"] = device
            if gpu_count:
                os.environ["GPU_NUM_DEVICES"] = gpu_count
            logger.info(
                f"XLA Environment set: PJRT_DEVICE={device}"
                f"{', GPU_NUM_DEVICES=' + gpu_count if gpu_count else ''}"
            )
            return xm.xla_device()

    # Default to CPU if no valid mode is found
    os.environ["PJRT_DEVICE"] = "CPU"
    logger.warning(f"Unknown or unspecified XLA mode '{mode}', defaulting to CPU.")
    return xm.xla_device()


def override_device_if_allowed(
    device_override: str, current_device: Optional[torch.device]
) -> Optional[torch.device]:
    """
    Applies the device override if specified and valid. If no override is provided, defaults to CUDA if available.

    Args:
        device_override (str): The device override specified in the conversion.
        current_device (Optional[torch.device]): The currently selected device, if any.

    Returns:
        Optional[torch.device]: The overridden device if valid, or CUDA/CPU fallback.
    """
    override_map = {
        "CUDA": "cuda",
        "MPS": "mps",
        "CPU": "cpu",
    }

    if device_override:
        if device_override in override_map:
            logger.info(f"Device override applied: {device_override}")
            return torch.device(override_map[device_override])
        logger.warning(
            f"Invalid device override '{device_override}', ignoring override."
        )

    # Default to CUDA if no override provided and current device isn't already CUDA
    if torch.cuda.is_available() and (
        not current_device or current_device.type != "cuda"
    ):
        logger.info("Defaulting to CUDA device.")
        return torch.device("cuda")

    return current_device or torch.device("cpu")


def fallback_device_selection(use_cuda: bool, use_mps: bool) -> torch.device:
    """
    Selects a fallback device based on CUDA, MPS, and CPU availability.

    Args:
        use_cuda (bool): Whether CUDA is enabled and available.
        use_mps (bool): Whether MPS is enabled and available.

    Returns:
        torch.device: The selected fallback device.
    """
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")

    if use_mps and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def setup_device(
    current_device: Optional[torch.device] = None,
    allow_cuda: Optional[bool] = True,
    allow_mps: Optional[bool] = True,
    allow_device_override: Optional[bool] = True,
    selected_conversion: Optional[ConversionOption] = None,
) -> torch.device:
    """
    Configures the appropriate device based on mode, user preferences, and conversion-specific options.

    Args:
        current_device (Optional[torch.device]): The currently selected device, if any.
        allow_cuda (bool): Whether to allow CUDA.
        allow_mps (bool): Whether to allow MPS.
        allow_device_override (bool): Whether to allow device overrides.
        selected_conversion (Optional[ConversionOption]): Conversion options for the current benchmark.

    Returns:
        torch.device: The configured device.
    """

    mode = selected_conversion.mode if selected_conversion else ""
    device_override = (
        selected_conversion.device_override if selected_conversion else None
    )

    logger.debug(
        f"Mode: {mode}, Allow CUDA: {allow_cuda}, Allow MPS: {allow_mps}, Allow Override: {allow_device_override}"
    )

    # Handle XLA devices if mode indicates XLA
    if "XLA" in mode:
        try:
            xla_device = handle_xla_device(mode, allow_cuda)
            logger.info(f"Chosen device: {xla_device} (XLA Mode: {mode})")
            return xla_device
        except ImportError as e:
            logger.error("torch_xla is unavailable.", exc_info=e)
            raise RuntimeError("XLA mode requested but torch_xla is not installed.")

    # Apply device override if specified and allowed
    if allow_device_override and selected_conversion and device_override:
        overridden_device = override_device_if_allowed(device_override, current_device)
        if overridden_device:
            logger.info(f"Chosen device: {overridden_device} (Device override applied)")
            return overridden_device

    # Fallback to CUDA, MPS, or CPU
    fallback_device = fallback_device_selection(allow_cuda, allow_mps)
    logger.info(f"Chosen device: {fallback_device} (Fallback selection)")
    return fallback_device
