import argparse
import logging
import os

import torch

from alma.conversions.conversion_options import ConversionOption

logger = logging.getLogger(__name__)


def set_xla_environment(mode: str):
    """Sets the environment variables for XLA modes."""
    xla_device_map = {
        "XLA_GPU": ("CUDA", "1"),
        "XLA_CPU": ("CPU", None),
        "XLA_TPU": ("TPU", None),
    }
    import torch_xla.core.xla_model

    for key, (device, gpu_count) in xla_device_map.items():
        if key in mode:
            os.environ["PJRT_DEVICE"] = device
            if gpu_count:
                os.environ["GPU_NUM_DEVICES"] = gpu_count
            logger.info(
                f"Environment set: PJRT_DEVICE={device}{', GPU_NUM_DEVICES=' + gpu_count if gpu_count else ''}"
            )
            return torch_xla.core.xla_model.xla_device()

    # Default to CPU if no valid XLA mode is found
    os.environ["PJRT_DEVICE"] = "CPU"
    logger.warning(f"Unknown XLA mode '{mode}', defaulting to CPU.")
    return torch_xla.core.xla_model.xla_device()


def select_device(use_cuda: bool, use_mps: bool) -> torch.device:
    """Selects the appropriate device based on CUDA, MPS, and CPU availability."""
    if use_cuda and torch.cuda.is_available():
        logger.info("CUDA device selected for benchmarking.")
        return torch.device("cuda")
    if use_mps and torch.backends.mps.is_available():
        logger.info("MPS device selected for benchmarking.")
        return torch.device("mps")
    logger.info("CPU device selected for benchmarking.")
    return torch.device("cpu")


def apply_device_override(device_override: str) -> torch.device:
    """Applies the device override specified in the conversion options."""
    override_map = {
        "CUDA": "cuda",
        "MPS": "mps",
        "CPU": "cpu",
    }
    if device_override in override_map:
        logger.info(f"Device override selected: {device_override}")
        return torch.device(override_map[device_override])

    logger.warning(
        f"Invalid device override '{device_override}', falling back to default selection."
    )
    return None


def setup_device(
    args: argparse.Namespace,
    selected_conversion: ConversionOption,
    current_device: torch.device,
) -> torch.device:
    """
    Configures the appropriate device based on the mode, user preferences,
    and conversion-specific options.
    """
    mode = selected_conversion.get("mode", "")
    device_override = selected_conversion.get("device_override")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    use_override = not args.no_device_override

    logger.debug(
        f"Mode: {mode}, Use CUDA: {use_cuda}, Use MPS: {use_mps}, Use Override: {use_override}"
    )

    # Keep current device if it exists and isn't overridden
    if current_device and not use_override:
        logger.info(f"Keeping current device: {current_device}")
        return current_device

    if "XLA" in mode:
        try:
            import torch_xla.core.xla_model as xm

            return set_xla_environment(mode)
        except ImportError as e:
            logger.error("torch_xla is not installed or unavailable.", exc_info=e)
            raise RuntimeError("XLA mode requested, but torch_xla is unavailable.")

    if use_override and device_override:
        device = apply_device_override(device_override)
        if device:
            return device

    return select_device(use_cuda, use_mps)
