import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Union

from pydantic import ValidationError

from ..conversions.conversion_options import MODEL_CONVERSION_OPTIONS, ConversionOption
from ..utils.ipdb_hook import ipdb_sys_excepthook

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def list_of_strings(arg: str) -> List[str]:
    """
    Parse a comma-separated string into a list of strings.
    """
    return arg.split(",")


def parse_benchmark_args() -> Tuple[argparse.Namespace, List[ConversionOption]]:
    """
    Parses command-line arguments for benchmarking PyTorch models and determines
    the selected conversions.

    Returns:
        Tuple[argparse.Namespace, List[ConversionOption]]: Parsed arguments and selected conversions.
    """
    # String representation of all available conversion options
    string_rep_of_conv_options: str = "\n".join(
        [f"{key}: {value.mode}" for key, value in MODEL_CONVERSION_OPTIONS.items()]
    )

    # Mapping from mode strings to their integer keys
    mode_to_key = {v.mode: k for k, v in MODEL_CONVERSION_OPTIONS.items()}

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Benchmark PyTorch Models")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the model file (e.g., mnist_cnn_quantized.pt, mnist_cnn_scripted.pt, mnist_cnn.pt)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to the directory containing samples for inference.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="Input batch size for benchmarking (default: 64).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2048,
        help="Total number of samples to process (default: 2048).",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="Disables CUDA acceleration.",
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="Disables MPS acceleration.",
    )
    parser.add_argument(
        "--no-device-override",
        action="store_true",
        default=False,
        help="Ignores any device_override in the selected conversions.",
    )
    parser.add_argument(
        "--conversions",
        type=list_of_strings,
        default=None,
        help=f"""The model options you would like to benchmark. These can be integers that correspond 
        to different transforms or their string names. Multiple options can be selected, e.g., --conversions
        0,2,EAGER. The mapping is this:\n{string_rep_of_conv_options}""",
    ),
    parser.add_argument(
        "--ipdb",
        action="store_true",
        default=False,
        help="Enable the ipdb system exception hook.",
    )

    args = parser.parse_args()

    # Enable ipdb system exception hook if requested
    if args.ipdb:
        ipdb_sys_excepthook()

    # Convert model path to a Path object if provided
    if args.model_path is not None:
        args.model_path = Path(args.model_path)

    # If no conversions are provided, select all available options
    if not args.conversions:
        conversions = list(MODEL_CONVERSION_OPTIONS.keys())
    else:
        conversions = args.conversions

    # Resolve the selected conversions into ConversionOption instances
    selected_conversions: List[ConversionOption] = []

    for conversion in conversions:
        if isinstance(conversion, str):
            # Handle numeric strings
            if conversion.isdigit():
                conversion_int = int(conversion)
                if conversion_int in MODEL_CONVERSION_OPTIONS:
                    selected_conversions.append(
                        MODEL_CONVERSION_OPTIONS[conversion_int]
                    )
                else:
                    raise ValueError(f"Invalid conversion key: {conversion_int}")
            # Handle mode string inputs
            elif conversion in mode_to_key:
                selected_conversions.append(
                    MODEL_CONVERSION_OPTIONS[mode_to_key[conversion]]
                )
            else:
                raise ValueError(f"Invalid conversion mode: '{conversion}'.")
        elif isinstance(conversion, int):
            # Handle integer keys directly
            if conversion in MODEL_CONVERSION_OPTIONS:
                selected_conversions.append(MODEL_CONVERSION_OPTIONS[conversion])
            else:
                raise ValueError(f"Invalid conversion key: {conversion}.")
        else:
            raise ValueError(f"Unexpected conversion type: {type(conversion)}")

    # Logging the selected conversions
    def format_list(conversions: List[ConversionOption]) -> str:
        """Formats a list of ConversionOption objects for logging."""
        modes = [c.mode for c in conversions]
        return (
            ", ".join(modes[:-1]) + " and " + modes[-1] if len(modes) > 1 else modes[0]
        )

    logger.info(
        f"Selected model conversions for benchmarking: {format_list(selected_conversions)}"
    )

    args.conversions = selected_conversions
    return args, selected_conversions
