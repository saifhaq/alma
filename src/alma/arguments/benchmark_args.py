import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Union

import torch
from typing_extensions import TypedDict

from ..conversions.conversion_options import MODEL_CONVERSION_OPTIONS
from ..utils.ipdb_hook import ipdb_sys_excepthook

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ConversionOption(TypedDict):
    mode: str
    device_override: Union[str, None]


def list_of_strings(arg: str) -> List[str]:
    """Parse a comma-separated string into a list of strings."""
    return arg.split(",")


def parse_benchmark_args() -> Tuple[argparse.Namespace, List[ConversionOption]]:
    """
    Parses command-line arguments for benchmarking PyTorch models and determines
    the selected conversions.

    Returns:
        Tuple[argparse.Namespace, List[ConversionOption]]: Parsed arguments and selected conversions.
    """
    # Create a string representation of the model conversion options
    string_rep_of_conv_options: str = "; \n".join(
        [f"{key}: {value['mode']}" for key, value in MODEL_CONVERSION_OPTIONS.items()]
    )

    # Construct a helper mapping from mode strings to their integer keys for validation
    mode_to_key = {v["mode"]: k for k, v in MODEL_CONVERSION_OPTIONS.items()}

    # Valid options are keys (ints) or mode strings
    valid_conversion_options = list(MODEL_CONVERSION_OPTIONS.keys()) + list(
        mode_to_key.keys()
    )

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
        help="Path to the directory containing samples for inference",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="Input batch size for benchmarking (default: 64)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2048,
        help="Total number of samples to process (default: 2048)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="Disables CUDA acceleration",
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="Disables MPS acceleration",
    )
    parser.add_argument(
        "--no-device-override",
        action="store_true",
        default=False,
        help="If set, ignores any device_override in the selected conversions.",
    )
    parser.add_argument(
        "--conversions",
        type=list_of_strings,
        default=None,
        help=f"""The model options you would like to benchmark. These can be integers that correspond 
        to different transforms or their string names. Multiple options can be selected, e.g., --conversions
        0,2,EAGER. The mapping is this:\n{string_rep_of_conv_options}""",
    )
    parser.add_argument(
        "--ipdb",
        action="store_true",
        default=False,
        help="Enable the ipdb system exception hook",
    )

    args = parser.parse_args()

    if args.ipdb:
        ipdb_sys_excepthook()

    # Convert model path to Path object if provided
    if args.model_path is not None:
        args.model_path = Path(args.model_path)

    # If no conversion options are provided, we use all available options
    if not args.conversions:
        conversions = valid_conversion_options
    else:
        conversions = args.conversions

    # Checks on the model conversion options
    assert isinstance(
        conversions, (list, int, str)
    ), "Please select a valid option for the model conversion"
    if not isinstance(conversions, list):
        conversions = [conversions]

    error_msg = (
        lambda conversion: f"Please select a valid option for the model conversion, {conversion} not in {valid_conversion_options}. Call `-h` for help."
    )

    selected_conversions: List[ConversionOption] = []
    for conversion in conversions:
        # Handle numeric strings
        if isinstance(conversion, str) and conversion.isnumeric():
            conversion_int = int(conversion)
            assert conversion_int in MODEL_CONVERSION_OPTIONS, error_msg(conversion_int)
            selected_conversions.append(MODEL_CONVERSION_OPTIONS[conversion_int])
        # Handle string modes
        elif isinstance(conversion, str) and not conversion.isnumeric():
            assert conversion in valid_conversion_options, error_msg(conversion)
            # If it's a known mode, find its int key and append the corresponding ConversionOption
            if conversion in mode_to_key:
                selected_conversions.append(
                    MODEL_CONVERSION_OPTIONS[mode_to_key[conversion]]
                )
            else:
                # This case theoretically shouldn't happen since we validated above
                raise ValueError(error_msg(conversion))
        # Handle ints directly
        elif isinstance(conversion, int):
            assert conversion in MODEL_CONVERSION_OPTIONS, error_msg(conversion)
            selected_conversions.append(MODEL_CONVERSION_OPTIONS[conversion])
        else:
            raise ValueError(error_msg(conversion))

    def format_list(lst: List[ConversionOption]) -> str:
        if not lst:
            return ""
        modes = [c["mode"] for c in lst]
        if len(modes) > 1:
            return ", ".join(modes[:-1]) + " and " + modes[-1]
        else:
            return modes[0]

    logger.info(
        f"{format_list(selected_conversions)} model conversions selected for benchmarking\n"
    )

    args.conversions = selected_conversions

    return args, selected_conversions
