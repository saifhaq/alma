import argparse
import logging
from pathlib import Path
from typing import Tuple, Union

import torch

from ..conversions.select import MODEL_CONVERSION_OPTIONS
from ..utils.ipdb_hook import ipdb_sys_excepthook

# Create a module-level logger
logger = logging.getLogger(__name__)
# Don't add handlers - let the application configure logging
logger.addHandler(logging.NullHandler())


# Define a custom argument type for a list of strings
def list_of_strings(arg):
    return arg.split(",")


def parse_benchmark_args() -> Tuple[argparse.Namespace, torch.device]:

    # Create a string represenation of the model conversion options
    # to add to the argparser description.
    string_rep_of_conv_options: str = "; \n".join(
        [f"{key}: {value}" for key, value in MODEL_CONVERSION_OPTIONS.items()]
    )
    valid_conversion_options = list(MODEL_CONVERSION_OPTIONS.keys()) + list(
        MODEL_CONVERSION_OPTIONS.values()
    )

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
        help="input batch size for benchmarking (default: 30)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2048,
        help="Total number of samples to process (default: 300)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA acceleration",
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables MPS acceleration",
    )
    parser.add_argument(
        "--conversions",
        type=list_of_strings,
        # choices=[str(i) for i in MODEL_CONVERSION_OPTIONS.keys()] + list(MODEL_CONVERSION_OPTIONS.values()),
        default=None,
        help=f"""The model option you would like to benchmark. These are integers that correspond 
to different transforms, or their string names. MUltiple options can be selected, e.g. --conversions
0,2,EAGER . The mapping is this:\n{string_rep_of_conv_options}""",
    )
    parser.add_argument(
        "--ipdb",
        action="store_true",
        default=False,
        help="Enable the ipdb system exception hook",
    )

    args = parser.parse_args()

    if args.ipdb:
        # Add an ipdb hook to the sys.excepthook, which will throw one into an ipdb shell when an
        # exception is raised.
        ipdb_sys_excepthook()

    if args.model_path is not None:
        args.model_path = Path(args.model_path)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

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
    # Convert all selected conversion options to a list of strings. I.e., all ints become strings
    # We also check that the provided conversion options are valid
    selected_conversions = []
    for conversion in conversions:
        if isinstance(conversion, str) and conversion.isnumeric():
            conversion = int(conversion)
            assert conversion in valid_conversion_options, error_msg(conversion)
            selected_conversions.append(MODEL_CONVERSION_OPTIONS[conversion])
        elif isinstance(conversion, str) and not conversion.isnumeric():
            assert conversion in valid_conversion_options, error_msg(conversion)
            selected_conversions.append(conversion)
        elif isinstance(conversion, int):
            assert conversion in valid_conversion_options, error_msg(conversion)
            selected_conversions.append(MODEL_CONVERSION_OPTIONS[conversion])
        else:
            raise ValueError(error_msg(conversion))

    # Convert the list of selected conversions to a "pretty" string for logging
    format_list = lambda lst: ", ".join(lst[:-1]) + (
        " and " + lst[-1] if len(lst) > 1 else lst[0] if lst else ""
    )
    logger.info(
        f"{format_list(selected_conversions)} model conversions selected for benchmarking\n"
    )

    args.conversions = selected_conversions

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return args, device
