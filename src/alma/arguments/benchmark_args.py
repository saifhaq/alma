import argparse
import logging
from pathlib import Path
from typing import Tuple, Union

import torch

from ..conversions.select import MODEL_CONVERSION_OPTIONS


# Define a custom argument type for a list of strings
def list_of_strings(arg):
    return arg.split(",")


def parse_benchmark_args(
    logger: Union[logging.Logger, None] = None
) -> Tuple[argparse.Namespace, torch.device]:
    if logger is None:
        logger = logging.getLogger(__name__)

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
        default=30,
        metavar="N",
        help="input batch size for benchmarking (default: 30)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
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

    args = parser.parse_args()

    if args.model_path is not None:
        args.model_path = Path(args.model_path)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    # If no conversion options are provided, we use all available options
    if not args.conversions:
        args.conversions = valid_conversion_options

    # Checks on the model conversion options
    assert isinstance(
        args.conversions, (list, int, str)
    ), "Please select a valid option for the model conversion"
    if not isinstance(args.conversions, list):
        conversions = [args.conversions]
    else:
        conversions = args.conversions

    error_msg = (
        lambda conversion: f"Please select a valid option for the model conversion, {conversion} not in {valid_conversion_options}. Call `-h` for help."
    )
    for conversion in conversions:
        if conversion.isnumeric():
            conversion = int(conversion)
        assert conversion in valid_conversion_options, error_msg(conversion)
        logger.info(
            f"{MODEL_CONVERSION_OPTIONS[conversion]} model selected for benchmarking"
        )

    # Convert all selected conversion options to a list of strings. I.e., all ints become strings
    new_conversions = []
    for conversion in conversions:
        if conversion.isnumeric():
            conversion = int(conversion)
        if isinstance(conversion, int):
            if MODEL_CONVERSION_OPTIONS[conversion] not in new_conversions:
                conversions.append(MODEL_CONVERSION_OPTIONS[conversion])
        else:
            if conversion not in new_conversions:
                new_conversions.append(conversion)

    args.conversions = new_conversions

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return args, device
