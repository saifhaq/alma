import argparse

import torch

from conversions.select import MODEL_CONVERSION_OPTIONS

def parse_benchmark_args(logger):
    # string_rep_of_conv_options = ""
    string_rep_of_conv_options = "; \n".join([f"{key}: {value}" for key, value in MODEL_CONVERSION_OPTIONS.items()])
    # for key, value in MODEL_CONVERSION_OPTIONS.items():
    #     string_rep_of_conv_options += f"{key}: {value}\n"


    parser = argparse.ArgumentParser(description="Benchmark PyTorch Models")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model file (e.g., mnist_cnn_quantized.pt, mnist_cnn_scripted.pt, mnist_cnn.pt)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing images for inference",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=30,
        metavar="N",
        help="input batch size for benchmarking (default: 30)",
    )
    parser.add_argument(
        "--n_images",
        type=int,
        default=5000,
        help="Total number of images to process (default: 300)",
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
        "--conversion",
        type=int,
        choices=MODEL_CONVERSION_OPTIONS.keys(),
        default=5,
        help=f"""The model option you would like to benchmark. These are integers that correspond 
to different transforms. The mapping is this:\n{string_rep_of_conv_options}""",
    )

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    assert isinstance(
        args.conversion, int
    ), "Please select a valid option for the model conversion"
    assert (
        args.conversion in MODEL_CONVERSION_OPTIONS.keys()
    ), "Please select a valid option for the model conversion"

    logger.info(f"{MODEL_CONVERSION_OPTIONS[args.conversion]} model selected for benchmarking")
    

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return args, device
