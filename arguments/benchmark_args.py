import argparse

import torch


def parse_benchmark_args():
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
        "--compile",
        action="store_true",
        default=False,
        help="run torch.compile on the model",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        default=False,
        help="run torch.export on the model",
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        default=False,
        help="export the model to ONNX",
    )
    parser.add_argument(
        "--tensorrt",
        action="store_true",
        default=False,
        help="when exporting, use TensorRT backend for torch.compile",
    )
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return args, device
