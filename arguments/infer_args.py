import argparse

import torch


def parse_infer_args():
    parser = argparse.ArgumentParser(description="Digit Classification Inference")
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="path to the trained TorchScript model",
        default="mnist_cnn_scripted.pt",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=False,
        help="directory with images for inference",
        default="data_for_inference",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    return args, device
