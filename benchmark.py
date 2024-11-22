import argparse
import logging
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments.benchmark_args import parse_benchmark_args
from conversions.select import select_forward_call_function
from data.datasets import BenchmarkCustomImageDataset
from data.loaders import CircularDataLoader
from data.transforms import InferenceTransform
from data.utils import get_sample_data
from model.model import Net
from utils.ipdb_hook import ipdb_sys_excepthook
from utils.setup_logging import setup_logging

ipdb_sys_excepthook()


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    if model_path.endswith(".pt"):
        try:
            model = torch.jit.load(model_path, map_location=device)
            model.eval()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model = Net()  # Define your model architecture (Net)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
    else:
        model = torch.jit.load(model_path, map_location=device)
        model.to(device)
        model.eval()
    return model


def benchmark_model(
    model: torch.nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    n_images: int,
    args: argparse.Namespace,
    logging,
) -> None:
    total_time = 0.0
    total_images = 0
    num_batches = 0

    # Get sample of data, used in some of the compilation methods
    data = get_sample_data(data_loader, device)

    # Get the forward call of the model, which we will benchmark
    forward_call = select_forward_call_function(model, args, data, logging)

    # Warmup
    counter = 0
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            _ = forward_call(data)
            counter += 1
            if counter > 3:
                break

    start_time = time.time()  # Start timing for the entire process

    with torch.no_grad():
        for data, _ in tqdm(data_loader, desc="Benchmarking"):
            if total_images >= n_images:
                break

            # data = data.to(device, non_blocking=True)
            data = data.to(device)
            batch_start_time = time.time()
            _ = forward_call(data)
            batch_end_time = time.time()

            batch_size = min(data.size(0), n_images - total_images)
            total_time += batch_end_time - batch_start_time
            total_images += batch_size
            num_batches += 1

            if total_images >= n_images:
                break

    end_time = time.time()  # End timing for the entire process

    total_elapsed_time = end_time - start_time
    throughput = total_images / total_elapsed_time if total_elapsed_time > 0 else 0
    logging.info(f"Total elapsed time: {total_elapsed_time:.4f} seconds")
    logging.info(f"Total inference time (model only): {total_time:.4f} seconds")
    logging.info(f"Total images: {total_images}")
    logging.info(f"Throughput: {throughput:.2f} images/second")


def main() -> None:
    setup_logging()
    args, device = parse_benchmark_args(logging)

    # Create dataset and data loader
    dataset = BenchmarkCustomImageDataset(
        img_dir=args.data_dir, transform=InferenceTransform
    )
    data_loader = CircularDataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    model = load_model(args.model_path, device)

    benchmark_model(model, device, data_loader, args.n_images, args, logging)


if __name__ == "__main__":
    main()
