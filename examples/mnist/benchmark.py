import argparse
import logging
import time

import torch
from model.model import Net
from utils.data.datasets import BenchmarkCustomImageDataset
from utils.data.loaders import CircularDataLoader
from utils.data.transforms import InferenceTransform

from alma.arguments.benchmark_args import parse_benchmark_args
from alma.benchmark_model import benchmark_model
from alma.utils.ipdb_hook import ipdb_sys_excepthook
from alma.utils.load_model import load_model
from alma.utils.setup_logging import setup_logging
from alma.utils.data import get_sample_data

# One needs to set their quantization backend engine to what is appropriate for their system.
# torch.backends.quantized.engine = 'x86'
torch.backends.quantized.engine = "qnnpack"


def main() -> None:
    args, device = parse_benchmark_args(logging)

    # Create dataset and data loader
    dataset = BenchmarkCustomImageDataset(
        img_dir=args.data_dir, transform=InferenceTransform
    )
    data_loader = CircularDataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    data = get_sample_data(data_loader, device)

    # Load model
    load_start_time = time.perf_counter()
    model = load_model(args.model_path, device, logger=logging, modelArchitecture=Net)
    load_end_time = time.perf_counter()
    logging.info(f"Model loading time: {load_end_time - load_start_time:.4f} seconds")

    # Benchmark the model. This will generate a dataloader that provides random tensors of the
    # same shape as `data`, which is used to benchmark the model.
    logging.info("Benchmarking model using random data")
    benchmark_model(model, device, data[0, :, :, :].squeeze(), args, logging)

    # Benchmark the model using the provided data loader.
    logging.info("Benchmarking model using provided data loader")
    benchmark_model(model, device, data, args, logging, data_loader=data_loader)


if __name__ == "__main__":
    # Adds an ipdb hook to the sys.excepthook, which will throw one into an ipdb shell when an
    # exception is raised
    ipdb_sys_excepthook()
    setup_logging()
    main()
