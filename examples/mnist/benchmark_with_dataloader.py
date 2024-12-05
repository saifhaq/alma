import logging
import time

import torch
from model.model import Net
from utils.data.datasets import BenchmarkCustomImageDataset
from utils.data.loaders import CircularDataLoader
from utils.data.transforms import InferenceTransform

from alma.arguments.benchmark_args import parse_benchmark_args
from alma.benchmark_model import benchmark_model
from alma.benchmark.log import display_all_results
from alma.utils.ipdb_hook import ipdb_sys_excepthook
from alma.utils.load_model import load_model
from alma.utils.setup_logging import setup_logging

# One needs to set their quantization backend engine to what is appropriate for their system.
# torch.backends.quantized.engine = 'x86'
torch.backends.quantized.engine = "qnnpack"


def main() -> None:
    # Adds an ipdb hook to the sys.excepthook, which will throw one into an ipdb shell when an
    # exception is raised. Comment out to have the program crash as normal during an unhandled exception
    ipdb_sys_excepthook()

    # Set up logging. DEBUG level will also log the model graphs
    # A `setup_logging` function is provided for convenience, but one can use whatever logging one
    # wishes, or none.
    setup_logging(log_file=None, level="INFO")

    args, device = parse_benchmark_args()

    # Create dataset and data loader
    assert args.data_dir is not None, "Please provide a data directory"
    dataset = BenchmarkCustomImageDataset(
        img_dir=args.data_dir, transform=InferenceTransform
    )
    data_loader = CircularDataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    assert args.model_path is not None, "Please provide a model path"
    load_start_time = time.perf_counter()
    model = load_model(args.model_path, device, logger=logging, modelArchitecture=Net)
    load_end_time = time.perf_counter()
    logging.info(f"Model loading time: {load_end_time - load_start_time:.4f} seconds")

    # Configuration for the benchmarking
    config = {
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
    }

    # Benchmark the model using the provided data loader.
    logging.info("Benchmarking model using provided data loader")
    results: Dict[str, Dict[str, Any]] = benchmark_model(model, config, conversions, data_loader=data_loader)

    # Display the results
    display_all_results(results, display_function=print, include_traceback_for_errors=True)


if __name__ == "__main__":
    main()
