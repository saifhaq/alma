import logging
import time
from typing import Any, Dict

import torch
from model.model import Net
from utils.data.datasets import BenchmarkCustomImageDataset
from utils.data.loaders import CircularDataLoader
from utils.data.transforms import InferenceTransform
from utils.file_utils import save_dict_to_json

from alma.arguments.benchmark_args import parse_benchmark_args
from alma.benchmark.benchmark_config import BenchmarkConfig
from alma.benchmark.log import display_all_results
from alma.benchmark_model import benchmark_model
from alma.utils.device import setup_device
from alma.utils.load_model import load_model
from alma.utils.setup_logging import setup_logging

# One needs to set their quantization backend engine to what is appropriate for their system.
# torch.backends.quantized.engine = 'x86'
torch.backends.quantized.engine = "qnnpack"


def main() -> None:
    # Set up logging. DEBUG level will also log the internal conversion logs (where available), as well
    # as the model graphs. A `setup_logging` function is provided for convenience, but one can use
    # whatever logging one wishes, or none.
    setup_logging(log_file=None, level="ERROR")

    args, conversions = parse_benchmark_args()

    # Create dataset and data loader
    assert args.data_dir is not None, "Please provide a data directory"
    dataset = BenchmarkCustomImageDataset(
        img_dir=args.data_dir, transform=InferenceTransform
    )

    device = setup_device()
    # Load model
    assert args.model_path is not None, "Please provide a model path"
    load_start_time = time.perf_counter()
    model = load_model(args.model_path, device, logger=logging, modelArchitecture=Net)
    load_end_time = time.perf_counter()
    logging.info(f"Model loading time: {load_end_time - load_start_time:.4f} seconds")

    # Configuration for the benchmarking
    config = BenchmarkConfig(
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        device=device,
        multiprocessing=True,
        fail_on_error=False,
    )

    # Benchmark the model using the provided data loader.
    logging.info("Benchmarking model using provided data loader")
    data_loader = CircularDataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    results: Dict[str, Dict[str, Any]] = benchmark_model(
        model=model,
        config=config,
        conversions=conversions,
        data=None,
        data_loader=data_loader,
    )

    # Display the results
    display_all_results(
        results, display_function=print, include_traceback_for_errors=True
    )
    save_dict_to_json(results, "result.json")


if __name__ == "__main__":
    main()
