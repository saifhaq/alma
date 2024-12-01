import logging
import time

import torch

from .arguments.benchmark_args import parse_benchmark_args
from .benchmark.benchmark_model import benchmark_model
from .constants import SEE_EXAMPLES_MESSAGE
from .utils.ipdb_hook import ipdb_sys_excepthook
from .utils.load_model import load_model
from .utils.setup_logging import setup_logging

# One needs to set their quantization backend engine to what is appropriate for their system.
# torch.backends.quantized.engine = 'x86'
torch.backends.quantized.engine = "qnnpack"


def main() -> None:
    args, device = parse_benchmark_args(logging)

    # THESE ELEMENTS MUST BE PROVIDED BY THE USER. See examples/mnist/benchmark.py for an example
    dataset, data_loader, modelArchitecture = None, None, None
    assert dataset is not None, f"Dataset is None. {SEE_EXAMPLES_MESSAGE}"
    assert data_loader is not None, f"Data loader is None. {SEE_EXAMPLES_MESSAGE}"
    assert (
        modelArchitecture is not None
    ), f"Model architecture is None. {SEE_EXAMPLES_MESSAGE}"

    # Load model
    load_start_time = time.perf_counter()
    model = load_model(
        args.model_path, device, logger=logging, modelArchitecture=modelArchitecture
    )
    load_end_time = time.perf_counter()
    logging.info(f"Model loading time: {load_end_time - load_start_time:.4f} seconds")

    # Benchmark the model
    benchmark_model(model, device, data_loader, args.n_samples, args, logging)


if __name__ == "__main__":
    # Adds an ipdb hook to the sys.excepthook, which will throw one into an ipdb shell when an
    # exception is raised
    ipdb_sys_excepthook()
    setup_logging()
    main()
