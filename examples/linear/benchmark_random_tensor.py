import logging
from typing import Any, Dict

import torch

from alma.arguments.benchmark_args import parse_benchmark_args
from alma.benchmark.log import display_all_results
from alma.benchmark_model import benchmark_model
from alma.utils.setup_logging import setup_logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def main() -> None:
    # Set up logging. DEBUG level will also log the internal conversion logs (where available), as well
    # as the model graphs. A `setup_logging` function is provided for convenience, but one can use
    # whatever logging one wishes, or none.
    setup_logging(log_file=None, level="INFO")

    # Parse the benchmarking arguments
    args, device = parse_benchmark_args()

    # Create a random model
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 3),
        torch.nn.ReLU(),
    )

    # Create a random tensor
    data = torch.rand(1, 512, 3)

    # Configuration for the benchmarking
    config = {
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "device": device,  # The device to benchmark on
        "multiprocessing": True,  # If True, we test each method in its own isolated environment,
        # which helps keep methods from contaminating the global torch state
        "fail_fast": False,  # If False, we fail gracefully and keep testing other methods
    }

    # Benchmark the model
    # Feeding in a tensor, and no dataloader, will cause the benchmark_model function to generate a
    # dataloader that provides random tensors of the same shape as `data`, which is used to
    # benchmark the model. As verbose logging is provided, it will log the benchmarking
    # at a DEBUG level.
    logging.info("Benchmarking model using random data")
    results: Dict[str, Dict[str, Any]] = benchmark_model(
        model, config, args.conversions, data=data.squeeze()
    )

    # Display the results
    display_all_results(
        results, display_function=print, include_traceback_for_errors=False
    )


if __name__ == "__main__":
    main()
