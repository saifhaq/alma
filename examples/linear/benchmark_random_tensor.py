import logging
from typing import Any, Dict

import torch

from alma.arguments.benchmark_args import parse_benchmark_args
from alma.benchmark import BenchmarkConfig
from alma.benchmark.log import display_all_results
from alma.benchmark_model import benchmark_model
from alma.utils.multiprocessing.lazyload import lazyload
from alma.utils.setup_logging import setup_logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def main() -> None:
    # Set up logging. DEBUG level will also log the internal conversion logs (where available), as well
    # as the model graphs. A `setup_logging` function is provided for convenience, but one can use
    # whatever logging one wishes, or none.
    setup_logging(log_file=None, level="INFO")

    # Parse the benchmarking arguments
    args, conversions = parse_benchmark_args()

    # Set the device one wants to benchmark on
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create a random model
    # NOTE: We use the lazyload wrapper to ensure this is only loaded once we call it within the `benchmark` function.
    # This is because under multiprocessing setups, where we use multiprocessing to ensure that each export/compilation
    # option occurs in a standalone environment to prevent global torch state contamination, there is a risk
    # that a model could be loaded here at initialisation, and again inside the child process. A lazyload ensures
    # that we only load it within the child processes, which uses half the memory for model storage compared to also
    # loading it here in the parent process.
    model = lazyload(
        lambda: torch.nn.Sequential(
            torch.nn.Linear(3, 3),
            torch.nn.ReLU(),
        )
    )

    # Create a random tensor
    data = torch.rand(1, 512, 3)

    # Set up the benchmarking configuration
    config = BenchmarkConfig(
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        device=device,
        multiprocessing=True,  # If True, we test each method in its own isolated environment,
        # which helps keep methods from contaminating the global torch state
        fail_on_error=False,  # If False, we fail gracefully and keep testing other methods
    )

    # Benchmark the model
    # Feeding in a tensor, and no dataloader, will cause the benchmark_model function to generate a
    # dataloader that provides random tensors of the same shape as `data`, which is used to
    # benchmark the model. As verbose logging is provided, it will log the benchmarking
    # at a DEBUG level.
    logging.info("Benchmarking model using random data")
    results: Dict[str, Dict[str, Any]] = benchmark_model(
        model, config, conversions, data=data.squeeze()
    )

    # Display the results
    display_all_results(
        results, display_function=print, include_traceback_for_errors=False
    )


if __name__ == "__main__":
    main()
