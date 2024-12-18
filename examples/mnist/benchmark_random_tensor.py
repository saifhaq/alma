import logging
from typing import Any, Dict

import torch
from model.model import Net
from utils.file_utils import save_dict_to_json

from alma.arguments.benchmark_args import parse_benchmark_args
from alma.benchmark import BenchmarkConfig
from alma.benchmark.log import display_all_results
from alma.benchmark_model import benchmark_model
from alma.conversions.conversion_options import conversions_to_modes
from alma.utils.setup_logging import setup_logging

# One needs to set their quantization backend engine to what is appropriate for their system.
# torch.backends.quantized.engine = 'x86'
torch.backends.quantized.engine = "qnnpack"


def main() -> None:
    # Set up logging. DEBUG level will also log the internal conversion logs (where available), as well
    # as the model graphs. A `setup_logging` function is provided for convenience, but one can use
    # whatever logging one wishes, or none.
    setup_logging(log_file=None, level="INFO")

    # Parse the benchmarking arguments
    args, conversions = parse_benchmark_args()

    # Set the device one wants to benchmark on
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create random tensor (of MNIST image size)
    data = torch.rand(1, 3, 28, 28)

    # Load model (random weights)
    model = Net()

    # Configuration for the benchmarking
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
    # benchmark the model.
    # NOTE: one needs to squeeze the data tensor to remove the batch dimension
    logging.info("Benchmarking model using random data")
    results: Dict[str, Dict[str, Any]] = benchmark_model(
        model, config, conversions, data=data.squeeze()
    )

    # Display the results
    display_all_results(
        results, display_function=print, include_traceback_for_errors=False
    )

    # Save the results to JSON for easy CI integration
    save_dict_to_json(results, "result.json")


if __name__ == "__main__":
    main()
