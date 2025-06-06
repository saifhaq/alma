import logging
from typing import Any, Dict

import torch
from model.model import Net
from utils.file_utils import save_dict_to_json

from alma.benchmark import BenchmarkConfig
from alma.benchmark.log import display_all_results
from alma.benchmark_model import benchmark_model
from alma.utils.multiprocessing.lazyload import lazyload
from alma.utils.setup_logging import setup_logging

# One needs to set their quantization backend engine to what is appropriate for their system.
# torch.backends.quantized.engine = 'x86'
torch.backends.quantized.engine = "qnnpack"


def main() -> None:
    # Set up logging. DEBUG level will also log the internal conversion logs (where available), as well
    # as the model graphs. A `setup_logging` function is provided for convenience, but one can use
    # whatever logging one wishes, or none.
    setup_logging(log_file=None, level="INFO")

    # Set the device one wants to benchmark on
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create random tensor (of MNIST image size)
    data = torch.rand(1, 3, 28, 28)

    # Load model (random weights)
    # NOTE: Here we don't use the lazyload decorator on purpose, which will cause the model to be initialised immediately.
    # Since we are using Multiprocessing, this will log a warning about inefficient memory usage.
    model = Net()

    # Configuration for the benchmarking
    config = BenchmarkConfig(
        n_samples=2048,
        batch_size=64,
        device=device,
        multiprocessing=True,  # If True, we test each method in its own isolated environment,
        # which helps keep methods from contaminating the global torch state
        fail_on_error=False,  # If False, we fail gracefully and keep testing other methods
        non_blocking=False,  # If True, we don't block the main thread when transferring data from host to device
    )

    # Hard-code a list of options. These can be provided as a list of strings, or a list of ConversionOption objects
    conversions = [
        "EAGER",
        "JIT_TRACE",
        "TORCH_SCRIPT",
        "COMPILE_INDUCTOR_DEFAULT",
        "COMPILE_OPENXLA",
        "COMPILE_INDUCTOR_MAX_AUTOTUNE",
        "COMPILE_CUDAGRAPHS",
    ]
    # conversions = [ConversionOption["EAGER"], ConversionOption["JIT_TRACE"], ConversionOption["TORCH_SCRIPT"], ConversionOption["COMPILE_INDUCTOR_DEFAULT"]]

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
