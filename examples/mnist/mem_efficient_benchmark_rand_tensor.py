import logging
from typing import Any, Dict

import torch
from model.model import Net

from alma.arguments.benchmark_args import parse_benchmark_args
from alma.benchmark import BenchmarkConfig
from alma.benchmark.log import display_all_results
from alma.benchmark_model import benchmark_model
from alma.utils.device import setup_device
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

    # Parse arguments and get conversions along with the default device
    args, conversions = parse_benchmark_args()

    # A provided util that will detect one's device and provide the appropriate torch.device object
    device = setup_device(
        None, allow_cuda=(not args.no_cuda), allow_mps=(not args.no_mps)
    )

    # It is a lot more memory efficient, if multi-processing is enabled, to create the model in a
    # callable function, which can be called later to create the model.
    # This allows us to initialise the model in each child process, rather than the parent
    # process. This is because the model is not pickled and sent to the child process (which would
    # require the program to sotre the model in memory twice), but rather created in the child
    # process. This is especially important if the model is large and two instances would not fit
    # on device.
    # We accomplish this via the lazyload function, which initializes a Lazyload instance, which will
    # cause the model to ony be loaded when called, not at initialisation.
# Get baseline memory

    import psutil
    import os
    def get_memory_usage():
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    baseline_memory = get_memory_usage()
    print(f"Baseline memory: {baseline_memory:.1f} MB\n")

# Method 1: Check the is_loaded() method
    print("=== Method 1: Using is_loaded() method ===")

    model = lazyload(Net)  # Note: corrected syntax
    print(f"Model created, is_loaded(): {model.is_loaded()}")
    print(f"Memory after creating lazy loader: {get_memory_usage():.1f} MB")
    # import ipdb, pprint; ipdb.set_trace();
    # model = model.load()
    # print(f"Memory after creating lazy loader: {get_memory_usage():.1f} MB")


    # Configuration for the benchmarking. Here we show of all of the options, including for device.
    # With `allow_device_override` we allow a device-specific conversion method to automtically assign
    # itself to that device, e.g. ONNX_CPU will automatically run on CPU even if device is CUDA.
    config = BenchmarkConfig(
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        multiprocessing=False,  # If True, we test each method in its own isolated environment,
        # which helps keep methods from contaminating the global torch state
        fail_on_error=False,  # If False, we fail gracefully and keep testing other methods
        non_blocking=False,  # If True, we don't block the main thread when transferring data from host to device
        # Device options:
        allow_device_override=not args.no_device_override,  # Allow device override for device-specific conversions
        allow_cuda=not args.no_cuda,  # True allows CUDA as an override option
        allow_mps=not args.no_mps,  # True allows MPS as an override option
        device=device,
    )

    # Prepare random data on the selected device for this conversion
    data = torch.rand(1, 3, 28, 28)

    # Benchmark the model, fed in as a callable
    # Feeding in a tensor, and no dataloader, will cause the benchmark_model function to generate a
    # dataloader that provides random tensors of the same shape as `data`, which is used to
    # benchmark the model.
    # NOTE: one needs to squeeze the data tensor to remove the batch dimension
    logging.info(
        "Benchmarking model using random data, passing in a callable to initialise the model"
    )
    results: Dict[str, Dict[str, Any]] = benchmark_model(
        model,
        config,
        conversions,
        data=data.squeeze(),
    )

    # Display results for this conversion (errors shown, but without tracebacks)
    display_all_results(
        results,
        display_function=print,
        include_errors=True,
        include_traceback_for_errors=False,
    )


if __name__ == "__main__":
    main()
