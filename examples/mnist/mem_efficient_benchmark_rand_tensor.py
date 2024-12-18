import logging
from typing import Any, Dict

import torch
from model.model import Net

from alma.arguments.benchmark_args import parse_benchmark_args
from alma.benchmark import BenchmarkConfig
from alma.benchmark.log import display_all_results
from alma.benchmark_model import benchmark_model
from alma.utils.setup_logging import setup_logging
from alma.utils.device import setup_device

# One needs to set their quantization backend engine to what is appropriate for their system.
# torch.backends.quantized.engine = 'x86'
torch.backends.quantized.engine = "qnnpack"


# It is a lot more memory efficienct, if multi-processing is enabled, to create the model in a
# callable function, which can be called later to create the model.
# This allows us to initialise the model in each child process, rather than the parent
# process. This is because the model is not pickled and sent to the child process (which would
# require the program to sotre the model in memory twice), but rather created in the child
# process. This is especially important if the model is large and two instances would not fit
# on device.
def model_init() -> torch.nn.Module:
    """
    A callable that returns the model to be benchmarked. This allows us to initialise the model
    at a later date, which is useful when using multiprocessing (in turn used to isolate each method
    in its own process, keeping them from contaminating the global torch state). By initialising
    the model inside the child process, we avoid having two instances of the model in memory at
    once.

    NOTE: THIS HAS TO BE DEFINED AT THE MODULE LEVEL, NOT NESTED INSIDE ANY FUNCTION. This is so
    that it is pickle-able, necessary for it to be passed to multi-processing.
    """
    return Net()


def main() -> None:

    # Set up logging. DEBUG level will also log the internal conversion logs (where available), as well
    # as the model graphs. A `setup_logging` function is provided for convenience, but one can use
    # whatever logging one wishes, or none.
    setup_logging(log_file=None, level="INFO")

    # Parse arguments and get conversions along with the default device
    args, conversions = parse_benchmark_args()

    # A provided util that will detect one's device and provide the appropriate torch.device object
    device = setup_device()

    # Configuration for the benchmarking. Here we show of all of the options, including for device.
    # With `allow_device_override` we allow a device-specific conversion method to automtically assign
    # itself to that device, e.g. ONNX_CPU will automatically run on CPU even if device is CUDA.
    config = BenchmarkConfig(
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        multiprocessing=True,  # If True, we test each method in its own isolated environment,
        # which helps keep methods from contaminating the global torch state
        fail_on_error=False,  # If False, we fail gracefully and keep testing other methods
        # Device options:
        allow_device_override=not args.no_device_override,  # Allow device override for device-specific conversions
        allow_cuda=not args.no_cuda,  # True allows CUDA as an override option
        allow_mps=not args.no_mps,  # True allows MPS as an override option
        device=device,
    )

    # Iterate through each selected conversion. we do this just tp show that this is an alternate
    # way to benchmark across multiple converisons, rather than passing in a list of conversions
    for conversion in conversions:

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
            model_init,
            config,
            [conversion],
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
