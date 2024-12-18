import logging
from typing import Any, Dict

import torch
from model.model import Net

from alma.arguments.benchmark_args import parse_benchmark_args
from alma.benchmark.benchmark_config import BenchmarkConfig
from alma.benchmark.log import display_all_results
from alma.benchmark_model import benchmark_model
from alma.utils.device import setup_device
from alma.utils.setup_logging import setup_logging

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

    # Create random input tensor on the default device
    current_device = setup_device(None, not args.no_cuda, not args.no_mps)

    # Iterate through each selected conversion (which is now a ConversionOption typed dict)
    for conversion in conversions:

        # mode = conversion["mode"]

        config = BenchmarkConfig(
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            multiprocessing=True,  # Test each method in isolation
            fail_on_error=False,  # Fail gracefully and continue testing other methods if one fails
            allow_device_override=not args.no_device_override,  # Allow device override for specific conversions, ie ONNX_CPU
            allow_cuda=not args.no_cuda,  # Allow CUDA if not disabled
            allow_mps=not args.no_mps,  # Allow MPS if not disabled
            device=current_device,
        )

        # Benchmark the model, fed in as a callable
        # Feeding in a tensor, and no dataloader, will cause the benchmark_model function to generate a
        # dataloader that provides random tensors of the same shape as `data`, which is used to
        # benchmark the model.
        # NOTE: one needs to squeeze the data tensor to remove the batch dimension

        data = torch.rand(1, 3, 28, 28)
        # Prepare random data on the selected device for this conversion
        random_data = data

        # Benchmark the model with the current conversion
        # Here we pass the conversion directly as `ConversionOption` and
        # assume benchmark_model will be updated to handle this format.
        results: Dict[str, Dict[str, Any]] = benchmark_model(
            model_init,
            config,
            [conversion],
            data=random_data.squeeze(),
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
