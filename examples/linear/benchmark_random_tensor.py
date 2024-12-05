import logging

import torch

from alma.arguments.benchmark_args import parse_benchmark_args
from alma.benchmark_model import benchmark_model
from alma.utils.ipdb_hook import ipdb_sys_excepthook
from alma.utils.setup_logging import setup_logging

# One needs to set their quantization backend engine to what is appropriate for their system.
# torch.backends.quantized.engine = 'x86'
torch.backends.quantized.engine = "qnnpack"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def main() -> None:

    # Adds an ipdb hook to the sys.excepthook, which will throw one into an ipdb shell when an
    # exception is raised. Comment out to have the program crash as normal during an unhandled exception
    ipdb_sys_excepthook()

    # Set up logging. DEBUG level will also log the model graphs
    # A `setup_logging` function is provided for convenience, but one can use whatever logging one
    # wishes, or none.
    setup_logging(log_file=None, level="INFO")
    # setup_logging(log_file=None, level="WARNING")

    # Parse the benchmarking arguments
    args, device = parse_benchmark_args()

    # Create a random model
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 3),
        torch.nn.ReLU(),
    ).to(device)

    # Create a random tensor
    data = torch.rand(1, 512, 3).to(device)

    # Configuration for the benchmarking
    config = {
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
    }

    # Benchmark the model
    # Feeding in a tensor, and no dataloader, will cause the benchmark_model function to generate a
    # dataloader that provides random tensors of the same shape as `data`, which is used to
    # benchmark the model. As verbose logging is provided, it will log the benchmarking
    # at a DEBUG level.
    logging.info("Benchmarking model using random data")
    results = benchmark_model(model, config, args.conversions, data=data.squeeze())
    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    main()
