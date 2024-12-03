import logging

import torch
from model.model import Net

from alma.arguments.benchmark_args import parse_benchmark_args
from alma.benchmark_model import benchmark_model
from alma.utils.ipdb_hook import ipdb_sys_excepthook
from alma.utils.setup_logging import setup_logging

# One needs to set their quantization backend engine to what is appropriate for their system.
# torch.backends.quantized.engine = 'x86'
torch.backends.quantized.engine = "qnnpack"


def main() -> None:
    args, device = parse_benchmark_args(logging)

    # Create random tensor (of MNIST image size)
    data = torch.rand(1, 3, 28, 28).to(device)

    # Load model (random weights)
    model = Net().to(device)

    # Benchmark the model
    # Feeding in a tensor, and no dataloader, will cause the benchmark_model function to generate a
    # dataloader that provides random tensors of the same shape as `data`, which is used to
    # benchmark the model.
    # NOTE: one needs to squeeze the data tensor to remove the batch dimension
    logging.info("Benchmarking model using random data")
    benchmark_model(model, device, args, args.conversions, data=data.squeeze())


if __name__ == "__main__":
    # Adds an ipdb hook to the sys.excepthook, which will throw one into an ipdb shell when an
    # exception is raised
    ipdb_sys_excepthook()
    setup_logging()
    main()
