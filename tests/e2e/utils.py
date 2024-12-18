import torch

from alma.benchmark.benchmark_config import BenchmarkConfig
from alma.benchmark_model import benchmark_model
from alma.conversions.conversion_options import mode_str_to_conversions


def simple_model_for_testing(conversions: list) -> None:
    """
    Test that eager mode benchmarking works successfully with a simple model.

    Inputs:
    - conversions (list): The list of conversion methods to benchmark.

    Outputs:
    None
    """
    # Create a random model
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 3),
        torch.nn.ReLU(),
    )

    # Create a random tensor
    data = torch.rand(1, 10, 3)

    # Configuration for the benchmarking
    config = BenchmarkConfig(
        n_samples=2,
        batch_size=2,
        device=torch.device("cpu"),
        multiprocessing=True,
        fail_on_error=True,
    )

    # Convert the modes to strings
    conversions = mode_str_to_conversions(conversions)

    # Benchmark the model
    results = benchmark_model(model, config, conversions, data=data.squeeze())

    return results
