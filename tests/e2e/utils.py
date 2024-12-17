import torch

from alma.benchmark_model import benchmark_model


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
    data = torch.rand(1, 512, 3)

    # Configuration for the benchmarking
    config = {
        "n_samples": 4,
        "batch_size": 2,
        "device": torch.device("cpu"),  # The device to benchmark on
        "multiprocessing": True,  # If True, we test each method in its own isolated environment,
        # which helps keep methods from contaminating the global torch state
        "fail_on_error": True,  # If False, we fail gracefully and keep testing other methods
    }

    # Benchmark the model
    results = benchmark_model(model, config, conversions, data=data.squeeze())

    return results
