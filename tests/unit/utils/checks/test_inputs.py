import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from alma.benchmark.benchmark_config import BenchmarkConfig
from alma.conversions.conversion_options import ConversionOption
from alma.utils.checks.inputs import check_input_type


def dummy_callable(x):
    return x


def test_valid_inputs():
    """Test with valid inputs of each allowed type."""
    # Test cases with different valid combinations
    conversion_options = [
        ConversionOption(mode="EAGER"),
        ConversionOption(mode="EXPORT+EAGER"),
    ]
    test_cases = [
        # Case 1: nn.Module model
        {
            "model": torch.nn.Linear(10, 10),
            "config": BenchmarkConfig(),
            "conversions": conversion_options,
            "data": torch.randn(10, 10),
            "data_loader": None,
        },
        # Case 2: Callable model
        {
            "model": dummy_callable,
            "config": BenchmarkConfig(),
            "conversions": [],
            "data": None,
            "data_loader": DataLoader(TensorDataset(torch.randn(10, 10))),
        },
    ]

    for case in test_cases:
        check_input_type(**case)  # Should not raise any exceptions


def test_invalid_model():
    """Test with invalid model types."""
    invalid_models = ["string_model", 123, [1, 2, 3], {"key": "value"}]

    for model in invalid_models:
        with pytest.raises(
            AssertionError, match="The model must be a torch.nn.Module or callable"
        ):
            check_input_type(
                model=model, config={}, conversions=[], data=None, data_loader=None
            )


def test_invalid_config():
    """Test with invalid config types."""
    invalid_configs = ["string_config", 123, [1, 2, 3], torch.nn.Linear(10, 10)]

    for config in invalid_configs:
        with pytest.raises(
            AssertionError, match="The config must be of type BenchmarkConfig"
        ):
            check_input_type(
                model=torch.nn.Linear(10, 10),
                config=config,
                conversions=[],
                data=None,
                data_loader=None,
            )


def test_invalid_conversions():
    """Test with invalid conversions types."""
    invalid_conversions = [
        "string_conversions",
        123,
        {"key": "value"},
        torch.nn.Linear(10, 10),
    ]

    for conversions in invalid_conversions:
        with pytest.raises(AssertionError):
            check_input_type(
                model=torch.nn.Linear(10, 10),
                config=BenchmarkConfig(),
                conversions=conversions,
                data=None,
                data_loader=None,
            )


def test_invalid_data():
    """Test with invalid data types."""
    invalid_data = ["string_data", 123, [1, 2, 3], {"key": "value"}]

    for data in invalid_data:
        with pytest.raises(AssertionError, match="The data must be a torch.Tensor"):
            check_input_type(
                model=torch.nn.Linear(10, 10),
                config=BenchmarkConfig(),
                conversions=[],
                data=data,
                data_loader=None,
            )


def test_invalid_data_loader():
    """Test with invalid data_loader types."""
    invalid_loaders = [
        "string_loader",
        123,
        [1, 2, 3],
        {"key": "value"},
        torch.randn(10, 10),
    ]

    for loader in invalid_loaders:
        with pytest.raises(
            AssertionError, match="The data_loader must be a DataLoader"
        ):
            check_input_type(
                model=torch.nn.Linear(10, 10),
                config=BenchmarkConfig(),
                conversions=[],
                data=None,
                data_loader=loader,
            )


def test_none_values():
    """Test that None is accepted for appropriate parameters."""
    check_input_type(
        model=torch.nn.Linear(10, 10),
        config=BenchmarkConfig(),
        conversions=None,  # None should be valid for conversions
        data=None,  # None should be valid for data
        data_loader=None,  # None should be valid for data_loader
    )
