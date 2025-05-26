from unittest.mock import patch

import pytest
import torch

from alma.benchmark.benchmark_config import BenchmarkConfig
from alma.utils.checks.model import check_model
from alma.utils.multiprocessing.lazyload import LazyLoader, lazyload


class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def base_config():
    """Fixture for creating a base BenchmarkConfig"""
    return BenchmarkConfig(n_samples=64, batch_size=32, device=torch.device("cpu"))


# Tests for check_model
def test_check_model_with_nn_module(base_config):
    """Test check_model with torch.nn.Module."""
    model = DummyModule()
    config = base_config.model_copy(update={"multiprocessing": True})

    with patch("logging.Logger.warning") as mock_warning:
        check_model(model, config)
        # Verify warning was logged about memory efficiency
        assert mock_warning.called
        assert (
            "Multiprocessing is enabled, but LazyLoader is not being used."
            in mock_warning.call_args[0][0]
        )


def test_check_model_with_lazyload_with_multiprocessing(base_config):
    """Test check_model with a lazyload wrapper with multiprocessing."""
    config = base_config.model_copy(update={"multiprocessing": True})
    model = lazyload(DummyModule())
    check_model(model, config)  # Should not raise any exceptions


def test_check_model_with_lazyload_no_multiprocessing(base_config):
    """Test check_model with a lazyload wrapper without multiprocessing."""
    model = lazyload(DummyModule())
    check_model(model, base_config)


def test_check_model_with_invalid_type(base_config):
    """Test check_model with invalid type."""
    config = base_config.model_copy(update={"multiprocessing": True})
    with pytest.raises(AssertionError):
        check_model("not_a_callable", config)


def test_check_model_multiprocessing_disabled(base_config):
    """Test check_model with multiprocessing disabled."""
    model = DummyModule()
    config = base_config.model_copy(update={"multiprocessing": False})

    with patch("logging.Logger.warning") as mock_warning:
        check_model(model, config)
        assert mock_warning.called


def test_check_model_multiprocessing_enabled(base_config):
    """Test check_model with multiprocessing enabled."""
    model = DummyModule()
    config = base_config.model_copy(update={"multiprocessing": True})

    with patch("logging.Logger.warning") as mock_warning:
        check_model(model, config)
        # Verify warning was logged
        assert mock_warning.called


def test_check_model_multiprocessing_and_lazyload_enabled(base_config):
    """Test check_model with multiprocessing and lazyload enabled."""
    model = lazyload(DummyModule())
    config = base_config.model_copy(update={"multiprocessing": True})

    with patch("logging.Logger.warning") as mock_warning:
        check_model(model, config)
        # Verify warning no was logged
        assert not mock_warning.called


# Test default config behavior
def test_check_model_default_config():
    """Test check_model with default config values."""
    model = DummyModule()
    config = BenchmarkConfig()  # Uses all default values

    with patch("logging.Logger.warning") as mock_warning:
        check_model(model, config)
        assert mock_warning.called
