import pytest
import torch
from pydantic import ValidationError

from alma.benchmark.benchmark_config import BenchmarkConfig
from alma.utils.checks.config import (
    check_config,
    check_consistent_batch_size,
    is_valid_torch_device,
)


# Test check_consistent_batch_size function
def test_check_consistent_batch_size_valid():
    """Test check_consistent_batch_size with valid inputs"""
    # Should not raise any exception
    check_consistent_batch_size("COMPILE", 4, 2)
    check_consistent_batch_size("COMPILE", 100, 25)
    check_consistent_batch_size("OTHER_METHOD", 5, 2)


def test_check_consistent_batch_size_invalid():
    """Test check_consistent_batch_size with invalid inputs"""
    with pytest.raises(ValueError) as exc_info:
        check_consistent_batch_size("COMPILE", 5, 2)
    assert "n_samples must be a multiple of batch_size" in str(exc_info.value)


# Test is_valid_torch_device function
def test_is_valid_torch_device_str():
    """Test is_valid_torch_device with valid string inputs"""
    valid_devices = ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]

    for device_str in valid_devices:
        device = is_valid_torch_device(device_str)
        assert isinstance(device, torch.device)
        assert str(device) == device_str


def test_is_valid_torch_device_device():
    """Test is_valid_torch_device with torch.device inputs"""
    cpu_device = torch.device("cpu")
    result = is_valid_torch_device(cpu_device)
    assert result == cpu_device


def test_is_valid_torch_device_invalid():
    """Test is_valid_torch_device with invalid inputs"""
    with pytest.raises(RuntimeError):
        is_valid_torch_device("invalid_device")

    with pytest.raises(AssertionError):
        is_valid_torch_device(None)


# Test BenchmarkConfig model and validation
def test_benchmark_config_defaults():
    """Test BenchmarkConfig with default values"""
    config = BenchmarkConfig()
    assert config.n_samples == 128
    assert config.batch_size == 128
    assert config.multiprocessing is True
    assert config.fail_on_error is True
    assert config.allow_device_override is True
    assert config.allow_cuda is True
    assert config.allow_mps is True
    assert isinstance(config.device, torch.device)


def test_benchmark_config_custom_values():
    """Test BenchmarkConfig with custom values"""
    config = BenchmarkConfig(
        n_samples=64, batch_size=32, multiprocessing=False, device=torch.device("cpu")
    )
    assert config.n_samples == 64
    assert config.batch_size == 32
    assert config.multiprocessing is False
    assert isinstance(config.device, torch.device)
    assert str(config.device) == "cpu"


def test_benchmark_config_validation():
    """Test BenchmarkConfig validation"""
    # Test invalid n_samples
    with pytest.raises(ValidationError) as exc_info:
        BenchmarkConfig(n_samples=0)
    assert "n_samples" in str(exc_info.value)
    assert "greater than" in str(exc_info.value)

    # Test invalid batch_size
    with pytest.raises(ValidationError) as exc_info:
        BenchmarkConfig(batch_size=-1)
    assert "batch_size" in str(exc_info.value)
    assert "greater than" in str(exc_info.value)


def test_benchmark_config_device_selection():
    """Test device selection behavior"""
    # Test with CUDA disabled
    config = BenchmarkConfig(allow_cuda=False, allow_mps=False)
    assert str(config.device) == "cpu"

    # Test with explicit device
    config = BenchmarkConfig(device=torch.device("cpu"))
    assert str(config.device) == "cpu"


def test_check_config():
    """Test check_config function with valid and invalid inputs"""
    # Valid config should not raise
    valid_config = BenchmarkConfig(
        n_samples=64, batch_size=32, device=torch.device("cpu")
    )
    check_config(valid_config)

    # Invalid config should raise ValidationError
    with pytest.raises(ValidationError):
        check_config(
            {"n_samples": -1, "batch_size": 32, "device": "cpu"}  # Invalid value
        )


@pytest.fixture
def valid_config():
    """Fixture for common test configurations"""
    return BenchmarkConfig(
        n_samples=64,
        batch_size=32,
        device=torch.device("cpu"),
        multiprocessing=True,
        fail_on_error=False,
    )
