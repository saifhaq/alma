import pytest
import torch
from pydantic import ValidationError

from alma.utils.checks.config import check_config, BenchmarkConfig, is_valid_torch_device, check_consistent_batch_size


# Test check_consistent_batch_size function
def test_check_consistent_batch_size_valid():
    """Test check_consistent_batch_size with valid inputs"""
    # Should not raise any exception
    check_consistent_batch_size("COMPILE", 4, 2)
    check_consistent_batch_size("COMPILE", 100, 25)
    check_consistent_batch_size(
        "OTHER_METHOD", 5, 2
    )  # Non-COMPILE methods don't need to be exact


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
    # Test invalid string
    with pytest.raises(RuntimeError):
        is_valid_torch_device("invalid_device")

    # Test invalid type
    with pytest.raises(AssertionError):
        is_valid_torch_device(None)


# Test check_config function
def test_check_config_valid():
    """Test check_config with valid input"""
    config = {
        "n_samples": 4,
        "batch_size": 2,
        "device": "cpu",
        "multiprocessing": True,
        "fail_on_error": False,
    }

    # Should not raise any exception
    check_config(config)
    assert isinstance(config["device"], torch.device)


def test_check_config_missing_required():
    """Test check_config with missing required fields"""
    # Missing device
    config = {
        "n_samples": 4,
        "batch_size": 2,
    }
    with pytest.raises(AssertionError) as exc_info:
        check_config(config)
    assert "`device` must be provided in config" in str(exc_info.value)


def test_check_config_invalid_values():
    """Test check_config with invalid values"""
    # Invalid types for required fields
    config = {
        "n_samples": "four",  # Should be int
        "batch_size": 2,
        "device": "cpu",
    }
    with pytest.raises(ValidationError):
        check_config(config)


def test_check_config_optional_fields():
    """Test check_config with optional fields"""
    # Minimal valid config
    config = {
        "n_samples": 4,
        "batch_size": 2,
        "device": "cpu",
    }
    # Should not raise any exception
    check_config(config)

    # All optional fields
    config = {
        "n_samples": 4,
        "batch_size": 2,
        "device": "cpu",
        "multiprocessing": True,
        "fail_on_error": True,
    }
    # Should not raise any exception
    check_config(config)


# Fixture for common test configurations
@pytest.fixture
def valid_config():
    return {
        "n_samples": 4,
        "batch_size": 2,
        "device": "cpu",
        "multiprocessing": True,
        "fail_on_error": False,
    }


def test_benchmark_config_model():
    """Test BenchmarkConfig pydantic model directly"""
    # Valid config
    config = BenchmarkConfig(
        n_samples=4,
        batch_size=2,
        device=torch.device("cpu"),
        multiprocessing=True,
        fail_on_error=False,
    )
    assert config.n_samples == 4
    assert config.batch_size == 2
    assert isinstance(config.device, torch.device)
    assert config.multiprocessing is True
    assert config.fail_on_error is False

    # Invalid config (wrong types)
    with pytest.raises(ValidationError):
        BenchmarkConfig(
            n_samples="four",  # Should be int
            batch_size=2,
            device=66,
            multiprocessing=True,
            fail_on_error=False,
        )
