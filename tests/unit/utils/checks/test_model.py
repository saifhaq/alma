from alma.utils.checks.model import (
    check_model,
    check_is_local_function,
    is_local_function_name,
    is_picklable,
)

import pytest
import torch
from unittest.mock import patch


# Helper functions and classes for testing
def module_level_function(x):
    return x


class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


def create_local_function():
    def local_function(x):
        return x

    return local_function


def return_unpicklable_function():
    # Create an unpicklable attribute
    def unpicklable_function():
        """This function is not picklable as it is locally scoped"""
        pass
    return unpicklable_function


# Tests for check_model
def test_check_model_with_nn_module():
    """Test check_model with torch.nn.Module."""
    model = DummyModule()
    config = {"multiprocessing": True}

    with patch("logging.Logger.warning") as mock_warning:
        check_model(model, config)
        # Verify warning was logged about memory efficiency
        assert mock_warning.called
        assert "not memory efficient" in mock_warning.call_args[0][0]


def test_check_model_with_callable():
    """Test check_model with a valid callable."""
    config = {"multiprocessing": True}
    check_model(module_level_function, config)  # Should not raise any exceptions


def test_check_model_with_local_function():
    """Test check_model with a local function (should fail)."""
    local_func = create_local_function()
    config = {"multiprocessing": True}

    with pytest.raises(AssertionError):
        check_model(local_func, config)


def test_check_model_with_invalid_type():
    """Test check_model with invalid type."""
    config = {"multiprocessing": True}
    with pytest.raises(AssertionError):
        check_model("not_a_model", config)


def test_check_model_multiprocessing_disabled():
    """Test check_model with multiprocessing disabled."""
    model = DummyModule()
    config = {"multiprocessing": False}

    with patch("logging.Logger.warning") as mock_warning:
        check_model(model, config)
        # Verify no warning was logged
        assert not mock_warning.called


# Tests for check_is_local_function
def test_check_is_local_function_valid():
    """Test check_is_local_function with valid function."""
    check_is_local_function(
        module_level_function, "error"
    )  # Should not raise any exceptions


def test_check_is_local_function_invalid():
    """Test check_is_local_function with local function."""
    local_func = create_local_function()
    with pytest.raises(AssertionError):
        check_is_local_function(local_func, "error")


def test_check_is_local_function_unpicklable():
    """Test check_is_local_function with unpicklable function."""
    with pytest.raises(AssertionError):
        check_is_local_function(return_unpicklable_function(), "error")


# Tests for is_local_function_name
def test_is_local_function_name_module_level():
    """Test is_local_function_name with module-level function."""
    assert not is_local_function_name(module_level_function)


def test_is_local_function_name_local():
    """Test is_local_function_name with local function."""
    local_func = create_local_function()
    assert is_local_function_name(local_func)


def test_is_local_function_name_invalid_input():
    """Test is_local_function_name with invalid input."""
    with pytest.raises(ValueError):
        is_local_function_name("not_a_function")


# Tests for is_picklable
def test_is_picklable_valid():
    """Test is_picklable with picklable function."""
    assert is_picklable(module_level_function)


def test_is_picklable_invalid():
    """Test is_picklable with unpicklable function."""
    assert not is_picklable(return_unpicklable_function())


def test_is_picklable_local():
    """Test is_picklable with local function."""
    local_func = create_local_function()
    assert not is_picklable(local_func)


# Test default config behavior
def test_check_model_default_config():
    """Test check_model with empty config (should default to multiprocessing=True)."""
    model = DummyModule()
    with patch("logging.Logger.warning") as mock_warning:
        check_model(model, {})
        assert mock_warning.called
