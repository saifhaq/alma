import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from alma.utils.checks.data import check_data_or_dataloader


def test_data_only():
    """Test when only data tensor is provided."""
    data = torch.randn(10, 3)  # Sample tensor
    check_data_or_dataloader(data=data, data_loader=None)


def test_dataloader_only():
    """Test when only DataLoader is provided."""
    # Create a dummy DataLoader
    dataset = TensorDataset(torch.randn(10, 3))
    data_loader = DataLoader(dataset, batch_size=2)
    check_data_or_dataloader(data=None, data_loader=data_loader)


def test_both_provided():
    """Test that an error is raised when both data and DataLoader are provided."""
    data = torch.randn(10, 3)
    dataset = TensorDataset(torch.randn(10, 3))
    data_loader = DataLoader(dataset, batch_size=2)

    with pytest.raises(
        AssertionError,
        match="If a data loader is provided, the data tensor must be None",
    ):
        check_data_or_dataloader(data=data, data_loader=data_loader)


def test_neither_provided():
    """Test that an error is raised when neither data nor DataLoader is provided."""
    with pytest.raises(
        AssertionError,
        match="If data is not provided, the data_loader must be provided",
    ):
        check_data_or_dataloader(data=None, data_loader=None)


def test_wrong_types():
    """Test that the function accepts only the correct types."""
    # Test with wrong type for data
    with pytest.raises(TypeError):
        check_data_or_dataloader(data=[1, 2, 3], data_loader=None)

    # Test with wrong type for data_loader
    with pytest.raises(TypeError):
        check_data_or_dataloader(data=None, data_loader=[1, 2, 3])
