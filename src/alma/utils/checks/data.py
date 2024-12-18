import torch
from torch.utils.data import DataLoader


def check_data_or_dataloader(data: torch.Tensor, data_loader: DataLoader) -> None:
    """
    Check that either data or data loader is provided, and not both.

    Args:
        data: The data to use for benchmarking (initialising the dataloader)
        data_loader: The DataLoader to get samples of data from

    Returns:
        None

    Raises:
        TypeError: If data is provided but is not a torch.Tensor, or if data_loader is provided but is not a DataLoader
        AssertionError: If neither or both inputs are provided
    """
    # Validate that data is a torch.Tensor if it's not None
    if data is not None and not isinstance(data, torch.Tensor):
        raise TypeError("The 'data' parameter must be of type torch.Tensor.")

    if data_loader is not None and not isinstance(data_loader, DataLoader):
        raise TypeError(
            "The 'data_loader' parameter must be of type torch.utils.data.DataLoader."
        )
    # Either the `data` Tensor must be provided, or a data loader
    if data is None:
        error_msg = "If data is not provided, the data_loader must be provided"
        assert data_loader is not None, error_msg
    if data_loader is None:
        error_msg = "If data_loader is not provided, the data tensor must be provided"
        assert data is not None, error_msg
    else:
        error_msg = "If a data loader is provided, the data tensor must be None"
        assert data is None, error_msg
