import torch
from torch.utils.data import DataLoader


def check_data_or_dataloader(data: torch.Tensor, data_loader: DataLoader) -> None:
    """
    Check that either data or data loader is provided, and not both.

    Inputs:
    - data (torch.Tensor): The data to use for benchmarking (initialising the dataloader).
    - data_loader (DataLoader): The DataLoader to get samples of data from.

    Outputs:
    None
    """

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
