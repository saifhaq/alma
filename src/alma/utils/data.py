import torch
from torch.utils.data import DataLoader


def get_sample_data(data_loader: DataLoader, device: torch.device) -> torch.Tensor:
    """
    Get a sample of data from the DataLoader.

    Inputs:
    - data_loader (DataLoader): The DataLoader to get a sample of data from
    - device (torch.device): The device the data tensor should live on

    Outputs:
    - data (torch.Tensor): A sample of data from the DataLoader
    """
    for data, _ in data_loader:
        data = data.to(device)
        return data
    raise ValueError("DataLoader is empty")
