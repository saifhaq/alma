import torch
from torch.utils.data import DataLoader
from ..conversions.conversion_options import ConversionOption


def get_sample_data(data_loader: DataLoader, device: torch.device, conversion: ConversionOption) -> torch.Tensor:
    """
    Get a sample of data from the DataLoader.

    Inputs:
    - data_loader (DataLoader): The DataLoader to get a sample of data from
    - device (torch.device): The device the data tensor should live on
    - conversion (ConversionOption): The conversion method to benchmark.

    Outputs:
    - data (torch.Tensor): A sample of data from the DataLoader
    """
    for data in data_loader:
        if isinstance(data, torch.Tensor):
            data = data.to(device)
            assert (
                data.dtype == conversion.data_dtype
            ), f"The data loader dtype ({data.dtype}) does not match the conversion dtype ({conversion.data_dtype})."

        return data
    raise ValueError("DataLoader is empty")
