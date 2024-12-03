import inspect
import re
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.fx as fx
from torch.utils.data import DataLoader


def process_batch(
    batch: torch.Tensor,
    model: Union[torch.nn.Module, fx.GraphModule],
    device: torch.device,
) -> np.ndarray:
    """
    Process a batch of data through a model.

    Inputs:
    - batch (torch.Tensor): A batch of data.
    - model (torch.nn.Module): The model to process the data with.
    - device (torch.device): The device to run the model on.

    Outputs:
    - preds (np.ndarray): The model predictions.
    """
    data = batch.to(device)
    output = model(data)
    preds = output.argmax(dim=1, keepdim=True)
    return preds.cpu().numpy()


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
