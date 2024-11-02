import os
import time
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.fx as fx
from torch.utils.data import DataLoader


def gather_image_paths(
    root_dir: str, times: Dict[str, float]
) -> Tuple[List[str], Dict[str, float]]:
    """
    Gather image paths from a directory.

    Inputs:
    - root_dir (str): The root directory to search for images
    - times (Dict[str, float]): A dictionary to store timing information

    Outputs:
    - image_paths (List[str]): A list of image paths
    - times (Dict[str, float]): An updated dictionary, with stored timing information
    """
    times["image_paths_gather_start_time"] = time.time()
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, file))

    times["image_paths_gather_end_time"] = time.time()
    return image_paths, times


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
