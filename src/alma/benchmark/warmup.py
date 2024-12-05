from typing import Callable

import torch
from torch.utils.data import DataLoader


def warmup(
    forward_call: Callable, data_loader: DataLoader, device: torch.device
) -> None:
    """
    Warms up the forward call for a few iterations.

    Inputs:
    - forward_call (Callable): The forward call we will be benchmarking.
    - data_loader (DataLoader): The data loader we use for the warmup. Should be the same as
        for the benchmarking.
    - device (torch.device): the device we are targetting.

    Outputs:
    None
    """
    counter = 0
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            _ = forward_call(data)
            counter += 1
            if counter > 10:
                return
