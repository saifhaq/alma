import concurrent.futures
import logging
import time
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.fx as fx
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
