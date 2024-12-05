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


def inference_time_benchmarking(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    times: Dict[str, float],
) -> Tuple[Dict[int, int], Dict[str, float]]:
    """
    perform inference on the given model using the given data loader.
    this will return the counts of each digit in the dataset.

    inputs:
    - data_loader (Dataloader): the dataloader to get the data from.
    - model (torch.nn.Module): the model to perform inference with.
    - device (torch.device): the device to run the model on.
    - times (Dict[str, float]): the dictionary to store the times in.

    returns:
    - digit_counts (Dict[int, int]): the counts of each digit in the dataset.
    - times (Dict[str, float]): the updated dictionary of times.
    """
    digit_counts = [0] * 10
    times["inference_start_time"] = time.time()
    with torch.no_grad():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for batch, _ in tqdm(data_loader, desc="processing batches", unit="batch"):
                futures.append(executor.submit(process_batch, batch, model, device))
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="aggregating results",
                unit="batch",
            ):
                preds = future.result()
                for pred in preds:
                    digit_counts[pred.item()] += 1
    times["inference_end_time"] = time.time()
    return digit_counts, times
