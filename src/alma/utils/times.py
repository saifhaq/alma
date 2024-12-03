import concurrent.futures
import time
from typing import Dict, Tuple

import torch
from tqdm import tqdm

from .data import process_batch

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def log_benchmark_times(
    logging: logging.Logger,
    times: Dict[str, float],
):
    """
    Log the benchmark times.

    Inputs:
    - logging (logging.Logger): The logger to use for logging.
    - times (Dict[str, float]): The dictionary of times.

    Returns:
    - total_execution_time (float): The total execution time.
    - inference_time (float): The inference time.
    - model_load_time (float): The model load time.
    - image_paths_gather_time (float): The image paths gather time.
    """
    total_execution_time = times["inference_end_time"] - times["model_load_start_time"]
    inference_time = times["inference_end_time"] - times["inference_start_time"]
    model_load_time = times["model_load_end_time"] - times["model_load_start_time"]

    image_paths_gather_time = (
        times["image_paths_gather_end_time"] - times["image_paths_gather_start_time"]
    )

    logging.info(f"model loading time: {model_load_time:.2f} seconds")
    logging.info(f"image paths gathering time: {image_paths_gather_time:.2f} seconds")
    logging.info(f"inference time: {inference_time:.2f} seconds")
    logging.info(f"total execution time: {total_execution_time:.2f} seconds")
    return (
        total_execution_time,
        inference_time,
        model_load_time,
        image_paths_gather_time,
    )


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
