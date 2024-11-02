import argparse
import logging
import time
from typing import Dict, Tuple

import torch

from model.model import Net


def load_model(
    model_path: str,
    device: torch.device,
    times: Dict[str, float],
    logging: logging.Logger,
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    """
    Load the model from the given path and return it. Also return the time
    it took to load the model.

    Inputs:
    - model_path (str): The path to the model to load.
    - device (torch.device): The device to run the model on.
    - times (Dict[str, float]): The dictionary to store the times in.
    - logging (logging.Logger): The logger to use for logging.

    Returns:
    - model (torch.nn.Module): The loaded model.
    - times (Dict[str, float]): The updated dictionary of times.
    """

    logging.info(f"Loading model from {model_path}")
    times["model_load_start_time"] = time.time()
    if "scripted" in model_path:
        model = torch.jit.load(model_path).to(device)
    else:
        model = Net()
        model.load_state_dict(torch.load(model_path))
        model.to(device)
    model.eval()
    times["model_load_end_time"] = time.time()

    return model, times
