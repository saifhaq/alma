import logging
import time

import torch
import torch.quantization as tq
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def PTQ(
    model: torch.nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
):
    """
    Perform PTQ on the model using the given dataloader.
    This works in place on the model.

    Inputs:
    - model (torch.nn.Module): The model to quantize.
    - device (torch.device): The device to run the model on.
    - dataloader (DataLoader): The DataLoader to get a sample of data from.
    """

    model.eval()
    # Ensure fake quantization enabled
    model.apply(tq.enable_fake_quant)

    # Ensure model is on the correct device
    model.to(device)

    # Enable PTQ observers
    for module in model.modules():
        if hasattr(module, "observer_enabled") or hasattr(module, "static_enabled"):
            module.enable_observer()

    start_time = time.time()
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="PTQ"):
            data, target = data.to(device), target.to(device)
            _ = model(data)
    end_time = time.time()
    logging.info(f"PTQ time: {end_time - start_time:.2f}s")

    # Disable PTQ observers
    for module in model.modules():
        if hasattr(module, "observer_enabled") or hasattr(module, "static_enabled"):
            module.disable_observer()
