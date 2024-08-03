import time

import torch
import torch.quantization as tq
import logging
from tqdm import tqdm

def PTQ(model, device, dataloader):
    """
    Perform PTQ on the model using the given dataloader.
    """

    model.eval()
    # Ensure fake quantization enabled
    model.apply(tq.enable_fake_quant)

    # Enable PTQ observers
    for module in model.modules():
        if hasattr(module, 'observer_enabled') or hasattr(module, 'static_enabled'):
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
        if hasattr(module, 'observer_enabled') or hasattr(module, 'static_enabled'):
            module.disable_observer()
