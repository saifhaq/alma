import argparse
import logging
import time
from typing import Callable

import torch
import torch.optim as optim
import torch.quantization as tq
from torch.optim.lr_scheduler import StepLR


def QAT(
    train: Callable,
    test: Callable,
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
):
    """
    Perform QAT on the model using the given dataloaders, and train adn test
    functions.
    This works in place on the model.

    Inputs:
    - train (function): The training function.
    - test (function): The testing function.
    - args (argparse.Namespace): The command line arguments.
    - model (torch.nn.Module): The model to quantize.
    - device (torch.device): The device to run the model on.
    - train_loader (DataLoader): The DataLoader to get a sample of data from.
    - test_loader (DataLoader): The DataLoader to get a sample of data from.

    Returns:
    None
    """

    # Reset the optimizer with included quantization parameters
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # NOTE: we may want seperate gamma, epochs and step_size for QAT
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Ensure fake quantization enabled
    model.apply(tq.enable_fake_quant)
    model.eval()

    # Disable PTQ observers
    for module in model.modules():
        if hasattr(module, "observer_enabled") or hasattr(module, "static_enabled"):
            module.disable_observer()

    # QAT training lopp
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    end_time = time.time()
    logging.info(f"QAT time: {end_time - start_time:.2f}s")
