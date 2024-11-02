import argparse
from typing import Callable

import torch
import torch.fx as fx

from quantization.PTQ import PTQ
from quantization.QAT import QAT
from quantization.qconfigs import fixed_0255
from quantization.utils import replace_node_with_target


def quantize_model(
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    train: Callable,
    test: Callable,
):
    """
    Quantize the model using PTQ and QAT. This works in place on the model.

    Inputs:
    args (argparse.Namespace): The command line arguments.
    model (torch.nn.Module): The model to quantize.
    device (torch.device): The device to run the model on.
    train_loader (DataLoader): The DataLoader to get a sample of data from.
    test_loader (DataLoader): The DataLoader to get a sample of data from.
    train (function): The training function.
    test (function): The testing function

    Returns:
    None
    """

    # Quantize the model
    # Eager mode quantization
    # model = model.eager_quantize()

    # FX graph mode quantization
    model: fx.GraphModule = model.fx_quantize()
    replace_node_with_target(model, "activation_post_process_0", fixed_0255())

    # Do PTQ
    PTQ(model, device, test_loader)

    # Do QAT
    QAT(train, test, args, model, device, train_loader, test_loader)
