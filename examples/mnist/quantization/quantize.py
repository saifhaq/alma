import argparse
import logging
from typing import Callable, List, Union

import torch
import torch.fx as fx
from torch.optim.lr_scheduler import LRScheduler, StepLR

from alma.quantization.fx_quantize import fx_quantize
from alma.quantization.PTQ import PTQ
from alma.quantization.QAT import QAT
from alma.quantization.qconfigs import fixed_0255
from alma.quantization.utils import replace_node_with_target
from alma.utils.data import get_sample_data


def quantize_model(
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    logger: logging.Logger,
    train_loader: Union[torch.utils.data.DataLoader, None] = None,
    test_loader: Union[torch.utils.data.DataLoader, None] = None,
    train: Union[Callable, None] = None,
    test: Union[Callable, None] = None,
):
    """
    Quantize the model using FX Graph mode, then applying PTQ and QAT. This works in place on the
    model.

    Inputs:
    args (argparse.Namespace): The command line arguments.
    model (torch.nn.Module): The model to quantize.
    device (torch.device): The device to run the model on.
    logger (logging.Logger): The logger to use for logging.
    train_loader (Union[DataLoader, None]): The DataLoader to get samples of train data from.
    test_loader (Union[DataLoader, None]): The DataLoader to get samples of test data from.
    train (Union[function, None]): The training function.
    test (Union[function, None]): The testing function.

    Returns:
    None
    """

    # Quantize the model
    # Eager mode quantization
    # model = eager_quantize(model, fuse_layers=fuse_layers, logger=logger)

    assert train_loader is not None, "train_loader must be provided"
    data = get_sample_data(train_loader, device)

    # FX graph mode quantization
    fx_model: fx.GraphModule = fx_quantize(model, data, act_scale=2, logger=logger)

    # Optional graph manipulation here.
    logger.info("Replacing activation_post_process_0 with fixed_0255")
    replace_node_with_target(model, "activation_post_process_0", fixed_0255())

    # Do PTQ
    PTQ(fx_model, device, train_loader)

    # Do QAT
    assert train is not None, "train function must be provided for QAT"
    assert test is not None, "test function must be provided for QAT"
    assert test_loader is not None, "test_loader must be provided for QAT"

    # Reset the optimizer with included quantization parameters
    qat_optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    # NOTE: we may want seperate gamma, epochs and step_size for QAT
    qat_scheduler: LRScheduler = StepLR(qat_optimizer, step_size=1, gamma=args.gamma)
    QAT(
        train,
        test,
        args,
        fx_model,
        device,
        train_loader,
        test_loader,
        qat_optimizer,
        qat_scheduler,
    )


def fuse_layers(model: torch.nn.Module) -> torch.nn.Module:
    """
    Fuses the layers of the model, used in eager mode quantization.

    Inputs:
    model (torch.nn.Module): The model to fuse the layers of.

    Outputs:
    fused_model (torch.nn.Module): The model with fused layers.
    """
    list_of_layers_to_fuse: List[List[str]] = [
        ["conv1", "relu1"],
        ["conv2", "relu2"],
        ["fc1", "relu3"],
    ]
    fused_model = torch.quantization.fuse_modules(
        model,
        list_of_layers_to_fuse,
        inplace=False,
    )
    return fused_model
