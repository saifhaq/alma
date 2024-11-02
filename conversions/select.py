import argparse
import logging
from typing import Any, Callable

import torch

from conversions import (
    get_export_aot_inductor_forward_call,
    get_export_compiled_forward_call,
    get_export_eager_forward_call,
    get_onnx_forward_call,
    get_tensorrt_dynamo_forward_call,
)


def select_forward_call_function(
    model: Any, args: argparse.Namespace, data: torch.Tensor, logging: logging.Logger
) -> Callable:
    """
    Get the forward call function for the model. The complexity is because there are multiple
    ways to export the model, and the forward call is different for each.

    Inputs:
    - model (Any): The model to get the forward call for.
    - args (argparse.Namespace): The command line arguments.
    - data (torch.Tensor): A sample of data to pass through the model, which may be needed for
    some of the export methods.
    - logging (logging.Logger): The logger to use for logging.

    Outputs:
    - forward (Callable): The forward call function for the model.
    """
    if args.export:
        if args.tensorrt:
            forward = get_tensorrt_dynamo_forward_call(model, data)
        else:
            # option = 'COMPILE'
            option = "AOTInductor"
            # option = 'EAGER'
            match option:
                case "COMPILE":
                    # This is torch compile, fed into torch export
                    forward = get_export_compiled_forward_call(model, data)

                case "AOTInductor":
                    forward = get_export_aot_inductor_forward_call(model, data)

                case "EAGER":
                    forward = get_export_eager_forward_call(model)
    elif args.compile:
        # Torch.compile without export. This is a regular forward call
        forward = model.forward

    elif args.onnx:
        forward = get_onnx_forward_call(model, data, logging)

    else:
        # Regular eager model forward call
        forward = model.forward

    return forward
