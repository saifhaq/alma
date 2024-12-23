"""
Focused on https://github.com/pytorch/ao for on-GPU quantization in PyTorch.
"""

import logging
from typing import Callable

import torch
from torch._dynamo.eval_frame import OptimizedModule

from ...utils.setup_logging import suppress_output
from .compile import get_compiled_model
from .utils.checks.type import check_model_type

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_torchao_autoquant_model(
    model: torch.nn.Module,
    data: torch.Tensor,
    backend: str,
    use_autoquant_default: bool = True,
) -> OptimizedModule:
    """
    Get the model with torchao autoquantization applied. Uses torch.compile and max-autone by
    default.

    Inputs:
    - model (torch.nn.Module): The model to apply autoquantization to.
    - data (torch.Tensor): The input data to pass through the model.
    - backend (str): the backend for torch.compile to use.
    - use_autoquant_default (bool): Whether to use the default autoquantization settings.

    Outputs:
    - model (OptimizedModule): The model with autoquantization applied.
    """
    import torchao
    from torchao.quantization import DEFAULT_INT4_AUTOQUANT_CLASS_LIST

    # Compile the model using torch.compile
    compiled_model = get_compiled_model(model, data, backend=backend)

    # Autoquantize the model, with suppressed internal logs if logging is above Debug level.
    logger.info("Running torchao autoquant on the model")
    with suppress_output(logger.root.level >= logging.DEBUG):
        if use_autoquant_default:
            # Perform autoquantization and torch.compile with default settings
            model = torchao.autoquant(compiled_model)
        elif not use_autoquant_default:
            # Perform autoquantization and torch.compile with int4 support
            model = torchao.autoquant(
                compiled_model,
                qtensor_class_list=DEFAULT_INT4_AUTOQUANT_CLASS_LIST,
            )

        # Pass in an input which is used in order to pick fastest quantization operations
        # and apply torch compilation.
        _ = model(data)

    check_model_type(model, OptimizedModule)

    return model


def get_torchao_autoquant_forward_call(
    model: torch.nn.Module,
    data: torch.Tensor,
    backend: str,
    use_autoquant_default: bool = True,
) -> Callable:
    """
    Get the forward call for the model with torchao autoquantization applied. This runs on a
    compiled model.

    Inputs:
    - model (torch.nn.Module): The model to apply autoquantization to.
    - data (torch.Tensor): The input data to pass through the model.
    - backend (str): The backend for torch.compile to use.
    - use_autoquant_default (bool): Whether to use the default autoquantization settings.

    Outputs:
    - forward (Callable): The forward call for the model with autoquantization applied.
    """
    model = get_torchao_autoquant_model(model, data, backend, use_autoquant_default)

    return model.forward
