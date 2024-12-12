import logging
from typing import Callable

import torch

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_jit_traced_model_forward_call(
    model: torch.nn.Module,
    data: torch.Tensor,
) -> Callable:
    """
    Jit traces the model, and returns a callable to the traced model.

    Inputs:
    - model (torch.nn.Module): The model to trace.
    - data (torch.Tensor): Sample data to feed through the model for tracing.

    Returns:
    - forward (Callable): the forward call representing the traced model.
    """
    traced_model = torch.jit.trace(model, data)
    return traced_model.forward
