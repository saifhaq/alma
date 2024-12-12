import logging
from typing import Callable

import torch

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_torch_scripted_model_forward_call(
    model: torch.nn.Module,
) -> Callable:
    """
    Jit scripts the model, and returns a callable to the scripted model. Not all models
    are scriptable, as it requires that the model be statictically compilable, e.g. no
    graph breaks, data-dependent control flows, branching, etc.

    Inputs:
    - model (torch.nn.Module): The model to script.

    Returns:
    - forward (Callable): the forward call representing the scripted model.
    """
    scripted_model = torch.jit.script(model)
    return scripted_model.forward
