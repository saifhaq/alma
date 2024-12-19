import logging
from typing import Callable

import torch

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_fp16_eager_forward_call(
    model: torch.nn.Module,
) -> Callable:
    """
    Cast the model to half precision and return a forward call for the model.

    Inputs:
    - model (torch.nn.Module): The model to cast to half precision.

    Returns:
    - forward (Callable): the forward call representing the fp16 model.
    """
    model = model.half()

    def forward(data: torch.Tensor) -> torch.Tensor:
        """
        Forward call for the fp16 model. We make sure to cast the input data to half precision before
        passing it through the model.

        Inputs:
        - data (torch.Tensor): The input data to pass through the model.

        Outputs:
        - output (torch.Tensor): The output from the model.
        """
        data = data.half()
        return model(data)

    return forward
