import logging
from typing import Callable

import torch

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_bfp16_eager_forward_call(
    model: torch.nn.Module,
) -> Callable:
    """
    Cast the model to bf16 precision and return a forward call for the model.

    Inputs:
    - model (torch.nn.Module): The model to cast to bf16 precision.

    Returns:
    - forward (Callable): the forward call representing the fp16 model.
    """
    model = model.to(torch.bfloat16)

    def forward(data: torch.Tensor) -> torch.Tensor:
        """
        Forward call for the bf16 model. We make sure to cast the input data to bf16 precision before
        passing it through the model.

        Inputs:
        - data (torch.Tensor): The input data to pass through the model.

        Outputs:
        - output (torch.Tensor): The output from the model.
        """
        data = data.to(torch.bfloat16)
        return model(data)

    return forward
