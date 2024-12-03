from typing import Any, Callable

import torch
import torch.fx as fx
import torch_tensorrt


def get_tensorrt_dynamo_forward_call(model: Any, data: torch.Tensor) -> Callable:
    """
    Get the forward call function for the model using TensorRT.

    Inputs:
    - model (Any): The model to get the forward call for.
    - data (torch.Tensor): A sample of data to pass through the model.

    Outputs:
    - forward (Callable): The forward call function for the model.
    """
    # Compile the model using TensorRT
    trt_gm: fx.GraphModule = torch_tensorrt.dynamo.compile(model, data)

    return trt_gm.forward
