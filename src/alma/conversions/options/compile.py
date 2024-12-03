import logging
from typing import Callable, Union

import torch
import torch.fx as fx

from .utils.check_type import check_model_type

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_compiled_model_forward_call(
    model: Union[torch.nn.Module, fx.GraphModule],
    data: torch.Tensor,
) -> Callable:
    """
    Compile the model using torch.compile.

    Inputs:
    - model (torch.nn.Module): The model to export
    - data (torch.Tensor): A sample of data to feed through the model

    Outputs:
    model (torch._dynamo.eval_frame.OptimizedModule): The compiled model

    """
    logging.info("Running torch.compile on the model")
    check_model_type(model, (torch.nn.Module, fx.GraphModule))

    torch._dynamo.reset()

    # See below for documentation on torch.compile and a discussion of modes
    # https://pytorch.org/get-started/pytorch-2.0/#user-experience
    compile_settings = {
        # 'mode': "reduce-overhead",  # Good for small models
        "mode": "max-autotune",  # Slow to compile, but should find the "best" option
        "fullgraph": True,  # Compiles entire program into 1 graph, but comes with restricted Python
    }

    model = torch.compile(model, **compile_settings)

    with torch.no_grad():
        _ = model(data)

    # # Print model graph
    # logging.debug("Model graph:")
    # logging.debug(model.graph.print_tabular())

    check_model_type(model, torch._dynamo.eval_frame.OptimizedModule)

    return model.forward
