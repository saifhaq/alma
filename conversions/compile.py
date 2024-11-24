from typing import Callable, Union

import torch
import torch.fx as fx

from .utils import check_model_type

def get_compiled_model_forward_call(
    model: Union[torch.nn.Module, fx.GraphModule],
    data: torch.Tensor,
    logging,
) -> Callable:
    """
    Compile the model using torch.compile.

    Inputs:
    - model (torch.nn.Module): The model to export
    - data (torch.Tensor): A sample of data to feed through the model
    - logging: The logger to use for logging

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

    _ = model(data)

    # Print model graph
    model.graph.print_tabular()

    check_model_type(model, torch._dynamo.eval_frame.OptimizedModule)

    return model.forward
