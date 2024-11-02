from typing import Union

import torch
import torch.fx as fx
from torch.utils.data import DataLoader

from data.utils import get_sample_data


def get_compiled_model(
    model: Union[torch.nn.Module, fx.GraphModule],
    data_loader: DataLoader,
    device: torch.device,
    logging,
) -> torch._dynamo.eval_frame.OptimizedModule:
    """
    Compile the model using torch.compile.

    Inputs:
    - model (torch.nn.Module): The model to export
    - data_loader (DataLoader): The DataLoader to get a sample of data from
    - device (torch.device): The device to run the model on
    - logging: The logger to use for logging

    Outputs:
    model (torch._dynamo.eval_frame.OptimizedModule): The compiled model

    """
    logging.info("Running torch.compile on the model")
    assert isinstance(
        model, (torch.nn.Module, fx.GraphModule)
    ), f"model must be of type torch.nn.Module or fx.GraphModule, got {type(model)}"

    torch._dynamo.reset()

    # See below for documentation on torch.compile and a discussion of modes
    # https://pytorch.org/get-started/pytorch-2.0/#user-experience
    compile_settings = {
        # 'mode': "reduce-overhead",  # Good for small models
        "mode": "max-autotune",  # Slow to compile, but should find the "best" option
        "fullgraph": True,  # Compiles entire program into 1 graph, but comes with restricted Python
    }

    model = torch.compile(model, **compile_settings)

    data = get_sample_data(data_loader, device)
    _ = model(data)

    # Print model graph
    model.graph.print_tabular()

    assert isinstance(
        model, torch._dynamo.eval_frame.OptimizedModule
    ), f"model must be of type OptimizedModule, got {type(model)}"

    return model


def get_export_compiled_forward_call(
    model: torch._dynamo.eval_frame.OptimizedModule, data: torch.Tensor
):
    """
    Get the forward call function for the model using torch.compile.

    Inputs:
    - model (torch._dynamo.eval_frame.OptimizedModule): The model to get the forward call for.
    - data (torch.Tensor): A sample of data to pass through the model.

    Outputs:
    - forward (Callable): The forward call function for the model.
    """

    # NOTE: not sure if this is correct. It may be correct for compile, or for compile+export.
    # If the latter, move this into the export sub directory.
    import ipdb

    ipdb.set_trace()
    assert isinstance(
        model, torch._dynamo.eval_frame.OptimizedModule
    ), f"model must be of type OptimizedModule, got {type(model)}"

    # Set the comilation settings
    compile_settings = {
        # Good for small models
        # 'mode': "reduce-overhead",
        # Slow to compile, but should find the "best" option
        "mode": "max-autotune",
        # Compiles entire program into 1 graph, but only works with a restricted subset of Python
        # (e.g. no data dependent control flow)
        "fullgraph": True,
    }

    # Compile the model, and get the forward call
    forward = torch.compile(model.module(), **compile_settings)  # , backend="inductor")

    return forward
