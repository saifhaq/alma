import logging
from typing import Callable, Dict, Literal

import torch
from torch.export.exported_program import ExportedProgram

from .utils.check_type import check_model_type
from .utils.export import get_exported_model

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_export_compiled_forward_call(
    model: torch.nn.Module, data: torch.Tensor, backend: Literal[str] = "inductor"
) -> Callable:
    """
    Get the forward call function for the exported model using torch.compile.

    Inputs:
    - model (torch.nn.Module): The model to get the forward call for.
    - data (torch.Tensor): A sample of data to pass through the model.
    - backend (Literal[str]): The backend to use for torch.compile. Currently supported options in
        PyTorch are given by torch._dynamo.list_backends():
        ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']

    Outputs:
    - forward (Callable): The forward call function for the model.
    """

    # Export the model
    model = get_exported_model(model, data)

    check_model_type(model, ExportedProgram)

    # Set the compilation settings
    compile_settings: Dict[str, str] = get_compile_settings(backend)

    # Log graph
    if logger.root.level <= logging.DEBUG:
        logger.debug("Model graph:")
        # logger.debug(model.module().)

    # Compile the model, and get the forward call
    forward = torch.compile(model.module(), **compile_settings).forward

    with torch.no_grad():
        _ = forward(data)

    return forward


def get_compile_settings(backend: Literal[str]) -> Dict[str, str]:
    """
    Get the compilation settings for each torch.dynamo backend choice.

    Inputs:
    - backend (Literal[str]): The backend to use for torch.compile. Currently supported options in
        PyTorch are given by torch._dynamo.list_backends():
        ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']

    Outputs:
    - compile_settings (Dict[str, str]): The returned compilation settings.
    """
    match backend:
        case "inductor":
            compile_settings = {
                # Good for small models
                # 'mode': "reduce-overhead",
                # Slow to compile, but should find the "best" option
                "mode": "max-autotune",
                # Compiles entire program into 1 graph, but only works with a restricted subset of Python
                # (e.g. no data dependent control flow)
                "fullgraph": True,
                "backend": backend,
            }
        case "cudagraphs" | "openxla" | "tvm" | "onnxrt":
            compile_settings = {
                "fullgraph": True,
                "backend": backend,
            }
        case _:
            raise ValueError(f"{backend} is not a valid option")

    return compile_settings
