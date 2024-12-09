import logging
from contextlib import contextmanager
from typing import Callable, Dict, Literal, Union

import torch
import torch._dynamo
import torch.fx as fx
from torch.export.exported_program import ExportedProgram

from ...utils.setup_logging import suppress_output
from .utils.checks.type import check_model_type

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
            # See below for documentation on torch.compile and a discussion of modes
            # https://pytorch.org/get-started/pytorch-2.0/#user-experience
            compile_settings = {
                # Good for small models
                # 'mode': "reduce-overhead",
                # Slow to compile, but should find the "best" option
                # "mode": "max-autotune",
                # More stable than max-autotune
                "mode": "reduce-overhead",
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


def get_compiled_model(
    model: Union[torch.nn.Module, fx.GraphModule, ExportedProgram],
    data: torch.Tensor,
    backend: Literal[str],
) -> Callable:
    """
    Compile the model using torch.compile.

    Inputs:
    - model (torch.nn.Module): The model to export
    - data (torch.Tensor): A sample of data to feed through the model
    - backend (Literal['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']): the backend for
        torch.compile to use.

    Outputs:
    model (torch._dynamo.eval_frame.OptimizedModule): The compiled model
    """
    logger.info(f"Running torch.compile [{backend} backend] on the model")
    check_model_type(model, (torch.nn.Module, fx.GraphModule, ExportedProgram))

    torch._dynamo.reset()

    # Set the compilation settings
    compile_settings: Dict[str, str] = get_compile_settings(backend)

    # Compile the model, with suppressed internal logs if logging is above Debug level.
    with suppress_output(logger.root.level >= logging.DEBUG):
        model = torch.compile(model, **compile_settings)

        # Test data can be fed through
        with torch.no_grad():
            _ = model(data)

    # Print model graph
    if logger.root.level <= logging.DEBUG:
        logger.debug("Model graph:")
        logger.debug(model.graph.print_tabular())

    check_model_type(model, torch._dynamo.eval_frame.OptimizedModule)

    return model


def get_compiled_model_forward_call(
    model: Union[torch.nn.Module, fx.GraphModule, ExportedProgram],
    data: torch.Tensor,
    backend: Literal[str],
) -> Callable:
    """
    Run torch.compile in the model, and get its forward call.

    Inputs:
    - model (torch.nn.Module): The model to export
    - data (torch.Tensor): A sample of data to feed through the model
    - backend (Literal['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']): the backend for
        torch.compile to use.

    Outputs:
    forward (Callable): the forward call of the compiled model.
    """

    model = get_compiled_model(model, data, backend)
    return model.forward


def get_compiled_forward_call_eager_fallback(
    model: Union[torch.nn.Module, fx.GraphModule, ExportedProgram],
    data: torch.Tensor,
    backend: Literal[str],
) -> Callable:
    """
    Run torch.compile in the model, and get its forward call. If dynamo errors occur, we fallback
    on eager mode by wrapping it with a context manager.

    Inputs:
    - model (torch.nn.Module): The model to export
    - data (torch.Tensor): A sample of data to feed through the model
    - backend (Literal['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']): the backend for
        torch.compile to use.

    Outputs:
    forward (Callable): the forward call of the compiled model.
    """
    with suppress_dynamo_errors():
        return get_compiled_model_forward_call(model, data, backend)


@contextmanager
def suppress_dynamo_errors():
    """
    Context manager to temporarily suppress torch._dynamo errors.
    This will have the execution fall back to eager mode.
    """
    # Store the original setting
    original_setting = torch._dynamo.config.suppress_errors
    try:
        # Set suppress_errors to True
        torch._dynamo.config.suppress_errors = True
        yield
    finally:
        # Restore the original setting
        torch._dynamo.config.suppress_errors = original_setting
