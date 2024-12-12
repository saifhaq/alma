import logging
from contextlib import contextmanager
from typing import Callable, Dict, Literal, Union

import torch
import torch._dynamo
import torch.fx as fx
from torch._dynamo.eval_frame import OptimizedModule
from torch.export.exported_program import ExportedProgram

from ...utils.setup_logging import suppress_output
from .utils.checks.type import check_model_type
from .utils.compile_settings import get_compile_settings

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_compiled_model(
    model: Union[torch.nn.Module, fx.GraphModule, ExportedProgram],
    data: torch.Tensor,
    backend: Literal[str],
) -> OptimizedModule:
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
        with torch.no_grad():
            model = torch.compile(model, **compile_settings)

            # Feed some data through the model to make sure it works
            _ = model(data)

    # Print model graph
    if logger.root.level <= logging.DEBUG:
        logger.debug("Model graph:")
        logger.debug(model.graph.print_tabular())

    check_model_type(model, OptimizedModule)

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
