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
        with torch.no_grad():
            model = torch.compile(model, **compile_settings)

            # Feed some data through the model to make sure it works
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


def get_compile_settings(backend: Literal[str] = "inductor-default") -> Dict[str, str]:
    """
    Get the compilation settings for each torch.dynamo backend choice.

    Inputs:
    - backend (Literal[str]): The backend to use for torch.compile. Currently supported options in
        PyTorch are given by torch._dynamo.list_backends():
        ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']
        However, we have also split 'inductor' into 'inductor-default', 'inductor-max-autotune', and
        'inductor-reduce-overhead'. See here for an explanation of each:
        https://pytorch.org/get-started/pytorch-2.0/#user-experience

    Outputs:
    - compile_settings (Dict[str, str]): The returned compilation settings.
    """
    match backend:
        case "inductor-default":
            compile_settings = {
                "mode": "default",
                "backend": "inductor",
            }
        case "inductor-reduce-overhead":
            compile_settings = {
                "mode": "reduce-overhead",
                "fullgraph": True,
                "backend": "inductor",
            }

        case "inductor-max-autotune":
            import torch._inductor.config

            torch._inductor.config.max_autotune_gemm_backends = "ATEN,TRITON,CPP"
            torch._inductor.config.max_autotune = True
            # TRITON backend fails with conv2d layers during max-autotune.
            torch._inductor.config.max_autotune_conv_backends = "ATEN"
            compile_settings = {
                "mode": "max-autotune",
                "fullgraph": True,
                "backend": "inductor",
                # Options can also be fed in via a dict, see here for options: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
                # However, I found this to be slower than using the "preset" options, so I think feeding
                # in any options will prevent any tuning from happening. This is why the settings
                # are controlled via global settings above.
                # options: {},
            }

        case "cudagraphs" | "openxla" | "tvm":
            compile_settings = {
                "fullgraph": True,
                "backend": backend,
            }
        case "onnxrt":
            # TODO: debug why CUDA does not work on this.
            # See here for the accepted options for ONNXRT backend:
            # https://github.com/pytorch/pytorch/blob/05c1f37188b1923ee3e48634cb1aaa3e976e2d0f/torch/onnx/_internal/onnxruntime.py#L689
            from torch.onnx._internal.onnxruntime import OrtBackendOptions

            # Configure the ORT backend options. See here for options:
            # https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
            ort_options = OrtBackendOptions(
                # Specify preferred execution providers in priority order
                preferred_execution_providers=[
                    # ("CUDAExecutionProvider", {"device_id": 0}),  # For GPU execution
                    "CPUExecutionProvider",  # Fallback to CPU
                ],
                # Automatically infer execution providers from input/output devices
                infer_execution_providers=True,
                # Optional: Set to True if you want to pre-allocate output memory
                preallocate_output=True,
                # Disable AOT autograd for training support
                use_aot_autograd=False,
            )
            compile_settings = {
                "fullgraph": True,
                "backend": backend,
                "options": ort_options,
            }
        case _:
            raise ValueError(f"{backend} is not a valid option")

    return compile_settings


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
