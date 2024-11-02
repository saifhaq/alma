import os
from typing import Callable

import torch._export
import torch._inductor
from torch.export.exported_program import ExportedProgram

CUDA_HOME = os.environ.get("CUDA_HOME")


def get_export_aot_inductor_forward_call(
    model: ExportedProgram, data: torch.Tensor
) -> Callable:
    """
    Get the forward call function for the model using AOTInductor.

    Inputs:
    - model (ExportedProram): The model to get the forward call for.
    - data (torch.Tensor): A sample of data to pass through the model.

    Outputs:
    - forward (Callable): The forward call function for the model.
    """
    if not CUDA_HOME:
        raise ValueError(
            "To use the AOTInductor option for export, set CUDA_HOME when you call the script, e.g. `CUDA_HOME='/usr/local/cuda' python benchmark.py`"
        )

    assert isinstance(
        model, ExportedProgram
    ), f"model must be of type ExportedProgram, got {type(model)}"

    # Compile the exported program to a `.so` using ``AOTInductor``
    # E.g. this can be run as a C++ file, or called from Python
    with torch.no_grad():
        so_path = torch._inductor.aot_compile(model.module(), (data,))

    # Load and run the .so file in Python.
    # To load and run it in a C++ environment, see:
    # https://pytorch.org/docs/main/torch.compiler_aot_inductor.html
    forward = torch._export.aot_load(so_path, device="cuda")

    return forward
