from typing import Callable

from torch.export.exported_program import ExportedProgram


def get_export_eager_forward_call(model: ExportedProgram) -> Callable:
    """
    Get eager mode forward call of export (shouldn't be much faster than basic eager
    mode, the only difference is we perhaps remove some of the Python wrapper functions
    around the Aten ops)

    Inputs:
    - model (ExportedProgram): The model to get the forward call for.

    Outputs:
    - forward (Callable): The forward call function for the model.

    """
    assert isinstance(
        model, ExportedProgram
    ), f"model must be of type ExportedProgram, got {type(model)}"
    forward = model.module().forward

    return forward
