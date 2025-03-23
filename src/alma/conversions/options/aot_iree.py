from typing import Callable

import torch
import iree.runtime as ireert
import iree.turbine.aot as aot

def get_aot_iree_forward_call(
    model: torch.nn.Module, data: torch.Tensor
) -> Callable:
    """
    Get an IREE-compiled model via the AOT export path.

    Inputs:
    - model (torch.nn.Module): The model to get the forward call for.
    - data (torch.Tensor): A sample of data to pass through the model.

    Outputs:
    - forward (Callable): The forward call function for the model.
    """
    # Get AOT exported model
    export_output = aot.export(model, data)

    # Compile to a deployable artifact.
    binary = export_output.compile(save_to=None)

    # Use the IREE runtime API to test the compiled program.
    config = ireert.Config("local-task")
    vm_module = ireert.load_vm_module(
        ireert.VmModule.copy_buffer(config.vm_instance, binary.map_memory()),
        config,
    )
    # result = vm_module.main(data)
    # print(result.to_host())

    def forward(data: torch.Tensor) -> torch.Tensor:
        """
        Forward call for the IREE-compiled model. We make sure to cast it back to Torch.

        Inputs:
        - data (torch.Tensor): The input data to pass through the model.

        Outputs:
        - output (torch.Tensor): The output from the model.
        """
        return vm_module.main(data)

    return forward
