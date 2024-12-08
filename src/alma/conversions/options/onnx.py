import logging
from pathlib import Path
from typing import Any, Callable, Union

import onnx
import onnxruntime
import torch

from ...utils.setup_logging import suppress_output
from .utils.checks.type import check_model_type

# Create a module-level logger
logger = logging.getLogger(__name__)
# Don't add handlers - let the application configure logging
logger.addHandler(logging.NullHandler())


def save_onnx_model(
    model,
    data: torch.Tensor,
    onnx_model_path: str = "model/model.onnx",
):
    """
    Export the model to ONNX using torch.onnx. Saves the model to `onnx_model_path`.

    Inputs:
    - model (torch.nn.Module): The model to export
    - data (torch.Tensor): A sample of data to pass through the model.
    - onnx_model_path (str): The path to save the ONNX model to

    Outputs:
    None
    """
    logger.info("Running torch.onnx on the model")

    # Providing input and output names sets the display names for values
    # within the model's graph. Setting these does not change the semantics
    # of the graph; it is only for readability.
    #
    # The inputs to the network consist of the flat list of inputs (i.e.
    # the values you would pass to the forward() method) followed by the
    # flat list of parameters. You can partially specify names, i.e. provide
    # a list here shorter than the number of inputs to the model, and we will
    # only set that subset of names, starting from the beginning.
    input_names = ["input_data"] + [name for name, _ in model.named_parameters()]
    output_names = ["output"]

    # torch.onnx.export(model, data, "model.onnx", verbose=True, input_names=input_names, output_names=output_names)
    model.eval()

    # Export the model
    logger.info(f"Saving the torch.onnx model to {onnx_model_path}")
    with suppress_output(logger.root.level >= logging.DEBUG):
        torch.onnx.export(
            model,  # model being run
            data,  # model input (or a tuple for multiple inputs)
            onnx_model_path,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            # opset_version=10,          # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=input_names,  # the model's input names
            output_names=output_names,  # the model's output names
            verbose=logger.root.level <= logging.DEBUG,
            # dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
            #             'output' : {0 : 'batch_size'}}
        )

        # Check the model is well formed
        loaded_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(loaded_model)

    # Print a human readable representation of the graph
    if logger.root.level <= logging.DEBUG:
        logger.debug("ONNX model graph:")
        logger.debug(onnx.helper.printable_graph(loaded_model.graph))


def _get_onnx_forward_call(
    model: Union[Path, torch.onnx.ONNXProgram],
    onnx_provider: str = "CPUExecutionProvider",
) -> Callable:
    """
    A helper function to return the ONNX forward call.

    Inputs:
    - model (Union[Path, torch.onnx.ONNXProgram]): this can be either a path to an ONNX model, or
        the ONNX model directly (if created via the torch.onnx.dynamo_export API).
    - onnx_provider (str): the ONNX execution provider to use.

    Outputs:
    - onnx_forward (Callable): The forward call function for the model.
    """
    # Create ONNX runtime session for ONNX model
    ort_session = onnxruntime.InferenceSession(model, providers=[onnx_provider])
    logger.info(f"Loaded ONNX model, using {onnx_provider}")
    # NOTE: see here for a list of Execution providers: https://onnxruntime.ai/docs/execution-providers/

    # Onnx runtime forward call
    def onnx_forward(data: torch.Tensor) -> Callable:
        ort_inputs = {ort_session.get_inputs()[0].name: data.cpu().numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        return ort_outs

    return onnx_forward


def get_onnx_forward_call(
    model: Any,
    data: torch.Tensor,
    onnx_model_path: str = "model/model.onnx",
    onnx_provider: str = "CPUExecutionProvider",
):
    """
    Get the forward call function for the model using ONNX.

    Inputs:
    - model (Any): The model to get the forward call for.
    - data (torch.Tensor): A sample of data to pass through the model.
    - onnx_model_path (str): the path to save the ONNX model to.
    - onnx_provider (str): the ONNX execution provider to use.

    Outputs:
    - onnx_forward (Callable): The forward call function for the model.
    """
    # We first save the ONNX model
    save_onnx_model(model, data, onnx_model_path)

    # Get onnx forward call
    onnx_forward: Callable = _get_onnx_forward_call(onnx_model_path, onnx_provider)

    return onnx_forward


def get_onnx_dynamo_forward_call(
    model: Any,
    data: torch.Tensor,
    onnx_model_path: str = "model/model.onnx",
):
    """
    Get the forward call function for the model using ONNX (dynamo API, in beta as of 24/11/2024).

    Inputs:
    - model (Any): The model to get the forward call for.
    - data (torch.Tensor): A sample of data to pass through the model.
    - onnx_model_path (str): the path to save the ONNX model to.

    Outputs:
    - forward (Callable): The forward call function for the model.
    """

    logger.info("Running torch.onnx.dynamo_export (beta) on the model")
    with suppress_output(logger.root.level >= logging.DEBUG):
        onnx_program = torch.onnx.dynamo_export(model, data)

    check_model_type(onnx_program, torch.onnx.ONNXProgram)

    # Save model, required as an intermediary
    onnx_program.save(onnx_model_path)

    # Get onnx forward call
    onnx_forward: Callable = _get_onnx_forward_call(onnx_model_path)

    return onnx_forward
