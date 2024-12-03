import logging

import torch
from torch.export.exported_program import ExportedProgram

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_exported_model(model, data: torch.Tensor) -> ExportedProgram:
    """
    Export the model using torch.export.

    Inputs:
    - model (torch.nn.Module): The model to export
    - data (torch.Tensor): A sample of data to feed through the model for tracing

    Outputs:
    model (torch.export.Model): The exported model
    """

    logging.info("Running torch.export on the model")

    # Call torch export, which decomposes the forward pass of the model
    # into a graph of Aten primitive operators
    model = torch.export.export(model, (data,))

    logging.debug("Model graph:")
    if logging.root.level <= logging.DEBUG:
        logging.debug(model.graph.print_tabular())

    return model
