import logging

import torch
from torch.export.exported_program import ExportedProgram
from torch.utils.data import DataLoader

from data.utils import get_sample_data


def get_exported_model(
    model, data_loader: DataLoader, device: torch.device, logging: logging.Logger
) -> ExportedProgram:
    """
    Export the model using torch.export.

    Inputs:
    - model (torch.nn.Module): The model to export
    - data_loader (DataLoader): The DataLoader to get a sample of data from
    - device (torch.device): The device to run the model on
    - logging (logging.Logger): The logger to use for logging

    Outputs:
    model (torch.export.Model): The exported model
    """

    logging.info("Running torch.export the model")

    # Get a sample of data to pass through the model
    data = get_sample_data(data_loader, device)

    # Call torch export, which decomposes the forward pass of the model
    # into a graph of Aten primitive operators
    model = torch.export.export(model, (data,))
    model.graph.print_tabular()

    return model
