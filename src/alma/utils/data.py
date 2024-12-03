from typing import Union, Dict, Any

import numpy as np
import torch
import torch.fx as fx
from torch.utils.data import DataLoader
import inspect
import re


def process_batch(
    batch: torch.Tensor,
    model: Union[torch.nn.Module, fx.GraphModule],
    device: torch.device,
) -> np.ndarray:
    """
    Process a batch of data through a model.

    Inputs:
    - batch (torch.Tensor): A batch of data.
    - model (torch.nn.Module): The model to process the data with.
    - device (torch.device): The device to run the model on.

    Outputs:
    - preds (np.ndarray): The model predictions.
    """
    data = batch.to(device)
    output = model(data)
    preds = output.argmax(dim=1, keepdim=True)
    return preds.cpu().numpy()


def get_sample_data(data_loader: DataLoader, device: torch.device) -> torch.Tensor:
    """
    Get a sample of data from the DataLoader.

    Inputs:
    - data_loader (DataLoader): The DataLoader to get a sample of data from
    - device (torch.device): The device the data tensor should live on

    Outputs:
    - data (torch.Tensor): A sample of data from the DataLoader
    """
    for data, _ in data_loader:
        data = data.to(device)
        return data
    raise ValueError("DataLoader is empty")


def args_to_dict(*args) -> Dict[str, Any]:
    """
    Capture the variable names of the arguments passed to the function.

    Example usage:
    a = 5
    b = "hello"
    >>> args_as_dict = args_to_dict(a, b)
    >>> print(args_as_dict)
    {'a': 5, 'b': 'hello'}

    Inputs:
    *args: The arguments passed to the function.

    Outputs:
    output (Dict[str, Any]): A dictionary containing the variable names of the arguments passed to the function.
    """
    frame = inspect.currentframe()
    outer_frame = frame.f_back
    calling_code = inspect.getframeinfo(outer_frame).code_context[0].strip()

    # Extract the argument string
    match = re.search(r"\((.*?)\)$", calling_code)
    if match:
        arg_string = match.group(1)

        # Split the argument string, handling potential commas within function calls or data structures
        arg_names = []
        paren_count = 0
        current_arg = ""
        for char in arg_string:
            if char == "(" or char == "[" or char == "{":
                paren_count += 1
            elif char == ")" or char == "]" or char == "}":
                paren_count -= 1

            if char == "," and paren_count == 0:
                arg_names.append(current_arg.strip())
                current_arg = ""
            else:
                current_arg += char

        if current_arg:
            arg_names.append(current_arg.strip())

        # Print the argument names and their values
        output = {}
        for name, value in zip(arg_names, args):
            output[name] = value
    else:
        raise ValueError("Couldn't parse the function call.")
    return output
