import logging
from pathlib import Path
from typing import Union

import torch

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def load_model(
    model_path: Path,
    device: torch.device,
    logger: logging.Logger,
    modelArchitecture: Union[torch.nn.Module, None] = None,
) -> torch.nn.Module:
    """
    Load the model from the given path. If the model is a torchscript model, load it using torch.jit.load.
    Otherwise, load the model using torch.load and load the state dict into the model.

    Inputs:
    - model_path (Union[Path, str]): Path to the model weights.
    - device (torch.device): Device to load the model on.
    - logger (logging.Logger): Logger to use for logging.
    - modelArchitecture (torch.nn.Module): The model architecture to use. This is required if the model is not a torchscript model.

    Outputs:
    - model (torch.nn.Module): The loaded model.
    """
    logger.info(f"Loading model from {model_path}")
    if model_path.suffix == ".pt":
        try:
            # Load the model using torch.jit.load
            model = torch.jit.load(model_path, map_location=device)
            model.eval()
        except RuntimeError:
            # Load the model using torch.load and load the state dict into the model
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            error_msg = "Model architecture must be provided to load model weights via the state dict"
            assert modelArchitecture is not None, error_msg
            model = modelArchitecture()
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
    else:
        # Load the model using torch.jit.load
        model = torch.jit.load(model_path, map_location=device)
        model.to(device)
        model.eval()
    return model
