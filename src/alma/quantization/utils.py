import logging
from typing import Optional, Tuple

import torch
import torch.fx as fx

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def save_fake_quantized_model(model: torch.nn.Module, path: str) -> None:
    """
    Saves the state dictionary of a quantized model to a specified path.

    Args:
        model (torch.nn.Module): The quantized model to be saved.
        path (str): The path where the model state dict will be saved.
    """
    model.eval()
    torch.save(model.state_dict(), path)
    logging.info(f"Quantized model saved to {path}")


def replace_node_with_target(
    fx_model: fx.GraphModule, target_node_name: str, target_module: torch.nn.Module
) -> None:
    """
    Replaces a node in the FX graph with a target module. For example, this can be useful for
    replacing quantization modules with custom modules.

    Args:
        fx_model (fx.GraphModule): The FX graph module containing the node to replace.
        target_node_name (str): The name of the target node to be replaced.
        target_module (torch.nn.Module): The new module to replace the target node.

    Raises:
        ValueError: If the node with the specified target name is not found.
    """

    def find_target_node() -> Optional[fx.Node]:
        """Find the node with the specified target name."""
        for node in fx_model.graph.nodes:
            if node.op == "call_module" and node.target == target_node_name:
                return node
        return None

    modules = dict(fx_model.named_modules())
    target_fx_node = find_target_node()

    if target_fx_node:
        replace_node_module(modules, target_fx_node, target_module)
        logging.info(f"Replaced node {target_node_name} with new module.")
    else:
        error_message = f"Could not find node with target {target_node_name}"
        logging.error(error_message)
        raise ValueError(error_message)


def replace_node_module(
    modules: dict, node: fx.Node, new_module: torch.nn.Module
) -> None:
    """
    Replaces a module in the given module dictionary with a new module.

    Args:
        modules (dict): A dictionary of modules.
        node (fx.Node): The node representing the target module to be replaced.
        new_module (torch.nn.Module): The new module to replace the target module.
    """
    assert isinstance(node.target, str), "Node target must be a string"
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, new_module)
    logging.info(f"Replaced module {node.target} in parent {parent_name}")


def _parent_name(target: str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.

    Args:
        target (str): The qualified name to split.

    Returns:
        Tuple[str, str]: The parent path and the last atom of the target.
    """
    *parent, name = target.rsplit(".", 1)
    return parent[0] if parent else "", name


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example model, path, and target node for demonstration
    class ExampleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 1, 3)

        def forward(self, x):
            return self.conv(x)

    example_model = ExampleModel()
    new_conv = torch.nn.Conv2d(1, 1, 3)
    fx_model = fx.symbolic_trace(example_model)

    # Replace the node in the fx model
    try:
        replace_node_with_target(fx_model, "conv", new_conv)
        logging.info("Node replacement completed successfully.")
    except ValueError as e:
        logging.error(f"Node replacement failed: {e}")
