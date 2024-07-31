import torch.fx as fx
from typing import Tuple
import torch

def save_fake_quantized_model(self, path):
    # Ensure the model is in evaluation mode
    self.eval()
    # Save the quantized model state dict
    torch.save(self.state_dict(), path)
    logging.info(f"Quantized model saved to {path}")

def replace_node_module(modules, node: fx.Node, new_module: torch.nn.Module):
    """
    Helper function for having `new_module` take the place of `node` in a dict of modules.
    """
    assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    #modules[node.target] = new_module
    setattr(modules[parent_name], name, new_module)

def _parent_name(target : str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`) 
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name


def replace_node_with_target(fx_model, target_node_name: str, target_node: torch.nn.Module):
    """
    Replaces a node in the FX graph with a target node.
    """
    replaced = False
    modules = dict(fx_model.named_modules())
    for node in fx_model.graph.nodes:
        # If the operation the node is doing is to call a module
        if node.op == 'call_module':
            if node.target == target_node_name:
                # This updates `modules` so that the fixed qparams takes the place of what was represented by `node`
                replace_node_module(modules, node, target_node)
                replaced = True
    if not replaced:
        raise ValueError(f"Could not find node with target {target_node_name}")
