from typing import Any


def check_model_type(model: Any, expected_type: Any):
    """
    Checks that the model is of the expected type.

    Inputs:
    - model (Any): the model whose type to check
    - expected_type (Any): the class we expect

    Outputs:
    None
    """

    assert isinstance(
        model, expected_type
    ), f"model must be of type {expected_type}, got {type(model)}"
