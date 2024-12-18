import pytest

from .utils import simple_model_for_testing


def test_eager_jit_torchscript():
    """
    Test that eager mode benchmarking works successfully with a simple model.
    """
    methods = ["EAGER", "JIT_TRACE", "TORCH_SCRIPT"]
    results = simple_model_for_testing(methods)

    assert (
        len(results) == 3
    ), "Expected 3 results for EAGER, JIT_TRACE, and TORCH_SCRIPT conversions"
    for method in methods:
        assert method in results, f"Expected {method} to be in results"
    for key in results.keys():
        assert (
            results[key]["status"] == "success"
        ), f"Expected {key} conversion to succeed"
