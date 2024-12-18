import pytest

from .utils import simple_model_for_testing


def test_eager_success():
    """
    Test that eager mode benchmarking works successfully with a simple model.
    """
    results = simple_model_for_testing(["EAGER"])

    assert len(results) == 1, "Expected 1 result for EAGER conversion"
    assert (
        results["EAGER"]["status"] == "success"
    ), "Expected EAGER conversion to succeed"


def test_jit_success():
    """
    Test that jit traced mode benchmarking works successfully with a simple model.
    """
    results = simple_model_for_testing(["JIT_TRACE"])

    assert len(results) == 1, "Expected 1 result for JIT_TRACE conversion"
    assert (
        results["JIT_TRACE"]["status"] == "success"
    ), "Expected JIT_TRACE conversion to succeed"


def test_torchscript_success():
    """
    Test that torchscript mode benchmarking works successfully with a simple model.
    """
    results = simple_model_for_testing(["TORCH_SCRIPT"])

    assert len(results) == 1, "Expected 1 result for TORCH_SCRIPT conversion"
    assert (
        results["TORCH_SCRIPT"]["status"] == "success"
    ), "Expected TORCH_SCRIPT conversion to succeed"
