import unittest

import torch

from alma.utils.multiprocessing.error_handling import benchmark_error_handler
from alma.utils.multiprocessing.multiprocessing import benchmark_process_wrapper


def failing_benchmark(*args, **kwargs):
    """
    A benchmark function that will always fail.
    """
    raise ValueError("Test error")


# Decorate the failing benchmark function with the error handler
decorated_failing_benchmark = benchmark_error_handler(failing_benchmark)


def create_benchmark_error(multiprocessing: bool) -> dict:
    """
    Provided a decorated benchmark function which will fail, we call the benchmark_process_wrapper
    to trigger the error and return the result. We cna then monitor the traceback to make sure
    it is as expected.

    Inputs:
    - multiprocessing (bool): whether to run the benchmark in a separate process

    Outputs:
    - result (dict): the result of the benchmark, in this case the error dictionary.
    """
    # Call benchmark_process_wrapper which should trigger the decorated benchmark
    result = benchmark_process_wrapper(
        multiprocessing=multiprocessing,
        benchmark_func=decorated_failing_benchmark,
        device=torch.device("cpu"),
        model=None,
        conversion="EAGER",
        data_loader=None,
        n_samples=None,
    )

    return result


def check_traceback_for_benchmark_error(result: dict, multiprocessing: bool) -> None:
    """
    Given an error in the `benchmark` function, we check that the traceback is as expected.

    Inputs:
    - result (dict): the result of the benchmark, in this case the error dictionary.
    - multiprocessing (bool): whether the benchmark was run in a separate process.

    Outputs:
    - None
    """
    traceback = result["traceback"].split("\n")
    assert "result = benchmark_process_wrapper" in traceback[-11]
    assert "alma/utils/multiprocessing/multiprocessing.py" in traceback[-10]
    if multiprocessing:
        assert "run_benchmark_process" in traceback[-10]
    else:
        assert "benchmark_process_wrapper" in traceback[-10]

    assert "result = benchmark_func(device, *args, **kwargs)" in traceback[-9]
    assert "failing_benchmark" in traceback[-8]
    assert "failing_benchmark(" in traceback[-7]
    assert "alma/utils/multiprocessing/error_handling.py" in traceback[-6]
    assert "result: dict = decorated_func(*args, **kwargs)" in traceback[-5]
    assert "failing_benchmark" in traceback[-4]


class TestBenchmark(unittest.TestCase):
    def test_error_handler_wrapper_no_multi(self):
        """
        Test the error handler with multiprocessing disabled.
        """
        multiprocessing: bool = False

        # Call create_benchmark_error, which should trigger the decorated benchmark
        result = create_benchmark_error(multiprocessing)

        # Check error message is correct
        assert result["error"] == "Test error"

        # Check the traceback is as expected
        check_traceback_for_benchmark_error(result, multiprocessing)

    def test_error_handler_wrapper_multi(self):
        """
        Test the error handler with multiprocessing enabled.
        """
        multiprocessing: bool = True

        # Call create_benchmark_error, which should trigger the decorated benchmark
        result = create_benchmark_error(multiprocessing)

        # Check error message is correct
        assert result["error"] == "Test error"

        # Check the traceback is as expected
        check_traceback_for_benchmark_error(result, multiprocessing)


if __name__ == "__main__":
    unittest.main()
