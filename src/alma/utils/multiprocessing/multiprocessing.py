import multiprocessing as mp
import traceback
from multiprocessing import Process, Queue
from typing import Any, Callable, Union

import torch

from .traceback import get_next_line_info


def run_benchmark_process(
    formatted_stacktrace: str,
    benchmark_func: Callable,
    device: torch.device,
    args: tuple,
    kwargs: dict,
    result_queue: Queue,
) -> None:
    """
    Helper function to run the benchmark in a separate process.

    Inputs:
    - formatted_stacktrace (str): the formatted stack trace to include, captuting the instructions
        up this point. This helps us to construct a coherent traceback through the multi-processing
        and error handling.
    - benchmark_func (callable): the benchmarking function to run
    - device (torch.device): the device to run the benchmark on
    - args (tuple): positional arguments for the benchmark function
    - kwargs (dict): keyword arguments for the benchmark function
    - result_queue (Queue): queue to store the results or errors
    """
    try:
        # Initialize the XLA environment if device is XLA
        if device.type == "xla":
            import torch_xla.core.xla_model as xm

            device = xm.xla_device()
    except Exception as e:
        formatted_stacktrace += f"\n{traceback.format_exc()}"

    # Get the next line to include in the traceback
    next_cmd_multi = get_next_line_info()
    result = benchmark_func(device, *args, **kwargs)
    result_queue.put((result, formatted_stacktrace + next_cmd_multi))


def benchmark_process_wrapper(
    multiprocessing: bool,
    benchmark_func: Callable,
    device: torch.device,
    *args: Any,
    **kwargs: Any,
) -> Union[Any, None]:
    """
    Wrapper to run benchmark in a fresh process (if multiprocessing enabled) and return its results. This allows us
    to run different conversion methods (whose imports may affect the glocal state of PyTorch)
    in isolation. This means that every method will be tested with a blank slate, at the cost
    of some overhead.

    However, multiprocessing can make debugging difficult. As such, we provide the option to turn it
    off, and just run the benchmark_func callable directly.

    We provide the `device` as an argument, as the CUDA device requires the start method to be set to 'spawn'.

    # Usage example:
    def benchmark(device, method_type, model_path, *args, **kwargs):
        # Your existing benchmark code here
        if method_type == "optimum_quanto":
            from optimum_quanto import xyz
            # ... benchmark code ...
        elif method_type == "torch.export":
            # ... benchmark code ...
        return results

    # Run benchmarks
    result1 = benchmark_process_wrapper(True, benchmark, device, "optimum_quanto", ...)
    result2 = benchmark_process_wrapper(True, benchmark, device, "torch.export", ...)

    Inputs:
    - multiprocessing (bool): whether ot not to activatr the multiprocessing / isolated environments.
    - benchmark_func (callable): the benchmarking (or any to-be-isolated) function
    - device (torch.device): the device to run the benchmark on
    - args (Any): the argumnts for the callable
    - kawargs (Any): the keyword arguments for the callable

    Outputs:
    - results (Union[Any, None]): the results from the callable.
    - formatted_stacktrace (str): the formatted stack trace from the callable.
    """
    # Capture the current stack trace, so that we can create a coherent traceback
    stack = traceback.extract_stack()
    formatted_stacktrace = "Traceback (most recent call last):\n"
    formatted_stacktrace += "".join(traceback.format_list(stack)[:-1])

    # If multiprocessing is disabled, we just return the callable directly
    if not multiprocessing:
        # Get next line to include in the traceback
        next_cmd_single = get_next_line_info()
        result = benchmark_func(device, *args, **kwargs)
        return result, formatted_stacktrace + next_cmd_single
    # If the device to benchmark on is CUDA, we need to set the start method to 'spawn'
    if device.type in ["cuda", "xla"]:
        # This is required for CUDA, as the default 'fork' method does not work with CUDA
        mp.set_start_method("spawn", force=True)

    # Queue to get results back from the process
    result_queue = Queue()

    # Start process
    p = Process(
        target=run_benchmark_process,
        args=(formatted_stacktrace, benchmark_func, device, args, kwargs, result_queue),
    )
    p.start()
    p.join()

    # Get result (if any)
    if not result_queue.empty():
        result, stacktrace = result_queue.get()
        return result, stacktrace
    raise RuntimeError("benchmarking process fialed to return a result")
