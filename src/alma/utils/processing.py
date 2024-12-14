import multiprocessing as mp
from multiprocessing import Process, Queue
from typing import Any, Callable, Union

import torch


def run_benchmark_process(
    benchmark_func: Callable,
    device: torch.device,
    args: tuple,
    kwargs: dict,
    result_queue: Queue,
) -> None:
    """
    Helper function to run the benchmark in a separate process.

    Inputs:
    - benchmark_func (callable): the benchmarking function to run
    - device (torch.device): the device to run the benchmark on
    - args (tuple): positional arguments for the benchmark function
    - kwargs (dict): keyword arguments for the benchmark function
    - result_queue (Queue): queue to store the results or errors
    """
    result = benchmark_func(device, *args, **kwargs)
    result_queue.put(result)


def process_wrapper(
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
    result1 = process_wrapper(True, benchmark, device, "optimum_quanto", ...)
    result2 = process_wrapper(True, benchmark, device, "torch.export", ...)

    Inputs:
    - multiprocessing (bool): whether ot not to activatr the multiprocessing / isolated environments.
    - benchmark_func (callable): the benchmarking (or any to-be-isolated) function
    - device (torch.device): the device to run the benchmark on
    - args (Any): the argumnts for the callable
    - kawargs (Any): the keyword arguments for the callable

    Outputs:
    results (Union[Any, None]): the results from the callable.
    """
    # If multiprocessing is disabled, we just return the callable directly
    if not multiprocessing:
        return benchmark_func(device, *args, **kwargs)

    # If the device to benchmark on is CUDA, we need to set the start method to 'spawn'
    if device.type == "cuda":
        # This is required for CUDA, as the default 'fork' method does not work with CUDA
        mp.set_start_method("spawn", force=True)

    # Queue to get results back from the process
    result_queue = Queue()

    # Start process
    p = Process(
        target=run_benchmark_process,
        args=(benchmark_func, device, args, kwargs, result_queue),
    )
    p.start()
    p.join()

    # Get result (if any)
    if not result_queue.empty():
        result = result_queue.get()
        return result
    return None
