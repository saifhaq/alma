import multiprocessing as mp
from multiprocessing import Process, Queue
from typing import Any, Callable, Union

import torch

if torch.cuda.is_available():
    # Set the start method to 'spawn' at the beginning of your script
    mp.set_start_method("spawn", force=True)


def run_benchmark_process(
    benchmark_func: Callable, args: tuple, kwargs: dict, result_queue: Queue
) -> None:
    """
    Helper function to run the benchmark in a separate process.

    Inputs:
    benchmark_func (callable): the benchmarking function to run
    args (tuple): positional arguments for the benchmark function
    kwargs (dict): keyword arguments for the benchmark function
    result_queue (Queue): queue to store the results or errors
    """
    result = benchmark_func(*args, **kwargs)
    result_queue.put(result)


def process_wrapper(
    benchmark_func: Callable, *args: Any, **kwargs: Any
) -> Union[Any, None]:
    """
    Wrapper to run benchmark in a fresh process and return its results. This allows us
    to run different conversion methods (whose imports may affect the glocal state of PyTorch)
    in isolation. This means that every method will be tested with a blank slate, at the cost
    of some overhead.

    # Usage example:
    def benchmark(method_type, model_path, *args, **kwargs):
        # Your existing benchmark code here
        if method_type == "optimum_quanto":
            from optimum_quanto import xyz
            # ... benchmark code ...
        elif method_type == "torch.export":
            # ... benchmark code ...
        return results

    # Run benchmarks
    result1 = process_wrapper(benchmark, "optimum_quanto", ...)
    result2 = process_wrapper(benchmark, "torch.export", ...)

    Inputs:
    benchmark_func (callable): the benchmarking (or any to-be-isolated) function
    args (Any): the argumnts for the callable
    kawargs (Any): the keyword arguments for the callable

    Outputs:
    results (Union[Any, None]): the results from the callable.
    """
    # Queue to get results back from the process
    result_queue = Queue()

    # Start process
    p = Process(
        target=run_benchmark_process, args=(benchmark_func, args, kwargs, result_queue)
    )
    p.start()
    p.join()

    # Get result (if any)
    if not result_queue.empty():
        result = result_queue.get()
        return result
    return None
