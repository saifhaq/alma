import inspect
import logging
import pickle
from typing import Callable, Union

import torch

from ..multiprocessing import LazyLoader

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def check_model(model: Callable, config: dict) -> None:
    """
    Checks if the model (or init function) is valid, and logs a warning if a memory inefficient
    process is being used.

    Inputs:
    - model (Callable): The model to benchmark, or a callable function to get the model.
    - config (dict): The configuration for the benchmarking.

    Outputs:
    None
    """
    assert callable(model), "The model should always be a callable"

    # If we are doing multiprocessing and the model is being LazyLoaded, that's perfect
    if isinstance(model, LazyLoader) and config.multiprocessing:
        # Unless it's already been loaded
        if model.is_loaded():
            warning_msg = """Multiprocessing is enabled, and a LazyLoader is being used, which is ideal. However,
the Lazyloader instance has already been loaded, which means there will be 2 copies of the model initialized in memory.
Iti is recommended you check your code to see why the model has been initialized already. See 
`examples/mnist/README.md` in section `Effect on model memory` for discussion on this topic."""
            logger.warning(warning_msg)
        return

    if not isinstance(model, LazyLoader) and config.multiprocessing:
        warning_msg = """Multiprocessing is enabled, but LazyLoader is not being used. This can cause
the model to be loaded into memory twice, once in the parent process and once in the child process.
See `examples/mnist/mem_efficient_benchmark_rand_tensor.py` for an example on how to use the `LazyLoader`
for more efficient memory usage."""
        logger.warning(warning_msg)

    if not config.multiprocessing:
        warning_msg = """Multiprocessing is not enabled. These is a risk that different export 
options, e.g. HF optimum quanto, will contaminate the global torch state and break other conversion options. For testing
multiple options, consider enabling multiprocessing to run each option in a dedicated subprocess. Also
consider using the `alma.utils.multiprocessing.LazyLoader class for more memory efficient loading of models
in a multiprocessing setup. See `examples/mnist/mem_efficient_benchmark_rand_tensor.py` for an example."""
        logger.warning(warning_msg)
        return


def check_is_local_function(func: Callable, error_msg: str) -> None:
    """
    Check if a function is locally scoped (defined inside another function). A locally scoped
    function is not pickle-able, which will interfere if one is rying to pass a callable to a
    child process during multiprocessing, where the callable is used to initialise the model.

    Inputs:
    - func (Callable): Function to check.
    - error_msg (str): the error msg to throw.

    Outputs:
    None

    Raises:
    ValueError: If the passed object is not a function
    """
    assert not is_local_function_name(func), error_msg
    assert is_picklable(func), error_msg


def is_local_function_name(func: Callable) -> bool:
    """
    Check if the name of the function is consistent with it being locally scoped.

    Inputs:
    func (Callable): Function to check

    Outputs:
    bool: True if function is locally scoped, False otherwise

    Raises:
    ValueError: If the passed object is not a function
    """
    if not inspect.isfunction(func):
        raise ValueError(f"Expected a function, got {type(func)}")

    # Get the qualified name (includes parent scope info)
    qualname = func.__qualname__

    # If it contains a dot and isn't a method (doesn't start with a capital)
    # then it's likely a local function
    if "." in qualname and not qualname[0].isupper():
        return True

    return False


def is_picklable(func: Callable) -> bool:
    """
    Test if a function can be pickled (and thus used with multiprocessing).

    Inputs:
    func (Callable): Function to test

    Outputs:
    bool: True if function can be pickled, False otherwise
    """
    try:
        # Try to pickle and unpickle the function
        _ = pickle.loads(pickle.dumps(func))
        return True
    except (pickle.PicklingError, AttributeError, TypeError):
        return False
