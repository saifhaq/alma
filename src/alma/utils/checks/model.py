import inspect
import logging
import pickle
from typing import Callable, Union

import torch

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def check_model(model: Union[torch.nn.Module, Callable], config: dict) -> None:
    """
    Checks if the model (or init function) is valid, and logs a warning if a memory inefficient
    process is being used.

    Inputs:
    - model (Union[torch.nn.Module, Callable]): The model to benchmark, or a callable function to get the model.
    - config (dict): The configuration for the benchmarking.

    Outputs:
    None
    """

    if not isinstance(model, torch.nn.Module) and config.multiprocessing:
        type_error_msg = "If not torch.nn.Module, the modle type should be a Callable that loads the model"
        assert isinstance(model, Callable), type_error_msg
        error_msg = """Please ensure that your 'load model' function is pickle-able. The function 
must not be locally sourced, but sourced at the module level."""
        check_is_local_function(model, error_msg)

    # If the model is a torch.nn.Mpodule and multiprocessing is enabled, we log a warning
    # that this is not memory efficient
    elif isinstance(model, torch.nn.Module) and config.multiprocessing:
        warning_msg = """Multiprocessing is enabled, and the model is a torch.nn.Module. This is not memory efficient,
as the model will be pickled and sent to each child process, which will require the model to be stored in memory
twice. If the model is large, this may cause memory issues. Consider using a callable to return the model, which
will be created in each child process, rather than the parent process. See `examples/mnist/mem_efficient_benchmark_rand_tensor.py`
for an example."""
        logger.warning(warning_msg)


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
