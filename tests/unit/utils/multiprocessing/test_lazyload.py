
import pytest
import torch
import os
import psutil

from alma.utils.multiprocessing.lazyload import lazyload, LazyLoader, init_lazy_model


class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = torch.nn.Linear(size, size, bias=False)

    def forward(self, x):
        return self.linear(x)


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def test_lazy_load_memory_usage():
    """
    Test the lazyload function successfully defers increased memory usage until the model
    is lazy initialised.
    """
    # Call create_benchmark_error, which should trigger the decorated benchmark
    initial_memory = get_memory_usage()

    # We define the model, but because of lazyload we don't yet load it into memory
    model = lazyload(lambda: Model(5000))
    assert isinstance(model, LazyLoader), "The model should be a LazyLoader instance"
    post_def_memory = get_memory_usage()
    assert post_def_memory < initial_memory + 0.1, "Minimal extra memory should have been consumed by the lazy model init"

    # Initialize the model (i.e. load into memory)
    model = init_lazy_model(model)
    post_init_memory = get_memory_usage()
    assert post_init_memory > initial_memory + 5, "Loading the model into memory should increase the program memory usage"


def test_lazy_load_loaded_status():
    """
    Test the LazyLoader attributes are as expected.
    """
    # We define the model, but because of lazyload we don't yet load it into memory
    model = lazyload(lambda: Model(5000))
    assert isinstance(model, LazyLoader), "The model should be a LazyLoader instance"
    assert not model.is_loaded()

    # Initialize the model (i.e. load into memory)
    model = init_lazy_model(model)
    assert not isinstance(model, LazyLoader), "The model should no longer be a LazyLoader instance"
    assert isinstance(model, Model), "The model should be a Model instance"

