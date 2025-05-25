from typing import Any, Callable

import torch


class LazyLoader:
    """A lazy loader that defers object creation until accessed."""

    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._instance = None
        self._loaded = False

    def load(self) -> Any:
        """Load the actual object and return it."""
        if not self._loaded:
            print("Factory", self._factory)
            self._instance = self._factory()
            self._loaded = True
        return self._instance

    def is_loaded(self) -> bool:
        """Check if the object has been loaded."""
        return self._loaded

    def unload(self):
        """Unload the object to free memory."""
        self._instance = None
        self._loaded = False

    def __call__(self, *args, **kwargs):
        """Allow the lazy loader to be called like the original object."""
        return self.load()(*args, **kwargs)

    def __getattr__(self, name):
        """Proxy attribute access to the loaded instance."""
        return getattr(self.load(), name)


def lazyload(factory: Callable[[], Any]) -> LazyLoader:
    """
    Create a lazy loader from a factory function.

    Usage:
        model = lazy(lambda: torch.nn.Sequential(...))
        # or
        model = lazy(lambda: MyModel(param1, param2))
    """
    return LazyLoader(factory)


def init_lazy_model(model: Any, device: torch.device) -> Any:
    """
    Load the model, if it is LazyLoader instance and has not yet been loaded.

    Args:
        model (Any): Instance that may be a LazyLoader instance or not.
        device (torch.device): the device to send the model to

    Returns:
        (Any): The loaded model.
    """
    if isinstance(model, LazyLoader):
        model = model.load()

    if hasattr(model, "to"):
        model.to(device)
    else:
        assert hasattr(model, "device") and model.device == device

    # Set to eval mode
    if hasattr(model, "eval"):
        model.eval()

    return model
