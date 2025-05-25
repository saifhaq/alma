from typing import Any, Callable


class LazyLoader:
    """A lazy loader that defers object creation until accessed."""

    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._instance = None
        self._loaded = False

    def load(self) -> Any:
        """Load the actual object and return it."""
        if not self._loaded:
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


def init_lazy_model(obj_or_cls: Any) -> Any:
    """
    Load the model, if it is LazyLoader instance and has not yet been loaded.

    Args:
        obj_or_cls (Any): Instance that may be a LazyLoader instance or not.

    Returns:
        (Any): The loaded model.
    """
    if isinstance(obj_or_cls, LazyLoader):
        return obj_or_cls.load()
    else:
        return obj_or_cls
