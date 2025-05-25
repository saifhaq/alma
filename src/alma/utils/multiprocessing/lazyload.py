import functools
from typing import Any, Callable, TypeVar, Type

T = TypeVar("T")


class LazyLoader:
    """A lazy loader that defers object creation until accessed."""

    def __init__(self, cls: Type[T], *args, **kwargs):
        self._cls = cls
        self._args = args
        self._kwargs = kwargs
        self._instance = None
        self._loaded = False

    def load(self) -> T:
        """Load the actual object and return it."""
        if not self._loaded:
            self._instance = self._cls(*self._args, **self._kwargs)
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
        """Allow the lazy loader to be called like the original class."""
        return self.load()(*args, **kwargs)

    def __getattr__(self, name):
        """Proxy attribute access to the loaded instance."""
        return getattr(self.load(), name)


def lazyload(cls: Type[T]) -> Type[T]:
    """
    Decorator that makes class instantiation lazy.

    Usage:
        @lazyload
        model = Model(param1, param2)

        # model is now a LazyLoader, actual Model() is not created yet
        # When you need the actual model:
        actual_model = model.load()

        # Or access attributes directly (auto-loads):
        result = model.some_method()  # This will load Model() first
    """

    class LazyWrapper:
        def __new__(cls, *args, **kwargs):
            return LazyLoader(cls._original_class, *args, **kwargs)

    LazyWrapper._original_class = cls
    LazyWrapper.__name__ = cls.__name__
    LazyWrapper.__qualname__ = cls.__qualname__

    return LazyWrapper


# Alternative implementation using a metaclass approach
class LazyMeta(type):
    """Metaclass that creates lazy-loading classes."""

    def __call__(cls, *args, **kwargs):
        if hasattr(cls, "_lazy_enabled"):
            return LazyLoader(cls._original_class, *args, **kwargs)
        else:
            return super().__call__(*args, **kwargs)


def lazyload_meta(cls: Type[T]) -> Type[T]:
    """Alternative implementation using metaclass."""

    class LazyClass(cls, metaclass=LazyMeta):
        _lazy_enabled = True
        _original_class = cls

    LazyClass.__name__ = cls.__name__
    LazyClass.__qualname__ = cls.__qualname__

    return LazyClass


# Example usage
if __name__ == "__main__":
    import time

    # Example model class
    class Model:
        def __init__(self, name="default", size=100):
            print(f"Loading model '{name}' with size {size}... (expensive operation)")
            time.sleep(0.5)  # Simulate expensive loading
            self.name = name
            self.size = size
            self.data = list(range(size))

        def predict(self, x):
            return f"Model {self.name} predicts: {x * 2}"

        def __str__(self):
            return f"Model(name={self.name}, size={self.size})"

    print("=== Without decorator (normal behavior) ===")
    normal_model = Model("normal", 50)
    print(f"Normal model: {normal_model}")

    print("\n=== With @lazyload decorator ===")

    @lazyload
    class Model:
        def __init__(self, name="default", size=100):
            print(f"Loading model '{name}' with size {size}... (expensive operation)")
            time.sleep(0.5)  # Simulate expensive loading
            self.name = name
            self.size = size
            self.data = list(range(size))

        def predict(self, x):
            return f"Model {self.name} predicts: {x * 2}"

        def __str__(self):
            return f"Model(name={self.name}, size={self.size})"

    # This looks exactly like normal instantiation, but no loading happens yet
    model = Model("lazy", 200)

    print(f"Model created (type: {type(model)})")
    print(f"Is loaded: {model.is_loaded()}")

    print("Accessing model attribute (triggers loading)...")
    result = model.predict(5)
    print(f"Prediction result: {result}")
    print(f"Is loaded: {model.is_loaded()}")

    print("\nManual loading example:")

    @lazyload
    class AnotherModel:
        def __init__(self, value):
            print(f"Actually creating AnotherModel with value {value}")
            self.value = value

    lazy_model = AnotherModel(42)
    print(f"Lazy model created, loaded: {lazy_model.is_loaded()}")

    actual_model = lazy_model.load()
    print(f"Actual model: {actual_model.value}")
    print(f"Now loaded: {lazy_model.is_loaded()}")

    # For multiprocessing, you can pass the LazyLoader object
    def worker_function(lazy_obj):
        # In subprocess, load the actual object
        obj = lazy_obj.load()
        return f"Worker processed: {obj}"

    result = worker_function(lazy_model)
    print(f"Worker result: {result}")
