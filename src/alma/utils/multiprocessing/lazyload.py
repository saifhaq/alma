from __future__ import annotations

"""Lazy loading utilities.

This module provides a *thread‑safe* :class:`LazyLoader` class that defers the
creation of heavyweight objects (e.g. large ML models) until the first time
they are accessed.  It is designed to work seamlessly with *multiprocessing* –
during pickling only the *factory* is serialised, so each child process
receives a **fresh, unloaded** loader.

The implementation is generic, light‑weight (uses ``__slots__``), and adds
quality‑of‑life dunder forwarding so the wrapper can be used as if it *were*
the wrapped object.
"""

import threading
from typing import Any, Callable, Generic, TypeVar, cast, overload

__all__ = [
    "LazyLoader",
    "lazyload",
    "init_lazy_model",
]

_T = TypeVar("_T")
_S = TypeVar("_S")


class LazyLoader(Generic[_T]):
    """A thread‑safe on‑demand loader for heavyweight objects.

    Parameters
    ----------
    factory
        A **zero‑argument** callable that creates and *returns* the target
        object when invoked.

    Notes
    -----
    * The first call to :py:meth:`load` executes the *factory* exactly once,
      even under heavy multithreading, thanks to a double‑checked lock.
    * Use :py:meth:`unload` to *explicitly* release the cached object and free
      memory.
    * The loader itself is **picklable**; only the *factory* is serialised so
      the loaded instance is never duplicated across processes.
    """

    __slots__ = ("_factory", "_instance", "_lock")

    _UNSET: object = object()  # Sentinel to mark the *not‑yet‑loaded* state

    # ---------------------------------------------------------------------
    # Construction & (de)serialisation
    # ---------------------------------------------------------------------
    def __init__(self, factory: Callable[[], _T]):
        self._factory: Callable[[], _T] = factory
        self._instance: _T | object = self._UNSET
        self._lock = threading.Lock()

    def __getstate__(self) -> dict[str, Callable[[], _T]]:
        """Return state for pickling; serialises only the factory."""
        return {"_factory": self._factory}

    def __setstate__(self, state: dict[str, Callable[[], _T]]) -> None:
        """Restore an **unloaded** loader from pickled *state*."""
        self._factory = state["_factory"]
        self._instance = self._UNSET
        self._lock = threading.Lock()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def is_loaded(self) -> bool:
        """Return ``True`` if the target object has been materialised."""
        return self._instance is not self._UNSET

    def load(self) -> _T:
        """Return the wrapped object, loading it *lazily* if required."""
        # Fast path: already loaded – no locking overhead
        if self._instance is self._UNSET:
            # Slow path: first caller takes the lock and builds the instance
            with self._lock:
                if self._instance is self._UNSET:  # Double‑checked locking
                    self._instance = self._factory()
        # `_instance` is guaranteed to hold the loaded object here.
        return cast(_T, self._instance)

    def unload(self) -> None:
        """Forget the cached instance so memory can be reclaimed."""
        with self._lock:
            self._instance = self._UNSET

    # ------------------------------------------------------------------
    # Dunder delegation – make the loader behave like the underlying obj
    # ------------------------------------------------------------------
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the loaded object, passing through arguments."""
        return self.load()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute lookup to the loaded object."""
        # Only invoked if *normal* attribute lookup fails.
        return getattr(self.load(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute, delegating to loaded object unless internal."""
        if name in self.__slots__:  # Bypass delegation for internal attrs
            object.__setattr__(self, name, value)
        else:
            setattr(self.load(), name, value)

    def __delattr__(self, name: str) -> None:
        """Delete an attribute from the loaded object."""
        if name in self.__slots__:
            raise AttributeError(f"Cannot delete internal attribute '{name}'.")
        delattr(self.load(), name)

    # ------------------------------------------------------------------
    # Representation helpers – handy for debugging/logging
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        """Return a string representation of the loader and its state."""
        state = "loaded" if self.is_loaded() else "unloaded"
        # Ensure factory repr is concise if it's a lambda or complex callable
        factory_repr = getattr(self._factory, "__name__", repr(self._factory))
        if factory_repr == "<lambda>":
            factory_repr = "lambda"  # Produce cleaner repr for simple lambdas
        return f"{self.__class__.__name__}({state}, factory={factory_repr})"


# -------------------------------------------------------------------------
# Convenience helpers – keep API parity with existing code bases
# -------------------------------------------------------------------------


def lazyload(factory: Callable[[], _T]) -> LazyLoader[_T]:
    """Create a :class:`LazyLoader` from *factory* (DSL sugar)."""
    return LazyLoader(factory)


@overload
def init_lazy_model(obj_or_cls: LazyLoader[_S]) -> _S: ...


@overload
def init_lazy_model(obj_or_cls: _S) -> _S: ...


def init_lazy_model(obj_or_cls: Any) -> Any:
    """Materialise *obj_or_cls* if it is a :class:`LazyLoader`.

    Designed for API parity with existing codebases that may use this pattern.
    """
    return obj_or_cls.load() if isinstance(obj_or_cls, LazyLoader) else obj_or_cls
