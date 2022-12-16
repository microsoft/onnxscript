"""Registry for aten functions."""

from __future__ import annotations

from typing import Any, Callable, Optional

import onnxscript


class OverloadedFunction:
    """Overloaded function."""

    def __init__(self, name: str):
        self.name = name
        self.default: Optional[Any] = None
        self.overloads: list[Any] = []


class Registry:
    """Registry for aten functions."""

    def __init__(self):
        self._registry: dict[str, OverloadedFunction] = {}

    def register(self, name: str, overload: bool = False):
        """Register a function."""

        def wrapper(func):
            if overload:
                self._registry.setdefault(name, OverloadedFunction(name)).overloads.append(
                    func
                )
            else:
                self._registry.setdefault(name, OverloadedFunction(name)).default = func
            return func

        return wrapper

    def __getitem__(self, name):
        return self._registry[name]

    def __contains__(self, name):
        return name in self._registry

    def __iter__(self):
        return iter(self._registry)

    def __len__(self):
        return len(self._registry)

    def __str__(self):
        return str(self._registry)

    def __repr__(self):
        return repr(self._registry)


# Default registry
default_registry = Registry()


def torch_op(name, overload: bool = False, registry: Optional[Registry] = None):
    """Register a torch op."""
    if registry is None:
        registry = default_registry

    def wrapper(func: Callable[..., Any]) -> onnxscript.OnnxFunction:

        # Compile the function
        compiled = onnxscript.script()(func)

        registry.register(name, overload=overload)(compiled)
        return compiled

    return wrapper
