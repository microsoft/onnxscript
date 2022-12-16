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

    def register(self, func: Any, name: str, *, overload: bool = False) -> None:
        """Register a function."""

        if overload:
            self._registry.setdefault(name, OverloadedFunction(name)).overloads.append(func)
        else:
            self._registry.setdefault(name, OverloadedFunction(name)).default = func

    def __getitem__(self, name):
        return self._registry[name]

    def __contains__(self, name):
        return name in self._registry

    def __iter__(self):
        return iter(self._registry)

    def __repr__(self):
        return repr(self._registry)


# Default registry
default_registry = Registry()


def torch_op(
    name, overload: bool = False, registry: Optional[Registry] = None
) -> Callable[[Callable[..., Any]], onnxscript.OnnxFunction]:
    """Register a torch op."""
    if registry is None:
        registry = default_registry

    def wrapper(func: Callable[..., Any]) -> onnxscript.OnnxFunction:

        # Compile the function
        compiled = onnxscript.script()(func)

        assert registry is not None
        registry.register(compiled, name, overload=overload)
        return compiled

    return wrapper
