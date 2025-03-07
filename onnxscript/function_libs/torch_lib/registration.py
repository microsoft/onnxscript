# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Registry for aten functions."""

from __future__ import annotations

import re
from typing import Any, Callable, Generator, Optional

import onnxscript
from onnxscript.function_libs.torch_lib import _constants

# Regex that will match "<namespace>::<op_name>[.<overload>]"
_QUALIFIED_OPERATOR_NAME_REGEX = re.compile(
    r"^(?P<namespace>[a-zA-Z0-9_]+)::(?P<name>[a-zA-Z0-9_]+)(?P<overload>\.[a-zA-Z0-9._]+)?$"
)


class OverloadedFunction:
    """Overloaded function.

    Attributes:
        name: Name of the op. E.g. "aten::add".
        overloads: Overloads function.
        privates: Private functions not exposed to users.
        complex: Support complex functions.
    """

    def __init__(self, name: str):
        self.name = name
        self.overloads: list[Any] = []
        self.privates: list[Any] = []
        self.complex: list[Any] = []


class Registry:
    """Registry for aten functions."""

    def __init__(self):
        self._registry: dict[str, OverloadedFunction] = {}

    def register(
        self, func: Any, name: str, *, private: bool = False, complex: bool = False
    ) -> None:
        """Register a function."""

        if private:
            self._registry.setdefault(name, OverloadedFunction(name)).privates.append(func)
        elif complex:
            self._registry.setdefault(name, OverloadedFunction(name)).complex.append(func)
        else:
            self._registry.setdefault(name, OverloadedFunction(name)).overloads.append(func)

    def __getitem__(self, name):
        return self._registry[name]

    def __contains__(self, name):
        return name in self._registry

    def __iter__(self):
        return iter(self._registry)

    def __repr__(self):
        return repr(self._registry)

    def items(self) -> Generator[tuple[str, OverloadedFunction], None, None]:
        yield from self._registry.items()

    def values(self) -> Generator[OverloadedFunction, None, None]:
        yield from self._registry.values()


# Default registry
default_registry = Registry()


def _check_and_normalize_names(name: str | tuple[str, ...]) -> tuple[str, ...]:
    names: tuple[str, ...]

    if isinstance(name, str):
        names = (name,)
    else:
        names = name
    if not isinstance(names, tuple):
        raise TypeError(f"Name must be a string or a tuple of strings, got {name}")
    for name_ in names:
        if name_.endswith(".default") or not _QUALIFIED_OPERATOR_NAME_REGEX.fullmatch(name_):
            raise ValueError(
                f"Invalid name '{name_}'. Must be in the form 'namespace::name' for default overloads "
                "or 'namespace::name.overload' for other overloads."
            )

    return names


def torch_op(
    name: str | tuple[str, ...],
    *,
    registry: Optional[Registry] = None,
    trace_only: bool = False,
    private: bool = False,
    complex: bool = False,
) -> Callable[[Callable], onnxscript.OnnxFunction | onnxscript.values.TracedOnnxFunction]:
    """Register a torch op.

    Args:
        name: Qualified ATen name of the function. E.g. "aten::relu", "aten::add.Tensor".
            Or a tuple of names e.g. ("aten::add.Scalar", "aten::add.Tensor").
            Default overloads should be specified by omitting the overload part,
            i.e. "aten::relu" instead of "aten::relu.default".
        registry: Registry to register the function to. If None, the default registry is used.
        trace_only: Whether the function should only be traced and not compiled.
        private: Whether the function is private (not directly exposed). It should
            be true for all functions with names starting with "_".
        complex: Whether the function expects complex-valued inputs.
    """
    if registry is None:
        registry = default_registry

    def wrapper(
        func: Callable,
    ) -> onnxscript.OnnxFunction | onnxscript.values.TracedOnnxFunction:
        # Compile the function
        custom_opset = onnxscript.values.Opset(domain=_constants.DOMAIN, version=1)

        processed_func: onnxscript.OnnxFunction | onnxscript.values.TracedOnnxFunction
        if trace_only:
            processed_func = onnxscript.values.TracedOnnxFunction(custom_opset, func)
        else:
            processed_func = onnxscript.script(opset=custom_opset)(func)

        assert registry is not None
        for name_ in _check_and_normalize_names(name):
            registry.register(processed_func, name_, private=private, complex=complex)
        return processed_func

    return wrapper
