# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any, Iterator

from onnxscript._internal.builder import GraphBuilder, OpBuilder
from onnxscript.nn._parameter import Parameter


class Module:
    """Base class for all onnxscript modules, mirroring PyTorch's nn.Module.

    Subclasses define ``forward()`` to build ONNX subgraphs. Child modules
    and parameters are registered automatically via ``__setattr__``.
    Because ``Parameter`` subclasses ``ir.Value``, parameters like
    ``self.weight`` can be passed directly to ONNX ops.

    Example::

        class Linear(onnxscript.nn.Module):
            def __init__(self, in_features, out_features, bias=True, name=None):
                super().__init__(name)
                self.weight = Parameter([out_features, in_features], name="weight")
                if bias:
                    self.bias = Parameter([out_features], name="bias")
                else:
                    self.bias = None

            def forward(self, op, x):
                w_t = op.Transpose(self.weight, perm=[1, 0])
                result = op.MatMul(x, w_t)
                if self.bias is not None:
                    result = op.Add(result, self.bias)
                return result
    """

    def __init__(self, name: str | None = None) -> None:
        # Use object.__setattr__ to avoid triggering our __setattr__ override
        # before _parameters and _modules dicts exist.
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})

    @property
    def name(self) -> str | None:
        return self._name

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            # Auto-register parameters; set default name from attribute name.
            if value.name is None:
                value.name = name
            self._parameters[name] = value
            # Also store on the instance so getattr works outside forward()
            object.__setattr__(self, name, value)
        elif isinstance(value, Module):
            # Auto-register child modules; inherit attribute name if unnamed.
            if value._name is None:
                object.__setattr__(value, "_name", name)
            self._modules[name] = value
            object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    def __call__(self, op: OpBuilder, *args: Any, **kwargs: Any) -> Any:
        builder: GraphBuilder = op.builder
        module_name = self._name or ""
        builder.push_module(module_name)
        try:
            # Realize parameters: qualify names and register as graph initializers.
            for param in self._parameters.values():
                param.realize(builder)

            result = self.forward(op, *args, **kwargs)
        finally:
            builder.pop_module()
        return result

    def forward(self, op: OpBuilder, *args: Any, **kwargs: Any) -> Any:
        """Define the computation performed by this module.

        Must be overridden by subclasses. Receives an ``OpBuilder`` as the
        first argument so that ONNX ops can be called as ``op.MatMul(x, w)``.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement forward()")

    # ------------------------------------------------------------------
    # Iterators
    # ------------------------------------------------------------------

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Return an iterator over module parameters."""
        yield from self._parameters.values()
        if recurse:
            for module in self._modules.values():
                yield from module.parameters(recurse=True)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        """Return an iterator over module parameters, yielding (name, Parameter) pairs."""
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, param
        if recurse:
            for mod_name, module in self._modules.items():
                sub_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
                yield from module.named_parameters(prefix=sub_prefix, recurse=True)

    def modules(self) -> Iterator[Module]:
        """Return an iterator over all modules in the tree (including self)."""
        yield self
        for module in self._modules.values():
            yield from module.modules()

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, Module]]:
        """Return an iterator over all modules, yielding (name, Module) pairs."""
        yield prefix, self
        for name, module in self._modules.items():
            sub_prefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_modules(prefix=sub_prefix)

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}("]
        for name, module in self._modules.items():
            mod_repr = repr(module).replace("\n", "\n  ")
            lines.append(f"  ({name}): {mod_repr}")
        for name, param in self._parameters.items():
            lines.append(f"  ({name}): {param!r}")
        lines.append(")")
        return "\n".join(lines)
