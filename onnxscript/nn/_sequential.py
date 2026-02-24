# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any

from onnxscript._internal import builder as _builder
from onnxscript.nn import _module_list


class Sequential(_module_list.ModuleList):
    """A sequential container that calls children in order, mirroring ``torch.nn.Sequential``.

    Children are registered with string keys ``"0"``, ``"1"``, etc., just like
    ``ModuleList``. The ``forward`` method passes the output of each child as
    the input to the next.

    Example::

        class SiLU(Module):
            def forward(self, op, x):
                return op.Mul(x, op.Sigmoid(x))

        # Produces parameter names: "mod.0.weight", "mod.0.bias"
        # SiLU at index 0 has no parameters.
        mod = Sequential([SiLU(), Linear(4, 4)])

        # Calling mod(op, x) is equivalent to:
        #   x = silu(op, x)
        #   x = linear(op, x)
    """

    def _set_name(self, name: str) -> None:
        """Set this container's name. Children keep simple ``"0"``, ``"1"`` names.

        Unlike ``ModuleList._set_name`` which fully qualifies children (used
        when ModuleList is iterated externally), Sequential is called via
        ``__call__`` which already pushes its own name onto the builder stack.
        Children must keep simple keys to avoid double-prefixing.
        """
        object.__setattr__(self, "_name", name)
        for key, child in self._modules.items():
            child._set_name(key)

    def _register_child(self, key: str, module: _module_list.Module) -> None:
        """Register a child module under the given string key.

        Unlike ``ModuleList._register_child`` which qualifies the child name
        with the parent name, Sequential keeps children with simple index
        names because ``__call__`` already pushes the Sequential's own name.
        """
        if module._name is None:  # pylint: disable=protected-access
            object.__setattr__(module, "_name", key)
        self._modules[key] = module
        object.__setattr__(self, key, module)

    def forward(self, op: _builder.OpBuilder, *args: Any, **kwargs: Any) -> Any:
        """Run each child module sequentially, passing output to the next."""
        if len(self) == 0:
            raise RuntimeError("Cannot call forward on an empty Sequential container")
        for i, module in enumerate(self):
            if i == 0:
                args = (module(op, *args, **kwargs),)
                kwargs = {}
            else:
                args = (module(op, *args),)
        return args[0]

    def __repr__(self) -> str:
        lines = ["Sequential("]
        for name, module in self._modules.items():
            mod_repr = repr(module).replace("\n", "\n  ")
            lines.append(f"  ({name}): {mod_repr}")
        lines.append(")")
        return "\n".join(lines)
