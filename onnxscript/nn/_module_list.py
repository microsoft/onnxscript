# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any, Iterator, overload

from onnxscript._internal import builder as _builder
from onnxscript.nn._module import Module


class ModuleList(Module):
    """Holds child modules in a list, mirroring ``torch.nn.ModuleList``.

    Children are registered with string keys ``"0"``, ``"1"``, etc., so that
    hierarchical parameter names use ``.0.``, ``.1.`` separators just like
    PyTorch.

    Example::

        class MyModel(Module):
            def __init__(self):
                super().__init__("model")
                self.layers = ModuleList([Linear(4, 4) for _ in range(3)])

            def forward(self, op, x):
                for layer in self.layers:
                    x = layer(op, x)
                return x

        # Parameters will be named:
        #   model.layers.0.weight, model.layers.1.weight, model.layers.2.weight
    """

    def __init__(self, modules: list[Module] | None = None) -> None:
        super().__init__()
        if modules is not None:
            for idx, module in enumerate(modules):
                self._register_child(str(idx), module)

    def _set_name(self, name: str) -> None:
        """Set this container's name and prefix all children's names."""
        object.__setattr__(self, "_name", name)
        for key, child in self._modules.items():
            child._set_name(f"{name}.{key}")  # pylint: disable=protected-access

    def _register_child(self, key: str, module: Module) -> None:
        """Register a child module under the given string key."""
        if module._name is None:  # pylint: disable=protected-access
            object.__setattr__(module, "_name", key)
        self._modules[key] = module
        object.__setattr__(self, key, module)

    def append(self, module: Module) -> ModuleList:
        """Append a module to the end of the list."""
        key = str(len(self._modules))
        self._register_child(key, module)
        return self

    def extend(self, modules: list[Module]) -> ModuleList:
        """Append modules from an iterable to the end of the list."""
        for module in modules:
            self.append(module)
        return self

    @overload
    def __getitem__(self, idx: int) -> Module: ...

    @overload
    def __getitem__(self, idx: slice) -> ModuleList: ...

    def __getitem__(self, idx: int | slice) -> Module | ModuleList:
        if isinstance(idx, slice):
            keys = list(self._modules.keys())[idx]
            new_list = ModuleList()
            for i, key in enumerate(keys):
                new_list._register_child(str(i), self._modules[key])
            return new_list
        if idx < 0:
            idx += len(self._modules)
        key = str(idx)
        if key not in self._modules:
            raise IndexError(f"index {idx} is out of range")
        return self._modules[key]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def forward(self, op: _builder.OpBuilder, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "ModuleList is not callable directly. "
            "Iterate over its children and call them individually."
        )
