# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Sequence

import onnx_ir as ir

from onnxscript._internal.builder import GraphBuilder


class Parameter(ir.Value):
    """A module parameter that is also an ``ir.Value``.

    Since ``Parameter`` subclasses ``ir.Value``, it can be passed directly
    to ONNX ops inside ``Module.forward()`` without any conversion.
    Calling :meth:`_realize` qualifies the name with the current module
    context and registers the parameter as a graph initializer.

    Args:
        shape: Shape of the parameter tensor.
        dtype: Data type of the parameter. Defaults to FLOAT.
        name: Name for the parameter. If None, the attribute name from
            the parent Module is used.
        data: Optional initial tensor data. If provided, the initializer
            will carry this as its const_value.
    """

    def __init__(
        self,
        shape: Sequence[int | ir.SymbolicDim | None],
        dtype: ir.DataType = ir.DataType.FLOAT,
        name: str | None = None,
        data: ir.TensorProtocol | None = None,
    ) -> None:
        super().__init__(
            name=name,
            shape=ir.Shape(shape),
            type=ir.TensorType(dtype),
            const_value=data,
        )
        self._realized = False

    @property
    def dtype(self) -> ir.DataType | None:  # type: ignore[override]
        """Return the element data type of this parameter."""
        return self.type.dtype if self.type is not None else None

    def _realize(self, builder: GraphBuilder) -> Parameter:
        """Qualify the name and register as a graph initializer.

        Uses direct assignment to ``graph.initializers[...]`` to skip the
        const_value check. Idempotent: subsequent calls are no-ops.
        """
        if self._realized:
            return self

        if self.name:
            self.name = builder.qualify_name(self.name)
        builder.graph.initializers[self.name] = self  # type: ignore[index]
        self._realized = True
        return self

    def __repr__(self) -> str:
        return f"Parameter(shape={list(self.shape)}, dtype={self.dtype}, name={self.name!r})"
