# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Sequence

import onnx_ir as ir

from onnxscript._internal import builder as _builder


class Parameter(ir.Value):
    """A module parameter that is also an ``ir.Value``.

    Since ``Parameter`` subclasses ``ir.Value``, it can be passed directly
    to ONNX ops inside ``Module.forward()`` without any conversion.
    Calling :meth:`_realize` qualifies the name with the current module
    context and registers the parameter as a graph initializer.

    Args:
        shape: Shape of the parameter tensor.
        dtype: Data type of the parameter. If None and data is not provided, defaults to float32.
        name: Name for the parameter. If None, the attribute name from
            the parent Module is used.
        data: Optional initial tensor data. If provided, the initializer
            will carry this as its const_value.
    """

    def __init__(
        self,
        shape: Sequence[int],
        dtype: ir.DataType | None = None,
        name: str | None = None,
        data: ir.TensorProtocol | None = None,
    ) -> None:
        if data is not None:
            if dtype is not None and data.dtype != dtype:
                raise ValueError(
                    f"Data type of provided data ({data.dtype}) does not match the specified dtype ({dtype})."
                )
            if dtype is None:
                dtype = data.dtype
        elif dtype is None:
            dtype = ir.DataType.FLOAT

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

    def _realize(self, builder: _builder.GraphBuilder) -> Parameter:
        """Qualify the name and register as a graph initializer.

        Uses direct assignment to ``graph.initializers[...]`` to skip the
        const_value check. Idempotent: subsequent calls are no-ops.
        """
        if self._realized:
            return self

        self_name = self.name
        if not self_name:
            raise ValueError(
                "Parameter._realize() called on a Parameter without a name. "
                "Ensure the Parameter is attached to a Module attribute or otherwise "
                "initialized with a name before realization."
            )
        self_name = self.name = builder._qualify_initializer_name(self_name)  # pylint: disable=protected-access
        builder.graph.initializers[self_name] = self
        self._realized = True
        return self

    def __repr__(self) -> str:
        return f"Parameter(shape={list(self.shape)}, dtype={self.dtype}, name={self.name!r})"
